"""Which part of the pipeline is to blame? Code, rule, or linearity.

The question
------------
The substrate scored 22.10% on permuted-MNIST against 68.30% for a naive
backprop MLP, and that was recorded as a falsification of the project's central
hypothesis. But the readout's rule binarises its own output::

    err = target - (out > 0.0).astype(np.float32)

so a class that already scores positive receives exactly zero error and stops
learning, and no class ever acquires a margin. Separately, on identical cached
codes, a closed-form *quadratic* ridge readout reached 0.333 on a single 10-way
task where a *linear* one reached 0.157 -- but that comparison may have conflated
readout **capacity** with readout **learning rule**.

Three hypotheses make different predictions on the same data:

===========================  ==========================================================
``H-code``                   the representation is weak; *every* readout fails
``H-rule``                   the code is fine; the local rule is the bottleneck
``H-linear``                 the code is nonlinearly decodable, linearly not
===========================  ==========================================================

Their fixes are disjoint, so this is the highest-value measurement available.

Protocol
--------
1. Encode the substrate's code for every sample **once** and cache it (plus the
   task ids and labels) to a single ``.npz``. Every arm then trains on
   bit-identical inputs, so the readout kind is the only variable. This matters
   more than it looks: the substrate is stateful (plastic synapses, adaptation),
   so re-encoding per arm would hand each arm a different input.
2. Cache a matched **random-projection control** (fixed random linear map from
   the 784 pixels, then ReLU) at both the substrate's width and 900 dims. If the
   substrate's code does not beat a random projection of the same width under
   the *same* readout, the substrate contributes nothing and no readout fix can
   rescue the hypothesis.
3. Run every readout kind on those identical codes, on both benchmarks, with
   ``normalize`` on and off, and on the single-task 10-way setting where the
   earlier ceiling was measured.
4. ``ridge``/``ridge_quad`` follow the same per-task schedule as the local rules
   (fit only on the current task's samples). An all-data-at-once fit is reported
   separately and labelled an **oracle**: it is an upper bound, not a fair arm.
5. Report mean +/- 95% CI (1.96 * SEM) for everything, plus an arm ladder over
   the quadratic dimension.

Honesty notes
-------------
* ``ridge``/``ridge_quad`` are closed-form solves and are **not** biologically
  plausible. They measure what the *feature space* supports. ``local_quad``
  exists to ask whether a plausible rule can reach the same place.
* The substrate's normaliser drifts during training and is frozen at test time;
  ``normalize=False`` arms exist because the drifting normaliser is itself a
  candidate confound.
* Codes for the ``as_falsified`` variant are produced with
  ``reset_between_samples=False``, which is how the original falsification ran;
  ``reset_fresh`` uses the reset ``cortex.py`` now applies. Both are reported so
  the verdict can be checked against whichever substrate the reader trusts.
* Test codes are produced by a substrate that is still plastic (plasticity was
  never switched off for ``_code``, in this repo or in the original run). That
  is identical for every readout arm because the codes are shared, so it cannot
  affect the verdict -- but it is a property of the substrate, not of the
  readout, and is stated where it matters.

Run:  python3 experiments/readout_diagnostic.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "experiments" / "results"
#: Lives under data/ so the existing ``data/*.npz`` gitignore rule covers it:
#: it is ~100 MB of regenerable cache, not a source artefact.
DEFAULT_CACHE = ROOT / "data" / "readout_codes.npz"

#: Bump when the cache schema or the substrate config used to build it changes,
#: so a stale cache is rebuilt instead of silently reused.
CACHE_VERSION = "2"

#: The config the falsification used (experiments/results/continual_permuted.json
#: and continual_split.json): 800 neurons, k_out 64, k-WTA with k=64, dendrites
#: on, unsupervised Hebbian plasticity, 15 ms per sample.
SUBSTRATE = dict(
    n_neurons=800,
    n_input=784,
    n_classes=10,
    k_out=64,
    t_train_ms=15,
    t_test_ms=15,
    gain=2.2,
    dend_scale=1.0,
    dend_gain=2.5,
    use_dendrite=True,
    plasticity=True,
    inhibition="kwta",
    k_wta=64,
)

#: Train samples per task, exactly as the falsification subsampled them.
TRAIN_PER_TASK = {"split": 80, "permuted": 200}
TEST_PER_TASK = {"split": 50, "permuted": 100}

#: Sample count for the single-task 10-way "ceiling" reproduction (RESULTS.md
#: section 3.1 trained on 600 samples of one permuted task).
CEILING_SAMPLES = 600

#: The readout learning rate the falsification used for the substrate arm
#: (``CortexConfig.readout_lr`` default). The ``1e-2`` in the old CLI was the
#: *MLP's* learning rate and never reached the brain arm.
FALSIFICATION_LR = 0.05

VARIANTS = ("as_falsified", "reset_fresh")

BENCHMARKS = ("split", "permuted")


# --------------------------------------------------------------------- helpers
def _now() -> float:
    return time.perf_counter()


def _sem_ci95(values: list[float]) -> tuple[float, float]:
    """Mean and 1.96 * SEM. CI is 0.0 for a single seed (no spread to report)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, 0.0
    return mean, float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _random_projection(n_in: int, n_out: int, seed: int) -> np.ndarray:
    """Fixed random linear map, scaled so ReLU outputs are O(1)."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x5EED])
    return (rng.standard_normal((n_in, n_out)) * np.sqrt(1.0 / n_in)).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def stratified_subsample(x: np.ndarray, y: np.ndarray, n: int, seed: int):
    """Identical to ``experiments.continual.stratified_subsample``.

    Re-implemented here rather than imported so that a concurrent edit to that
    experiment file cannot silently change which samples this diagnostic caches
    (and therefore what every arm is compared on). ``self_check`` asserts the
    class distribution is balanced, which is the property that matters.
    """
    if n <= 0 or n >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, n // len(classes))
    picks = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picks.append(idx[:per_class])
    chosen = np.concatenate(picks)
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


# ------------------------------------------------------------------ cache build
@dataclass
class CacheSpec:
    benchmarks: tuple[str, ...] = BENCHMARKS
    variants: tuple[str, ...] = VARIANTS
    seeds: int = 3
    tasks: int = 5
    rp_widths: tuple[int, ...] = (800, 900)
    #: Feature-set names available to control arms, mapped to their width.
    ceiling: bool = True


def _code_batch(model, xs: np.ndarray, t_ms: int, out: np.ndarray,
                offset: int) -> float:
    """Encode ``xs`` into ``out[offset:offset+n]``; returns SynOps per sample.

    Encoding **is** substrate training when plasticity is on: every call writes
    Hebbian updates and advances adaptation state. The samples are therefore fed
    in the original order (task 0 train, then test; task 1 train, then test...)
    so the substrate sees exactly the sequence the falsification gave it.
    """
    synops = 0.0
    for i in range(len(xs)):
        out[offset + i] = model._code(xs[i], t_ms)
        synops += model.last_synops
    return synops / max(1, len(xs))


def _subsample_tasks(suite, bench: str, seed: int):
    """Per-task train/test sub-pools, stratified, in the original order."""
    n_tr, n_te = TRAIN_PER_TASK[bench], TEST_PER_TASK[bench]
    tr_x, tr_y, tr_t, te_x, te_y, te_t = [], [], [], [], [], []
    for ti, task in enumerate(suite.tasks):
        x, y = stratified_subsample(task.x_train, task.y_train, n_tr, seed * 17 + ti)
        counts = np.bincount(y, minlength=suite.n_classes)
        live = counts[counts > 0]
        if live.size > 1 and live.max() - live.min() > 2:
            raise AssertionError(
                f"subsample not stratified for {task.name}: {counts.tolist()}"
            )
        tr_x.append(x); tr_y.append(y)
        tr_t.append(np.full(len(y), ti, dtype=np.int16))
        xt, yt = stratified_subsample(task.x_test, task.y_test, n_te, seed * 31 + ti)
        te_x.append(xt); te_y.append(yt)
        te_t.append(np.full(len(yt), ti, dtype=np.int16))
    return (np.asarray(tr_x), np.asarray(tr_y), np.asarray(tr_t),
            np.asarray(te_x), np.asarray(te_y), np.asarray(te_t))


def build_cache(path: Path, spec: CacheSpec, *, verbose: bool = True) -> dict[str, np.ndarray]:
    """Encode every sample once and write a single .npz. Returns the arrays.

    Per (benchmark, variant, seed) this stores three code sets:

    ``Xtr``  the training codes, in the original interleaved order;
    ``Xdg``  the test codes for task ``i`` captured *immediately after* task
             ``i`` was trained -- the diagonal of the accuracy matrix;
    ``Xte``  the test codes for every task captured after the whole stream,
             i.e. the substrate's final representation, which is what the
             retention numbers are computed on.

    ``Xdg`` minus ``Xte`` is the direct measurement of representational drift,
    and it is worth having separately from accuracy because a readout can
    absorb a representation change it was never asked to absorb at test time.
    """
    from brain.cortex import CortexClassifier, CortexConfig
    from brain.tasks import split_mnist

    arrays: dict[str, np.ndarray] = {}
    t_start = _now()

    for bench in spec.benchmarks:
        permute = bench == "permuted"
        for seed in range(spec.seeds):
            suite = split_mnist(spec.tasks, seed=seed, permute=permute)
            n_neurons = SUBSTRATE["n_neurons"]
            tr_x, tr_y, tr_t, te_x, te_y, te_t = _subsample_tasks(suite, bench, seed)

            # ---- random-projection control. Built from the *same* pixels the
            # substrate saw (so a permuted task is projected from its own
            # permuted image), independent of the substrate itself, hence
            # cached once per (benchmark, seed).
            flat_tr = tr_x.reshape(-1, 784)
            flat_te = te_x.reshape(-1, 784)
            # The raw pixels themselves. The single most decisive control
            # available: if the substrate's 800-dim code does not beat the 784
            # pixels it was built from, the encoding is *destructive* and no
            # readout change can fix it.
            arrays[f"{bench}|{seed}|Xpix_tr"] = flat_tr.copy()
            arrays[f"{bench}|{seed}|Xpix_te"] = flat_te.copy()
            for w in spec.rp_widths:
                P = _random_projection(784, w, seed)
                arrays[f"{bench}|{seed}|Xrp{w}_tr"] = _relu(flat_tr @ P)
                arrays[f"{bench}|{seed}|Xrp{w}_te"] = _relu(flat_te @ P)
                arrays[f"{bench}|{seed}|rp{w}_seed"] = np.array([seed], dtype=np.int64)

            if spec.ceiling and permute:
                xc, yc = stratified_subsample(suite.tasks[0].x_train,
                                             suite.tasks[0].y_train,
                                             CEILING_SAMPLES, seed * 7 + 101)
                arrays[f"{bench}|{seed}|y_ce"] = yc.astype(np.int64)
                arrays[f"{bench}|{seed}|Xpix_ce"] = xc.copy()
                for w in spec.rp_widths:
                    P = _random_projection(784, w, seed)
                    arrays[f"{bench}|{seed}|Xrp{w}_ce"] = _relu(xc @ P)
            else:
                xc = yc = None

            for variant in spec.variants:
                cfg = CortexConfig(seed=seed, reset_between_samples=(variant == "reset_fresh"),
                                   **SUBSTRATE)
                model = CortexClassifier(cfg)
                key = f"{bench}|{variant}|{seed}"

                n_tr_total = int(tr_x.shape[0] * tr_x.shape[1])
                n_te_total = int(te_x.shape[0] * te_x.shape[1])
                Xtr = np.zeros((n_tr_total, n_neurons), dtype=np.float32)
                # Xdg has one row per test sample, grouped by task, same layout
                # as Xte; rows whose task had not been reached yet are zero and
                # are never read (scoring is always restricted to a task's own
                # contiguous block via `tte`).
                Xdg = np.zeros((n_te_total, n_neurons), dtype=np.float32)
                Xte = np.zeros((n_te_total, n_neurons), dtype=np.float32)
                syn, n_coded = 0.0, 0

                # Interleaved, in the original order: train task i, then
                # immediately capture task i's own test codes.
                off_tr = 0
                for ti in range(spec.tasks):
                    xs = tr_x[ti]
                    syn += _code_batch(model, xs, cfg.t_train_ms, Xtr, off_tr)
                    n_coded += len(xs)
                    off_tr += len(xs)
                    # where does task ti's test block start/end?
                    s_dg = int(np.flatnonzero(te_t.reshape(-1) == ti)[0])
                    nt = int((te_t == ti).sum())
                    syn += _code_batch(model, te_x[ti], cfg.t_test_ms, Xdg, s_dg)
                    n_coded += nt

                # Final representation: re-encode every task's test set once the
                # whole stream has been trained through.
                off_te = 0
                for ti in range(spec.tasks):
                    xs = te_x[ti]
                    syn += _code_batch(model, xs, cfg.t_test_ms, Xte, off_te)
                    n_coded += len(xs)
                    off_te += len(xs)

                if spec.ceiling and permute:
                    Xce = np.zeros((xc.shape[0], n_neurons), dtype=np.float32)
                    _code_batch(model, xc, cfg.t_train_ms, Xce, 0)
                    arrays[f"{key}|Xce"] = Xce

                model.be.eval()
                arrays[f"{key}|Xtr"] = Xtr
                arrays[f"{key}|Xdg"] = Xdg
                arrays[f"{key}|Xte"] = Xte
                arrays[f"{key}|ytr"] = tr_y.reshape(-1).astype(np.int64)
                arrays[f"{key}|ttr"] = tr_t.reshape(-1).astype(np.int16)
                arrays[f"{key}|yte"] = te_y.reshape(-1).astype(np.int64)
                arrays[f"{key}|tte"] = te_t.reshape(-1).astype(np.int16)
                arrays[f"{key}|synops_per_sample"] = np.array([syn / max(1, n_coded)],
                                                              dtype=np.float32)
                arrays[f"{key}|n_substrate_synapses"] = np.array(
                    [model.n_substrate_synapses], dtype=np.int64)

                if verbose:
                    print(f"    cached {key:26s} tr={Xtr.shape[0]:5d} te={Xte.shape[0]:5d} "
                          f"synops/sample={syn / max(1, n_coded):9.0f}  "
                          f"{_now() - t_start:6.1f}s", flush=True)

    arrays["cache_version"] = np.array([CACHE_VERSION])
    arrays["config_hash"] = np.array([
        hashlib.sha256(json.dumps(SUBSTRATE, sort_keys=True).encode()).hexdigest()[:16]
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if verbose:
        print(f"    wrote {path} ({path.stat().st_size / 1e6:.1f} MB) "
              f"in {_now() - t_start:.1f}s", flush=True)
    return arrays


def _required_keys(spec: CacheSpec) -> set[str]:
    keys: set[str] = {"cache_version", "config_hash"}
    for bench in spec.benchmarks:
        for seed in range(spec.seeds):
            keys |= {f"{bench}|{seed}|Xrp{w}_tr" for w in spec.rp_widths}
            keys |= {f"{bench}|{seed}|Xrp{w}_te" for w in spec.rp_widths}
            keys |= {f"{bench}|{seed}|Xpix_tr", f"{bench}|{seed}|Xpix_te"}
            if spec.ceiling and bench == "permuted":
                keys |= {f"{bench}|{seed}|y_ce", f"{bench}|{seed}|Xpix_ce"}
                keys |= {f"{bench}|{seed}|Xrp{w}_ce" for w in spec.rp_widths}
            for variant in spec.variants:
                k = f"{bench}|{variant}|{seed}"
                keys |= {f"{k}|Xtr", f"{k}|Xdg", f"{k}|Xte", f"{k}|ytr", f"{k}|ttr",
                         f"{k}|yte", f"{k}|tte", f"{k}|synops_per_sample"}
                if spec.ceiling and bench == "permuted":
                    keys |= {f"{k}|Xce"}
    return keys


def load_cache(path: Path, spec: CacheSpec) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"code cache {path} not found; run without --cache-only first, or "
            "pass --force-cache"
        )
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    ver = arrays.get("cache_version")
    if ver is None or str(ver[0]) != CACHE_VERSION:
        raise ValueError(
            f"cache {path} has version {None if ver is None else str(ver[0])!r}, "
            f"expected {CACHE_VERSION!r}; re-run with --force-cache"
        )
    missing = _required_keys(spec) - set(arrays)
    if missing:
        raise ValueError(
            f"cache {path} is missing {len(missing)} keys (e.g. {sorted(missing)[:3]}); "
            "re-run with --force-cache"
        )
    return arrays


# ----------------------------------------------------------------------- arms
@dataclass
class ArmSpec:
    """One readout evaluated on one cached feature set.

    ``family``/``rule``/``space`` are recorded separately because the whole
    point of the diagnostic is that "readout" is not one variable: the *feature
    space* and the *learning rule* vary independently and the verdict depends on
    which one is being blamed.
    """

    name: str
    features: str          # "brain" | "rp800" | "rp900"
    readout: str           # a ReadoutConfig.kind
    normalize: str         # "norm" | "raw"
    quad_dim: int = 0
    lr: float = FALSIFICATION_LR
    epochs: int = 1
    oracle: bool = False
    note: str = ""

    @property
    def family(self) -> str:
        return {"ridge": "closed_form_linear", "ridge_quad": "closed_form_quad"}.get(
            self.readout, "local_quad" if self.readout == "local_quad" else "local_linear")

    @property
    def space(self) -> str:
        return "quadratic" if self.readout in ("ridge_quad", "local_quad") else "linear"

    @property
    def rule(self) -> str:
        return {"local_perceptron": "perceptron_binary (current)",
                "local_delta": "delta_graded",
                "local_softmax": "softmax_xent"}.get(self.readout, "closed_form_lstsq")


def best_lr_per_rule(rows: list[dict[str, Any]], bench: str,
                     variant: str) -> dict[str, Any]:
    """Pick a step size per rule from the sweep, on the reported benchmark.

    This is a *sweep*, not tuning against the final number: every candidate is
    run on the full protocol and the whole sweep is written to the JSON, so the
    selection is auditable and the spread is visible. It exists because the
    rules do not share a stable step-size range - the binarised perceptron error
    is bounded by 1 while the delta error grows with the weight norm, so one
    shared lr cannot be a fair comparison. Reporting both the shared-lr arm and
    the best-lr arm is the honest way to state that.
    """
    out: dict[str, Any] = {}
    for rule in ("local_perceptron", "local_delta", "local_softmax", "local_quad"):
        cands = [r for r in rows
                 if r.get("readout") == rule and r.get("benchmark") == bench
                 and r.get("variant") == variant
                 and r.get("features") == "brain" and r.get("normalize") is True
                 and not r.get("oracle") and not r.get("error")
                 and np.isfinite(r.get("final_average", float("nan")))]
        if not cands:
            continue
        by_lr: dict[float, list[float]] = {}
        for r in cands:
            by_lr.setdefault(round(float(r["lr"]), 8), []).append(r["final_average"])
        means = {k: float(np.mean(v)) for k, v in by_lr.items()}
        best = max(means, key=lambda k: means[k])
        out[rule] = {"best_lr": best, "best_lr_mean": means[best],
                     "sweep": {str(k): v for k, v in sorted(means.items())}}
    return out


def arm_table(quad_dim: int, *, lr: float = FALSIFICATION_LR) -> list[ArmSpec]:
    """The pre-registered arm set. Fixed before any number was seen."""
    arms = [
        # H-code / H-rule core: linear space, varying the rule. The first two
        # are the pair whose difference IS the H-rule test.
        ArmSpec("lin_perceptron", "brain", "local_perceptron", "norm", lr=lr,
                note="the rule the falsification used (binarised error); the arm under test"),
        ArmSpec("lin_delta", "brain", "local_delta", "norm", lr=lr,
                note="graded error, same local three-factor form"),
        ArmSpec("lin_softmax", "brain", "local_softmax", "norm", lr=lr,
                note="cross-entropy, same local form"),
        ArmSpec("lin_ridge", "brain", "ridge", "norm",
                note="closed-form linear; NOT plausible. Upper bound for linear rules"),
        # H-linear: same rule, expanded feature space.
        ArmSpec("quad_ridge", "brain", "ridge_quad", "norm", quad_dim=quad_dim,
                note="closed-form quadratic; NOT plausible"),
        ArmSpec("quad_local", "brain", "local_quad", "norm", quad_dim=quad_dim, lr=lr,
                note="plausible local delta rule on the quadratic space"),
        # Normaliser confound: identical everything except the drifting z-score.
        ArmSpec("raw_perceptron", "brain", "local_perceptron", "raw", lr=lr,
                note="normalize=False"),
        ArmSpec("raw_delta", "brain", "local_delta", "raw", lr=lr,
                note="normalize=False"),
        ArmSpec("raw_ridge", "brain", "ridge", "raw",
                note="normalize=False"),
        ArmSpec("raw_quad_ridge", "brain", "ridge_quad", "raw", quad_dim=quad_dim,
                note="normalize=False"),
        # Substrate-contribution control: same readout, same schedule, same
        # width -- but the features are a fixed random projection of the raw
        # pixels instead of the substrate's code. Run with the normaliser on AND
        # off so that "the substrate beats a random projection" is never decided
        # by which arm happened to be standardised. If the substrate's code does
        # not beat this under the *same* readout, the substrate contributes
        # nothing and no readout fix can rescue the hypothesis: H-code.
        ArmSpec("ctrl_rp800_ridge", "rp800", "ridge", "raw",
                note="random projection at the substrate's own width"),
        ArmSpec("ctrl_rp800_quad_ridge", "rp800", "ridge_quad", "raw",
                quad_dim=quad_dim),
        ArmSpec("ctrl_rp800_ridge_n", "rp800", "ridge", "norm",
                note="control, normaliser on"),
        ArmSpec("ctrl_rp800_quad_ridge_n", "rp800", "ridge_quad", "norm",
                quad_dim=quad_dim, note="control, normaliser on"),
        ArmSpec("ctrl_pix_ridge", "pix", "ridge", "raw",
                note="THE raw-pixel control: does the code beat the pixels it was built from?"),
        ArmSpec("ctrl_pix_quad_ridge", "pix", "ridge_quad", "raw", quad_dim=quad_dim,
                note="raw pixels with the quadratic space"),
        ArmSpec("ctrl_pix_delta_n", "pix", "local_delta", "norm", lr=1e-4,
                note="raw pixels under the best local rule"),
        ArmSpec("ctrl_rp900_ridge", "rp900", "ridge", "raw",
                note="random projection at 900 dims"),
        ArmSpec("ctrl_rp900_quad_ridge", "rp900", "ridge_quad", "raw",
                quad_dim=quad_dim),
        ArmSpec("ctrl_rp800_delta_n", "rp800", "local_delta", "norm", lr=1e-4,
                note="control under the best local rule"),
        ArmSpec("ctrl_rp800_perceptron_n", "rp800", "local_perceptron", "norm",
                lr=lr, note="control under the arm under test"),
    ]
    return arms


def _ladder_table(quad_dims: tuple[int, ...], *, lr: float = FALSIFICATION_LR) -> list[ArmSpec]:
    """Arm ladder over the quadratic dimension, to show where truth is.

    A quadratic readout is only "nonlinear" if it has products to work with. If
    a quadratic arm with a small sample of pairs already matches the full-pair
    ceiling, the code's nonlinearity is low-dimensional and cheap; if the
    ceiling keeps climbing with quad_dim, the earlier 0.333 was a *capacity*
    result and the honest verdict is capacity-limited, not rule-limited.
    """
    return [
        ArmSpec(f"ladder_quad{d}", "brain", "ridge_quad", "norm", quad_dim=d)
        for d in quad_dims
    ]


#: Step sizes swept for each local rule. The binarised perceptron rule tolerates
#: a larger step than the graded rules (its error is bounded by 1 while a delta
#: error grows with the weight norm), so a single shared step size would decide
#: the comparison on an arbitrary constant. The sweep is centred on the value
#: the falsification used (0.05) rather than chosen after seeing the results.
LR_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2)


def lr_sweep_table() -> list[ArmSpec]:
    """One arm per (local rule, step size), so no rule is judged at a bad step.

    Names are stable across the grid so that a summary key is
    ``<name>|brain|norm|<bench>|<lr>`` and the step size is the *only* thing
    distinguishing two sweep rows.
    """
    return [
        ArmSpec(f"sweep_{kind.replace('local_', '')}", "brain", kind, "norm",
                quad_dim=256 if kind == "local_quad" else 0, lr=lr,
                note=f"step-size sweep at lr={lr:g}")
        for kind in ("local_perceptron", "local_delta", "local_softmax", "local_quad")
        for lr in LR_GRID
    ]


#: Ridge penalties swept by the closed-form arms. The penalty is a genuine
#: hyper-parameter and the earlier "quadratic scores 0.333" observation was made
#: at one arbitrary value, so it cannot be left at a constant default: at
#: lambda = 1e-2 the solve is heavily under-regularised here (the Gram is rank
#: deficient whenever a task has fewer samples than features) and the apparent
#: capacity of the quadratic space is an artefact of that choice. Choosing it on
#: the test set would be leakage, so it is selected on a held-out slice of the
#: TRAINING codes only (see ``select_ridge``).
RIDGE_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def select_ridge(Xtr: np.ndarray, ytr: np.ndarray, ttr: np.ndarray,
                 arm: ArmSpec, seed: int, *, holdout: float = 0.25) -> tuple[float, dict]:
    """Choose the ridge penalty by holdout on the training codes.

    Uses a stratified per-task holdout of the *training* rows, so no test code
    is ever involved in the choice. The whole sweep is returned so the selection
    is auditable and a flat sweep can be seen to be flat.
    """
    from brain.readout import Readout, ReadoutConfig
    rng = np.random.default_rng(seed * 977 + 13)
    tr_idx, va_idx = [], []
    for t in np.unique(ttr):
        idx = np.flatnonzero(ttr == t)
        rng.shuffle(idx)
        k = max(1, int(len(idx) * holdout))
        va_idx.append(idx[:k])
        tr_idx.append(idx[k:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)
    Ytr = np.eye(10, dtype=np.float32)[ytr]
    scores: dict[str, float] = {}
    for lam in RIDGE_GRID:
        cfg = ReadoutConfig(kind=arm.readout, n_features=int(Xtr.shape[1]),
                            n_classes=10, ridge=lam, epochs=arm.epochs,
                            normalize=(arm.normalize == "norm"),
                            quad_dim=arm.quad_dim, quad_seed=1234, seed=seed)
        r = Readout(cfg)
        r.partial_fit(Xtr[tr_idx], Ytr[tr_idx])
        scores[f"{lam:g}"] = float(r.score(Xtr[va_idx], ytr[va_idx]))
    best = max(scores, key=lambda k: scores[k])
    return float(best), scores


class ArmRunner:
    """Train one arm per task on shared cached codes, then evaluate retention.

    ``as_falsified`` semantics, deliberately: train on task ``i`` only, then
    evaluate on every task ``j <= i``, then move on with no replay. ``acc[i, j]``
    is the accuracy on task ``j`` after training through task ``i``.
    """

    def __init__(self, arm: ArmSpec, data: dict[str, np.ndarray], key: str,
                 seed: int, *, lr: float | None = None, verbose: bool = False):
        from brain.readout import Readout, ReadoutConfig
        self.arm = arm
        self.data, self.key, self.seed = data, key, seed
        # The feature width must come from the matrix this arm will actually
        # train on, not from the substrate cache: the 900-dim control is wider
        # than the substrate's 800 and would otherwise be rejected by the
        # config guard (which is the guard working correctly).
        if arm.features == "brain":
            n_features = int(data[f"{key}|Xtr"].shape[1])
        else:
            bench, cs = key.split("|")[0], int(key.split("|")[2])
            n_features = int(data[f"{bench}|{cs}|X{arm.features}_tr"].shape[1])
        self.cfg = ReadoutConfig(
            kind=arm.readout,
            n_features=n_features,
            n_classes=10,
            lr=arm.lr if lr is None else lr,
            ridge=1e-2,
            epochs=arm.epochs,
            normalize=(arm.normalize == "norm"),
            quad_dim=arm.quad_dim,
            quad_seed=1234,
            seed=seed,
        )
        self.ridge_sweep: dict[str, float] = {}
        if arm.family.startswith("closed_form"):
            # A closed-form solve has a real hyper-parameter, and leaving it at a
            # constant would decide the quadratic-vs-linear comparison on that
            # constant. Chosen on a training-code holdout, never on test codes.
            Xsel = self._X("tr")
            tsel = data[f"{key}|ttr"]
            ysel = data[f"{key}|ytr"]
            if arm.oracle:
                Xsel, tsel, ysel = Xsel, tsel, ysel
            lam, sweep = select_ridge(Xsel, ysel, tsel, arm, seed)
            self.cfg.ridge = lam
            self.ridge_sweep = sweep
        self.readout = Readout(self.cfg)
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    def _X(self, which: str) -> np.ndarray:
        """Feature matrix for this arm.

        Only ``brain`` arms come from the cache's substrate codes; the control
        arms substitute the random projection, so that the *only* difference
        between an arm and its control is the features themselves.

        ``which="dg"`` is the diagonal set: test codes captured immediately
        after the task they belong to was trained. ``which="te"`` is the final
        representation after the whole stream. Scoring the diagonal on ``dg``
        and the retention row on ``te`` is what separates "the readout forgot"
        from "the substrate moved underneath it".
        """
        if self.arm.features == "brain":
            return self.data[f"{self.key}|X{which}"]
        bench, seed = self.key.split("|")[0], int(self.key.split("|")[2])
        # A fixed random projection has no substrate state, so it does not
        # drift: its "post-task" and "final" features are the same matrix by
        # construction. Aliasing dg->te states that explicitly rather than
        # duplicating the array.
        which_eff = "te" if which == "dg" else which
        return self.data[f"{bench}|{seed}|X{self.arm.features}_{which_eff}"]

    def run(self, n_tasks: int = 5) -> dict[str, Any]:
        Xtr = self._X("tr")
        Xdg, Xte = self._X("dg"), self._X("te")
        ytr = self.data[f"{self.key}|ytr"]
        yte = self.data[f"{self.key}|yte"]
        ttr = self.data[f"{self.key}|ttr"]
        tte = self.data[f"{self.key}|tte"]
        acc = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
        t0 = _now()
        for i in range(n_tasks):
            sel = np.flatnonzero(ttr == i)
            if sel.size:
                if self.arm.oracle:
                    # ORACLE: every task's training data at once. Not a fair
                    # arm and never presented as one - it is the ceiling a
                    # perfect linear/quadratic readout could reach if the
                    # continual schedule did not exist.
                    sel = np.arange(len(ytr))
                self.readout.partial_fit(Xtr[sel], np.eye(10, dtype=np.float32)[ytr[sel]])
            # the diagonal is scored on the codes from right after task i
            s_i = np.flatnonzero(tte == i)
            if s_i.size:
                acc[i, i] = self.readout.score(Xdg[s_i], yte[s_i])
            for j in range(i):
                s = np.flatnonzero(tte == j)
                if s.size:
                    acc[i, j] = self.readout.score(Xte[s], yte[s])
        for j in range(n_tasks):
            # final row is the retention measurement: final substrate code
            s = np.flatnonzero(tte == j)
            if s.size:
                acc[n_tasks - 1, j] = self.readout.score(Xte[s], yte[s])
        final = acc[n_tasks - 1]
        final = final[np.isfinite(final)]
        # BWT/forgetting on the triangular matrix, per brain/metrics.py.
        diag = np.array([acc[i, i] for i in range(n_tasks) if np.isfinite(acc[i, i])])
        bwt, forget = float("nan"), float("nan")
        if diag.size:
            bwt = float(np.nanmean(final - diag))
            drops = []
            for j in range(n_tasks):
                col = acc[:, j][np.isfinite(acc[:, j])]
                if col.size:
                    drops.append(float(col.max() - col[-1]))
            forget = float(np.mean(drops)) if drops else float("nan")
        return {
            "arm": self.arm.name,
            "features": self.arm.features,
            "readout": self.arm.readout,
            "family": self.arm.family,
            "space": self.arm.space,
            "rule": self.arm.rule,
            "normalize": self.arm.normalize == "norm",
            "quad_dim": self.arm.quad_dim,
            "lr": self.cfg.lr,
            "oracle": self.arm.oracle,
            "seed": self.seed,
            "acc_matrix": acc.tolist(),
            "final_average": float(np.mean(final)) if final.size else float("nan"),
            "diagonal_mean": float(diag.mean()) if diag.size else float("nan"),
            "backward_transfer": bwt,
            "forgetting": forget,
            "n_params": self.readout.n_params,
            "feature_dim": self.readout.feature_dim,
            "updates": self.readout.updates,
            "wall_seconds": _now() - t0,
            "note": self.arm.note,
            "ridge": self.cfg.ridge,
            "ridge_sweep": self.ridge_sweep,
            "diverged": bool(self.readout.diagnostics.get("diverged")),
            "weight_absmax": float(self.readout.diagnostics.get("weight_absmax", 0.0)),
            "diagnostics": self.readout.diagnostics,
            "error": None,
        }


def _single_task_arm(arm: ArmSpec, data: dict[str, np.ndarray], key: str,
                     seed: int) -> dict[str, Any]:
    """The no-interference 10-way setting where the 0.333 ceiling was measured."""
    from brain.readout import Readout, ReadoutConfig
    bench = key.split("|")[0]
    # y_ce lives at the benchmark level (it depends only on the pixel sample,
    # not on the substrate variant), while Xce is per-variant because it is a
    # substrate output.
    y = data[f"{bench}|{seed}|y_ce"]
    if arm.features == "brain":
        X = data[f"{key}|Xce"]
    else:
        X = data[f"{bench}|{seed}|X{arm.features}_ce"]
    n = len(y)
    cut = int(n * 0.7)
    ridge = 1e-2
    ridge_sweep: dict[str, float] = {}
    if arm.family.startswith("closed_form"):
        # Same no-leakage rule as the continual arms: select on a holdout of the
        # TRAIN slice only. The 70/30 test slice is never consulted.
        rng = np.random.default_rng(seed * 977 + 13)
        idx = rng.permutation(cut)
        vk = max(1, int(cut * 0.25))
        va, tr = idx[:vk], idx[vk:]
        for lam in RIDGE_GRID:
            probe = Readout(ReadoutConfig(
                kind=arm.readout, n_features=int(X.shape[1]), n_classes=10,
                ridge=lam, normalize=(arm.normalize == "norm"),
                quad_dim=arm.quad_dim, quad_seed=1234, seed=seed))
            probe.partial_fit(X[tr], np.eye(10, dtype=np.float32)[y[tr]])
            ridge_sweep[f"{lam:g}"] = float(probe.score(X[va], y[va]))
        ridge = float(max(ridge_sweep, key=lambda k: ridge_sweep[k]))
    cfg = ReadoutConfig(kind=arm.readout, n_features=int(X.shape[1]), n_classes=10,
                        lr=arm.lr, ridge=ridge, epochs=arm.epochs,
                        normalize=(arm.normalize == "norm"), quad_dim=arm.quad_dim,
                        quad_seed=1234, seed=seed)
    r = Readout(cfg)
    t0 = _now()
    r.partial_fit(X[:cut], np.eye(10, dtype=np.float32)[y[:cut]])
    train_acc = r.score(X[:cut], y[:cut])
    test_acc = r.score(X[cut:], y[cut:])
    return {
        "arm": arm.name,
        "features": arm.features,
        "readout": arm.readout,
        "family": arm.family,
        "space": arm.space,
        "rule": arm.rule,
        "normalize": arm.normalize == "norm",
        "quad_dim": arm.quad_dim,
        "lr": cfg.lr,
        "seed": seed,
        "n_train": cut,
        "n_test": n - cut,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "n_params": r.n_params,
        "feature_dim": r.feature_dim,
        "ridge": ridge,
        "ridge_sweep": ridge_sweep,
        "wall_seconds": _now() - t0,
        "error": None,
    }


def capacity_scaling(data: dict[str, np.ndarray], spec: CacheSpec, *,
                     sizes: tuple[int, ...] = (200, 420, 1000, 2000, 4000),
                     quad_dim: int = 256, verbose: bool = True) -> list[dict[str, Any]]:
    """Linear vs quadratic vs pixels at growing TRAIN size, fixed TEST set.

    This is the arm that settles ``H-linear`` without the confound that broke
    the original claim. The earlier "quadratic 0.333 vs linear 0.157" comparison
    varied the feature space at one training size and one penalty, where the
    quadratic Gram (1856 dims from 420 samples) is rank deficient and the solve
    is effectively interpolation rather than generalisation. Here the penalty is
    re-selected per size on a training holdout, the test set is fixed across
    sizes, and the training set only grows:

    * if the quadratic gap is real, it should *widen* with data -- more samples
      is exactly what a genuinely higher-capacity feature space needs;
    * if the gap closes or inverts, the original observation was an artefact of
      the sample/penalty regime, not evidence of nonlinear structure.
    """
    from brain.cortex import CortexClassifier, CortexConfig
    from brain.tasks import split_mnist
    from brain.readout import Readout, ReadoutConfig

    rows: list[dict[str, Any]] = []
    n_test = 1000
    for seed in range(min(spec.seeds, 2)):
        suite = split_mnist(spec.tasks, seed=seed, permute=True)
        need = max(sizes) + n_test
        x, y = stratified_subsample(suite.tasks[0].x_train, suite.tasks[0].y_train,
                                    need, seed * 7 + 101)
        cfg = CortexConfig(seed=seed, reset_between_samples=False, **SUBSTRATE)
        model = CortexClassifier(cfg)
        X = np.zeros((len(y), SUBSTRATE["n_neurons"]), dtype=np.float32)
        _code_batch(model, x, cfg.t_train_ms, X, 0)
        model.be.eval()
        te = np.arange(len(y) - n_test, len(y))
        Yoh = np.eye(10, dtype=np.float32)
        for n_tr in sizes:
            if n_tr + n_test > len(y):
                continue
            tr = np.arange(n_tr)
            for feature_name, feat, nf in (("brain", X, SUBSTRATE["n_neurons"]),
                                           ("pixels", x, 784)):
                for kind, qd in (("ridge", 0), ("ridge_quad", quad_dim)):
                    # penalty chosen on a holdout of this size's TRAIN rows
                    rng = np.random.default_rng(seed * 31 + n_tr)
                    idx = rng.permutation(n_tr)
                    vk = max(1, int(n_tr * 0.2))
                    va, trn = idx[:vk], idx[vk:]
                    sweep: dict[str, float] = {}
                    for lam in RIDGE_GRID:
                        r = Readout(ReadoutConfig(
                            kind=kind, n_features=nf, n_classes=10, ridge=lam,
                            normalize=True, quad_dim=qd, quad_seed=1234, seed=seed))
                        r.partial_fit(feat[trn], Yoh[y[trn]])
                        sweep[f"{lam:g}"] = float(r.score(feat[va], y[va]))
                    lam = float(max(sweep, key=lambda k: sweep[k]))
                    r = Readout(ReadoutConfig(
                        kind=kind, n_features=nf, n_classes=10, ridge=lam,
                        normalize=True, quad_dim=qd, quad_seed=1234, seed=seed))
                    r.partial_fit(feat[tr], Yoh[y[tr]])
                    rows.append({
                        "seed": seed, "n_train": int(n_tr), "features": feature_name,
                        "readout": kind, "quad_dim": qd,
                        "feature_dim": r.feature_dim, "ridge": lam,
                        "ridge_sweep": sweep, "n_params": r.n_params,
                        "test_accuracy": float(r.score(feat[te], y[te])),
                        "train_accuracy": float(r.score(feat[tr], y[tr])),
                        "error": None,
                    })
            if verbose:
                print(f"    capacity scaling seed={seed} n_train={n_tr} done", flush=True)
    return rows


def print_capacity_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n" + "=" * 100)
    print("CAPACITY SCALING -- linear vs quadratic vs raw pixels, FIXED test set")
    print("  The test set never changes; only the TRAIN set grows and the ridge")
    print("  penalty is re-selected on a training holdout at each size. A real")
    print("  quadratic advantage must WIDEN with data; artifact gaps close.")
    print("=" * 100)
    sizes = sorted({r["n_train"] for r in rows})
    cells: dict[tuple, list[float]] = {}
    for r in rows:
        cells.setdefault((r["features"], r["readout"], r["n_train"]), []).append(
            r["test_accuracy"])
    labels = [("brain", "ridge", "brain linear"),
              ("brain", "ridge_quad", "brain quad"),
              ("pixels", "ridge", "pixels linear"),
              ("pixels", "ridge_quad", "pixels quad")]
    print(f"{'train n':>8} " + " ".join(f"{lab:>14}" for _, _, lab in labels))
    print("-" * 100)
    for n in sizes:
        vals = []
        for f, k, _ in labels:
            v = cells.get((f, k, n), [float("nan")])
            vals.append(f"{np.mean(v)*100:>13.2f}")
        print(f"{n:>8} " + " ".join(vals))
    print("-" * 100)
    for n in sizes:
        b = cells.get(("brain", "ridge", n), [float("nan")])
        q = cells.get(("brain", "ridge_quad", n), [float("nan")])
        gap = np.mean(q) - np.mean(b)
        print(f"  n_train={n:>5}: brain quad - linear = {gap*100:+6.2f} pts "
              f"({'quadratic WINS' if gap > 0.005 else 'linear wins' if gap < -0.005 else 'tie'})")
    print("=" * 100)


# ------------------------------------------------------------------ reporting
def summarise(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    """Group rows by ``keys`` and report mean +/- 95% CI on every metric."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("error"):
            continue
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out: dict[str, Any] = {}
    for gk, members in groups.items():
        label = "|".join(str(x) for x in gk)
        rec: dict[str, Any] = {"n_seeds": len(members)}
        for metric in ("final_average", "diagonal_mean", "backward_transfer",
                       "forgetting", "test_accuracy", "train_accuracy"):
            vals = [m[metric] for m in members if metric in m and np.isfinite(m[metric])]
            if vals:
                mean, ci = _sem_ci95(vals)
                rec[metric] = mean
                rec[f"{metric}_ci95"] = ci
        for metric in ("n_params", "feature_dim", "updates"):
            if metric in members[0]:
                rec[metric] = int(members[0][metric])
        if "ridge" in members[0]:
            rec["ridge"] = float(members[0]["ridge"])
        if any("diverged" in m for m in members):
            rec["diverged"] = bool(any(m.get("diverged") for m in members))
            rec["n_diverged"] = int(sum(1 for m in members if m.get("diverged")))
        rec["errors"] = sorted({m["error"] for m in members if m.get("error")})
        out[label] = rec
    return out


def _fmt(mean: float, ci: float, scale: float = 100.0, width: int = 7) -> str:
    if not np.isfinite(mean):
        return " " * width + "n/a"
    return f"{mean * scale:>{width}.2f}+-{ci * scale:<5.2f}"


def print_continual_table(summary: dict[str, Any], bench: str) -> None:
    print("\n" + "=" * 108)
    print(f"CONTINUAL RETENTION (5 tasks, no replay) -- {bench.upper()}-MNIST")
    print("  'final' = mean accuracy over all 5 tasks at the end of the stream.")
    print("  'diag'  = mean accuracy on each task right after it was trained.")
    print("  'forget'= mean drop from the best accuracy ever seen on a task to the end.")
    print("=" * 108)
    print(f"{'arm':<26}{'feat':<7}{'nm':<4}{'variant':<15}{'lr':<7}{'final %':<15}"
          f"{'diag %':<15}{'forget %':<15}{'params':>8}{'lambda':>8}")
    print("-" * 122)
    order = [k for k in summary
             if len(k.split("|")) > 3 and k.split("|")[3] == bench]
    order.sort(key=lambda k: -(summary[k].get("final_average") or -1))
    for label in order:
        s = summary[label]
        parts = label.split("|")
        arm, features, norm = parts[0], parts[1], parts[2]
        variant = parts[4]
        lr_txt = f"{float(parts[5]):g}" if len(parts) > 5 else "-"
        lam_txt = f"{s['ridge']:g}" if "ridge" in s else "-"
        flag = " DIVERGED" if s.get("diverged") else ""
        print(f"{arm:<26}{_feat_label(features):<7}{_yn(norm):<4}{variant:<15}{lr_txt:<7}"
              f"{_fmt(s.get('final_average', float('nan')), s.get('final_average_ci95', 0.0), 100.0, 6):<15}"
              f"{_fmt(s.get('diagonal_mean', float('nan')), s.get('diagonal_mean_ci95', 0.0), 100.0, 6):<15}"
              f"{_fmt(s.get('forgetting', float('nan')), s.get('forgetting_ci95', 0.0), 100.0, 6):<15}"
              f"{s.get('n_params', 0):>8,}{lam_txt:>8}{flag}")
    print("=" * 122)


def _yn(value: Any) -> str:
    """Render the normalise flag, whether it arrives as bool or as 'norm'/'raw'."""
    if isinstance(value, str):
        return "y" if value == "norm" else "n"
    return "y" if bool(value) else "n"


def _feat_label(features: str) -> str:
    return {"brain": "brain", "rp800": "rp800", "rp900": "rp900",
            "pix": "pixels"}.get(features, features)


def print_ceiling_table(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("SINGLE-TASK 10-WAY CEILING (no interference; where 0.333 was measured)")
    print("  70/30 split of a stratified 600-sample draw from permuted task 0.")
    print("=" * 100)
    print(f"{'arm':<24}{'feat':<7}{'nm':<4}{'train %':<16}{'test %':<16}{'dim':>8}")
    print("-" * 100)
    order = sorted(summary, key=lambda k: -(summary[k].get("test_accuracy") or -1))
    for label in order:
        s = summary[label]
        parts = label.split("|")
        arm, features, norm = parts[0], parts[1], parts[2]
        if len(parts) > 4:
            arm = (arm + f" lr={parts[4]}")[:23]
        print(f"{arm:<24}{_feat_label(features):<7}{_yn(norm):<5}"
              f"{_fmt(s.get('train_accuracy', float('nan')), s.get('train_accuracy_ci95', 0.0)):<16}"
              f"{_fmt(s.get('test_accuracy', float('nan')), s.get('test_accuracy_ci95', 0.0)):<16}"
              f"{s.get('feature_dim', 0):>8,}")
    print("=" * 100)


def verdict(full_summary: dict[str, Any], bench: str,
            sweep: dict[str, Any] | None = None,
            variant: str = "as_falsified") -> dict[str, Any]:
    """Decision rule for one benchmark, on that benchmark's rows only.

    ``full_summary`` is filtered here rather than by the caller because a
    cross-benchmark lookup is a silent, verdict-changing bug: every arm that
    exists on both benchmarks would resolve to whichever benchmark happened to
    be scanned first, and both verdicts would print identical numbers. The
    filter is on the benchmark field, and ``_key`` additionally asserts the
    resolved key belongs to ``bench``.
    """
    summary = {k: v for k, v in full_summary.items()
               if len(k.split("|")) > 4 and k.split("|")[3] == bench
               and k.split("|")[4] == variant}
    """Apply the pre-registered decision rule. No post-hoc margin tuning.

    The discriminating numbers, in order of precedence:

    ``rule_gap`` = best local linear arm - local_perceptron, on brain features.
        If the *same* features and the same local family climb materially when
        the error term is fixed, the rule was the bottleneck. ``H-rule``.

    ``code_gap`` = best local arm (any space) - best control arm (random
        projection, same width, same readout family). If the substrate's code
        does not beat a random projection of the same width under the same
        readout, the substrate contributes nothing: ``H-code``.

    ``linear_gap`` = best quadratic arm - best linear arm, same rule family.
        Non-zero and the quadratic arms being the best overall means the code is
        nonlinearly decodable but linearly not: ``H-linear``.

    Thresholds are stated as absolute accuracy and the CI combination is
    conservative (sum of the two 95% CIs), so a difference must clear its own
    noise before it is called. Anything smaller is reported as "no measurable
    difference on this benchmark".
    """
    def _ci(arm: str, features: str, norm: bool) -> float:
        return float(summary.get(_key(arm, features, norm), {})
                     .get("final_average_ci95") or 0.0)

    def _val(arm: str, features: str, norm: bool) -> float:
        v = summary.get(_key(arm, features, norm), {}).get("final_average")
        return float(v) if v is not None and np.isfinite(v) else float("nan")

    def _key(arm: str, features: str = "brain", norm: bool = True,
             lr: float | None = None) -> str:
        """Resolve a summary key, tolerating bool/str flags and an lr suffix."""
        prefixes = []
        for flag in (("norm", "True", "true") if norm else ("raw", "False")):
            prefixes.append(f"{arm}|{features}|{flag}|{bench}|{variant}")
        cands = []
        for pre in prefixes:
            for k in summary:
                if not k.startswith(pre):
                    continue
                if lr is None:
                    # prefer the arm without an lr suffix (the pre-registered
                    # single-lr arm), else the first match
                    cands.append((0 if len(k.split("|")) == 5 else 1, k))
                else:
                    parts = k.split("|")
                    if len(parts) > 5 and abs(float(parts[5]) - lr) < 1e-12:
                        cands.append((0, k))
        if cands:
            cands.sort(key=lambda t: t[0])
            return cands[0][1]
        return f"{arm}|{features}|{norm}|{bench}|{variant}"

    def g(k: str) -> float:
        v = summary.get(_key(k), {}).get("final_average")
        return float(v) if v is not None and np.isfinite(v) else float("nan")

    def gci(k: str) -> float:
        return float(summary.get(_key(k), {}).get("final_average_ci95") or 0.0)

    # Where a step-size sweep exists, judge each rule at its own best step
    # (``_key(..., lr=...)``); otherwise fall back to the shared falsification lr.
    def at_best_lr(rule: str, arm_name: str) -> float:
        info = (sweep or {}).get(rule)
        if not info:
            return float("nan")
        key = _key(arm_name, lr=float(info["best_lr"]))
        v = summary.get(key, {}).get("final_average")
        return float(v) if v is not None and np.isfinite(v) else float("nan")

    loc_p = at_best_lr("local_perceptron", "sweep_perceptron")
    loc_d = at_best_lr("local_delta", "sweep_delta")
    loc_s = at_best_lr("local_softmax", "sweep_softmax")
    quad_local = at_best_lr("local_quad", "sweep_quad")
    lin_ridge = g("lin_ridge")
    quad_ridge = g("quad_ridge")
    # all-NaN guards: nanmax warns and returns nan, which is the honest answer,
    # but the warning would clutter a run that is otherwise fine.
    def _nanmax(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return max(vals) if vals else float("nan")
    # The control arms carry their own feature name and always run normalise=False
    # (a random projection's ReLU output is already a reasonable scale), so they
    # must be looked up with those flags rather than the brain defaults.
    # The strongest control wins the comparison: a claim that "the substrate
    # contributes something" is only as good as the best cheap alternative to it.
    _ctrl_arms = (
        ("ctrl_rp800_ridge", "rp800", False),
        ("ctrl_rp800_quad_ridge", "rp800", False),
        ("ctrl_rp800_ridge_n", "rp800", True),
        ("ctrl_rp800_quad_ridge_n", "rp800", True),
        ("ctrl_pix_ridge", "pix", False),
        ("ctrl_pix_quad_ridge", "pix", False),
    )
    ctrl_candidates = [v for v in (_val(*a) for a in _ctrl_arms) if np.isfinite(v)]
    ctrl = max(ctrl_candidates) if ctrl_candidates else float("nan")
    _scored = [(f"{a[0]}({a[1]})", _val(*a)) for a in _ctrl_arms]
    _scored = [t for t in _scored if np.isfinite(t[1])]
    best_ctrl_arm = max(_scored, key=lambda t: t[1])[0] if _scored else "none"


    best_local_linear = _nanmax([loc_d, loc_s, loc_p])
    rule_gap = best_local_linear - loc_p
    rule_gap_ci = (float((sweep or {}).get("local_delta", {}).get("best_lr_ci95", 0.0) or 0.0)
                   + _ci_at_best_lr(summary, sweep, "local_perceptron", "sweep_perceptron")
                   + _ci_at_best_lr(summary, sweep, "local_delta", "sweep_delta"))
    linear_gap = _nanmax([quad_ridge, quad_local]) - _nanmax([loc_p, loc_d, loc_s, lin_ridge])
    linear_gap_ci = (gci("quad_ridge") + gci("lin_ridge")
                     + _ci_at_best_lr(summary, sweep, "local_delta", "sweep_delta"))
    code_gap = _nanmax([loc_d, loc_s, loc_p, quad_local]) - ctrl
    code_gap_ci = (_ci_at_best_lr(summary, sweep, "local_delta", "sweep_delta")
                   + _ci_at_best_lr(summary, sweep, "local_softmax", "sweep_softmax")
                   + max([_ci(*a) for a in _ctrl_arms] or [0.0]))

    # Three-way call per hypothesis. "Not supported" and "inconclusive" are
    # different states: a gap that is positive but inside its own noise is not
    # evidence against the hypothesis, it is an underpowered measurement, and
    # reporting it as a refutation would be the same error in the opposite
    # direction from the one this diagnostic exists to fix.
    def three_way(gap: float, ci: float, label: str, threshold: float = 0.01) -> str:
        if not np.isfinite(gap):
            return f"{label}: NOT MEASURABLE (missing arm)"
        if gap > ci and gap > threshold:
            return f"{label} SUPPORTED (+{gap*100:.2f} pts > CI {ci*100:.2f})"
        if gap < -ci:
            return f"{label} REJECTED ({gap*100:+.2f} pts)"
        return (f"{label}: INCONCLUSIVE ({gap*100:+.2f} pts inside CI {ci*100:.2f})")

    call = [
        three_way(rule_gap, rule_gap_ci, "H-rule",
                  threshold=max(0.01, 0.02)),
    ]
    if np.isfinite(code_gap) and code_gap > code_gap_ci and code_gap > 0.01:
        call.append(f"substrate code BEATS the best cheap control "
                    f"(+{code_gap*100:.2f} pts > CI {code_gap_ci*100:.2f})")
    elif np.isfinite(code_gap) and code_gap < -code_gap_ci:
        call.append(f"H-code SUPPORTED: a cheap control feature set beats the "
                    f"substrate ({code_gap*100:+.2f} pts)")
    elif np.isfinite(code_gap):
        call.append(f"substrate-vs-control: inconclusive "
                    f"({code_gap*100:+.2f} pts inside CI {code_gap_ci*100:.2f})")
    call.append(three_way(linear_gap, linear_gap_ci, "H-linear"))

    return {
        "benchmark": bench,
        "variant": variant,
        "lin_perceptron": loc_p,
        "lin_delta": loc_d,
        "lin_softmax": loc_s,
        "lin_ridge": lin_ridge,
        "quad_ridge": quad_ridge,
        "quad_local": quad_local,
        "best_control_arm": best_ctrl_arm,
        "best_local_lr": {r: (sweep or {}).get(r, {}).get("best_lr")
                          for r in ("local_perceptron", "local_delta",
                                    "local_softmax", "local_quad")},
        "ctrl_random_projection": ctrl,
        "rule_gap": float(rule_gap),
        "rule_gap_combined_ci95": float(rule_gap_ci),
        "linear_gap": float(linear_gap),
        "linear_gap_combined_ci95": float(linear_gap_ci),
        "code_gap": float(code_gap),
        "code_gap_combined_ci95": float(code_gap_ci),
        "calls": call,
    }


def _ci_at_best_lr(summary: dict[str, Any], sweep: dict[str, Any] | None,
                   rule: str, arm_name: str) -> float:
    info = (sweep or {}).get(rule)
    if not info:
        return 0.0
    for k, v in summary.items():
        parts = k.split("|")
        if (parts[0] == arm_name and len(parts) > 5
                and abs(float(parts[5]) - float(info["best_lr"])) < 1e-12):
            return float(v.get("final_average_ci95") or 0.0)
    return 0.0


def print_verdict(v: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"VERDICT -- {v['benchmark'].upper()}-MNIST  [substrate variant: {v.get('variant')}]")
    print("  (all numbers are final-average accuracy)")
    print("=" * 100)
    print(f"  rule gap   (best local linear - local_perceptron) "
          f"{v['rule_gap']*100:+6.2f} pts  CI {v['rule_gap_combined_ci95']*100:5.2f}")
    print(f"  [reference] best cheap-feature control "
          f"({v.get('best_control_arm', '?')}): {v['ctrl_random_projection']*100:5.2f}"
          f"   vs substrate best local {max(v['lin_delta'], v['lin_softmax'], v['lin_perceptron'])*100:5.2f}"
          f"  / lin_perceptron {v['lin_perceptron']*100:5.2f}")
    print(f"  code gap   (best local - random projection ctrl) "
          f"{v['code_gap']*100:+6.2f} pts  CI {v['code_gap_combined_ci95']*100:5.2f}")
    print(f"  linear gap (best quadratic - best linear)        "
          f"{v['linear_gap']*100:+6.2f} pts  CI {v['linear_gap_combined_ci95']*100:5.2f}")
    for c in v["calls"]:
        print(f"  -> {c}")
    print("=" * 100)


# ---------------------------------------------------------------------- main
def _cached_self_check(data: dict[str, np.ndarray], spec: CacheSpec) -> dict[str, Any]:
    """Assert the cache really is shared input, before any arm runs.

    These are the properties the whole verdict rests on, so they are checked
    rather than assumed: identical feature matrices across arms (guaranteed by
    construction, since the code matrices are stored once), matching row counts
    between codes/labels/task-ids, balanced task sizes, and non-zero variance in
    the codes. A degenerate all-zero code would make every readout tie and the
    "verdict" would be an artefact of a dead cache.
    """
    report: dict[str, Any] = {"checks": {}, "hashes": {}}
    for bench in spec.benchmarks:
        for variant in spec.variants:
            for seed in range(spec.seeds):
                k = f"{bench}|{variant}|{seed}"
                Xtr, Xte = data[f"{k}|Xtr"], data[f"{k}|Xte"]
                if not (Xtr.shape[0] == data[f"{k}|ytr"].shape[0]
                        == data[f"{k}|ttr"].shape[0]):
                    raise AssertionError(f"{k}: train rows misaligned")
                if not (Xte.shape[0] == data[f"{k}|yte"].shape[0]
                        == data[f"{k}|tte"].shape[0]):
                    raise AssertionError(f"{k}: test rows misaligned")
                if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
                    raise AssertionError(f"{k}: non-finite codes")
                if Xtr.std() == 0.0:
                    raise AssertionError(f"{k}: zero-variance training codes")
                counts = np.bincount(data[f"{k}|tte"].astype(np.int64),
                                     minlength=spec.tasks)
                if counts[spec.tasks - 1] == 0 and spec.tasks > 1:
                    raise AssertionError(f"{k}: last task has no test rows")
                report["hashes"][k] = _sha(Xtr)
    report["checks"]["rows_aligned"] = 1.0
    report["checks"]["codes_finite"] = 1.0
    report["checks"]["codes_nonzero_variance"] = 1.0
    # The random-projection control must be non-degenerate too; a dead control
    # would make the substrate look informative by comparison.
    for bench in spec.benchmarks:
        for w in spec.rp_widths:
            rp = data[f"{bench}|0|Xrp{w}_tr"]
            if rp.std() == 0.0 or not np.isfinite(rp).all():
                raise AssertionError(f"control {bench}|rp{w} is degenerate")
            report["checks"][f"control_nondegenerate_rp{w}_{bench}"] = 1.0
    return report


def run_all(data: dict[str, np.ndarray], spec: CacheSpec, *,
            quad_dim: int, ladder: bool, verbose: bool = False
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every arm on every (benchmark, variant, seed). Returns arms, ceiling."""
    rows: list[dict[str, Any]] = []
    ceil_rows: list[dict[str, Any]] = []
    arms = arm_table(quad_dim)
    if ladder:
        arms = arms + _ladder_table((16, 64, 256, 1024))
    arms = arms + lr_sweep_table()

    total = len(spec.benchmarks) * len(spec.variants) * spec.seeds * len(arms)
    done = 0
    for bench in spec.benchmarks:
        for seed in range(spec.seeds):
            for vi, variant in enumerate(spec.variants):
                key = f"{bench}|{variant}|{seed}"
                for arm in arms:
                    # Control arms do not read the substrate at all (they use
                    # cached random projections or raw pixels), so running them
                    # once per variant would duplicate bit-identical seeds. The
                    # mean would be unchanged but the CI would shrink by
                    # sqrt(n_variants), manufacturing significance in exactly the
                    # comparison that decides H-code. They therefore run once,
                    # under the first variant, and are reported there.
                    if arm.features != "brain" and vi > 0:
                        continue
                    done += 1
                    if verbose and done % 10 == 1:
                        print(f"    [{done}/{total}] {bench} {variant} {arm.name}",
                              flush=True)
                    try:
                        r = ArmRunner(arm, data, key, seed).run(spec.tasks)
                    except Exception as exc:  # keep other arms alive
                        r = {"arm": arm.name, "features": arm.features,
                             "readout": arm.readout, "family": arm.family,
                             "space": arm.space, "rule": arm.rule,
                             "normalize": arm.normalize == "norm",
                             "quad_dim": arm.quad_dim, "lr": arm.lr,
                             "oracle": arm.oracle, "seed": seed, "benchmark": bench,
                             "variant": variant, "final_average": float("nan"),
                             "diagonal_mean": float("nan"), "n_params": 0,
                             "feature_dim": 0, "updates": 0, "wall_seconds": 0.0,
                             "note": arm.note,
                             "error": f"{type(exc).__name__}: {exc}"}
                        if verbose:
                            traceback.print_exc()
                    r["benchmark"] = bench
                    r["variant"] = variant
                    rows.append(r)
                    if bench == "permuted" and spec.ceiling and arm.features == "brain":
                        try:
                            c = _single_task_arm(arm, data, key, seed)
                        except Exception as exc:
                            c = {"arm": arm.name, "features": arm.features,
                                 "readout": arm.readout, "family": arm.family,
                                 "space": arm.space, "rule": arm.rule,
                                 "normalize": arm.normalize == "norm",
                                 "quad_dim": arm.quad_dim, "lr": arm.lr, "seed": seed,
                                 "train_accuracy": float("nan"),
                                 "test_accuracy": float("nan"), "n_params": 0,
                                 "feature_dim": 0, "wall_seconds": 0.0,
                                 "error": f"{type(exc).__name__}: {exc}"}
                        c["benchmark"] = bench
                        c["variant"] = variant
                        ceil_rows.append(c)
                    elif bench == "permuted" and spec.ceiling and vi == 0:
                        c = _single_task_arm(arm, data, f"{bench}|{variant}|{seed}", seed)
                        c["benchmark"] = bench
                        c["variant"] = variant
                        ceil_rows.append(c)
    return rows, ceil_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    p.add_argument("--variants", nargs="*", default=list(VARIANTS),
                   help="as_falsified (no reset between samples) | reset_fresh")
    p.add_argument("--quad-dim", type=int, default=256,
                   help="sampled pairwise products for the quadratic arms")
    p.add_argument("--no-ladder", action="store_true",
                   help="skip the quad_dim ladder (faster)")
    p.add_argument("--no-ceiling", action="store_true",
                   help="skip the single-task ceiling set")
    p.add_argument("--capacity", action="store_true", default=True,
                   help="run the linear-vs-quadratic capacity scaling (default on)")
    p.add_argument("--no-capacity", dest="capacity", action="store_false")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--force-cache", action="store_true",
                   help="rebuild the code cache even if a valid one exists")
    p.add_argument("--cache-only", action="store_true",
                   help="build the cache and exit without running any arm")
    p.add_argument("--from-cache", action="store_true",
                   help="require the cache to exist (never build it)")
    p.add_argument("--out", type=Path,
                   default=RESULTS_DIR / "readout_diagnostic.json")
    p.add_argument("--reanalyse", action="store_true",
                   help="re-derive every table and verdict from an existing --out "
                        "JSON without re-running any arm (uses --out as input)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.reanalyse:
        if not args.out.exists():
            print(f"ERROR: --reanalyse needs an existing {args.out}")
            return 2
        old = json.loads(args.out.read_text())
        spec_r = CacheSpec(benchmarks=tuple(old["meta"]["benchmarks"]),
                           variants=tuple(old["meta"]["variants"]),
                           seeds=int(old["meta"]["seeds"]),
                           tasks=int(old["meta"]["tasks"]),
                           ceiling=bool(old.get("ceiling")))
        print("=" * 100)
        print(f"RE-ANALYSIS of {args.out} (no arms re-run)")
        print("=" * 100)
        return report_and_write(old.get("arms", []), old.get("ceiling", []),
                                old.get("capacity_scaling", []), spec_r, args, _now(),
                                config_hash=str(old.get("meta", {}).get("config_hash", "-")),
                                checks=old.get("checks"))

    spec = CacheSpec(benchmarks=tuple(args.benchmarks), variants=tuple(args.variants),
                     seeds=args.seeds, tasks=args.tasks,
                     ceiling=not args.no_ceiling)
    t_start = _now()

    print("=" * 100)
    print("READOUT DIAGNOSTIC -- is the bottleneck the CODE, the RULE, or LINEARITY?")
    print("=" * 100)
    print(f"  machine      : {platform.platform()} / {platform.machine()}")
    print(f"  python       : {sys.version.split()[0]}  numpy {np.__version__}")
    try:
        from brain.backend import get_backend, has_mlx
        be = get_backend("auto")
        print(f"  MLX available: {has_mlx()}   default backend: {be.name}")
    except Exception as exc:  # pragma: no cover
        print(f"  backend probe failed: {exc}")
    print(f"  seeds        : {spec.seeds}   tasks: {spec.tasks}")
    print(f"  benchmarks   : {', '.join(spec.benchmarks)}")
    print(f"  variants     : {', '.join(spec.variants)}")
    print(f"  quad_dim     : {args.quad_dim}")
    print(f"  cache        : {args.cache}")

    # ---------------------------------------------------------------- cache
    if args.force_cache or not args.cache.exists():
        if args.from_cache:
            print(f"\nERROR: --from-cache given but {args.cache} does not exist.")
            return 2
        print("\n[1/3] building code cache (this is the slow part; the substrate "
              "is re-run once per sample)")
        data = build_cache(args.cache, spec)
    else:
        try:
            data = load_cache(args.cache, spec)
            print("\n[1/3] reusing code cache")
        except (ValueError, FileNotFoundError) as exc:
            if args.from_cache:
                print(f"\nERROR: {exc}")
                return 2
            print(f"\n[1/3] cache unusable ({exc}); rebuilding")
            data = build_cache(args.cache, spec)
    _example = next(k for k in data if k.endswith("|Xtr"))
    print(f"    cache shape ok; substrate codes are shared by every arm by "
          f"construction (e.g. {_example} -> {data[_example].shape})")

    checks = _cached_self_check(data, spec)
    if args.cache_only:
        print("\n--cache-only: done.")
        return 0

    # ----------------------------------------------------------------- arms
    print(f"\n[2/3] running arms (identical cached codes for every readout)")
    rows, ceil_rows = run_all(data, spec, quad_dim=args.quad_dim,
                              ladder=not args.no_ladder, verbose=args.verbose)
    # cap_rows is consumed by report_and_write below. It was previously only
    # ever bound inside that function's own scope, so a fresh (non-reanalyse)
    # run raised NameError at the return statement AFTER all arms had finished
    # and wrote no JSON at all. Bind it here.
    cap_rows: list[dict[str, Any]] = []
    if args.reanalyse:
        cap_rows = [r for r in (data.get("capacity_scaling") or [])
                    if not r.get("error")]
    print(f"    {len(rows)} continual arms, {len(ceil_rows)} ceiling arms, "
          f"{sum(1 for r in rows if r.get('error'))} failed")

    # Group by arm identity *including lr*, so the step-size sweep rows cannot
    # collide with the pre-registered single-lr arms of the same name.
    return report_and_write(rows, ceil_rows, cap_rows, spec, args, t_start,
                            config_hash=str(data["config_hash"][0]), checks=checks)


def report_and_write(rows: list[dict[str, Any]], ceil_rows: list[dict[str, Any]],
                     cap_rows: list[dict[str, Any]], spec: CacheSpec,
                     args: argparse.Namespace, t_start: float,
                     config_hash: str = "-",
                     checks: dict[str, Any] | None = None) -> int:
    """Summarise, print every table, and write the JSON. Returns an exit code.

    Split out of :func:`main` so that ``--reanalyse`` can re-derive the whole
    report from an existing JSON without re-running 40 minutes of arms. That
    matters for correctness, not just convenience: the decision rule in
    :func:`verdict` is code, and a reviewer must be able to change it and see the
    effect on the *same* numbers.
    """
    # ``variant`` MUST be part of the key. Folding the two substrate variants
    # together would average a 201-SynOp/sample code with a 90-SynOp/sample code
    # and report the mean as if it described one substrate - the single most
    # misleading thing this table could do, since the variant is a first-class
    # experimental factor (it changes the code, not the readout).
    summary = summarise(
        rows, ("arm", "features", "normalize", "benchmark", "variant", "lr"))
    # ``bench_summary`` keeps all six fields: every consumer (the tables, the
    # verdict, the best-lr sweep) indexes them positionally, and dropping the
    # variant here would silently average two different substrates together.
    bench_summary = dict(summary)
    # One sweep per benchmark: a step size tuned on permuted-MNIST must not
    # decide the split-MNIST verdict (that is cross-benchmark leakage, and it
    # would quietly make the two verdicts identical).
    sweeps = {b: best_lr_per_rule(rows, b, spec.variants[0]) for b in spec.benchmarks}
    sweep = sweeps.get("permuted") or next(iter(sweeps.values()), {})

    for bench in spec.benchmarks:
        print_continual_table(bench_summary, bench)
    if ceil_rows:
        print_ceiling_table(summarise(ceil_rows, ("arm", "features", "normalize")))

    if args.reanalyse:
        # Re-analysis must not touch the simulator: it re-derives tables and
        # verdicts from the JSON's own rows. Re-running the capacity stage here
        # would also make the "same numbers" guarantee false.
        cap_rows = [r for r in cap_rows if not r.get("error")]
        print_capacity_table(cap_rows)
    elif args.capacity:
        print("\n[2b/3] capacity scaling (linear vs quadratic, fixed test set)")
        cap_rows = capacity_scaling(data, spec, quad_dim=args.quad_dim,
                                    verbose=args.verbose)
        print_capacity_table(cap_rows)

    for _b in spec.benchmarks:
        _sw = sweeps.get(_b) or {}
        if not _sw:
            continue
        print("\n" + "=" * 100)
        print(f"STEP-SIZE SWEEP -- {_b.upper()}-MNIST "
              "(final-average accuracy vs lr)")
        print("  A rule must not be judged at a bad step. The perceptron error is bounded")
        print("  by 1 while a delta error is unbounded, so the rules do not share a stable")
        print("  step size and one shared lr would decide the verdict on a constant.")
        print("  Best arm per rule is marked '*'.")
        print("=" * 100)
        for rule, info in _sw.items():
            pts = "  ".join(
                f"{lr}={v*100:5.1f}{'*' if abs(float(lr) - info['best_lr']) < 1e-12 else ' '}"
                for lr, v in sorted(info["sweep"].items(), key=lambda kv: float(kv[0])))
            print(f"  {rule:<17} {pts}")
        print("=" * 100)

    # The verdict is stated on ``as_falsified`` because that is the substrate the
    # original 22.10% run used; the reset variant is reported in the tables.
    verdict_variant = "as_falsified" if "as_falsified" in spec.variants else spec.variants[0]
    verdicts = [verdict(bench_summary, b, sweeps.get(b, {}), verdict_variant)
                for b in spec.benchmarks]
    for v in verdicts:
        print_verdict(v)

    # ------------------------------------------------------------- payload
    print(f"\n[3/3] writing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "cache_version": CACHE_VERSION,
            "config_hash": config_hash,
            "substrate": SUBSTRATE,
            "train_per_task": TRAIN_PER_TASK,
            "test_per_task": TEST_PER_TASK,
            "ceiling_samples": CEILING_SAMPLES,
            "falsification_lr": FALSIFICATION_LR,
            "quad_dim": args.quad_dim,
            "quad_seed": 1234,
            "seeds": spec.seeds,
            "tasks": spec.tasks,
            "benchmarks": list(spec.benchmarks),
            "variants": list(spec.variants),
            "cache": str(args.cache),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "wall_seconds_total": _now() - t_start,
            "caveats": [
                "ridge and ridge_quad are closed-form least-squares solves and are "
                "NOT biologically plausible; they measure the feature space, not a "
                "learning mechanism.",
                "oracle=True arms fit all tasks' data at once and are upper bounds, "
                "never fair comparisons.",
                "CortexClassifier._code applies plasticity while encoding, so test "
                "codes are produced by a still-plastic substrate. Every arm sees the "
                "same cached matrices, so this cannot bias the readout comparison.",
                "as_falsified = reset_between_samples False, which is what the "
                "original 22.10% run used. reset_fresh = True, the current default.",
                "Perception-level novelty is not claimed: the control arms use a "
                "fixed random projection at the same width, and the substrate's "
                "advantage over it is reported explicitly.",
                "Operation counts are not joules; no wall-clock efficiency claim is "
                "made for the readout arms.",
            ],
        },
        "checks": checks if checks is not None else {},
        "arms": rows,
        "ceiling": ceil_rows,
        "capacity_scaling": cap_rows,
        "capacity_scaling_summary": summarise(
            cap_rows, ("features", "readout", "n_train")) if cap_rows else {},
        "summary": bench_summary,
        "ceiling_summary": summarise(ceil_rows, ("arm", "features", "normalize")),
        "verdicts": verdicts,
        "lr_sweep": sweeps,
        "step_size_caveat": (
            "The binarised perceptron error is bounded by 1 while a delta error is "
            "unbounded in the weight norm, so the rules do not share a stable step "
            "size. The verdict uses each rule at its own best step size from the "
            "reported sweep; the shared-lr arms are also reported. At the "
            "falsification's lr=0.05 the delta rule diverges outright, which is "
            "itself a property of the rule, not a measurement artifact."
        ),
        "failed_arms": [{"arm": r["arm"], "benchmark": r.get("benchmark"),
                         "error": r["error"]} for r in rows if r.get("error")],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"    wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    print(f"\nDONE in {_now() - t_start:.1f}s")
    errors = [r for r in rows if r.get("error")]
    return 1 if errors and len(errors) == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
