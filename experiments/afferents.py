"""Do cortical afferent projections fix the substrate's representation?

The claim under test
--------------------
``brain/cortex.py`` routes the image into the cortex with a 1:1 identity map::

    drive[: cfg.n_input] = pix * cfg.gain

so neuron ``i`` receives pixel ``i`` and nothing else, and neurons
``n_input..n_neurons`` receive nothing at all. No neuron combines two pixels,
so no neuron can be selective for any multi-pixel feature. Real cortical
afferents are massively convergent and divergent instead.

A previous throwaway measurement (single seed, non-stratified split) reported
that replacing the identity map with a random mixed projection raised a linear
probe on the raw spike-count code from **0.395 to 0.690** on 10-way MNIST. This
file re-measures that rigorously and then asks the two questions that decide
whether it matters:

1. **Is the effect really about *mixing*, or only about *activity level*?** The
   reported gain response is non-monotonic (gain 6.0 beats gain 12.0), and it
   tracks the non-monotonic activity response of the dendritic nonlinearity.
   That is exactly the signature of an activity artefact, so a
   **matched-activity control** is the decisive measurement: compare a
   *zero-mixing* projection against a *mixed* one at the same active fraction.
2. **Does a better projection rescue the continual-learning result?** The
   published falsification is that the substrate retains 22.10% against 68.30%
   for a parameter-matched backprop MLP on Permuted-MNIST. A projection that
   boosts a linear probe from 0.40 to 0.69 must be re-run end-to-end before any
   claim is made about it.

Protocol
--------
* Data: local MNIST cache, stratified subsample of 500 train / 200 held-out
  test samples, so every class is equally represented and the probe has a large
  same-class budget. Train and test are disjoint (different MNIST splits).
* Codes: spike counts over 15 ms from a 900-neuron population, exactly the
  substrate ``brain/cortex.py`` builds, with the drive supplied by
  ``brain/afferents.py``. The projection is FROZEN (never trained); only the
  probe is fitted.
* Probe: closed-form least squares on ``[code, 1]`` (linear) and
  ``[code, code**2, 1]`` (quadratic), trained on the 500 train codes and scored
  on the 200 held-out codes. Matches the earlier measurement's convention.
* Seeds: >= 3 independent seeds; every number is mean +/- 95% CI
  (``1.96 * SEM``). Different seeds re-draw the frozen projection AND the
  neuron noise, which is the honest unit of replication.
* ``active_frac`` is the fraction of ``(sample, neuron)`` cells that spike at
  least once in the 15 ms window; ``mean_count`` is spikes per sample.

Consequence test (monkeypatch, stated plainly)
----------------------------------------------
``brain/cortex.py`` is **not modified**. ``experiments/continual.py`` is
imported and its ``brain`` arms are replaced at run time by a subclass whose
``_code`` feeds a projection from ``brain/afferents.py`` into the same
``Brain.step(external_dend=...)`` call. The patched arm runs the identical
task order, subsampling, update rule, evaluation schedule and metric code as the
published benchmark. The published protocol ran *before* ``cortex.py`` gained
``reset_between_samples`` and the ``delta`` readout rule, so the patched arms
are configured with ``reset_between_samples=False, readout_rule="binary"`` to
match the run that produced the published numbers; setting
``--modern-protocol`` re-runs everything with the current defaults so the
comparison can be checked against either substrate.

Baselines are re-run live through ``experiments/continual.py`` (not copied), so
the MLP arm is bit-identical to the published benchmark. The published JSON
numbers are re-read and asserted before use.

Backend note (MLX lazy-eval trap)
---------------------------------
The sweep runs on the NumPy backend. That is deliberate, not a fallback: on the
M4 Max it is ~5x faster per sample than MLX at this network size (a 900-neuron
step does not amortise GPU dispatch) and it scales ~10x across processes, whereas
MLX does not. Every array read goes through ``Backend.to_numpy``, which calls
``mx.eval`` first, so no un-evaluated graph is ever read. Part 0 measures
backend parity on identical configurations so the choice is evidence-backed
rather than asserted.

Run:  python3 experiments/afferents.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "experiments" / "results"
SCRATCH_DIR = Path("/Volumes/T9/human-brain/scratch")
N_INPUT = 784
N_CLASSES = 10
N_NEURONS = 900            # matches the earlier measurement and cortex defaults
T_MS = 15
N_TRAIN = 500
N_TEST = 200
#: Neuron count for the end-to-end consequence test (the published benchmark's).
CONS_NEURONS = 800
PUBLISHED_PERMUTED = RESULTS_DIR / "continual_permuted.json"
PUBLISHED_SPLIT = RESULTS_DIR / "continual_split.json"

#: Published MLP reference numbers, asserted against the JSON before use.
EXPECTED_PUBLISHED = {
    "permuted": {"mlp": 0.6830, "mlp_replay": 0.7290},
    "split": {"mlp": 0.1867, "mlp_replay": 0.3093},
}


# --------------------------------------------------------------------- helpers
_CONTINUAL = None


def _load_continual():
    """Import ``experiments/continual.py`` by path without a package install."""
    global _CONTINUAL
    if _CONTINUAL is not None:
        return _CONTINUAL
    spec = importlib.util.spec_from_file_location("continual", ROOT / "experiments" / "continual.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continual"] = mod
    spec.loader.exec_module(mod)
    _CONTINUAL = mod
    return mod


def _stratified(x: np.ndarray, y: np.ndarray, n: int, seed: int):
    """Stratified subsample, reusing the published benchmark's implementation."""
    cont = _load_continual()
    return cont.stratified_subsample(x, y, n, seed)


def _probe_all(c_train, y_train, c_test, y_test, *, quadratic: bool = False) -> dict:
    """Probe accuracies under several solve conventions.

    The headline number follows the earlier measurement's convention
    (``lstsq`` with ``rcond=1e-3``). That convention is reported alongside a
    properly regularised solve because it is NOT numerically stable at these
    shapes: the code matrix is ~800 columns wide with many near-zero-variance
    columns, so the effective rank changes with the sample count and accuracy
    becomes non-monotonic in training-set size (measured: 0.74 at 50
    samples/class, 0.46 at 100, 0.89 at 400 for the same arm). Any comparison
    that depends on which side of that instability an arm lands on is an
    artefact. Every probe here is therefore reported together, and a ranking is
    only treated as real if it survives all of them. ``rcond=None`` (full
    least squares, no rank truncation) is included as the third convention.
    """
    a = np.concatenate([c_train, c_train ** 2], 1) if quadratic else c_train
    b = np.concatenate([c_test, c_test ** 2], 1) if quadratic else c_test
    a = np.concatenate([a, np.ones((len(a), 1), dtype=np.float32)], 1)
    b = np.concatenate([b, np.ones((len(b), 1), dtype=np.float32)], 1)
    y = np.eye(N_CLASSES)[y_train]
    out: dict[str, float] = {}

    w = np.linalg.lstsq(a, y, rcond=1e-3)[0]
    out["lstsq_rcond1e-3"] = float(((b @ w).argmax(1) == y_test).mean())

    w = np.linalg.lstsq(a, y, rcond=None)[0]
    out["lstsq_full"] = float(((b @ w).argmax(1) == y_test).mean())

    # Ridge on the normal equations, intercept unpenalised. float64 throughout.
    gram = (a.T @ a).astype(np.float64)
    xty = (a.T @ y).astype(np.float64)
    d = gram.shape[0]
    for lam in (1.0, 10.0, 100.0):
        reg = lam * np.eye(d)
        reg[-1, -1] = 0.0
        try:
            wr = np.linalg.solve(gram + reg, xty)
        except np.linalg.LinAlgError:
            wr = np.linalg.lstsq(gram + reg, xty, rcond=None)[0]
        out[f"ridge_{lam:g}"] = float(((b @ wr).argmax(1) == y_test).mean())
    return out


def _probe(c_train, y_train, c_test, y_test, *, quadratic: bool = False) -> float:
    """Headline probe: the original measurement's convention."""
    return _probe_all(c_train, y_train, c_test, y_test, quadratic=quadratic)["lstsq_rcond1e-3"]


def _mean_ci(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v.mean()), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(v.size))


# --------------------------------------------------- decomposition controls (local)
class _ControlProjection:
    """Experiment-local projection controls that are NOT library kinds.

    These decompose the gain of a random projection into two factors that have
    nothing to do with convergence:

    * ``ctrl_perm``  - the SAME 1:1 routing (neuron ``i`` <- pixel ``i``, unit
      weight) but the pixel indices are randomly permuted. Zero mixing, zero
      polarity change: the drive vector is a permutation of the identity drive.
    * ``ctrl_sign``  - identity routing with a random per-neuron ``+/-1``
      polarity. Zero mixing; half the population now responds to *darkness*.
    * ``ctrl_both``  - permutation *and* polarity, still zero mixing. This is
      the honest name for what ``sparse``/``fan_in=1`` actually computes.
    * ``ctrl_centered`` - identity routing, all-positive, but the image mean is
      subtracted so the drive is zero-mean yet still single-signed per neuron.
      This separates "the input is not zero-mean" from "receptive fields have no
      sign" -- measured: centering alone does NOT reproduce the gain, polarity
      does.

    ``ctrl_both`` matching a fully mixed projection means the effect is input
    *randomisation and signed receptive fields*, not *convergence* (combining
    several inputs). Kept here rather than in ``brain/afferents.py`` so the
    shipped interface stays exactly as specified.

    Biological grounding for ``ctrl_sign``: cortical pyramidal neurons receive
    both excitatory and inhibitory input, so their linear receptive fields are
    signed, not rectified. A purely positive input pathway is an artefact of the
    identity routing, not a property of cortex.
    """

    def __init__(self, kind: str, *, n_input: int, n_neurons: int, gain: float, seed: int):
        rng = np.random.default_rng(seed)
        self.kind = kind
        self.n_input, self.n_neurons, self.gain = n_input, n_neurons, gain
        self.perm = rng.permutation(n_input) if kind in ("ctrl_perm", "ctrl_both") else None
        self.sign = None
        self.mu = float(load_pixel_mean()) if kind == "ctrl_centered" else 0.0
        if kind in ("ctrl_sign", "ctrl_both"):
            self.sign = (rng.integers(0, 2, n_neurons) * 2 - 1).astype(np.float32)
        self.n_connections = n_input
        self.n_weights = n_input + (n_neurons if self.sign is not None else 0)
        self.macs_per_sample = 0

    def project(self, x):
        pix = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.perm is not None:
            pix = pix[self.perm]
        if self.mu:
            pix = pix - self.mu
        out = np.zeros(self.n_neurons, dtype=np.float32)
        seg = pix[: self.n_neurons]
        if self.sign is not None:
            seg = seg * self.sign[: len(seg)]
        out[: len(seg)] = seg * self.gain
        return out

    def stats(self):
        return {"kind": self.kind, "n_input": self.n_input, "n_neurons": self.n_neurons,
                "gain": self.gain, "n_weights": self.n_weights,
                "n_connections": self.n_connections, "macs_per_sample": 0,
                "control": True}


_PIXEL_MEAN = None


def load_pixel_mean() -> float:
    """MNIST training-set pixel mean, read from the local cache (cached)."""
    global _PIXEL_MEAN
    if _PIXEL_MEAN is None:
        from brain.tasks import load_mnist
        x_train = load_mnist()[0]
        _PIXEL_MEAN = float(x_train.mean())
    return _PIXEL_MEAN


def make_projection(kind, fan_in, gain, seed, *, n_input=N_INPUT, n_neurons=N_NEURONS):
    """Frozen projection factory: library kinds plus the local controls."""
    if kind.startswith("ctrl_"):
        return _ControlProjection(kind, n_input=n_input, n_neurons=n_neurons,
                                  gain=gain, seed=seed)
    from brain.afferents import Afferents, AfferentConfig
    return Afferents(AfferentConfig(n_input=n_input, n_neurons=n_neurons, kind=kind,
                                    fan_in=fan_in or n_input, gain=gain, seed=seed))


# ---------------------------------------------------------------- code capture
def _codes(aff, xs, be_name: str, *, reset: bool, seed: int, t_ms: int = T_MS):
    """Spike-count codes for ``xs`` with a frozen projection feeding the dendrite.

    Returns ``(codes, diag)``; ``diag`` carries activity and operation counts.
    All array reads go through ``Backend.to_numpy`` (which evaluates MLX first).
    """
    from brain.cortex import CortexClassifier, CortexConfig

    model = CortexClassifier(
        CortexConfig(n_neurons=N_NEURONS, n_input=N_INPUT, n_classes=N_CLASSES,
                     t_train_ms=t_ms, t_test_ms=t_ms, seed=seed,
                     reset_between_samples=reset),
        backend=be_name,
    )
    be, cfg = model.be, model.cfg
    out = np.empty((len(xs), cfg.n_neurons), dtype=np.float32)
    spikes = 0
    synops = 0
    for i, x in enumerate(xs):
        if reset:
            model.brain.reset()
        db = be.array(aff.project(x))
        counts = np.zeros(cfg.n_neurons, dtype=np.float32)
        for _ in range(t_ms):
            st = model.brain.step(external_dend=db, neuromod=1.0)
            counts += be.to_numpy(model.brain.last_spike_mask).astype(np.float32)
            synops += st.synops_active
            spikes += st.spikes
        out[i] = counts
        # MLX lazy-eval trap: force the graph before the next iteration reads it.
        be.eval()
    diag = {
        "active_frac": float((out > 0).mean()),
        "mean_count": float(out.sum(1).mean()),
        "spikes_per_sample": float(spikes / max(1, len(xs))),
        "synops_per_sample": float(synops / max(1, len(xs))),
        "n_finite": bool(np.isfinite(out).all()),
    }
    return out, diag


# ------------------------------------------------------------------ experiment 1
def sweep_job(spec):
    """One (kind, fan_in, gain, seed) sweep cell."""
    warnings.filterwarnings("ignore")
    kind, fan_in, gain, seed = spec
    from brain.tasks import load_mnist

    x_train, y_train, x_test, y_test = load_mnist()
    xtr, ytr = _stratified(x_train, y_train, N_TRAIN, 1000 + seed)
    xte, yte = _stratified(x_test, y_test, N_TEST, 2000 + seed)

    aff = make_projection(kind, fan_in, gain, seed)
    c_tr, d_tr = _codes(aff, xtr, "numpy", reset=True, seed=seed)
    c_te, _ = _codes(aff, xte, "numpy", reset=True, seed=seed)
    lin = _probe_all(c_tr, ytr, c_te, yte, quadratic=False)
    qd = _probe_all(c_tr, ytr, c_te, yte, quadratic=True)
    return {
        "kind": kind, "fan_in": int(fan_in), "gain": float(gain), "seed": int(seed),
        "n_weights": int(aff.n_weights),
        "n_connections": int(aff.n_connections),
        "macs_per_sample": int(aff.macs_per_sample),
        "linear": lin["lstsq_rcond1e-3"],
        "linear_probes": lin,
        "linear_robust": float(np.mean([lin["lstsq_full"], lin["ridge_1"],
                                        lin["ridge_10"], lin["ridge_100"]])),
        "quadratic": qd["lstsq_rcond1e-3"],
        "quadratic_probes": qd,
        "quadratic_robust": float(np.mean([qd["lstsq_full"], qd["ridge_1"],
                                           qd["ridge_10"], qd["ridge_100"]])),
        **d_tr,
    }


def build_sweep():
    """The full (kind, fan_in, gain) grid; small fan-ins double as no-mixing controls."""
    specs = []
    for gain in (1.0, 2.2, 3.0, 6.0, 12.0):
        specs.append(("identity", 0, gain))
    # Zero-mixing decomposition controls: routing and polarity, never convergence.
    for ctrl in ("ctrl_perm", "ctrl_sign", "ctrl_both", "ctrl_centered"):
        for gain in (1.0, 2.2, 3.0, 6.0, 12.0):
            specs.append((ctrl, 0, gain))
    for gain in (1.0, 2.2, 3.0, 4.0, 6.0, 8.0, 12.0):
        specs.append(("dense", N_INPUT, gain))
    for fan in (1, 2, 4, 16, 64, 256, 784):
        for gain in (1.0, 2.2, 3.0, 4.0, 6.0, 8.0, 12.0):
            specs.append(("sparse", fan, gain))
    for fan in (9, 25, 64, 169):
        for gain in (1.0, 2.2, 3.0, 4.0, 6.0, 8.0, 12.0):
            specs.append(("conv", fan, gain))
    return specs


# ------------------------------------------------------------------ experiment 2
def backend_parity_job(spec):
    """Same configuration on MLX and NumPy: the backend must not change the science."""
    warnings.filterwarnings("ignore")
    kind, fan_in, gain, seed = spec
    from brain.tasks import load_mnist

    x_train, y_train, x_test, y_test = load_mnist()
    xtr, ytr = _stratified(x_train, y_train, N_TRAIN, 1000 + seed)
    xte, yte = _stratified(x_test, y_test, N_TEST, 2000 + seed)
    out = {}
    for be_name in ("numpy", "mlx"):
        aff = make_projection(kind, fan_in, gain, seed)
        c_tr, d_tr = _codes(aff, xtr, be_name, reset=True, seed=seed)
        c_te, _ = _codes(aff, xte, be_name, reset=True, seed=seed)
        out[be_name] = {
            "linear": _probe(c_tr, ytr, c_te, yte, quadratic=False),
            "linear_robust": float(np.mean(list(_probe_all(c_tr, ytr, c_te, yte).values())[1:])),
            "active_frac": d_tr["active_frac"],
            "n_finite": d_tr["n_finite"],
        }
    return {"kind": kind, "fan_in": int(fan_in), "gain": float(gain), "seed": int(seed),
            "by_backend": out}


# ------------------------------------------------------------------ experiment 3
def _projection_arm(spec):
    """One end-to-end continual-learning run with a frozen afferent projection.

    ``brain/cortex.py`` is untouched: the projection is injected by replacing
    this instance's ``_code`` method, which is the only member that constructs
    the drive. Everything else (task order, subsampling, readout update,
    evaluation schedule) is the published benchmark's own code.

    The neuron count is the published benchmark's (800), and every arm in this
    part - projection and baseline alike - runs in its own process with the
    published protocol's flags (``reset_between_samples=False``, binary readout)
    unless ``--modern-protocol`` is given.
    """
    warnings.filterwarnings("ignore")
    kind, fan_in, gain, seed, permute, modern, arm_tag = spec
    cont = _load_continual()
    from brain.cortex import CortexClassifier, CortexConfig
    from brain.metrics import ContinualCurve, accuracy
    from brain.tasks import split_mnist

    train_per_task, test_per_task = (200, 100) if permute else (80, 50)
    suite = split_mnist(5, seed=seed, permute=permute)
    for ti, task in enumerate(suite.tasks):
        task.x_train, task.y_train = cont.stratified_subsample(
            task.x_train, task.y_train, train_per_task, seed * 17 + ti)
        task.x_test, task.y_test = cont.stratified_subsample(
            task.x_test, task.y_test, test_per_task, seed * 31 + ti)

    aff = make_projection(kind, fan_in, gain, seed, n_neurons=CONS_NEURONS)
    cfg = CortexConfig(n_neurons=CONS_NEURONS, n_input=N_INPUT, n_classes=N_CLASSES,
                       k_out=64, seed=seed, use_dendrite=True, inhibition="kwta",
                       k_wta=64, plasticity=True)
    if not modern:
        # The published benchmark predates these two CortexConfig fields.
        cfg.reset_between_samples = False
        cfg.readout_rule = "binary"
    model = CortexClassifier(cfg, backend="numpy")

    def _code(self, x, t_ms):
        be = self.be
        pix = np.asarray(x, dtype=np.float32).reshape(-1)
        db = be.array(aff.project(pix))
        counts = np.zeros(self.cfg.n_neurons, dtype=np.float32)
        synops = 0
        for _ in range(t_ms):
            st = self.brain.step(external_dend=db, neuromod=1.0)
            counts += be.to_numpy(self.brain.last_spike_mask).astype(np.float32)
            synops += st.synops_active
        be.eval()
        self.last_synops = float(synops)
        return counts

    # Bind the patched projection encoder onto this instance only.
    model._code = _code.__get__(model, CortexClassifier)
    acc = np.full((5, 5), np.nan, dtype=np.float64)
    n_finite = True
    for i, task in enumerate(suite.tasks):
        model.fit_task(task, epochs=1)
        n_finite = n_finite and bool(np.isfinite(model.W).all())
        for j in range(i + 1):
            tj = suite.tasks[j]
            acc[i, j] = accuracy(model.predict(tj.x_test), tj.y_test)
    am = np.nan_to_num(acc)
    curve = ContinualCurve(acc=am)
    return {
        "benchmark": "permuted" if permute else "split",
        "protocol": "modern" if modern else "published",
        "arm": arm_tag, "kind": kind, "fan_in": int(fan_in), "gain": float(gain),
        "seed": int(seed),
        "acc_matrix": am.tolist(),
        "final_average": curve.final_average(),
        "diagonal_mean": float(np.mean([am[i, i] for i in range(5)])),
        "backward_transfer": curve.backward_transfer(),
        "forgetting": curve.forgetting(),
        "n_params": int(model.n_params),
        "synops_per_sample": float(model.synops_per_sample),
        "weights_finite": n_finite,
    }


def baseline_job(spec):
    """Published baseline arms, executed by the published runner itself."""
    warnings.filterwarnings("ignore")
    arm, seed, permute = spec
    cont = _load_continual()
    from brain.tasks import split_mnist

    train_per_task, test_per_task = (200, 100) if permute else (80, 50)
    suite = split_mnist(5, seed=seed, permute=permute)
    for ti, task in enumerate(suite.tasks):
        task.x_train, task.y_train = cont.stratified_subsample(
            task.x_train, task.y_train, train_per_task, seed * 17 + ti)
        task.x_test, task.y_test = cont.stratified_subsample(
            task.x_test, task.y_test, test_per_task, seed * 31 + ti)
    args = SimpleNamespace(neurons=N_NEURONS - 100, k_out=64, hidden=128, lr=0.01,
                           batch_size=64, inhibition="kwta", k_wta=64, epochs=1,
                           verbose=False)
    r = cont.run_arm(arm, suite, seed, args)
    return {
        "benchmark": "permuted" if permute else "split",
        "protocol": "published", "arm": arm, "seed": int(seed),
        "final_average": r.final_average, "diagonal_mean": r.diagonal_mean,
        "backward_transfer": r.backward_transfer, "forgetting": r.forgetting,
        "n_params": int(r.n_params), "synops_per_sample": float(r.synops_per_sample),
        "error": r.error,
    }


# ------------------------------------------------------------------- reporting
def summarise_sweep(rows) -> list[dict]:
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["kind"], r["fan_in"], r["gain"]), []).append(r)
    out = []
    for (kind, fan, gain), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        lin_m, lin_ci = _mean_ci([r["linear"] for r in rs])
        quad_m, quad_ci = _mean_ci([r["quadratic"] for r in rs])
        linr_m, linr_ci = _mean_ci([r["linear_robust"] for r in rs])
        quadr_m, _ = _mean_ci([r["quadratic_robust"] for r in rs])
        af_m, af_ci = _mean_ci([r["active_frac"] for r in rs])
        mc_m, _ = _mean_ci([r["mean_count"] for r in rs])
        out.append({
            "kind": kind, "fan_in": fan, "gain": gain, "n_seeds": len(rs),
            "linear": lin_m, "linear_ci95": lin_ci,
            "quadratic": quad_m, "quadratic_ci95": quad_ci,
            "linear_robust": linr_m, "linear_robust_ci95": linr_ci,
            "quadratic_robust": quadr_m,
            "quadratic_minus_linear": quad_m - lin_m,
            "quadratic_minus_linear_robust": quadr_m - linr_m,
            "active_frac": af_m, "active_frac_ci95": af_ci,
            "mean_count": mc_m,
            "n_weights": rs[0]["n_weights"], "n_connections": rs[0]["n_connections"],
            "macs_per_sample": rs[0]["macs_per_sample"],
            "synops_per_sample": float(np.mean([r["synops_per_sample"] for r in rs])),
            "seeds_finite": all(r["n_finite"] for r in rs),
        })
    return out


def select_winner(table) -> dict:
    """Best mean linear-probe accuracy among MIXED projections.

    The frozen random projection is fixed by the seed; only the probe is fitted,
    so this is not a train/test leak, but the reported accuracy is still an
    in-sample choice over configs. The consequence test therefore re-runs the
    top configs rather than only the argmax.
    """
    zero = ("identity", "ctrl_perm", "ctrl_sign", "ctrl_both", "ctrl_centered")
    mixed = [r for r in table if r["kind"] not in zero]
    return max(mixed, key=lambda r: r["linear"])


def matched_activity_control(table) -> dict:
    """THE decisive control: same active fraction, mixing vs no mixing.

    The question is whether the accuracy of a mixed projection is reachable
    *without* any input convergence. Every candidate no-mixing arm here has
    exactly one presynaptic partner per neuron and an explicit routing matrix,
    so ``mean_count`` (spikes per sample) equals the number of active neurons:
    activity is controlled directly, with no gain extrapolation.

    Arms compared, all with one input per neuron and zero convergence:

    * ``ctrl_perm``  - random pixel->neuron routing, unit polarity.
    * ``ctrl_sign``  - identity routing, random +/- polarity.
    * ``ctrl_both``  - random routing AND random polarity (what ``sparse`` with
      ``fan_in=1`` actually computes).
    * ``sparse fan_in=1`` - the library equivalent of ``ctrl_both``, included as
      a cross-check that the local control reproduces a shipped kind.

    For every mixed arm we report the zero-mixing arm whose activity is closest,
    plus the activity and accuracy gaps. A finding is *about mixing* only if the
    mixed arm beats its activity-matched zero-mixing counterpart by more than the
    combined 95% CI.
    """
    zero_kinds = ("ctrl_perm", "ctrl_sign", "ctrl_both", "ctrl_centered")
    zero_mixing = [r for r in table
                   if r["kind"] in zero_kinds or (r["kind"] == "sparse" and r["fan_in"] == 1)]
    identity = [r for r in table if r["kind"] == "identity"]
    pairs = []
    for r in table:
        if r["kind"] in zero_kinds or r["kind"] == "identity":
            continue
        if r["kind"] == "sparse" and r["fan_in"] == 1:
            continue
        if not zero_mixing:
            continue
        best = min(zero_mixing, key=lambda z: abs(z["active_frac"] - r["active_frac"]))
        pairs.append({
            "mixed": {"kind": r["kind"], "fan_in": r["fan_in"], "gain": r["gain"]},
            "mixed_active_frac": r["active_frac"], "mixed_linear": r["linear"],
            "mixed_linear_ci95": r["linear_ci95"],
            "zero_mixing": {"kind": best["kind"], "fan_in": best["fan_in"],
                            "gain": best["gain"]},
            "zero_mixing_active_frac": best["active_frac"],
            "zero_mixing_linear": best["linear"],
            "zero_mixing_linear_ci95": best["linear_ci95"],
            "activity_gap": abs(best["active_frac"] - r["active_frac"]),
            "linear_gap": r["linear"] - best["linear"],
            "both_cis_disjoint": abs(r["linear"] - best["linear"]) >
                                (r["linear_ci95"] + best["linear_ci95"]),
        })
    identity_ceiling = max((r["active_frac"] for r in identity), default=float("nan"))
    identity_at_ceiling = max(identity, key=lambda r: r["active_frac"]) if identity else None
    return {
        "pairs": pairs,
        "zero_mixing_arms": [
            {"kind": z["kind"], "fan_in": z["fan_in"], "gain": z["gain"],
             "active_frac": z["active_frac"], "mean_count": z["mean_count"],
             "linear": z["linear"], "linear_ci95": z["linear_ci95"]}
            for z in zero_mixing
        ],
        "identity_max_active_frac": float(identity_ceiling),
        "identity_at_max": ({"gain": identity_at_ceiling["gain"],
                             "linear": identity_at_ceiling["linear"],
                             "linear_ci95": identity_at_ceiling["linear_ci95"]}
                            if identity_at_ceiling else None),
    }


def verify_published(path: Path, key: str) -> dict:
    """Load and assert the published baseline numbers before comparing against them."""
    if not path.is_file():
        raise FileNotFoundError(f"published results missing: {path}")
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {})
    out = {"path": str(path.relative_to(ROOT)), "n_seeds": {}, "final_average": {},
           "final_average_ci95": {}, "asserted": True}
    for arm, expected in EXPECTED_PUBLISHED[key].items():
        s = summary.get(arm, {})
        if s.get("status") != "ok":
            raise AssertionError(f"{path.name}: arm {arm} not ok: {s}")
        got = float(s["final_average"])
        if abs(got - expected) > 5e-4:
            raise AssertionError(
                f"{path.name}: arm {arm} final_average is {got:.4f}, "
                f"documentation claims {expected:.4f}")
        out["n_seeds"][arm] = int(s["n_seeds"])
        out["final_average"][arm] = got
        out["final_average_ci95"][arm] = float(s["final_average_ci95"])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3, help="replicate seeds for the sweep")
    p.add_argument("--workers", type=int,
                   default=min(12, os.cpu_count() or 4),
                   help="parallel worker processes")
    p.add_argument("--quick", action="store_true", help="reduced grid for a smoke run")
    p.add_argument("--modern-protocol", action="store_true",
                   help="use current cortex.py defaults (reset + delta readout) instead "
                        "of the published-protocol compatibility flags")
    p.add_argument("--skip-parity", action="store_true")
    args = p.parse_args(argv)
    workers = max(1, int(args.workers))

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("=" * 100)
    print("AFFERENT PROJECTIONS: rigorous re-measurement, matched-activity control, consequence test")
    print("=" * 100)

    # ---- published references, verified before anything is compared against them
    ref = {"permuted": verify_published(PUBLISHED_PERMUTED, "permuted"),
           "split": verify_published(PUBLISHED_SPLIT, "split")}
    for bench, info in ref.items():
        arms = ", ".join(f"{a}={v*100:.2f}%" for a, v in info["final_average"].items())
        print(f"verified published {bench:9s} baselines: {arms}  (JSON matches the docs)")

    # ---- part 0: backend parity (MLX lazy-eval trap)
    parity = []
    if not args.skip_parity:
        print("\n[0/4] backend parity (NumPy vs MLX, identical configs)...")
        p_specs = [("identity", 0, 2.2, 0), ("sparse", 64, 6.0, 0)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            parity = list(ex.map(backend_parity_job, p_specs))
        for row in parity:
            a, b = row["by_backend"]["numpy"], row["by_backend"]["mlx"]
            print(f"    {row['kind']:8s} fan={row['fan_in']:4d} g={row['gain']:4.1f}: "
                  f"numpy lin={a['linear']:.3f} act={a['active_frac']*100:.1f}%  |  "
                  f"mlx lin={b['linear']:.3f} act={b['active_frac']*100:.1f}%  "
                  f"(|d|={abs(a['linear']-b['linear']):.3f})")

    # ---- part 1: the sweep
    specs = build_sweep()
    if args.quick:
        specs = [s for s in specs if s[2] in (2.2, 6.0) and s[1] in (0, 1, 64, 784)]
    jobs = [(*s, seed) for s in specs for seed in range(args.seeds)]
    print(f"\n[1/4] projection sweep: {len(specs)} configs x {args.seeds} seeds = "
          f"{len(jobs)} runs, {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(sweep_job, jobs, chunksize=1))
    table = summarise_sweep(rows)
    sweep_seconds = time.perf_counter() - t_start
    print(f"    done in {sweep_seconds:.0f}s")
    print(f"\n    {'kind':9s} {'fan':>4s} {'gain':>5s} {'linear probe':>16s} "
          f"{'quadratic':>16s} {'act%':>6s} {'count':>7s}")
    for r in table:
        print(f"    {r['kind']:9s} {r['fan_in']:>4d} {r['gain']:>5.1f} "
              f"{r['linear']:>8.3f} +/- {r['linear_ci95']:.3f} "
              f"{r['quadratic']:>8.3f} +/- {r['quadratic_ci95']:.3f} "
              f"{r['active_frac']*100:>5.1f} {r['mean_count']:>7.1f}")

    winner = select_winner(table)
    print(f"\n    WINNER (mixed, best mean linear probe): {winner['kind']} fan={winner['fan_in']} "
          f"gain={winner['gain']} -> {winner['linear']:.3f} +/- {winner['linear_ci95']:.3f} "
          f"at {winner['active_frac']*100:.1f}% active")

    # ---- part 2: matched-activity control
    print("\n[2/4] MATCHED-ACTIVITY CONTROL (is the effect mixing, or just firing rate?)...")
    mac = matched_activity_control(table)
    ident = next((r for r in table if r["kind"] == "identity" and r["gain"] == 2.2), None)
    ident_hi = next((r for r in table if r["kind"] == "identity" and r["gain"] == 6.0), None)
    if ident:
        print(f"    identity g=2.2: {ident['linear']:.3f} at {ident['active_frac']*100:.1f}% active")
    if ident_hi:
        print(f"    identity g=6.0: {ident_hi['linear']:.3f} at {ident_hi['active_frac']*100:.1f}% active"
              f"   <- 1:1 routing cannot raise activity; dCaAP attenuates strong input")
    print(f"    identity's highest reachable activity: {mac['identity_max_active_frac']*100:.1f}%"
          f"   (identity gain 2.2 is already activity-matched to dense g=2.2)")
    print(f"\n    {'mixed arm':30s} {'mixed':>7s} {'act%':>6s} | {'zero-mixing mate':24s} "
          f"{'zm':>7s} {'act%':>6s} {'gap':>7s} {'disjoint':>9s}")
    shown = set()
    for pr in mac["pairs"]:
        m, z = pr["mixed"], pr["zero_mixing"]
        if abs(pr["mixed_active_frac"] - winner["active_frac"]) > 0.08 and m["gain"] not in (2.2, 6.0):
            continue
        key = (m["kind"], m["fan_in"], m["gain"])
        if key in shown:
            continue
        shown.add(key)
        print(f"    {m['kind']+'/'+str(m['fan_in'])+' g='+str(m['gain']):30s} "
              f"{pr['mixed_linear']:>7.3f} {pr['mixed_active_frac']*100:>5.1f} | "
              f"{z['kind']+'/'+str(z['fan_in'])+' g='+str(z['gain']):24s} "
              f"{pr['zero_mixing_linear']:>7.3f} {pr['zero_mixing_active_frac']*100:>5.1f} "
              f"{pr['linear_gap']:>+7.3f} {str(pr['both_cis_disjoint']):>9s}")

    # ---- part 2b: is the probe itself stable? (controls for a readout artefact)
    print("\n[2b] PROBE-STABILITY CONTROL (is the ranking a solve artefact?)...")
    print(f"    {'arm':28s} {'lstsq_rcond':>12s} {'lstsq_full':>11s} {'ridge 1':>8s} "
          f"{'ridge 10':>9s} {'ridge 100':>10s}")
    stability = {"probe_conventions": {}, "ranking_agrees": {}}
    probe_keys = ["lstsq_rcond1e-3", "lstsq_full", "ridge_1", "ridge_10", "ridge_100"]
    for kind, fan, gain in (("identity", 0, 2.2), ("ctrl_perm", 0, 2.2),
                            ("ctrl_centered", 0, 2.2), ("ctrl_sign", 0, 2.2),
                            ("ctrl_both", 0, 2.2), ("sparse", 1, 2.2),
                            ("dense", 784, 2.2), ("dense", 784, 6.0), ("sparse", 64, 6.0)):
        rs = [r for r in rows if (r["kind"], r["fan_in"], r["gain"]) == (kind, fan, gain)]
        if not rs:
            continue
        means = {k: float(np.mean([r["linear_probes"][k] for r in rs])) for k in probe_keys}
        for k, v in means.items():
            stability["probe_conventions"].setdefault(k, {})[f"{kind}_f{fan}_g{gain}"] = v
        print(f"    {kind+'/'+str(fan)+'/g'+str(gain):28s} "
              + "".join(f"{means[k]:>12.3f}" if i == 0 else f"{means[k]:>11.3f}" if i == 1
                        else f"{means[k]:>8.3f}" if i == 2 else f"{means[k]:>9.3f}"
                        if i == 3 else f"{means[k]:>10.3f}"
                        for i, k in enumerate(probe_keys)))
    # Ranking agreement: does the winner stay the winner under every solve?
    for k in probe_keys:
        scores = {}
        for r in rows:
            t = (r["kind"], r["fan_in"], r["gain"])
            scores.setdefault(t, []).append(r["linear_probes"][k])
        mixed = {t: float(np.mean(v)) for t, v in scores.items()
                 if t[0] not in ("identity", "ctrl_perm", "ctrl_sign",
                                 "ctrl_both", "ctrl_centered")}
        best_t = max(mixed, key=lambda t: mixed[t])
        stability["ranking_agrees"][k] = {"winner": list(best_t), "score": mixed[best_t]}
        print(f"    winner under {k:16s}: {best_t[0]}/f{best_t[1]}/g{best_t[2]} "
              f"({mixed[best_t]:.3f})")

    # ---- part 3: consequence test
    print("\n[3/4] CONSEQUENCE TEST: full continual learning, published protocol, "
          "monkeypatched drive...")
    top_mixed = sorted([r for r in table if r["kind"] != "identity"],
                       key=lambda r: -r["linear"])[:2]
    chosen = [("identity_control", "identity", 0, 2.2)]
    for i, r in enumerate(top_mixed):
        chosen.append((f"winner{i+1}_{r['kind']}_f{r['fan_in']}_g{r['gain']}",
                       r["kind"], r["fan_in"], r["gain"]))
    # The decisive comparison: the activity-matched zero-convergence arm.
    ctrl_pick = None
    if mac["pairs"]:
        best_pair = max(mac["pairs"], key=lambda pr: pr["mixed_linear"])
        ctrl_pick = (best_pair["zero_mixing"]["kind"], best_pair["zero_mixing"]["fan_in"],
                     best_pair["zero_mixing"]["gain"])
    if ctrl_pick is None:
        ctrl_pick = ("ctrl_both", 0, 2.2)
    chosen.append((f"zero_mixing_{ctrl_pick[0]}_g{ctrl_pick[2]}", *ctrl_pick))

    cons_jobs = []
    for permute in (True, False):
        seeds = range(args.seeds)
        for tag, kind, fan, gain in chosen:
            for s in seeds:
                cons_jobs.append((kind, fan, gain, s, permute, args.modern_protocol, tag))
        for arm in ("mlp", "mlp_replay"):
            for s in seeds:
                cons_jobs.append((arm, s, permute))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        cons_rows = list(ex.map(_projection_arm,
                                [j for j in cons_jobs if len(j) == 7], chunksize=1))
        base_rows = list(ex.map(baseline_job,
                                [j for j in cons_jobs if len(j) == 3], chunksize=1))
    cons_rows = [r for r in cons_rows if r is not None]
    base_rows = [r for r in base_rows if r is not None]

    consequence = {}
    for bench in ("permuted", "split"):
        rows_b = [r for r in cons_rows if r["benchmark"] == bench]
        rows_base = [r for r in base_rows if r["benchmark"] == bench]
        arms = {}
        for tag in [c[0] for c in chosen]:
            rs = [r for r in rows_b if r["arm"] == tag]
            fa, ci = _mean_ci([r["final_average"] for r in rs])
            dg, _ = _mean_ci([r["diagonal_mean"] for r in rs])
            fg, _ = _mean_ci([r["forgetting"] for r in rs])
            arms[tag] = {
                "kind": rs[0]["kind"] if rs else None,
                "fan_in": rs[0]["fan_in"] if rs else None,
                "gain": rs[0]["gain"] if rs else None,
                "n_seeds": len(rs), "final_average": fa, "final_average_ci95": ci,
                "diagonal_mean": dg, "forgetting": fg,
                "synops_per_sample": float(np.mean([r["synops_per_sample"] for r in rs])) if rs else None,
                "n_params": int(rs[0]["n_params"]) if rs else None,
                "weights_finite": all(r["weights_finite"] for r in rs) if rs else None,
            }
        for arm in ("mlp", "mlp_replay"):
            rs = [r for r in rows_base if r["arm"] == arm and not r["error"]]
            fa, ci = _mean_ci([r["final_average"] for r in rs])
            arms[arm] = {
                "n_seeds": len(rs), "final_average": fa, "final_average_ci95": ci,
                "diagonal_mean": _mean_ci([r["diagonal_mean"] for r in rs])[0],
                "forgetting": _mean_ci([r["forgetting"] for r in rs])[0],
                "synops_per_sample": 0.0,
                "n_params": int(rs[0]["n_params"]) if rs else None,
                "published_final_average": ref[bench]["final_average"].get(arm),
            }
        identity_fa = arms["identity_control"]["final_average"]
        arms["identity_control"]["delta_vs_identity_pp"] = 0.0
        for tag, a in arms.items():
            if tag != "identity_control" and a["final_average"] == a["final_average"]:
                a["delta_vs_identity_pp"] = (a["final_average"] - identity_fa) * 100
        consequence[bench] = {"arms": arms}

    for bench in ("permuted", "split"):
        print(f"\n    {bench.upper()}-MNIST (5 tasks, single pass, no replay, "
              f"{args.seeds} seeds, {'published' if not args.modern_protocol else 'modern'} protocol)")
        print(f"      {'arm':34s} {'final retained':>18s} {'just-trained':>13s} "
              f"{'SynOps/sample':>14s}")
        for tag, a in consequence[bench]["arms"].items():
            fa = f"{a['final_average']*100:6.2f} +/- {a['final_average_ci95']*100:4.2f}"
            print(f"      {tag:34s} {fa:>18s} {a['diagonal_mean']*100:>12.2f}% "
                  f"{(a['synops_per_sample'] or 0):>14,.0f}")

    # ---- part 4: the quadratic-readout anomaly
    print("\n[4/4] QUADRATIC-READOUT ANOMALY CHECK (was it compensating for missing mixing?)...")
    quad = {"rows": []}
    for r in table:
        if r["kind"] == "identity" and r["gain"] not in (2.2, 6.0):
            continue
        if r["kind"] != "identity" and r["gain"] not in (2.2, 6.0):
            continue
        quad["rows"].append({
            "kind": r["kind"], "fan_in": r["fan_in"], "gain": r["gain"],
            "linear": r["linear"], "quadratic": r["quadratic"],
            "delta": r["quadratic_minus_linear"], "active_frac": r["active_frac"],
        })
    for r in quad["rows"]:
        print(f"    {r['kind']:9s} fan={r['fan_in']:4d} g={r['gain']:4.1f}: "
              f"lin={r['linear']:.3f} quad={r['quadratic']:.3f} "
              f"delta={r['delta']:+.3f} active={r['active_frac']*100:5.1f}%")
    deltas_ident = [r["delta"] for r in quad["rows"] if r["kind"] == "identity"]
    deltas_mixed = [r["delta"] for r in quad["rows"] if r["kind"] != "identity"]
    quad["mean_delta_identity"] = float(np.mean(deltas_ident)) if deltas_ident else None
    quad["mean_delta_mixed"] = float(np.mean(deltas_mixed)) if deltas_mixed else None

    # ---- verdict
    # Three separate questions, each decided by its own comparison. Conflating
    # them is how a real effect gets written up with the wrong cause.
    ident_base = next((r for r in table if r["kind"] == "identity" and r["gain"] == 2.2), None)
    zero_sorted = sorted(mac["zero_mixing_arms"], key=lambda z: -z["linear"])
    best_zero = zero_sorted[0] if zero_sorted else None
    # The fair comparison for "is it mixing?": the best zero-mixing arm whose
    # activity is within a couple of points of the winning mixed arm.
    near = [z for z in mac["zero_mixing_arms"]
            if abs(z["active_frac"] - winner["active_frac"]) < 0.02]
    best_zero_matched = max(near, key=lambda z: z["linear"]) if near else best_zero
    mixed_minus_zero = (winner["linear"] - best_zero_matched["linear"]) if best_zero_matched else None
    mixed_ci = winner["linear_ci95"] + (best_zero_matched["linear_ci95"] if best_zero_matched else 0.0)
    is_mixing = bool(mixed_minus_zero is not None and mixed_minus_zero > max(mixed_ci, 0.05))

    consequences = consequence["permuted"]["arms"]
    proj_arms = {t: a for t, a in consequences.items() if t not in ("mlp", "mlp_replay")}
    best_proj_tag = max(proj_arms, key=lambda t: proj_arms[t]["final_average"])
    best_proj = proj_arms[best_proj_tag]
    mlp_fa = consequences["mlp"]["final_average"]
    ident_fa = consequences["identity_control"]["final_average"]
    rescued = best_proj["final_average"] > mlp_fa
    gap = (mlp_fa - best_proj["final_average"]) * 100

    split_arms = consequence["split"]["arms"]
    split_proj = max((a for t, a in split_arms.items() if t not in ("mlp", "mlp_replay")),
                     key=lambda a: a["final_average"])

    # Polarity vs routing vs centering, straight from the control arms.
    def _at(kind, gain=2.2):
        return next((r for r in table if r["kind"] == kind and r["gain"] == gain), None)

    ident_c = _at("identity")
    perm_c, sign_c, both_c, centered_c = (_at("ctrl_perm"), _at("ctrl_sign"),
                                          _at("ctrl_both"), _at("ctrl_centered"))
    zero_best = max(
        [r for r in (ident_c, perm_c, sign_c, both_c, centered_c) if r],
        key=lambda r: r["linear"])
    polarity = {
        "identity": ident_c["linear"] if ident_c else None,
        "ctrl_perm": perm_c["linear"] if perm_c else None,
        "ctrl_centered": centered_c["linear"] if centered_c else None,
        "ctrl_sign": sign_c["linear"] if sign_c else None,
        "ctrl_both": both_c["linear"] if both_c else None,
        "best_zero_convergence_kind": zero_best["kind"],
        "best_zero_convergence_linear": zero_best["linear"],
        "polarity_effect": (max(sign_c["linear"], both_c["linear"]) - ident_c["linear"])
                           if (sign_c and both_c and ident_c) else None,
        "routing_effect": (perm_c["linear"] - ident_c["linear"])
                          if (perm_c and ident_c) else None,
        "centering_effect": (centered_c["linear"] - ident_c["linear"])
                            if (centered_c and ident_c) else None,
        "all_arms": [
            {"kind": r["kind"], "gain": r["gain"], "linear": r["linear"],
             "linear_ci95": r["linear_ci95"], "active_frac": r["active_frac"],
             "mean_count": r["mean_count"]}
            for r in (ident_c, perm_c, centered_c, sign_c, both_c) if r
        ],
    }

    verdict = {
        "q1_projection_effect_real": {
            "identity_g2.2_linear": ident_base["linear"] if ident_base else None,
            "identity_g2.2_ci95": ident_base["linear_ci95"] if ident_base else None,
            "best_mixed_linear": winner["linear"],
            "best_mixed_linear_ci95": winner["linear_ci95"],
            "delta": (winner["linear"] - ident_base["linear"]) if ident_base else None,
            "answer": bool(ident_base and winner["linear"] > ident_base["linear_ci95"] * 3),
        },
        "q2_is_it_convergence": {
            "polarity_decomposition": polarity,
            "best_mixed_linear": winner["linear"],
            "best_zero_mixing_linear": best_zero_matched["linear"] if best_zero_matched else None,
            "best_zero_mixing_kind": (best_zero_matched["kind"] if best_zero_matched else None),
            "best_zero_mixing_gain": (best_zero_matched["gain"] if best_zero_matched else None),
            "best_zero_mixing_active_frac": (best_zero_matched["active_frac"]
                                             if best_zero_matched else None),
            "mixed_active_frac": winner["active_frac"],
            "gap": mixed_minus_zero,
            "combined_ci95": mixed_ci,
            "answer_is_convergence": is_mixing,
        },
        "q3_consequence": {
            "best_projection_arm": best_proj_tag,
            "best_projection_permuted_final": best_proj["final_average"],
            "best_projection_permuted_ci95": best_proj["final_average_ci95"],
            "identity_control_permuted_final": ident_fa,
            "mlp_permuted_final": mlp_fa,
            "mlp_permuted_ci95": consequences["mlp"]["final_average_ci95"],
            "gap_to_mlp_pp": gap,
            "rescues_continual_learning": bool(rescued),
            "best_projection_split_final": split_proj["final_average"],
            "identity_control_split_final": split_arms["identity_control"]["final_average"],
            "mlp_split_final": split_arms["mlp"]["final_average"],
            "mlp_replay_split_final": split_arms["mlp_replay"]["final_average"],
        },
        "quadratic_anomaly": {
            "mean_delta_identity": quad["mean_delta_identity"],
            "mean_delta_mixed": quad["mean_delta_mixed"],
            "explained_by_missing_mixing": bool(
                quad["mean_delta_identity"] is not None
                and quad["mean_delta_mixed"] is not None
                and quad["mean_delta_identity"] > quad["mean_delta_mixed"]),
        },
    }

    pol = polarity["polarity_effect"]
    rout = polarity["routing_effect"]
    polarity_is_cause = bool(pol is not None and pol > 0.15 and (rout is None or pol > 2 * max(rout, 0.0)))
    if verdict["q1_projection_effect_real"]["answer"] and is_mixing:
        headline = ("STANDS (with a corrected cause): the identity 1:1 input map is a real "
                    "bottleneck AND convergence -- not merely gain, activity, routing or "
                    "polarity -- accounts for part of the gain.")
    elif verdict["q1_projection_effect_real"]["answer"] and polarity_is_cause:
        headline = ("PARTIALLY RETRACTED, CAUSE CORRECTED: the frozen random projection is a "
                    "real and large fix to the representation (identity "
                    f"{polarity['identity']:.3f} -> {winner['linear']:.3f}), but the operative "
                    "variable is NOT afferent convergence. It is receptive-field POLARITY: an "
                    "absolute-value projection with one pixel per neuron and NO convergence "
                    f"scores {max(polarity['ctrl_sign'], polarity['ctrl_both']):.3f}, while the "
                    f"same projection with unit (all-positive) polarity scores "
                    f"{polarity['identity']:.3f} and a pure pixel-routing permutation scores "
                    f"{polarity['ctrl_perm']:.3f}. The identity input path is rectified "
                    "(pixels are in [0,1], so every receptive field is positive), whereas real "
                    "cortical receptive fields are signed.")
    else:
        headline = ("RETRACTED: the mixed projection does not outperform identity once "
                    "activity and routing are controlled; the earlier measurement does not "
                    "replicate rigorously.")
    verdict["verdict"] = headline
    verdict["consequence_verdict"] = (
        "RETRACTED: the projection rescues continual learning"
        if rescued else
        "STANDS: the projection does NOT rescue continual learning -- the substrate remains "
        f"{gap:.2f} points behind the parameter-matched MLP on Permuted-MNIST"
    )

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    q1, q2, q3 = verdict["q1_projection_effect_real"], verdict["q2_is_it_convergence"], verdict["q3_consequence"]
    print(f"  Q1. Is the identity input map a real bottleneck?")
    print(f"      identity g=2.2 {q1['identity_g2.2_linear']:.3f} -> best mixed "
          f"{q1['best_mixed_linear']:.3f}  (delta {q1['delta']:+.3f})  => {'YES' if q1['answer'] else 'NO'}")
    print(f"  Q2. Is the gain due to CONVERGENCE (mixing several inputs)?")
    print(f"      identity {polarity['identity']:.3f} | permuted routing {polarity['ctrl_perm']:.3f} "
          f"| centered {polarity['ctrl_centered']:.3f} | SIGNED {polarity['ctrl_sign']:.3f} "
          f"| signed+permuted {polarity['ctrl_both']:.3f}   (all 1 pixel/neuron except identity)")
    print(f"      polarity effect {polarity['polarity_effect']:+.3f} vs routing effect "
          f"{polarity['routing_effect']:+.3f}")
    print(f"      best mixed {q2['best_mixed_linear']:.3f} at {q2['mixed_active_frac']*100:.1f}% active")
    print(f"      best zero-convergence {q2['best_zero_mixing_linear']:.3f} "
          f"({q2['best_zero_mixing_kind']} g={q2['best_zero_mixing_gain']}) at "
          f"{q2['best_zero_mixing_active_frac']*100:.1f}% active  "
          f"(gap {q2['gap']:+.3f}, combined CI {q2['combined_ci95']:.3f})")
    print(f"      => {'MIXING IS THE CAUSE' if is_mixing else 'NOT SUPPORTED: random routing/polarity suffice'}")
    print(f"  Q3. Does the winning projection rescue continual learning?")
    print(f"      {q3['best_projection_arm']}: {q3['best_projection_permuted_final']*100:.2f}% "
          f"vs identity {q3['identity_control_permuted_final']*100:.2f}% "
          f"vs MLP {q3['mlp_permuted_final']*100:.2f}% (permuted)")
    print(f"      => {verdict['consequence_verdict']}")
    print(f"\n  HEADLINE: {headline}")

    payload = {
        "config": {
            "n_input": N_INPUT, "n_neurons": N_NEURONS, "t_ms": T_MS,
            "n_train": N_TRAIN, "n_test": N_TEST, "seeds": args.seeds,
            "quick": bool(args.quick), "modern_protocol": bool(args.modern_protocol),
            "backend": "numpy", "workers": workers,
        },
        "published_reference": ref,
        "backend_parity": parity,
        "sweep": table,
        "sweep_raw_runs": len(rows),
        "winner": winner,
        "matched_activity": mac,
        "consequence": consequence,
        "quadratic_check": quad,
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - t_start,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = RESULTS_DIR / "afferents.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)} ({out.stat().st_size/1024:.0f} KB) "
          f"in {payload['wall_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
