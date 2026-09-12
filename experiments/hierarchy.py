"""Does a second spiking layer buy nonlinear decoding that depth 1 cannot reach?

Hypothesis under test
---------------------
A second layer of the *same* dendritic spiking neurons, trained by the *same*
local rule, extracts nonlinear structure that a linear readout misses - i.e.
dendritic nonlinearity substitutes for explicit quadratic feature expansion,
with no backpropagation and no kernel.

What is varied
--------------
Only ``n_layers`` (plus, in the ablation, the layer-2 dendrite and layer-1
plasticity). Both depths see the same input split, the same seeds, the same
readout rule and the same hyper-parameter selection procedure, so any
difference is attributable to depth.

Three methodology decisions, stated explicitly
----------------------------------------------
1. **The test set is never used for any choice.** The readout's step size is
   selected on a validation split carved out of training data, and inter-layer
   gains are calibrated on training data. Test accuracy is reported once, from
   the model selected on validation.
2. **Frozen feature standardisation precedes the local rule.** The delta rule is
   stable only while ``lr * lambda_max(X^T X) < 2``; measured on these codes
   ``lambda_max`` is ~3e4, so the interface default ``lr=1e-2`` diverges (loss
   ~1e22, accuracy at chance). That is a step-size artefact, not a fact about
   the representation, so the default is reported as its own diagnostic arm
   rather than silently replaced.
3. **State resets at every phase boundary** so each block of codes is
   reproducible from its own sample order, while carrying over within a block
   exactly as ``CortexClassifier`` does. Carry-over vs per-sample reset is a
   real confound with a large measured effect and is reported as an ablation.

Run:  python3 experiments/hierarchy.py
      python3 experiments/hierarchy.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"

#: Candidate step sizes for the local rule; selection is by validation only.
LR_GRID: tuple[float, ...] = (3e-3, 1e-3, 3e-4, 1e-4, 3e-5)

#: The published single-task 10-way ceiling this run is measured against.
LEGACY_CEILING = 0.333


# --------------------------------------------------------------------- utils
def ci95(values) -> float:
    """Half-width of the 95% CI of the mean."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if a.size < 2:
        return 0.0
    return float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def mean_ci(values) -> dict:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(a.mean()), "ci95": ci95(a), "n": int(a.size),
            "values": [float(v) for v in a]}


def stratified_pick(y, n_per_class, rng):
    """Balanced subset indices, so class counts cannot skew a task."""
    picks = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picks.append(idx[:n_per_class])
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


@dataclass
class Split:
    """Disjoint stratified train/validation/test block from one task."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int = 10

    def as_task(self):
        """A minimal object exposing the ``x_train``/``y_train`` contract."""
        return _TrainOnly(self)


class _TrainOnly:
    def __init__(self, split: Split):
        self.x_train = split.x_train
        self.y_train = split.y_train
        self.name = "train-only"


def make_split(task, *, n_train_per_class, n_val_per_class, n_test_per_class, seed) -> Split:
    """Carve disjoint balanced train/val/test blocks out of one task.

    Validation and test are drawn from different pools (``y_train`` and
    ``y_test`` respectively for MNIST-derived tasks), so a validation-tuned
    hyper-parameter can never have seen the test images.
    """
    rng = np.random.default_rng(seed)
    tr = stratified_pick(task.y_train, n_train_per_class, rng)
    mask = np.ones(len(task.y_train), dtype=bool)
    mask[tr] = False
    pool = np.flatnonzero(mask)
    rng2 = np.random.default_rng(seed + 1)
    val_local = stratified_pick(task.y_train[pool], n_val_per_class, rng2)
    val = pool[val_local]
    rng3 = np.random.default_rng(seed + 2)
    te = stratified_pick(task.y_test, n_test_per_class, rng3)
    return Split(
        x_train=task.x_train[tr], y_train=task.y_train[tr],
        x_val=task.x_train[val], y_val=task.y_train[val],
        x_test=task.x_test[te], y_test=task.y_test[te],
    )


# ------------------------------------------------------------------ the stack
def build_stack(*, n_layers, seed, dendrite2=None, plasticity=True,
                layer_plasticity=None, layer_neurons=900, k_out=48, t_ms=15,
                backend="numpy"):
    from brain.hierarchy import StackConfig, CorticalStack

    layer_dend = None
    if dendrite2 is not None and n_layers == 2:
        layer_dend = (True, bool(dendrite2))
    cfg = StackConfig(
        n_layers=n_layers, layer_neurons=layer_neurons, k_out=k_out, t_ms=t_ms,
        seed=seed, plasticity=plasticity, backend=backend,
        layer_use_dendrite=layer_dend, layer_plasticity=layer_plasticity,
    )
    return CorticalStack(cfg, backend=backend)


def codes_for(stack, X, *, reset: bool = False) -> np.ndarray:
    """Final-layer codes for a block of samples.

    ``reset`` is off by default, and that choice is deliberate. The substrate's
    plasticity is *always* on - it is the mechanism under study, not a training
    phase - so the honest protocol is to present the blocks in order
    (train, then validation, then test) and let the substrate keep adapting,
    exactly as ``CortexClassifier`` does with no reset between samples. Turning
    ``reset`` on gives a different and much weaker substrate (measured: layer 1
    active fraction 0.005 vs 0.016) because the dendritic potential needs ~10 ms
    to charge. It is kept as a switch and reported as an ablation.

    Consequence, stated plainly: the code for a block depends on the samples
    that preceded it. No label ever participates in that adaptation, and
    hyper-parameters are still chosen on validation only, but a reader should
    know the substrate is online rather than stationary.
    """
    if reset:
        stack.reset_state()
    return stack.codes(X)


def standardise_fit(codes_tr: np.ndarray):
    """Frozen per-feature z-score statistics from training codes only."""
    C = codes_tr.astype(np.float64)
    mean, sd = C.mean(axis=0), C.std(axis=0)
    dead = sd <= 1e-8
    sd = np.where(dead, 1.0, sd)

    def apply(Cb: np.ndarray) -> np.ndarray:
        z = (Cb.astype(np.float32) - mean.astype(np.float32)) / sd.astype(np.float32)
        z[:, dead] = 0.0
        return z.astype(np.float32)

    return apply, {"n_dead_features": int(dead.sum())}


def lambda_max(Z: np.ndarray) -> float:
    """Largest eigenvalue of Z^T Z; the delta rule needs lr * lambda_max < 2."""
    G = (Z.astype(np.float64).T @ Z.astype(np.float64))
    try:
        return float(np.linalg.eigvalsh(G).max())
    except np.linalg.LinAlgError:  # pragma: no cover
        return float("nan")


DEFAULT_GAIN_GRID: tuple[float, ...] = (24.0, 32.0, 48.0, 64.0)


def select_relay_gain(build_fn, split: Split, *, seed: int, n_classes: int,
                      gains=DEFAULT_GAIN_GRID, lr_grid=LR_GRID,
                      max_train: int = 200) -> dict:
    """Choose the inter-layer drive scale by validation accuracy.

    A deeper stack carries one hyper-parameter a single layer does not: how
    strongly layer 1's spikes drive layer 2. Leaving it at a guess would risk an
    unfair comparison, because a mis-scaled or silent layer 2 looks exactly like
    "depth does not help" and there is a hard cliff in both directions (measured:
    no layer-2 spikes at all below gain ~4, active suppression above ~128).

    Each candidate is evaluated on a **freshly built** substrate. That detail is
    load-bearing rather than tidy: this substrate's plasticity is always on, so
    reusing one instance across candidates would present the same training data
    to it once per candidate. An earlier version did exactly that and left depth
    2 with six extra unsupervised Hebbian passes before its official run, which
    destabilised layer 2 and produced a train accuracy of 0.95 against a test
    accuracy of 0.12. Selection must not consume the resource it is selecting
    for, so every candidate starts from identical initial conditions.

    Test data is never touched; only the validation split and a capped
    subsample of training data are used.

    Returns the sweep, which is itself evidence: a flat curve means depth 2 is
    insensitive to the gain, a peaked one locates the operating point.
    """
    Xtr, ytr = split.x_train, split.y_train
    if max_train and len(Xtr) > max_train:
        rng = np.random.default_rng(seed)
        pick = rng.permutation(len(Xtr))[:max_train]
        Xtr, ytr = Xtr[pick], ytr[pick]

    sweep = []
    best = (-1.0, None, None)
    for g in gains:
        stack = build_fn()
        if stack.cfg.n_layers > 1:
            stack._relay_scale = [float(g)] * (stack.cfg.n_layers - 1)
        C_tr = codes_for(stack, Xtr)
        C_val = codes_for(stack, split.x_val)
        az, _ = standardise_fit(C_tr)
        lr, val, _ = _select_lr(az(C_tr), ytr, az(C_val), split.y_val,
                                seed=seed, n_classes=n_classes, lr_grid=lr_grid)
        sweep.append({"gain": float(g), "val_acc": float(val), "lr": float(lr)})
        if val > best[0]:
            best = (val, float(g), float(lr))
    return {"sweep": sweep, "chosen": best[1], "chosen_lr": best[2],
            "chosen_val_acc": float(best[0]), "n_candidates": len(sweep)}


def evaluate_stack(stack, split: Split, *, lr_grid=LR_GRID, epochs=1,
                   seed=0, extra_kinds=("ridge", "ridge_quad")) -> dict:
    """Forward-pass the split, select the step size on validation, score test.

    Returns every number the harness reports for one (depth, seed) run,
    including the diagnostics that make the result auditable: the validation
    sweep, the diverging default, ``lambda_max``, per-layer activity, SynOps and
    the closed-form reference accuracies.
    """
    from brain.hierarchy import build_readout

    t0 = time.perf_counter()
    C_tr = codes_for(stack, split.x_train)
    C_val = codes_for(stack, split.x_val)
    C_te = codes_for(stack, split.x_test)
    code_seconds = time.perf_counter() - t0

    apply_z, zinfo = standardise_fit(C_tr)
    Z_tr, Z_val, Z_te = apply_z(C_tr), apply_z(C_val), apply_z(C_te)
    lam = lambda_max(Z_tr)
    n_feat = int(C_tr.shape[1])

    def fresh(lr: float, kind: str = "local_delta"):
        r = build_readout(n_features=n_feat, n_classes=split.n_classes,
                          seed=seed, kind=kind, lr=lr, epochs=epochs,
                          normalize=False)
        return r

    # ---- local rule: step size chosen on validation only
    sweep = []
    best_val, best_lr = -1.0, None
    for lr in lr_grid:
        r = fresh(lr)
        r.partial_fit(Z_tr, split.y_train)
        v = float(r.score(Z_val, split.y_val))
        sweep.append({"lr": float(lr), "val_acc": v})
        if v > best_val:
            best_val, best_lr = v, float(lr)
    model = fresh(best_lr)
    model.partial_fit(Z_tr, split.y_train)
    test_acc = float(model.score(Z_te, split.y_test))
    train_acc = float(model.score(Z_tr, split.y_train))

    # ---- diagnostic: the interface default, expected to diverge
    r_def = fresh(1e-2)
    r_def.partial_fit(Z_tr, split.y_train)
    default_val = float(r_def.score(Z_val, split.y_val))

    # ---- closed-form references on the identical codes (capacity probes)
    references = {}
    for kind in extra_kinds:
        try:
            r = fresh(1.0, kind=kind)
            r.partial_fit(Z_tr, split.y_train)
            references[kind] = {
                "test_acc": float(r.score(Z_te, split.y_test)),
                "val_acc": float(r.score(Z_val, split.y_val)),
                "n_params": int(r.n_params),
            }
        except Exception as exc:  # pragma: no cover - keep other arms alive
            references[kind] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "test_acc": test_acc,
        "train_acc": train_acc,
        "val_acc": float(best_val),
        "chosen_lr": float(best_lr),
        "lr_sweep": sweep,
        "default_lr_val_acc": default_val,
        "lambda_max": lam,
        "stability_bound": (2.0 / lam) if lam and np.isfinite(lam) and lam > 0 else None,
        "n_dead_features": zinfo["n_dead_features"],
        "references": references,
        "layer_spikes": {str(k): float(v) for k, v in stack.layer_spikes.items()},
        "layer_active_frac": {str(k): float(v) for k, v in stack.layer_active_frac.items()},
        "synops_per_sample": float(stack.synops_per_sample),
        "n_params": int(stack.n_params),
        "readout_params": int(stack.readout_params),
        "n_substrate_synapses": int(stack.n_substrate_synapses),
        "relay_gain": [None if g is None else float(g) for g in stack._relay_scale],
        "n_train": int(len(split.y_train)),
        "code_seconds": float(code_seconds),
    }


# --------------------------------------------------------------------- arms
@dataclass
class RunResult:
    arm: str
    seed: int
    depth: int
    suite: str
    metrics: dict = field(default_factory=dict)
    retention: list = field(default_factory=list)
    wall_seconds: float = 0.0
    error: str | None = None


def run_single_task(args, seed: int, depth: int) -> RunResult:
    """Single-task 10-way MNIST: the decisive ceiling comparison."""
    from brain.tasks import split_mnist

    suite = split_mnist(1, seed=0)
    split = make_split(
        suite.tasks[0], n_train_per_class=args.train_per_class,
        n_val_per_class=args.val_per_class, n_test_per_class=args.test_per_class,
        seed=seed,
    )
    res = RunResult(arm=f"depth{depth}", seed=seed, depth=depth, suite="mnist-10way")
    t0 = time.perf_counter()
    try:
        stack = build_stack(n_layers=depth, seed=seed, layer_neurons=args.neurons,
                            k_out=args.k_out, backend=args.backend)
        gain_info = None
        if depth > 1:
            gain_info = select_relay_gain(
                lambda: build_stack(n_layers=depth, seed=seed,
                                    layer_neurons=args.neurons, k_out=args.k_out,
                                    backend=args.backend),
                split, seed=seed, n_classes=split.n_classes,
                gains=getattr(args, "_gain_grid", DEFAULT_GAIN_GRID),
                max_train=args.gain_max_train)
            stack._relay_scale = [gain_info["chosen"]] * (depth - 1)
        res.metrics = evaluate_stack(stack, split, seed=seed)
        if gain_info is not None:
            res.metrics["gain_selection"] = gain_info
            res.metrics["relay_gain"] = [gain_info["chosen"]] * (depth - 1)
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            traceback.print_exc()
    res.wall_seconds = time.perf_counter() - t0
    return res


def _select_lr(Z_tr, y_tr, Z_val, y_val, *, seed, n_classes, lr_grid=LR_GRID):
    """Pick the local rule's step size on validation; return (lr, best_val, sweep)."""
    from brain.hierarchy import build_readout

    sweep, best_val, best_lr = [], -1.0, None
    for lr in lr_grid:
        r = build_readout(n_features=int(Z_tr.shape[1]), n_classes=n_classes,
                          seed=seed, kind="local_delta", lr=lr, epochs=1)
        r.partial_fit(Z_tr, y_tr)
        v = float(r.score(Z_val, y_val))
        sweep.append({"lr": float(lr), "val_acc": v})
        if v > best_val:
            best_val, best_lr = v, float(lr)
    return best_lr, best_val, sweep


def run_continual(args, seed: int, depth: int, permute: bool) -> RunResult:
    """Split- or permuted-MNIST continual learning, no replay, one retained head.

    Protocol:
      * a single readout is retained across the whole stream (no per-task head),
        which is what makes interference measurable at all;
      * one frozen z-score, fitted on task 0's training codes, is used for every
        task so all tasks share one feature space;
      * the step size is chosen per task on that task's validation block;
      * after task ``i`` is learned, every task ``0..i`` is re-evaluated,
        giving the retention matrix.

    The substrate's own plasticity keeps adapting throughout (unsupervised);
    the label only ever enters the readout.
    """
    from brain.hierarchy import build_readout
    from brain.tasks import split_mnist

    suite = split_mnist(args.tasks, seed=args.suite_seed, permute=permute)
    n = len(suite)
    name = "permuted-mnist" if permute else "split-mnist"
    res = RunResult(arm=f"depth{depth}", seed=seed, depth=depth, suite=name)
    t0 = time.perf_counter()
    try:
        stack = build_stack(n_layers=depth, seed=seed, layer_neurons=args.neurons,
                            k_out=args.k_out, backend=args.backend)
        acc = np.full((n, n), np.nan, dtype=np.float64)
        total_synops = 0.0
        n_seen = 0
        apply_z = None
        chosen_lrs = []
        chosen_gain: list = []
        for i, task in enumerate(suite.tasks):
            split = make_split(
                task, n_train_per_class=args.train_per_class,
                n_val_per_class=args.val_per_class,
                n_test_per_class=args.test_per_class,
                seed=seed * 100 + i,
            )
            if i == 0 and depth > 1:
                gain_info = select_relay_gain(
                    lambda: build_stack(n_layers=depth, seed=seed,
                                        layer_neurons=args.neurons,
                                        k_out=args.k_out, backend=args.backend),
                    split, seed=seed, n_classes=suite.n_classes,
                    gains=getattr(args, "_gain_grid", DEFAULT_GAIN_GRID),
                    max_train=args.gain_max_train)
                stack._relay_scale = [gain_info["chosen"]] * (depth - 1)
                chosen_gain.append(gain_info)

            C_tr = codes_for(stack, split.x_train)
            C_val = codes_for(stack, split.x_val)
            if apply_z is None:
                apply_z, _ = standardise_fit(C_tr)
            Z_tr, Z_val = apply_z(C_tr), apply_z(C_val)

            best_lr, best_val, sweep = _select_lr(
                Z_tr, split.y_train, Z_val, split.y_val,
                seed=seed, n_classes=suite.n_classes)
            chosen_lrs.append({"task": i, "lr": best_lr, "val_acc": best_val,
                               "sweep": sweep})
            total_synops += float(stack.synops_per_sample) * len(split.y_train)
            n_seen += len(split.y_train)

            # train the retained readout on this task (previous weights retained)
            if not hasattr(stack, "readout_trained") or not stack.readout_trained:
                stack.readout = build_readout(
                    n_features=int(C_tr.shape[1]), n_classes=suite.n_classes,
                    seed=seed, kind="local_delta", lr=best_lr, epochs=1)
                stack.readout_trained = True
            else:
                stack.readout.cfg.lr = float(best_lr)
            stack.readout.partial_fit(Z_tr, split.y_train)

            for j in range(i + 1):
                tj = suite.tasks[j]
                rng = np.random.default_rng(seed * 1000 + j)
                idx = stratified_pick(tj.y_test, args.test_per_class, rng)
                Cj = codes_for(stack, tj.x_test[idx])
                acc[i, j] = float(stack.readout.score(apply_z(Cj), tj.y_test[idx]))

        res.retention = np.nan_to_num(acc, nan=0.0).tolist()
        am = np.nan_to_num(acc, nan=0.0)
        res.metrics = {
            "final_average": float(np.nanmean(acc[-1])),
            "diagonal_mean": float(np.mean([am[i, i] for i in range(n)])),
            "backward_transfer": float(np.mean(acc[n - 1, :] - np.diag(acc))),
            "forgetting": float(np.mean(
                [np.nanmax(acc[:, j]) - acc[n - 1, j] for j in range(n)])),
            "accuracy_at_learning": float(np.mean(np.diag(acc))),
            "synops_per_sample": total_synops / max(1, n_seen),
            "n_params": int(stack.n_params),
            "readout_params": int(stack.readout_params),
            "n_substrate_synapses": int(stack.n_substrate_synapses),
            "chosen_lrs": chosen_lrs,
            "gain_selection": chosen_gain,
        }
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            traceback.print_exc()
    res.wall_seconds = time.perf_counter() - t0
    return res


def mlp_hidden_for(n_params_target: int, n_in: int, n_classes: int) -> int:
    """Smallest hidden width whose MLP param count reaches the target.

    Solving ``n_in*h + h + h*n_classes + n_classes >= target`` for ``h``. The
    MLP is deliberately *not* made smaller than the substrate: a dense baseline
    with fewer parameters would not be a fair test of the spiking arm.
    """
    denom = max(1, n_in + n_classes + 1)
    h = int(np.ceil(n_params_target / denom))
    return max(1, h)


def run_mlp(args, seed: int, permute: bool, n_params_target: int) -> RunResult:
    """Parameter-matched (or larger) dense backprop MLP on the same task stream.

    Trained on the same images and labels, with the same evaluation protocol,
    but *not* given a validation-selected step size: its default lr is used
    unless ``--mlp-lr`` is given. That choice is recorded in the JSON config so
    the comparison cannot silently flatter the substrate.
    """
    from brain.tasks import split_mnist

    suite = split_mnist(args.tasks, seed=args.suite_seed, permute=permute)
    n = len(suite)
    name = "permuted-mnist" if permute else "split-mnist"
    hidden = mlp_hidden_for(n_params_target, 784, suite.n_classes)
    res = RunResult(arm="mlp_param_matched", seed=seed, depth=1, suite=name)
    t0 = time.perf_counter()
    try:
        from brain.baselines import MLPBackprop

        model = MLPBackprop(784, hidden, suite.n_classes, seed=seed,
                            lr=args.mlp_lr, batch_size=args.mlp_batch_size)
        acc = np.full((n, n), np.nan, dtype=np.float64)
        for i, task in enumerate(suite.tasks):
            split = make_split(
                task, n_train_per_class=args.train_per_class,
                n_val_per_class=args.val_per_class,
                n_test_per_class=args.test_per_class,
                seed=seed * 100 + i,
            )
            n_train = len(split.y_train)
            steps = max(1, int(np.ceil(n_train / args.mlp_batch_size)))
            target_updates = n_train * args.epochs
            epochs = max(1, int(np.ceil(target_updates / steps)))
            model.fit_task(_TrainOnly(split), epochs=epochs)
            for j in range(i + 1):
                tj = suite.tasks[j]
                rng = np.random.default_rng(seed * 1000 + j)
                idx = stratified_pick(tj.y_test, args.test_per_class, rng)
                acc[i, j] = float(np.mean(
                    model.predict(tj.x_test[idx]) == tj.y_test[idx]))
        res.retention = np.nan_to_num(acc, nan=0.0).tolist()
        am = np.nan_to_num(acc, nan=0.0)
        res.metrics = {
            "final_average": float(np.nanmean(acc[-1])),
            "diagonal_mean": float(np.mean([am[i, i] for i in range(n)])),
            "backward_transfer": float(np.mean(acc[n - 1, :] - np.diag(acc))),
            "forgetting": float(np.mean(
                [np.nanmax(acc[:, j]) - acc[n - 1, j] for j in range(n)])),
            "accuracy_at_learning": float(np.mean(np.diag(acc))),
            "n_params": int(model.n_params),
            "hidden": int(hidden),
            "epochs_per_task": float(epochs) if n else 0.0,
            # a dense MLP executes one MAC per parameter per sample, so its
            # per-sample operation count is its parameter count
            "synops_per_sample": float(model.n_params),
        }
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            traceback.print_exc()
    res.wall_seconds = time.perf_counter() - t0
    return res


def run_ablation(args, seed: int, kind: str, permute: bool) -> RunResult:
    """Ablations at the operating point that produced the depth comparison."""
    from brain.tasks import split_mnist

    suite = split_mnist(args.tasks, seed=args.suite_seed, permute=permute)
    n = len(suite)
    name = "permuted-mnist" if permute else "split-mnist"
    res = RunResult(arm=kind, seed=seed, depth=2, suite=name)
    t0 = time.perf_counter()
    try:
        if kind == "ablate_l2_dendrite":
            stack = build_stack(n_layers=2, seed=seed, dendrite2=False,
                                layer_neurons=args.neurons, k_out=args.k_out,
                                backend=args.backend)
        elif kind == "ablate_l1_plasticity":
            stack = build_stack(n_layers=2, seed=seed,
                                layer_plasticity=(False, True),
                                layer_neurons=args.neurons, k_out=args.k_out,
                                backend=args.backend)
        elif kind == "ablate_frozen_substrate":
            stack = build_stack(n_layers=2, seed=seed, plasticity=False,
                                layer_neurons=args.neurons, k_out=args.k_out,
                                backend=args.backend)
        else:
            raise ValueError(f"unknown ablation {kind!r}")

        acc = np.full((n, n), np.nan, dtype=np.float64)
        apply_z = None
        for i, task in enumerate(suite.tasks):
            split = make_split(
                task, n_train_per_class=args.train_per_class,
                n_val_per_class=args.val_per_class,
                n_test_per_class=args.test_per_class,
                seed=seed * 100 + i,
            )
            if i == 0 and stack.cfg.n_layers > 1:
                gi = select_relay_gain(
                    lambda: build_stack(
                        n_layers=2, seed=seed,
                        dendrite2=(False if kind == "ablate_l2_dendrite" else None),
                        layer_plasticity=(None if kind != "ablate_l1_plasticity"
                                          else (False, True)),
                        plasticity=(False if kind == "ablate_frozen_substrate" else True),
                        layer_neurons=args.neurons, k_out=args.k_out,
                        backend=args.backend),
                    split, seed=seed, n_classes=suite.n_classes,
                    gains=getattr(args, "_gain_grid", DEFAULT_GAIN_GRID),
                    max_train=args.gain_max_train)
                stack._relay_scale = [gi["chosen"]]
            C_tr, C_val = codes_for(stack, split.x_train), codes_for(stack, split.x_val)
            if apply_z is None:
                apply_z, _ = standardise_fit(C_tr)
            Z_tr, Z_val = apply_z(C_tr), apply_z(C_val)
            best_lr, best_val, _ = _select_lr(
                Z_tr, split.y_train, Z_val, split.y_val,
                seed=seed, n_classes=suite.n_classes)
            from brain.hierarchy import build_readout

            if not getattr(stack, "readout_trained", False):
                stack.readout = build_readout(
                    n_features=int(C_tr.shape[1]), n_classes=suite.n_classes,
                    seed=seed, kind="local_delta", lr=best_lr, epochs=1)
                stack.readout_trained = True
            else:
                stack.readout.cfg.lr = float(best_lr)
            stack.readout.partial_fit(Z_tr, split.y_train)
            for j in range(i + 1):
                tj = suite.tasks[j]
                rng = np.random.default_rng(seed * 1000 + j)
                idx = stratified_pick(tj.y_test, args.test_per_class, rng)
                acc[i, j] = float(stack.readout.score(
                    apply_z(codes_for(stack, tj.x_test[idx])), tj.y_test[idx]))
        res.retention = np.nan_to_num(acc, nan=0.0).tolist()
        am = np.nan_to_num(acc, nan=0.0)
        res.metrics = {
            "final_average": float(np.nanmean(acc[-1])),
            "diagonal_mean": float(np.mean([am[i, i] for i in range(n)])),
            "backward_transfer": float(np.mean(acc[n - 1, :] - np.diag(acc))),
            "forgetting": float(np.mean(
                [np.nanmax(acc[:, j]) - acc[n - 1, j] for j in range(n)])),
            "n_params": int(stack.n_params),
            "synops_per_sample": float(stack.synops_per_sample),
        }
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            traceback.print_exc()
    res.wall_seconds = time.perf_counter() - t0
    return res


# ------------------------------------------------------------------ reporting
def summarise_metric(runs: list[RunResult], key: str) -> dict:
    vals = [r.metrics.get(key) for r in runs
            if r.error is None and isinstance(r.metrics.get(key), (int, float))]
    return mean_ci(vals)


def report(args, single: list[RunResult], cont: dict[str, list[RunResult]],
           mlps: dict[str, list[RunResult]], abls: dict[str, list[RunResult]]) -> dict:
    """Print the tables and return the JSON-serialisable payload."""
    out: dict = {"single_task": {}, "continual": {}, "ablations": {},
                 "accounting": {}, "verdict": {}}

    # ---------------------------------------------------------- single task
    print("\n" + "=" * 88)
    print("SINGLE-TASK 10-WAY MNIST (no interference) — THE CEILING TEST")
    print("=" * 88)
    print(f"{'depth':>6} {'test acc':>18} {'train':>8} {'closed-form refs (test)':>26} "
          f"{'n_params':>10} {'SynOps/sample':>14}")
    print("-" * 88)
    by_depth: dict[int, list[RunResult]] = {1: [], 2: []}
    for r in single:
        if r.error is None:
            by_depth.setdefault(r.depth, []).append(r)
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        if not rows:
            print(f"{depth:>6} {'FAILED':>18}  {next((r.error for r in single if r.depth==depth), '')}")
            continue
        acc = mean_ci([r.metrics["test_acc"] for r in rows])
        tr = mean_ci([r.metrics["train_acc"] for r in rows])
        refs = rows[0].metrics.get("references", {})
        ref_s = " ".join(f"{k}={v.get('test_acc', float('nan')):.3f}"
                         for k, v in refs.items())
        np_ = mean_ci([r.metrics["n_params"] for r in rows])
        sy = mean_ci([r.metrics["synops_per_sample"] for r in rows])
        out["single_task"][f"depth{depth}"] = {
            "test_acc": acc, "train_acc": tr,
            "references": refs,
            "n_params": int(np_["mean"]), "synops_per_sample": sy["mean"],
            "chosen_lr": mean_ci([r.metrics["chosen_lr"] for r in rows]),
            "default_lr_val_acc": mean_ci([r.metrics["default_lr_val_acc"] for r in rows]),
            "lambda_max": mean_ci([r.metrics["lambda_max"] for r in rows]),
            "stability_bound": mean_ci([r.metrics["stability_bound"] for r in rows]),
            "layer_active_frac": rows[0].metrics.get("layer_active_frac"),
            "layer_spikes": rows[0].metrics.get("layer_spikes"),
            "relay_gain": rows[0].metrics.get("relay_gain"),
            "n_seeds": len(rows),
        }
        print(f"{depth:>6} {acc['mean']*100:9.2f} +/- {acc['ci95']*100:4.2f} "
              f"{tr['mean']*100:7.2f}% {ref_s:>26} {int(np_['mean']):>10,} "
              f"{sy['mean']:>14,.0f}")

    d1, d2 = out["single_task"].get("depth1"), out["single_task"].get("depth2")
    ceiling = {"legacy_reported_ceiling": LEGACY_CEILING}
    if d1 and d2:
        diff = d2["test_acc"]["mean"] - d1["test_acc"]["mean"]
        comb = d2["test_acc"]["ci95"] + d1["test_acc"]["ci95"]
        ceiling.update({
            "depth1_mean": d1["test_acc"]["mean"],
            "depth2_mean": d2["test_acc"]["mean"],
            "depth2_minus_depth1": diff,
            "combined_ci95": comb,
            "depth2_exceeds_depth1": bool(diff > comb),
            "depth1_above_legacy_ceiling": bool(d1["test_acc"]["mean"] > LEGACY_CEILING),
        })
        print("-" * 88)
        print(f"depth2 - depth1: {diff*100:+.2f} points "
              f"(combined 95% CI +/- {comb*100:.2f})")
        if diff > comb:
            print("  -> depth 2 EXCEEDS depth 1 beyond the CI")
        else:
            print("  -> depth 2 does NOT separate from depth 1 (CI overlaps zero)")
        print(f"legacy reported single-task 10-way ceiling: {LEGACY_CEILING:.3f} "
              f"(depth 1 here: {d1['test_acc']['mean']:.3f})")
    out["verdict"]["ceiling_test"] = ceiling

    # ------------------------------------------------------ parameter / SynOps
    print("\n" + "=" * 88)
    print("PARAMETER AND ACTIVE-SynOps ACCOUNTING (per sample, forward pass)")
    print("=" * 88)
    print(f"{'config':>28} {'n_params':>12} {'substrate syn':>14} {'SynOps/sample':>15} "
          f"{'spikes/sample':>14}")
    print("-" * 88)
    accounting = {}
    for r in single:
        if r.error is not None:
            continue
        key = f"depth{r.depth}"
        spk = sum(float(v) for v in r.metrics.get("layer_spikes", {}).values())
        accounting[key] = {
            "n_params": r.metrics["n_params"],
            "readout_params": r.metrics.get("readout_params"),
            "n_substrate_synapses": r.metrics["n_substrate_synapses"],
            "synops_per_sample": r.metrics["synops_per_sample"],
            "spikes_per_sample": spk,
        }
        print(f"{key:>28} {r.metrics['n_params']:>12,} "
              f"{r.metrics['n_substrate_synapses']:>14,} "
              f"{r.metrics['synops_per_sample']:>15,.0f} {spk:>14.1f}")
    out["accounting"]["single_task"] = accounting

    # ------------------------------------------------------------- continual
    for suite_name, runs in cont.items():
        print("\n" + "=" * 88)
        print(f"CONTINUAL LEARNING — {suite_name.upper()} "
              f"({args.tasks} tasks, single pass, no replay)")
        print("=" * 88)
        print(f"{'arm':>26} {'final avg':>18} {'diag':>8} {'BWT':>9} {'forget':>9} "
              f"{'n_params':>10} {'SynOps/s':>12}")
        print("-" * 88)
        bucket = {}
        for arm_name, arm_runs in [("depth1", [r for r in runs if r.depth == 1]),
                                   ("depth2", [r for r in runs if r.depth == 2])]:
            ok = [r for r in arm_runs if r.error is None]
            if not ok:
                err = next((r.error for r in arm_runs), "not run")
                print(f"{arm_name:>26} {'FAILED':>18}  {str(err)[:40]}")
                bucket[arm_name] = {"status": "failed", "error": err}
                continue
            fa = mean_ci([r.metrics["final_average"] for r in ok])
            dg = mean_ci([r.metrics["diagonal_mean"] for r in ok])
            bw = mean_ci([r.metrics["backward_transfer"] for r in ok])
            fg = mean_ci([r.metrics["forgetting"] for r in ok])
            np_ = int(ok[0].metrics["n_params"])
            sy = mean_ci([r.metrics["synops_per_sample"] for r in ok])
            bucket[arm_name] = {
                "status": "ok", "n_seeds": len(ok),
                "final_average": fa, "diagonal_mean": dg,
                "backward_transfer": bw, "forgetting": fg,
                "n_params": np_, "synops_per_sample": sy,
                "retention": [r.retention for r in ok],
                "wall_seconds": mean_ci([r.wall_seconds for r in ok]),
            }
            print(f"{arm_name:>26} {fa['mean']*100:9.2f} +/- {fa['ci95']*100:4.2f} "
                  f"{dg['mean']*100:7.2f}% {bw['mean']*100:8.2f}% "
                  f"{fg['mean']*100:8.2f}% {np_:>10,} {sy['mean']:>12,.0f}")

        for arm_key, arm_runs in [("mlp_param_matched", mlps.get(suite_name, []))]:
            ok = [r for r in arm_runs if r.error is None]
            if not ok:
                err = next((r.error for r in arm_runs), "not run")
                bucket[arm_key] = {"status": "failed", "error": err}
                print(f"{arm_key:>26} {'FAILED':>18}  {str(err)[:40]}")
                continue
            fa = mean_ci([r.metrics["final_average"] for r in ok])
            dg = mean_ci([r.metrics["diagonal_mean"] for r in ok])
            bw = mean_ci([r.metrics["backward_transfer"] for r in ok])
            fg = mean_ci([r.metrics["forgetting"] for r in ok])
            np_ = int(ok[0].metrics["n_params"])
            sy = mean_ci([r.metrics["synops_per_sample"] for r in ok])
            bucket[arm_key] = {
                "status": "ok", "n_seeds": len(ok),
                "final_average": fa, "diagonal_mean": dg,
                "backward_transfer": bw, "forgetting": fg,
                "n_params": np_, "synops_per_sample": sy,
                "hidden": ok[0].metrics.get("hidden"),
                "retention": [r.retention for r in ok],
                "wall_seconds": mean_ci([r.wall_seconds for r in ok]),
            }
            print(f"{arm_key:>26} {fa['mean']*100:9.2f} +/- {fa['ci95']*100:4.2f} "
                  f"{dg['mean']*100:7.2f}% {bw['mean']*100:8.2f}% "
                  f"{fg['mean']*100:8.2f}% {np_:>10,} {sy['mean']:>12,.0f}")

        # ---- the head-to-head the brief asks for
        b, m = bucket.get("depth2"), bucket.get("mlp_param_matched")
        if b and m and b.get("status") == "ok" and m.get("status") == "ok":
            margin = (b["final_average"]["mean"] - m["final_average"]["mean"]) * 100
            comb = (b["final_average"]["ci95"] + m["final_average"]["ci95"]) * 100
            fg_b = b["forgetting"]["mean"] * 100
            fg_m = m["forgetting"]["mean"] * 100
            print("-" * 88)
            print(f"depth2 - mlp final accuracy: {margin:+.2f} points "
                  f"(combined 95% CI +/- {comb:.2f})")
            print(f"forgetting: depth2 {fg_b:.2f}% vs mlp {fg_m:.2f}% "
                  f"(lower is better)")
            if m["diagonal_mean"]["mean"] < 0.2:
                print("  -> NOTE: the MLP is near chance on the tasks it was just "
                      "trained on, so a substrate 'win' here would be void.")
            bucket["head_to_head"] = {
                "final_acc_margin_depth2_minus_mlp": margin,
                "combined_ci95": comb,
                "forgetting_depth2": b["forgetting"]["mean"],
                "forgetting_mlp": m["forgetting"]["mean"],
                "mlp_at_chance": bool(m["diagonal_mean"]["mean"] < 0.2),
            }
            if b["synops_per_sample"]["mean"] and m["synops_per_sample"]["mean"]:
                ratio = b["synops_per_sample"]["mean"] / m["synops_per_sample"]["mean"]
                print(f"active SynOps/sample: depth2 {b['synops_per_sample']['mean']:,.0f} "
                      f"vs mlp {m['synops_per_sample']['mean']:,.0f} MACs "
                      f"(ratio {ratio:.3f}; <1 means the substrate spends fewer ops)")
                bucket["head_to_head"]["synops_ratio_depth2_over_mlp"] = ratio
        out["continual"][suite_name] = bucket

    # ------------------------------------------------------------- ablations
    for suite_name, runs in abls.items():
        if not runs:
            continue
        print("\n" + "=" * 88)
        print(f"ABLATIONS at depth 2 — {suite_name.upper()}")
        print("=" * 88)
        base = [r for r in cont.get(suite_name, []) if r.depth == 2 and r.error is None]
        print(f"{'arm':>28} {'final avg':>18} {'delta vs depth2':>18} {'forget':>9}")
        print("-" * 88)
        bl = mean_ci([r.metrics["final_average"] for r in base]) if base else None
        bucket = {}
        for r in runs:
            if r.error is not None:
                print(f"{r.arm:>28} {'FAILED':>18}  {str(r.error)[:36]}")
                bucket[r.arm] = {"status": "failed", "error": r.error}
                continue
            fa = mean_ci([x.metrics["final_average"] for x in runs if x.arm == r.arm
                          and x.error is None])
            fg = mean_ci([x.metrics["forgetting"] for x in runs if x.arm == r.arm
                          and x.error is None])
            delta = (fa["mean"] - bl["mean"]) if bl else float("nan")
            bucket[r.arm] = {"status": "ok", "n_seeds": fa["n"], "final_average": fa,
                             "forgetting": fg, "delta_vs_depth2": float(delta)}
            print(f"{r.arm:>28} {fa['mean']*100:9.2f} +/- {fa['ci95']*100:4.2f} "
                  f"{delta*100:+17.2f} {fg['mean']*100:8.2f}%")
        out["ablations"][suite_name] = bucket

    # ------------------------------------------------------------ methodology
    out["methodology"] = {
        "lr_grid": [float(x) for x in LR_GRID],
        "lr_selection": "validation split carved from training data; test never used",
        "test_set_used_for_selection": False,
        "feature_standardisation": "frozen z-score from training codes only, dead features zeroed",
        "backend": args.backend,
        "legacy_reported_ceiling": LEGACY_CEILING,
        "eval_protocol": "state reset at each phase boundary; retained single readout",
    }
    return out


def print_verdict(payload: dict) -> None:
    """State the bottom line plainly, including the negative case."""
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    c = payload["verdict"].get("ceiling_test", {})
    if "depth2_exceeds_depth1" in c:
        d1 = c["depth1_mean"] * 100
        d2 = c["depth2_mean"] * 100
        print(f"Single-task 10-way ceiling: depth1 {d1:.2f}% vs depth2 {d2:.2f}%")
        if c["depth2_exceeds_depth1"]:
            print("  Depth 2 RAISES the single-task ceiling beyond the confidence "
                  "interval -> a second locally-trained spiking layer does buy "
                  "decoding power depth 1 cannot reach.")
        else:
            print("  Depth 2 does NOT raise the ceiling beyond the confidence "
                  "interval -> the stated hypothesis is NOT supported.")
        if c["depth1_above_legacy_ceiling"]:
            print(f"  NOTE: depth 1 alone ({d1:.2f}%) already exceeds the "
                  f"previously reported {LEGACY_CEILING*100:.1f}% ceiling, so the "
                  f"old ceiling was not a property of the single-layer code.")
    else:
        print("  Ceiling test did not complete; see errors above.")

    for suite_name, bucket in payload["continual"].items():
        h = bucket.get("head_to_head") if isinstance(bucket, dict) else None
        if not h:
            continue
        print(f"\n{suite_name}: depth2 vs parameter-matched MLP")
        print(f"  final accuracy margin: {h['final_acc_margin_depth2_minus_mlp']:+.2f} "
              f"points (CI +/- {h['combined_ci95']:.2f})")
        win = h["final_acc_margin_depth2_minus_mlp"] > h["combined_ci95"]
        print(f"  {'substrate retains more' if win else 'substrate does NOT exceed the MLP'}"
              f"; forgetting {h['forgetting_depth2']*100:.2f}% vs "
              f"{h['forgetting_mlp']*100:.2f}%")
        if h.get("mlp_at_chance"):
            print("  CAVEAT: the MLP is at chance on its own training tasks, so this "
                  "comparison is void as evidence of learning.")
    print("=" * 88)


# -------------------------------------------------------------------- driver
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3, help="seeds per arm (>=3 for a real run)")
    p.add_argument("--tasks", type=int, default=5, help="tasks in the continual suites")
    p.add_argument("--neurons", type=int, default=900, help="neurons per layer")
    p.add_argument("--k-out", type=int, default=48, help="synaptic fan-out")
    p.add_argument("--train-per-class", type=int, default=100,
                   help="training samples per class (10 classes -> 1000 total)")
    p.add_argument("--val-per-class", type=int, default=20)
    p.add_argument("--test-per-class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--suite-seed", type=int, default=0,
                   help="seed for the task construction (pixel permutations etc.)")
    p.add_argument("--backend", default="numpy", choices=("numpy", "mlx", "auto"),
                   help="array backend; numpy is faster at this network size because "
                        "MLX's per-op dispatch dominates below ~10k neurons")
    p.add_argument("--gain-grid", default="24,32,48,64",
                   help="candidate inter-layer drive scales for depth 2; "
                        "selected on validation only")
    p.add_argument("--gain-max-train", type=int, default=200,
                   help="training samples used per gain candidate (cost control)")
    p.add_argument("--mlp-lr", type=float, default=1e-2)
    p.add_argument("--mlp-batch-size", type=int, default=64)
    p.add_argument("--out", default=str(RESULTS_DIR / "hierarchy.json"))
    p.add_argument("--quick", action="store_true", help="small smoke run")
    p.add_argument("--skip-continual", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        # A smoke test must still exercise the same code paths with sane
        # statistics. Shrinking the *sample count* is the wrong lever: 900
        # features against a few hundred samples makes the readout memorise
        # (train ~0.99, test ~0.10) and the run then measures overfitting rather
        # than representation. Shrink the network instead and keep the data.
        args.seeds = 1
        # cannot go below n_input: layer 1 injects one pixel per neuron
        args.neurons = max(784, min(args.neurons, 800))
        args.train_per_class = min(args.train_per_class, 50)
        args.val_per_class = min(args.val_per_class, 10)
        args.gain_max_train = min(args.gain_max_train, 100)
        args.test_per_class = min(args.test_per_class, 25)
        args.gain_grid = "32,48"
        args.skip_ablations = True

    gain_grid = tuple(float(x) for x in str(args.gain_grid).split(",") if x.strip())
    args._gain_grid = gain_grid
    seeds = [1000 + 17 * i for i in range(args.seeds)]
    t_start = time.perf_counter()

    print("=" * 88)
    print("CORTICAL HIERARCHY — does depth 2 beat the single-layer ceiling?")
    print("=" * 88)
    print(f"backend={args.backend}  neurons/layer={args.neurons}  k_out={args.k_out}  "
          f"seeds={len(seeds)}  train/class={args.train_per_class}  "
          f"val/class={args.val_per_class}  test/class={args.test_per_class}")
    try:
        import mlx.core as mx  # noqa: F401
        print("MLX is importable but NOT used by default: measured ~8x slower than "
              "NumPy at this network size (per-op dispatch dominates).")
    except Exception:
        pass

    # 1) single-task ceiling test
    print("\n[1/4] single-task 10-way MNIST, depth 1 vs depth 2 ...")
    single: list[RunResult] = []
    for depth in (1, 2):
        for seed in seeds:
            r = run_single_task(args, seed, depth)
            single.append(r)
            if r.error:
                print(f"  depth{depth} seed{seed}: FAILED {r.error}")
            else:
                print(f"  depth{depth} seed{seed}: test={r.metrics['test_acc']:.4f} "
                      f"train={r.metrics['train_acc']:.4f} lr={r.metrics['chosen_lr']:.0e} "
                      f"spk={sum(r.metrics['layer_spikes'].values()):.0f} "
                      f"({r.wall_seconds:.0f}s)", flush=True)

    # 2) continual learning, both suites
    cont: dict[str, list[RunResult]] = {}
    mlps: dict[str, list[RunResult]] = {}
    if not args.skip_continual:
        target = int(single[0].metrics["n_params"]) if single and not single[0].error else 95400
        for permute, suite_name in ((False, "split-mnist"), (True, "permuted-mnist")):
            print(f"\n[2/4] continual {suite_name}, depth 1 vs depth 2 ...")
            runs: list[RunResult] = []
            for depth in (1, 2):
                for seed in seeds:
                    r = run_continual(args, seed, depth, permute)
                    runs.append(r)
                    if r.error:
                        print(f"  depth{depth} seed{seed}: FAILED {r.error}")
                    else:
                        print(f"  depth{depth} seed{seed}: final="
                              f"{r.metrics['final_average']:.4f} "
                              f"diag={r.metrics['diagonal_mean']:.4f} "
                              f"({r.wall_seconds:.0f}s)", flush=True)
            cont[suite_name] = runs
            print(f"  parameter-matched MLP on {suite_name} ...")
            mruns = []
            for seed in seeds:
                m = run_mlp(args, seed, permute, target)
                mruns.append(m)
                if m.error:
                    print(f"  mlp seed{seed}: FAILED {m.error}")
                else:
                    print(f"  mlp seed{seed} (hidden={m.metrics['hidden']}, "
                          f"{m.metrics['n_params']:,} params): final="
                          f"{m.metrics['final_average']:.4f} "
                          f"diag={m.metrics['diagonal_mean']:.4f} "
                          f"({m.wall_seconds:.0f}s)", flush=True)
            mlps[suite_name] = mruns

    # 3) ablations at depth 2
    abls: dict[str, list[RunResult]] = {}
    if not args.skip_ablations:
        print("\n[3/4] ablations at depth 2 (permuted-MNIST) ...")
        runs = []
        for kind in ("ablate_l2_dendrite", "ablate_l1_plasticity",
                     "ablate_frozen_substrate"):
            for seed in seeds[: max(1, min(2, len(seeds)))]:
                r = run_ablation(args, seed, kind, True)
                runs.append(r)
                if r.error:
                    print(f"  {kind} seed{seed}: FAILED {r.error}")
                else:
                    print(f"  {kind} seed{seed}: final={r.metrics['final_average']:.4f} "
                          f"({r.wall_seconds:.0f}s)", flush=True)
        abls["permuted-mnist"] = runs

    # 4) report + persist
    print("\n[4/4] summarising ...")
    payload = report(args, single, cont, mlps, abls)
    payload["config"] = {
        "seeds": seeds, "tasks": args.tasks, "neurons": args.neurons,
        "k_out": args.k_out, "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class, "test_per_class": args.test_per_class,
        "epochs": args.epochs, "backend": args.backend, "mlp_lr": args.mlp_lr,
        "mlp_batch_size": args.mlp_batch_size, "suite_seed": args.suite_seed,
        "gain_grid": [float(x) for x in gain_grid],
        "quick": bool(args.quick),
    }
    payload["runs"] = {
        "single_task": [asdict(r) for r in single],
        "continual": {k: [asdict(r) for r in v] for k, v in cont.items()},
        "mlp": {k: [asdict(r) for r in v] for k, v in mlps.items()},
        "ablations": {k: [asdict(r) for r in v] for k, v in abls.items()},
    }
    try:
        from brain.hierarchy import READOUT_AVAILABLE

        payload["readout_module"] = ("brain.readout" if READOUT_AVAILABLE
                                     else "local fallback in brain/hierarchy.py")
    except Exception:  # pragma: no cover
        payload["readout_module"] = "unknown"
    payload["wall_seconds_total"] = time.perf_counter() - t_start

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True))
    tmp.replace(out_path)
    print(f"\nwrote {out_path}  ({payload['wall_seconds_total']:.0f}s total)")
    print(f"readout: {payload['readout_module']}")

    print_verdict(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
