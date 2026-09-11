"""The falsification experiment: single-pass continual learning, no replay.

Arms
----
======================  ==========================================================
``brain``               dendritic substrate + k-WTA + three-factor local plasticity
``brain_nodend``        same, dendrites ablated (``dend_mode="linear"``)
``brain_noplast``       same, plasticity disabled (frozen substrate)
``mlp``                 backprop MLP, parameter-matched
``mlp_replay``          backprop + experience replay (the STRONG baseline)
``frozen``              frozen random features + ridge readout (is plasticity causal?)
``shuffled``            substrate with randomised labels (must NOT learn)
======================  ==========================================================

Every arm sees the same tasks in the same order with the same budget and no
access to earlier tasks afterwards (except ``mlp_replay``, which is allowed a
replay buffer and is therefore strictly advantaged).

Falsification conditions are documented in ``docs/PARADIGM.md`` section 6. This
runner reports the numbers; it does not decide what they mean.

Run:  python3 experiments/continual.py --tasks 5 --seeds 3 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


@dataclass
class ArmResult:
    arm: str
    seed: int
    acc_matrix: list[list[float]] = field(default_factory=list)
    final_average: float = 0.0
    backward_transfer: float = 0.0
    forgetting: float = 0.0
    n_params: int = 0
    synops_per_sample: float = 0.0
    wall_seconds: float = 0.0
    diagonal_mean: float = 0.0
    n_updates: int = 0
    error: str | None = None


def _make_arm(name: str, suite, seed: int, args):
    """Instantiate one experimental arm. Imports are local so a missing module
    for one arm cannot prevent the others from running."""
    from brain.cortex import CortexClassifier, CortexConfig

    n_input = suite.n_input
    n_classes = suite.n_classes
    n_neurons = args.neurons

    if name == "brain":
        return CortexClassifier(CortexConfig(
            n_neurons=n_neurons, n_input=n_input, n_classes=n_classes,
            k_out=args.k_out, seed=seed, use_dendrite=True,
            inhibition=args.inhibition, k_wta=args.k_wta, plasticity=True))
    if name == "brain_nodend":
        return CortexClassifier(CortexConfig(
            n_neurons=n_neurons, n_input=n_input, n_classes=n_classes,
            k_out=args.k_out, seed=seed, use_dendrite=False,
            inhibition=args.inhibition, k_wta=args.k_wta, plasticity=True))
    if name == "brain_noplast":
        return CortexClassifier(CortexConfig(
            n_neurons=n_neurons, n_input=n_input, n_classes=n_classes,
            k_out=args.k_out, seed=seed, use_dendrite=True,
            inhibition=args.inhibition, k_wta=args.k_wta, plasticity=False))
    if name == "shuffled":
        return CortexClassifier(CortexConfig(
            n_neurons=n_neurons, n_input=n_input, n_classes=n_classes,
            k_out=args.k_out, seed=seed, use_dendrite=True,
            inhibition=args.inhibition, k_wta=args.k_wta, plasticity=True))

    from brain.baselines import FrozenFeaturesReadout, MLPBackprop, ReplayMLP

    if name == "mlp":
        return MLPBackprop(n_input, args.hidden, n_classes, seed=seed, lr=args.lr,
                           batch_size=args.batch_size)
    if name == "mlp_replay":
        return ReplayMLP(n_input, args.hidden, n_classes, seed=seed, lr=args.lr,
                         batch_size=args.batch_size, replay_frac=0.10)
    if name == "frozen":
        return FrozenFeaturesReadout(n_input, args.hidden, n_classes, seed=seed)
    raise ValueError(f"unknown arm {name!r}")


def run_arm(name: str, suite, seed: int, args) -> ArmResult:
    from brain.metrics import ContinualCurve, accuracy, synop_comparison

    res = ArmResult(arm=name, seed=seed)
    t0 = time.perf_counter()
    try:
        model = _make_arm(name, suite, seed, args)
        n_tasks = len(suite)
        acc = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
        total_synops = 0.0
        n_samples = 0

        for i, task in enumerate(suite.tasks):
            if name == "shuffled":
                # Destroy the label->input association while keeping the task
                # structure. NB: with only 2 classes per task a permutation is a
                # coin-flip between "identity" (a NO-OP that tests nothing) and
                # "swap" (a trivially learnable inversion), which is why the
                # separation from `brain` came out small. Randomising over the
                # full global label space guarantees a genuinely unlearnable
                # target for every task instead.
                rng = np.random.default_rng(seed * 1000 + i)
                # Wrap rather than mutate: `Task.__post_init__` enforces that
                # `classes == unique(y_train)`, so writing global random labels
                # into the Task directly raises. A plain namespace preserves the
                # corrupt-label control without violating the contract.
                task = SimpleNamespace(
                    name=task.name,
                    x_train=task.x_train,
                    y_train=rng.integers(0, suite.n_classes, size=len(task.y_train)),
                    x_test=task.x_test,
                    y_test=task.y_test,
                    classes=list(range(suite.n_classes)),
                )

            # FAIRNESS: the brain arm performs one online update per training
            # sample. A minibatch MLP with epochs=1 on the same data performs
            # only n_train/batch_size gradient steps - up to 60x fewer updates.
            # Comparing those two is not a comparison of learning rules, it is a
            # comparison of update budgets, so the MLP's epochs are scaled to
            # equalise the number of weight updates.
            epochs = args.epochs
            if name in ("mlp", "mlp_replay"):
                n_train = len(task.y_train)
                steps_per_epoch = max(1, int(np.ceil(n_train / args.batch_size)))
                target_updates = n_train * args.epochs
                epochs = max(1, int(np.ceil(target_updates / steps_per_epoch)))
            model.fit_task(task, epochs=epochs)

            # evaluate on every task seen so far -> lower-triangular acc matrix
            for j in range(i + 1):
                tj = suite.tasks[j]
                pred = model.predict(tj.x_test)
                acc[i, j] = accuracy(pred, tj.y_test)
                n_samples += len(tj.y_test)

        res.acc_matrix = np.nan_to_num(acc, nan=0.0).tolist()
        curve = ContinualCurve(acc=np.nan_to_num(acc, nan=0.0))
        res.final_average = curve.final_average()
        res.backward_transfer = curve.backward_transfer()
        res.forgetting = curve.forgetting()
        am = np.nan_to_num(acc, nan=0.0)
        res.diagonal_mean = float(np.mean([am[i, i] for i in range(n_tasks)]))
        res.n_params = int(model.n_params)
        if hasattr(model, "synops_per_sample"):
            res.synops_per_sample = float(model.synops_per_sample)
        elif hasattr(model, "last_synops") and n_samples:
            res.synops_per_sample = float(model.last_synops) / max(1, n_samples)
    except Exception as exc:  # keep other arms alive
        res.error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            traceback.print_exc()
    res.wall_seconds = time.perf_counter() - t0
    return res


ARMS = ["brain", "brain_nodend", "brain_noplast", "mlp", "mlp_replay", "frozen",
        "shuffled"]


def stratified_subsample(x: np.ndarray, y: np.ndarray, n: int,
                         seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Take ``n`` samples while preserving the class distribution.

    Slicing the first ``n`` rows is a trap: in permuted-MNIST every task carries
    all 10 classes, and the file order is not grouped by class, so a head-slice
    leaves a badly skewed label distribution (measured: counts of 5-16 per class
    in an 80-row slice). That silently turns the task into an unbalanced one and
    makes accuracy hard to interpret. Stratifying keeps the subsets honest, and
    falls back to an even split when a class has too few members.
    """
    if n <= 0 or n >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, n // len(classes))
    picks: list[np.ndarray] = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picks.append(idx[:per_class])
    chosen = np.concatenate(picks)
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


def summarise(results: list[ArmResult]) -> dict:
    out: dict[str, dict] = {}
    for arm in ARMS:
        rows = [r for r in results if r.arm == arm and r.error is None]
        if not rows:
            errs = [r.error for r in results if r.arm == arm]
            out[arm] = {"status": "failed", "error": errs[0] if errs else "not run"}
            continue
        fa = np.array([r.final_average for r in rows])
        bwt = np.array([r.backward_transfer for r in rows])
        fg = np.array([r.forgetting for r in rows])
        out[arm] = {
            "status": "ok",
            "n_seeds": len(rows),
            "final_average": float(fa.mean()),
            "final_average_ci95": float(1.96 * fa.std(ddof=1) / np.sqrt(len(fa)))
            if len(fa) > 1 else 0.0,
            "backward_transfer": float(bwt.mean()),
            "forgetting": float(fg.mean()),
            "diagonal_mean": float(np.mean([r.diagonal_mean for r in rows])),
            "n_params": int(rows[0].n_params),
            "synops_per_sample": float(np.mean([r.synops_per_sample for r in rows])),
            "wall_seconds": float(np.mean([r.wall_seconds for r in rows])),
        }
    return out


def print_table(summary: dict) -> None:
    print("\n" + "=" * 96)
    print("SINGLE-PASS CONTINUAL LEARNING — no replay, no task ID at test")
    print("=" * 96)
    hdr = (f"{'arm':14s} {'final acc':>16s} {'task-diag':>10s} {'BWT':>9s} "
           f"{'forget':>9s} {'params':>10s} {'SynOps/sample':>15s}")
    print(hdr)
    print("-" * 96)
    for arm in ARMS:
        s = summary.get(arm, {})
        if s.get("status") != "ok":
            print(f"{arm:14s} {'FAILED':>16s}  {s.get('error', '')[:60]}")
            continue
        acc = f"{s['final_average']*100:6.2f} +/- {s['final_average_ci95']*100:4.2f}"
        print(f"{arm:14s} {acc:>16s} {s['diagonal_mean']*100:9.2f}% "
              f"{s['backward_transfer']*100:8.2f}% "
              f"{s['forgetting']*100:8.2f}% {s['n_params']:>10,d} "
              f"{s['synops_per_sample']:>15,.0f}")
    print("=" * 96)

    b, a = summary.get("brain", {}), summary.get("mlp", {})
    if b.get("status") == "ok" and a.get("status") == "ok":
        margin = (b["final_average"] - a["final_average"]) * 100
        ci = (b["final_average_ci95"] + a["final_average_ci95"]) * 100
        print(f"\nbrain - mlp final accuracy: {margin:+.2f} points (combined 95% CI {ci:.2f})")
        # Guard against declaring a win on a void comparison: a baseline sitting
        # at chance has not been beaten, it has merely failed to run.
        chance = 1.0 / max(1, summary["brain"].get("n_classes", 10))
        if a["diagonal_mean"] <= chance * 1.5:
            print(f"  -> INVALID COMPARISON: the MLP baseline is at chance on what it was "
                  f"just trained on (diagonal {a['diagonal_mean']*100:.1f}%), so it never "
                  f"learned. Any 'win' here is void. Fix the baseline and re-run.")
        elif margin > ci:
            print("  -> brain substrate retains more; falsification condition A>=B NOT met")
        else:
            print("  -> FALSIFIED: backprop is statistically no worse (A >= B) "
                  "on this benchmark.")

    # --- explicit ablation readouts, since a null result here is informative
    for arm in ("brain_nodend", "brain_noplast", "shuffled", "frozen"):
        t_ = summary.get(arm, {})
        if t_.get("status") == "ok" and b.get("status") == "ok":
            d = (b["final_average"] - t_["final_average"]) * 100
            print(f"  brain - {arm:<13s}: {d:+6.2f} points "
                  f"({'ablation makes no difference -> claim NOT supported' if abs(d) < 1.0 else 'ablation matters'})")

    # --- criterion (f): active-synaptic-operation comparison vs the dense baseline
    from brain.metrics import synop_comparison

    if b.get("status") == "ok" and a.get("status") == "ok" and b["synops_per_sample"]:
        mlp_macs = float(a["n_params"])  # one MAC per parameter per sample
        comp = synop_comparison(
            {"synops_per_sample": b["synops_per_sample"]},
            {"synops_per_sample": mlp_macs},
        )
        print(f"\nACTIVE SYNOP COMPARISON (operation counts, NOT joules)")
        print(f"  brain: {comp['brain_synops_per_sample']:>14,.0f} active SynOps/sample "
              f"(spikes x fan-out)")
        print(f"  mlp  : {comp['mlp_synops_per_sample']:>14,.0f} MACs/sample "
              f"(= n_params, dense)")
        print(f"  ratio brain/mlp: {comp['brain_to_mlp_ratio']:.4f} "
              f"({'brain spends FEWER ops' if comp['brain_to_mlp_ratio'] < 1 else 'brain spends MORE ops'})")
        print(f"  NOTE: {comp['caveat'][:150]}...")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--permute", action="store_true",
                   help="use permuted-MNIST (harder: same classes, new pixel map)")
    p.add_argument("--neurons", type=int, default=2000)
    p.add_argument("--k-out", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--inhibition", default="kwta")
    p.add_argument("--k-wta", type=int, default=64)
    p.add_argument("--arms", nargs="*", default=ARMS)
    p.add_argument("--train-per-task", type=int, default=400,
                   help="subsample each task's training set (0 = use all)")
    p.add_argument("--test-per-task", type=int, default=200,
                   help="subsample each task's test set (0 = use all)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.inhibition == "kwta" and args.k_wta <= 0:
        args.k_wta = max(1, args.neurons // 32)

    from brain.tasks import split_mnist

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[ArmResult] = []
    for seed in range(args.seeds):
        suite = split_mnist(args.tasks, seed=seed, permute=args.permute)
        if args.train_per_task or args.test_per_task:
            for ti, t in enumerate(suite.tasks):
                if args.train_per_task:
                    t.x_train, t.y_train = stratified_subsample(
                        t.x_train, t.y_train, args.train_per_task, seed * 17 + ti)
                if args.test_per_task:
                    t.x_test, t.y_test = stratified_subsample(
                        t.x_test, t.y_test, args.test_per_task, seed * 31 + ti)
        print(f"\n### seed {seed} | {'permuted' if args.permute else 'split'}-MNIST "
              f"| {len(suite)} tasks | {suite.n_classes} classes")
        for arm in args.arms:
            r = run_arm(arm, suite, seed, args)
            status = r.error if r.error else f"acc={r.final_average*100:.2f}%"
            print(f"    {arm:14s} {status}  ({r.wall_seconds:.1f}s)")
            all_results.append(r)

    summary = summarise(all_results)
    print_table(summary)

    payload = {
        "config": vars(args),
        "summary": summary,
        "runs": [asdict(r) for r in all_results],
    }
    out = RESULTS_DIR / f"continual_{'permuted' if args.permute else 'split'}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
