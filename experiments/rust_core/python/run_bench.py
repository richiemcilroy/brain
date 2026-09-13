"""Matched-configuration benchmark: Python/MLX path vs Rust event-driven core.

Fairness design
---------------
Both engines run the SAME network and the SAME initial conditions. Python builds
the connectivity, times its own steps, then exports the exact graph it used; the
Rust core loads that fixture. So a difference in wall-clock cannot be blamed on a
different random graph, a different drive, or a different initial state.

Protocol (mirrors `bench/bench_scale.py` so the numbers are comparable):
  * `noise_std = 0`  -- the condition under which event-driven skipping is exact
  * plasticity disabled
  * 20 warmup steps, then timed steps
  * Python: `force_eval` (mx.eval + synchronize) after EVERY step, so no lazy
    evaluation can hide work
  * 1 step = 1 ms of biological time (dt = 1.0)

What is varied: the fraction of neurons receiving tonic drive. Each level
reports the ACHIEVED active fraction (spikes/neuron/step), not the intended one.

Usage:
    python3 python/run_bench.py --out bench
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parent
REPO = CORE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.connectivity import SynapseConfig, Synapses          # noqa: E402
from brain.neurons import NeuronConfig                          # noqa: E402
from brain.plasticity import PlasticityConfig                   # noqa: E402
from brain.simulator import Brain, SimConfig                    # noqa: E402

MAGIC = b"RBRAIN01"
DEND_MODE_IDS = {"dcaap": 0, "linear": 1, "none": 2}
# Overridable so the benchmark does not depend on the external SSD being mounted.
# Set RC_BIN / RC_SCRATCH to relocate. Defaults live under /tmp so a run never
# writes hundreds of MB to the internal disk.
DEFAULT_BIN = os.environ.get(
    "RC_BIN", "/tmp/zz_rust_target/release/rust_core")
DEFAULT_SCRATCH = Path(
    os.environ.get("RC_SCRATCH", "/tmp/zz_fixtures"))


def rust_binary() -> Path:
    p = Path(DEFAULT_BIN)
    if p.exists():
        return p
    raise SystemExit(
        f"rust binary not found at {p}\n"
        "build it first:\n"
        "  CARGO_TARGET_DIR=/tmp/zz_rust_target cargo build --release\n"
        "or set RC_BIN to the binary's path."
    )


def build_network(n: int, k: int, seed: int, w_exc: float):
    """Build with MLX (the project's fastest path) and noise off."""
    neu = NeuronConfig(n=n, dt=1.0, noise_std=0.0)
    syn = SynapseConfig(
        n_pre=n, n_post=n, k_out=k, seed=seed, delay_max=4,
        w_exc=w_exc, w_inh=w_exc * 4,
    )
    cfg = SimConfig(
        n_neurons=n, k_out=k, seed=seed, init_noise=0.0,
        neu=neu, syn=syn, plas=PlasticityConfig(enabled=False),
    )
    return Brain(cfg, backend="mlx")


def force_eval(brain: Brain) -> None:
    be = brain.be
    if be.is_mlx:
        import mlx.core as mx
        mx.eval(
            brain.neurons.v_soma, brain.neurons.v_dend, brain.neurons.adapt,
            brain.neurons.refrac, brain.neurons.spike_count,
            brain.synapses.weights, brain.delay.flat,
        )
        mx.synchronize()


def make_drive(n: int, frac: float, lo: float, hi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 777)
    n_driven = max(1, int(round(frac * n)))
    d = np.zeros((n,), dtype=np.float32)
    if n_driven > 0:
        idx = np.sort(rng.choice(n, size=n_driven, replace=False))
        d[idx] = rng.uniform(lo, hi, size=n_driven).astype(np.float32)
    return d


def time_python(brain: Brain, drive: np.ndarray, warm: int, steps: int) -> dict:
    """Time `steps` steps after `warm` warmup steps, with force_eval every step."""
    be = brain.be
    dv = be.array(drive)
    for _ in range(warm):
        brain.step(external_soma=dv)
        force_eval(brain)

    per_step = []
    spikes = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        s0 = time.perf_counter()
        st = brain.step(external_soma=dv)
        force_eval(brain)
        per_step.append((time.perf_counter() - s0) * 1e3)
        spikes += int(st.spikes)
    wall = time.perf_counter() - t0
    per_step.sort()
    return {
        # MINIMUM is the load-robust statistic on a shared machine; see BENCH.md.
        "ms_per_step_min": float(per_step[0]),
        "ms_per_step_median": float(per_step[len(per_step) // 2]),
        "ms_per_step_mean": float(sum(per_step) / len(per_step)),
        "ms_per_step_p90": float(per_step[int(0.9 * (len(per_step) - 1))]),
        "wall_s": wall,
        "steps": steps,
        "spikes_total": spikes,
        "spikes_per_step": spikes / steps,
        "active_frac": spikes / steps / brain.cfg.n_neurons,
        "bio_ms_per_wall_s": 1000.0 * (steps / wall),
        "bio_ms_per_wall_s_min": (1000.0 / per_step[0]) if per_step[0] > 0 else 0.0,
        "synops_active_per_s": (spikes / wall) * brain.cfg.k_out,
        "ram_bytes": sum(brain.memory_bytes().values()),
    }


def reference_run(n: int, k: int, seed: int, w_exc: float, drive: np.ndarray,
                  warm: int, steps: int) -> dict:
    """Reproduce the EXACT window the Rust core measures, on a fresh network.

    The Rust core is timed after `warm` warmup steps, so the reference trajectory
    must be the window [warm, warm+steps) of a run that also started from rest.
    A fresh Brain with the same seed has identical connectivity, and the reset
    state is the rest state, so this window is directly comparable.

    Returns the rest state to export as the fixture's initial condition, the
    per-step spike counts for the measured window, and per-neuron counts summed
    over that same window (not including warmup).
    """
    brain = build_network(n, k, seed, w_exc)
    be = brain.be
    brain.reset()
    dv = be.array(drive)

    for _ in range(warm):
        brain.step(external_soma=dv)

    counts0 = np.asarray(be.to_numpy(brain.neurons.spike_count)).astype(np.int64)
    per_step = []
    for _ in range(steps):
        per_step.append(int(brain.step(external_soma=dv).spikes))
    counts1 = np.asarray(be.to_numpy(brain.neurons.spike_count)).astype(np.int64)
    neuron_counts = (counts1 - counts0).astype(np.uint32)

    return {
        "brain": brain,
        "step_spikes": per_step,
        "neuron_spikes": neuron_counts,
    }


def export_fixture(path: Path, brain: Brain, drive: np.ndarray, steps: int,
                   step_spikes: list[int], neuron_spikes: np.ndarray) -> None:
    be = brain.be
    neu = brain.neuron_cfg
    targets = np.asarray(be.to_numpy(brain.synapses.targets)).astype("<u4")
    delays = np.asarray(be.to_numpy(brain.synapses.delays)).astype("<u4")
    weights = np.asarray(be.to_numpy(brain.synapses.weights), dtype="<f4")
    n = brain.cfg.n_neurons
    k = brain.syn_cfg.k_out
    with open(path, "wb") as f:
        f.write(MAGIC)
        # Declare the LARGEST delay actually present, not the configured bound:
        # MLX 0.31.0's randint returns values equal to `high` on large draws, so
        # a configured delay_max of 4 can yield real delays of 5. See
        # export_fixture.py::syn_max_delay for the measurement.
        real_max_delay = int(max(int(brain.syn_cfg.delay_max), int(delays.max())))
        f.write(struct.pack("<IIIIII", n, k, real_max_delay, steps,
                            DEND_MODE_IDS[neu.dend_mode], 1))
        f.write(struct.pack("<14f",
                            neu.tau_soma, neu.tau_dend, neu.tau_adapt, neu.e_leak,
                            neu.e_dend, neu.v_reset, neu.v_thresh, neu.adapt_base,
                            neu.adapt_inc, neu.dend_scale, neu.dend_gain,
                            neu.refractory_ms, neu.noise_std, neu.dt))
        f.write(targets.tobytes())
        f.write(delays.tobytes())
        f.write(weights.tobytes())
        # initial state: the network is at rest with v = e_leak
        f.write(np.full(n, neu.e_leak, dtype="<f4").tobytes())
        f.write(np.full(n, neu.e_dend, dtype="<f4").tobytes())
        f.write(np.full(n, neu.adapt_base, dtype="<f4").tobytes())
        f.write(np.zeros(n, dtype="<f4").tobytes())
        f.write(np.asarray(drive, dtype="<f4").tobytes())
        f.write(np.asarray(neuron_spikes, dtype="<u4").tobytes())
        f.write(np.asarray(step_spikes, dtype="<u4").tobytes())


def run_rust(binpath: Path, fixture: Path, extra: list[str]) -> dict:
    out = subprocess.run(
        [str(binpath), "bench-fixture", str(fixture), *extra],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(
            f"rust core failed (exit {out.returncode}) on {fixture}\n"
            f"--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}"
        )
    return json.loads(out.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORE / "bench")
    ap.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", type=str, default=None,
                    help="override, e.g. '100000:64:0.4,1000000:64:0.4'")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    binpath = rust_binary()

    # (n, k_out, w_exc). w_exc=0.40 (vs the library default 0.08) raises the
    # recurrent gain so that delivery materially affects the outcome; at 0.08 the
    # measurement is insensitive to whether delivery happens at all.
    if args.configs:
        configs = []
        for chunk in args.configs.split(","):
            n, k, w = chunk.split(":")
            configs.append((int(n), int(k), float(w)))
    else:
        configs = [(100_000, 64, 0.40), (1_000_000, 64, 0.40)]

    # (driven_frac, amplitude). Chosen from a measured calibration sweep so the
    # ACHIEVED active fraction spans ~0.1% to ~10%+, which is the range the
    # sparsity claim is about. The pair is only a knob; active fraction is
    # measured and reported per row.
    levels = [
        (0.0, 0.0),
        (0.001, 1.25),
        (0.005, 1.25),
        (0.01, 3.0),
        (0.02, 1.25),
        (0.05, 1.25),
        (0.20, 5.0),
        (0.50, 5.0),
        (1.00, 2.0),
    ]

    rows = []
    partial = args.out / "results.json"

    def flush() -> None:
        partial.write_text(json.dumps(
            {"rows": rows, "protocol": {"partial": len(rows)}}, indent=1))

    for (n, k, w_exc) in configs:
        for (frac, amp) in levels:
            label = f"n{n}_k{k}_d{frac:g}_a{amp:g}"
            fixture = args.scratch / f"{label}.bin"
            brain = build_network(n, k, args.seed, w_exc)
            drive = make_drive(n, frac, amp, amp, args.seed)

            py = time_python(brain, drive, args.warm, args.steps)
            del brain

            ref = reference_run(n, k, args.seed, w_exc, drive, args.warm, args.steps)
            export_fixture(fixture, ref["brain"], drive, args.steps,
                           ref["step_spikes"], ref["neuron_spikes"])

            # The Python timing run and the reference run must agree: the physics
            # is deterministic, so a mismatch means the two runs saw different
            # networks and the fixture would not describe the timed trajectory.
            # The reference run must reproduce the timed run, but not bit-exactly:
            # MLX accumulates `scatter_add` with atomics, so two identical runs
            # differ by ~1 ULP in the soma state and can differ by a spike or two
            # over thousands of steps. `bench/results_scale.json` documents the
            # same behaviour ("spike_trajectories_identical: true" for 25 steps at
            # n=20000, but not guaranteed at scale). A tiny mismatch is therefore
            # expected and is RECORDED rather than hidden; a large one means the
            # fixture does not describe the timed run and the comparison is void.
            ref_total = int(sum(ref["step_spikes"]))
            delta = abs(ref_total - py["spikes_total"])
            rel_delta = delta / max(1, py["spikes_total"])
            if rel_delta > 1e-4:
                raise SystemExit(
                    f"reference run disagrees with timed run: {ref_total} vs "
                    f"{py['spikes_total']} spikes (rel {rel_delta:.2e}); "
                    "fixture is not comparable"
                )
            del ref["brain"]

            rust = run_rust(binpath, fixture, ["--steps", str(args.steps),
                                               "--warm", str(args.warm)])

            row = {
                "python_timed_vs_reference_spike_delta": delta,
                "python_timed_vs_reference_spike_rel_delta": rel_delta,
                "loadavg_after": list(__import__("os").getloadavg()),
                "label": label, "n_neurons": n, "k_out": k, "w_exc": w_exc,
                "driven_frac": frac, "drive_amp": amp,
                "python": py, "rust": rust,
                "fixture": str(fixture),
            }
            rows.append(row)

            flush()
            print(f"{label:22s} py_min {py['ms_per_step_min']:8.4f} "
                  f"evt_min {rust['event_driven']['ms_per_step_min']:8.4f} "
                  f"dense_min {rust['dense']['ms_per_step_min']:8.4f} "
                  f"pyshaped_min {rust['python_shaped']['ms_per_step_min']:9.4f} "
                  f"| upd/step {rust['event_driven']['neuron_updates_per_step']:9.1f} "
                  f"active {py['active_frac']*100:7.5f}% "
                  f"| spikes py={py['spikes_total']} rust={rust['event_driven']['spikes_total']}",
                  flush=True)

            try:
                fixture.unlink()
            except OSError:
                pass

    load1, load5, load15 = __import__("os").getloadavg()
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "loadavg_at_end": [load1, load5, load15],
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "protocol": {
            "levels": levels,
            "warmup_steps": args.warm,
            "timed_steps": args.steps,
            "noise_std": 0.0,
            "plasticity": "disabled",
            "python_forces_eval_every_step": True,
            "dt_ms": 1.0,
            "note": "both engines run identical connectivity/state/drive from a shared fixture",
        },
        "rows": rows,
    }
    outfile = args.out / "results.json"
    outfile.write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
