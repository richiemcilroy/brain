"""Export a Python-simulator trajectory as a fixture for the Rust core.

The comparison is deliberately structured so that a divergence can only come
from the DYNAMICS, never from a different random graph:

  * Python builds connectivity and initial state exactly as `brain/` does.
  * Both engines are given the SAME targets/delays/weights/state/drive.
  * `noise_std = 0`, which is both the condition under which the Rust core's
    event-driven skipping is exact, and the condition that makes the comparison
    meaningful at all -- with noise on, the two RNGs differ and the trajectories
    would diverge for reasons that say nothing about the port's correctness.

Usage:
    python3 experiments/rust_core/python/export_fixture.py OUT.bin \
        [--n 3000] [--k 32] [--steps 300] [--drive-frac 0.05]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.connectivity import SynapseConfig, Synapses          # noqa: E402
from brain.neurons import NeuronConfig, NeuronState             # noqa: E402
from brain.plasticity import PlasticityConfig                   # noqa: E402
from brain.simulator import Brain, SimConfig                    # noqa: E402

MAGIC = b"RBRAIN01"
DEND_MODE_IDS = {"dcaap": 0, "linear": 1, "none": 2}


def build(n: int, k: int, drive_frac: float, seed: int, w_exc: float = 0.40):
    """Build a Brain with plasticity off and noise off (the comparison regime).

    `w_exc` is raised above the library default of 0.08 on purpose. At the
    default the recurrent current is far too weak to change the outcome, so a
    correctness comparison would pass even if delivery were completely broken
    (measured: 1,788 spikes with weights vs 1,787 with all weights zeroed).
    At 0.40 with k_out=1024 the same measurement is 2,154 vs 1,099, so the
    comparison actually exercises the delay ring and the scatter.
    """
    neu = NeuronConfig(n=n, dt=1.0, noise_std=0.0)
    syn = SynapseConfig(n_pre=n, n_post=n, k_out=k, seed=seed, delay_max=4,
                        w_exc=w_exc, w_inh=w_exc * 4)
    cfg = SimConfig(
        n_neurons=n, k_out=k, seed=seed, init_noise=0.0,
        neu=neu, syn=syn, plas=PlasticityConfig(enabled=False),
    )
    return Brain(cfg, backend="numpy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--drive-frac", type=float, default=0.05)
    ap.add_argument("--drive-low", type=float, default=1.15)
    ap.add_argument("--drive-high", type=float, default=1.35)
    ap.add_argument("--w-exc", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    brain = build(args.n, args.k, args.drive_frac, args.seed, args.w_exc)

    # Drive: a fixed random subset of neurons, identical in both engines.
    rng = np.random.default_rng(args.seed + 777)
    n_driven = max(1, int(round(args.drive_frac * args.n)))
    driven = np.sort(rng.choice(args.n, size=n_driven, replace=False))
    amps = rng.uniform(args.drive_low, args.drive_high, size=n_driven)
    drive = np.zeros((args.n,), dtype=np.float32)
    drive[driven] = amps.astype(np.float32)

    # Record the exact initial state, BEFORE any step is taken.
    be = brain.be
    v_soma0 = np.array(be.to_numpy(brain.neurons.v_soma), dtype=np.float32)
    v_dend0 = np.array(be.to_numpy(brain.neurons.v_dend), dtype=np.float32)
    adapt0 = np.array(be.to_numpy(brain.neurons.adapt), dtype=np.float32)
    refrac0 = np.array(be.to_numpy(brain.neurons.refrac), dtype=np.float32)

    step_spikes: list[int] = []
    for _ in range(args.steps):
        st = brain.step(external_soma=be.array(drive))
        step_spikes.append(int(st.spikes))

    neuron_spikes = np.array(be.to_numpy(brain.neurons.spike_count), dtype=np.uint32)

    targets = np.asarray(be.to_numpy(brain.synapses.targets))
    delays = np.asarray(be.to_numpy(brain.synapses.delays))
    weights = np.asarray(be.to_numpy(brain.synapses.weights), dtype=np.float32)

    neu = brain.neuron_cfg
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIIII", args.n, args.k, syn_max_delay(brain),
                            args.steps, DEND_MODE_IDS[neu.dend_mode], 1))
        f.write(struct.pack("<14f",
                            neu.tau_soma, neu.tau_dend, neu.tau_adapt, neu.e_leak,
                            neu.e_dend, neu.v_reset, neu.v_thresh, neu.adapt_base,
                            neu.adapt_inc, neu.dend_scale, neu.dend_gain,
                            neu.refractory_ms, neu.noise_std, neu.dt))
        f.write(targets.astype("<u4").tobytes())
        f.write(delays.astype("<u4").tobytes())
        f.write(weights.astype("<f4").tobytes())
        f.write(v_soma0.astype("<f4").tobytes())
        f.write(v_dend0.astype("<f4").tobytes())
        f.write(adapt0.astype("<f4").tobytes())
        f.write(refrac0.astype("<f4").tobytes())
        f.write(drive.astype("<f4").tobytes())
        f.write(neuron_spikes.astype("<u4").tobytes())
        f.write(np.array(step_spikes, dtype="<u4").tobytes())

    total = int(neuron_spikes.sum())
    print(f"wrote {args.out}")
    print(f"  n={args.n} k={args.k} synapses={args.n * args.k} steps={args.steps}")
    print(f"  python total spikes = {total}")
    print(f"  python neurons that spiked = {int((neuron_spikes > 0).sum())}")
    print(f"  python spikes/step (first 10) = {step_spikes[:10]}")
    print(f"  active fraction = {total / args.n / args.steps:.6f}")

    # Control: how much does recurrent delivery actually matter here? If this
    # ratio is ~1.0 the fixture cannot detect a broken delivery path.
    b2 = build(args.n, args.k, args.drive_frac, args.seed, args.w_exc)
    b2.synapses.weights = b2.be.zeros((args.n, args.k))
    zero_total = 0
    for _ in range(args.steps):
        zero_total += int(b2.step(external_soma=b2.be.array(drive)).spikes)
    print(f"  control (all weights zeroed): {zero_total} spikes")
    print(f"  recurrence sensitivity ratio = {total / max(zero_total, 1):.2f}"
          "   (<1.2 means this fixture cannot detect broken delivery)")
    return 0


def syn_max_delay(brain: Brain) -> int:
    """Largest delay actually present, not the configured bound.

    These differ in practice. `Backend.randint(low, high)` documents a half-open
    interval, but MLX 0.31.0 returns values equal to `high` on large draws: at
    shape (1000000, 64) with `randint(1, 5)` it produced nine 5s, while the same
    call at shape (1000, 64) never did. With `delay_max = 4` the ring has
    `max_delay + 1 = 5` slots, so a delay of 5 wraps onto slot 0 and collides
    with "arrive now" -- the spike is delivered a full ring-period late and
    corrupts the slot it lands in.

    The Rust core refuses to load a fixture whose delays exceed the declared
    bound (it validates every delay), which is how this surfaced. Using the
    observed maximum here keeps the fixture's declaration truthful, so the Rust
    core builds a ring large enough to reproduce the Python behaviour exactly.
    """
    d = brain.be.to_numpy(brain.synapses.delays)
    return int(max(int(brain.syn_cfg.delay_max), int(d.max())))


if __name__ == "__main__":
    raise SystemExit(main())
