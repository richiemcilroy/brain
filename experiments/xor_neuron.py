"""Single-neuron XOR: the core causal claim about dendrites.

A point neuron cannot compute XOR at any width of a single layer. A neuron with
a *non-monotonic* (tuned) dendritic subunit can, because the dendritic response
rises to a peak and then falls again, so "two inputs at once" produces LESS
drive than "exactly one input".

This script measures the dendritic transfer function and then tests whether a
real spiking neuron separates the XOR truth table, for both the intact
(``dcaap``) and ablated (``linear``) dendrite. It is the empirical basis for the
claim that the dendritic nonlinearity is load-bearing rather than decorative.

Run:  python3 experiments/xor_neuron.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.backend import get_backend  # noqa: E402
from brain.neurons import NeuronConfig, NeuronState  # noqa: E402

DRIVE_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
XOR_TABLE = [(0.0, 0.0, 0), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)]


def dendritic_transfer(mode: str, be) -> list[float]:
    """Steady-state dendritic drive as a function of total input magnitude."""
    cfg = NeuronConfig(n=1, dend_mode=mode, dend_scale=1.0, dend_gain=1.0,
                       tau_dend=8.0, noise_std=0.0)
    state = NeuronState(cfg, be)
    out = []
    for mag in DRIVE_MAGNITUDES:
        state.v_dend = be.zeros(1)
        for _ in range(300):  # relax to steady state
            state.v_dend = state.v_dend + (1.0 / cfg.tau_dend) * (
                -(state.v_dend - cfg.e_dend) + mag
            )
        drive = state.dendritic_activation()
        be.eval(drive)
        out.append(round(float(be.to_numpy(drive)[0]), 4))
    return out


def fires(mode: str, x1: float, x2: float, threshold: float, be,
          n_ms: int = 80) -> bool:
    """Does the neuron spike when driven with inputs (x1, x2)?"""
    cfg = NeuronConfig(n=1, dend_mode=mode, dend_scale=1.0, dend_gain=1.0,
                       v_thresh=threshold, noise_std=0.0, refractory_ms=5.0,
                       tau_dend=8.0, tau_soma=12.0)
    state = NeuronState(cfg, be)
    total = 0
    for _ in range(n_ms):
        i_dend = be.array([x1 + x2])
        spikes = state.update(be.zeros(1), i_dend)
        be.eval(spikes)
        total += int(be.to_numpy(spikes)[0])
    return total > 0


def best_threshold(drive: list[float]) -> float | None:
    """A threshold separating XOR iff drive is non-monotonic in the right way."""
    # need drive(0) < t <= drive(1) and drive(2) < t
    lo = drive[DRIVE_MAGNITUDES.index(1.0)]
    hi = max(drive[DRIVE_MAGNITUDES.index(2.0)], drive[0])
    if lo > hi:
        return (lo + hi) / 2.0
    return None


def main() -> int:
    be = get_backend("numpy", seed=0)
    print("=" * 72)
    print("Single-neuron XOR: dendritic nonlinearity as the causal mechanism")
    print("=" * 72)

    results: dict[str, bool] = {}
    for mode in ("dcaap", "linear"):
        drive = dendritic_transfer(mode, be)
        mono = all(b >= a for a, b in zip(drive, drive[1:]))
        print(f"\ndend_mode = {mode}")
        print(f"  magnitudes      : {DRIVE_MAGNITUDES}")
        print(f"  dendritic drive : {drive}")
        print(f"  monotonic       : {mono}")
        theta = best_threshold(drive)
        if theta is None:
            print("  separable threshold for XOR: NONE (drive never falls below "
                  "its one-input value) -> XOR impossible")
            results[mode] = False
            continue
        print(f"  separable threshold for XOR: {theta:.4f}")
        pattern = []
        for x1, x2, _target in XOR_TABLE:
            pattern.append(1 if fires(mode, x1, x2, theta, be) else 0)
        expected = [target for _a, _b, target in XOR_TABLE]
        ok = pattern == expected
        results[mode] = ok
        print(f"  observed spikes : {pattern}")
        print(f"  XOR truth table : {expected}")
        print(f"  XOR computed    : {ok}")

    print("\n" + "-" * 72)
    verdict = results["dcaap"] and not results["linear"]
    print(f"dcaap computes XOR          : {results['dcaap']}")
    print(f"linear ablation computes XOR: {results['linear']}")
    print(f"VERDICT: dendritic nonlinearity is causal: {verdict}")
    print("-" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
