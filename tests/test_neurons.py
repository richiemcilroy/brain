"""Biophysical checks on the two-compartment adaptive neuron.

The central claim under test is the dendritic ablation: ``dcaap`` must be a
genuinely *non-monotonic* (tuned) transfer function while ``linear`` is
monotonic, because the whole novelty argument rests on that difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.backend import Backend
from brain.neurons import NeuronConfig, NeuronState


def _state(be, n=8, **kwargs):
    cfg = NeuronConfig(n=n, noise_std=kwargs.pop("noise_std", 0.0), **kwargs)
    return NeuronState(cfg, be, init_noise=0.0)


def _total_spikes(be, mask) -> int:
    return int(be.to_numpy(be.sum(be.astype(mask, be.float_dtype))))


def test_zero_input_produces_no_spikes(be):
    st = _state(be, n=16)
    total = 0
    for _ in range(300):
        total += _total_spikes(be, st.update(be.zeros((16,)), be.zeros((16,))))

    assert total == 0, f"a neuron with zero input spiked {total} times over 300 ms"
    np.testing.assert_allclose(
        be.to_numpy(st.v_soma),
        np.full(16, st.cfg.e_leak, dtype=np.float32),
        atol=1e-6,
        err_msg="soma potential drifted away from leak with zero input",
    )


def test_strong_tonic_input_produces_sustained_plausible_rate(be):
    st = _state(be, n=4)
    drive = be.full((4,), 2.0)
    per_step = []
    for _ in range(500):
        per_step.append(_total_spikes(be, st.update(drive, be.zeros((4,)))))
    per_step = np.asarray(per_step, dtype=np.int64)

    rate_hz = st.mean_rate_hz(500)
    assert 10.0 <= rate_hz <= 200.0, (
        f"tonic drive of 2.0 gave {rate_hz:.1f} Hz per neuron, outside a plausible LIF band"
    )
    first_half, second_half = int(per_step[:250].sum()), int(per_step[250:].sum())
    assert first_half > 0 and second_half > 0, (
        f"spiking was not sustained: first-half={first_half}, second-half={second_half}"
    )
    assert second_half >= 0.5 * first_half, (
        f"spiking died out over the run: first-half={first_half}, second-half={second_half}"
    )
    assert np.all(be.to_numpy(st.spike_count) > 0), "some neurons never spiked under tonic drive"


def test_refractory_period_is_never_violated(be):
    st = _state(be, n=4)
    drive = be.full((4,), 2.0)
    spike_times = [[] for _ in range(4)]
    for t in range(400):
        mask = be.to_numpy(st.update(drive, be.zeros((4,))))
        for i in range(4):
            if mask[i]:
                spike_times[i].append(t)

    refractory = st.cfg.refractory_ms
    for i, times in enumerate(spike_times):
        assert len(times) >= 2, f"neuron {i} spiked fewer than twice; cannot check refractory"
        intervals = np.diff(times)
        assert intervals.min() >= refractory, (
            f"neuron {i} fired again after {intervals.min()} ms, below "
            f"refractory_ms={refractory}"
        )


def test_refractory_gates_a_suprathreshold_soma(be):
    st = _state(be, n=1)
    st.v_soma = be.full((1,), 5.0)
    st.refrac = be.full((1,), 2.0)
    blocked = st.update(be.zeros((1,)), be.zeros((1,)))
    assert _total_spikes(be, blocked) == 0, "spike escaped while neuron was refractory"

    st.v_soma = be.full((1,), 5.0)
    st.refrac = be.full((1,), 1.0)
    released = st.update(be.zeros((1,)), be.zeros((1,)))
    assert _total_spikes(be, released) == 1, (
        "neuron did not spike once its refractory period had elapsed"
    )


def test_dcaap_is_non_monotonic_and_peaks_at_dend_scale(be):
    xs = np.linspace(0.0, 4.0, 401, dtype=np.float32)
    cfg = NeuronConfig(n=len(xs), dend_mode="dcaap", noise_std=0.0)
    st = NeuronState(cfg, be, init_noise=0.0)
    st.v_dend = be.array(xs, dtype=be.float_dtype)
    phi = be.to_numpy(st.dendritic_activation()).astype(np.float64)

    peak = int(np.argmax(phi))
    expected_peak = cfg.e_dend + cfg.dend_scale
    assert 0 < peak < len(xs) - 1, "dCaAP peak is at the edge of the sweep, not interior"
    assert xs[peak] == pytest.approx(expected_peak, abs=5e-3), (
        f"dCaAP peaks at v_dend={xs[peak]}, expected e_dend + dend_scale={expected_peak}"
    )
    assert phi[peak] == pytest.approx(1.0, abs=1e-3), (
        f"dCaAP peak value is {phi[peak]}, expected 1.0"
    )

    tail = phi[peak:]
    assert np.all(np.diff(tail) < 0.0), (
        "dCaAP is not decreasing beyond its peak; the non-monotonic claim fails"
    )
    assert np.any(phi[peak + 1:] < phi[peak] - 0.05), (
        "no numerically lower response beyond the peak: phi is effectively monotonic"
    )
    beyond = phi[int(np.argmin(np.abs(xs - 4.0 * cfg.dend_scale)))]
    assert beyond < 0.25 * phi[peak], (
        f"dCaAP attenuation is too weak: phi(4*dend_scale)={beyond:.4f}"
    )


def test_dcaap_peak_tracks_configured_dend_scale(be):
    xs = np.linspace(0.0, 8.0, 801, dtype=np.float32)
    cfg = NeuronConfig(n=len(xs), dend_mode="dcaap", dend_scale=2.0, noise_std=0.0)
    st = NeuronState(cfg, be, init_noise=0.0)
    st.v_dend = be.array(xs, dtype=be.float_dtype)
    phi = be.to_numpy(st.dendritic_activation()).astype(np.float64)

    peak = int(np.argmax(phi))
    assert xs[peak] == pytest.approx(cfg.e_dend + cfg.dend_scale, abs=5e-3), (
        f"dCaAP peak did not track dend_scale: peak at {xs[peak]}, expected {cfg.dend_scale}"
    )
    assert np.all(np.diff(phi[peak:]) < 0.0), "dCaAP tail is not decreasing at dend_scale=2.0"


def test_linear_dendritic_mode_is_monotonic(be):
    xs = np.linspace(0.0, 4.0, 401, dtype=np.float32)
    cfg = NeuronConfig(n=len(xs), dend_mode="linear", noise_std=0.0)
    st = NeuronState(cfg, be, init_noise=0.0)
    st.v_dend = be.array(xs, dtype=be.float_dtype)
    phi = be.to_numpy(st.dendritic_activation()).astype(np.float64)

    assert np.all(np.diff(phi) > 0.0), "linear dendritic mode is not strictly increasing"
    assert int(np.argmax(phi)) == len(xs) - 1, "linear mode did not peak at the largest input"
    assert phi[-1] == pytest.approx(4.0 / cfg.dend_scale, rel=1e-6), (
        "linear mode did not scale input by 1/dend_scale"
    )


def test_dend_mode_none_yields_zero_dendritic_drive(be):
    cfg = NeuronConfig(n=8, dend_mode="none", noise_std=0.0)
    st = NeuronState(cfg, be, init_noise=0.0)
    st.v_dend = be.full((8,), 3.0)
    drive = be.to_numpy(st.dendritic_activation())
    np.testing.assert_array_equal(
        drive, np.zeros(8, dtype=np.float32),
        err_msg="dend_mode='none' produced non-zero dendritic drive",
    )
    for _ in range(50):
        st.update(be.zeros((8,)), be.full((8,), 3.0))
    np.testing.assert_allclose(
        be.to_numpy(st.v_soma), np.zeros(8, dtype=np.float32), atol=1e-6,
        err_msg="dendritic input leaked into the soma with dend_mode='none'",
    )


def test_adaptation_current_grows_with_spiking(be):
    spiking = _state(be, n=2)
    quiet = _state(be, n=2)

    for _ in range(30):
        spiking.update(be.full((2,), 2.0), be.zeros((2,)))
        quiet.update(be.zeros((2,)), be.zeros((2,)))
    adapt_mid = float(be.to_numpy(spiking.adapt).max())

    for _ in range(270):
        spiking.update(be.full((2,), 2.0), be.zeros((2,)))
        quiet.update(be.zeros((2,)), be.zeros((2,)))
    adapt_end = float(be.to_numpy(spiking.adapt).max())
    adapt_quiet = float(np.abs(be.to_numpy(quiet.adapt)).max())

    assert adapt_quiet == 0.0, f"adaptation grew without spikes: {adapt_quiet}"
    assert adapt_end > 0.05, f"adaptation current stayed negligible after spiking: {adapt_end}"
    assert adapt_end > adapt_mid, (
        f"adaptation did not accumulate with continued spiking: {adapt_mid} -> {adapt_end}"
    )
    assert float(be.to_numpy(spiking.spike_count).min()) > 0.0, "test neuron never spiked"


def test_soma_leak_follows_tau_soma(be):
    st = _state(be, n=1)
    st.v_soma = be.full((1,), 0.5)
    steps = int(st.cfg.tau_soma)
    for _ in range(steps):
        st.update(be.zeros((1,)), be.zeros((1,)))

    got = float(be.to_numpy(st.v_soma)[0])
    expected = 0.5 * float(np.exp(-1.0))
    assert got == pytest.approx(expected, rel=0.06), (
        f"soma decayed to {got:.4f} in {steps} ms, expected ~{expected:.4f} for tau_soma="
        f"{st.cfg.tau_soma}"
    )


def test_dendrite_follows_tau_dend(be):
    cfg = NeuronConfig(n=1, tau_dend=10.0, dend_gain=0.0, noise_std=0.0)
    st = NeuronState(cfg, be, init_noise=0.0)
    for _ in range(20):
        st.update(be.zeros((1,)), be.full((1,), 1.0))

    got = float(be.to_numpy(st.v_dend)[0])
    expected = 1.0 - float(np.exp(-2.0))
    assert got == pytest.approx(expected, rel=0.05), (
        f"dendrite charged to {got:.4f} after 20 ms, expected ~{expected:.4f} for tau_dend=10"
    )


def test_mean_rate_hz_has_correct_units(be):
    steps = 500
    st = _state(be, n=4)
    for _ in range(steps):
        st.update(be.full((4,), 2.0), be.zeros((4,)))

    counts = be.to_numpy(st.spike_count)
    expected = float(counts.mean()) / (steps * st.cfg.dt / 1000.0)
    assert st.mean_rate_hz(steps) == pytest.approx(expected, rel=1e-5), (
        "mean_rate_hz is not spike_count / (n * seconds)"
    )
