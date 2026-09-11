"""End-to-end integration: a small Brain runs, reads out, and resets.

These tests exercise the public ``Brain`` API on both available backends and
check that reported statistics stay coherent with the underlying state.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.backend import Backend, has_mlx
from brain.connectivity import DelayBuffer, SynapseConfig, Synapses
from brain.neurons import NeuronConfig
from brain.plasticity import PlasticityConfig
from brain.simulator import Brain, SimConfig


def _brain(be, *, n=128, k=8, frac=0.25, seed=11, drive_value=1.5):
    return Brain(
        SimConfig(
            n_neurons=n,
            k_out=k,
            seed=seed,
            spike_capacity_frac=frac,
            init_noise=0.2,
            neu=NeuronConfig(n=n, noise_std=0.05),
            syn=SynapseConfig(n_pre=n, n_post=n, k_out=k, seed=seed),
            plas=PlasticityConfig(),
        ),
        be,
    )


def test_small_brain_runs_100ms_and_reports_sane_activity(backend_name):
    be = Backend(backend_name, seed=11)
    brain = _brain(be)
    n = brain.cfg.n_neurons

    result = brain.run(100, drive=be.full((n,), 1.5))

    assert result["steps"] == 100
    assert result["total_spikes"] > 0, "brain produced no spikes at all under tonic drive"
    assert 0.0 < result["mean_active_frac"] < 0.5, (
        f"mean_active_frac={result['mean_active_frac']} is not a sane sparse activity level"
    )
    assert len(result["spikes_per_step"]) == 100
    assert all(0 <= s <= n for s in result["spikes_per_step"]), (
        "a step reported an impossible spike count"
    )
    assert sum(result["spikes_per_step"]) == result["total_spikes"], (
        "total_spikes does not match the per-step spike counts"
    )
    assert result["overflow"] >= 0, "overflow is negative"
    k_out = brain.syn_cfg.k_out
    assert result["total_synops_active"] == sum(s * k_out for s in result["spikes_per_step"]), (
        "total_synops_active does not match the per-step spike-count * fan-out"
    )
    assert result["total_synops_executed"] == 100 * brain.capacity * k_out, (
        "total_synops_executed does not match steps * capacity * fan-out"
    )


def test_step_stats_are_consistent_with_state(backend_name):
    be = Backend(backend_name, seed=11)
    brain = _brain(be)
    n = brain.cfg.n_neurons

    stats = brain.step(external_soma=be.full((n,), 1.5))

    assert 0.0 <= stats.active_frac <= 1.0, f"active_frac={stats.active_frac} outside [0, 1]"
    assert stats.active_frac == pytest.approx(stats.spikes / n), (
        "active_frac is not spikes / n_neurons"
    )
    assert stats.synops_active == stats.spikes * brain.syn_cfg.k_out, (
        "synops_active does not match spikes * k_out"
    )
    assert stats.synops_executed == brain.capacity * brain.syn_cfg.k_out, (
        "synops_executed does not match capacity * k_out"
    )
    assert np.isfinite(stats.mean_v), f"mean_v is not finite: {stats.mean_v}"


def test_population_vector_and_spike_rates_are_sane(backend_name):
    be = Backend(backend_name, seed=11)
    brain = _brain(be)
    n = brain.cfg.n_neurons
    brain.run(20, drive=be.full((n,), 1.5))

    vector = be.to_numpy(brain.population_vector(8))
    rates = be.to_numpy(brain.spike_rates())

    assert vector.shape == (8,), f"population_vector shape {vector.shape}, expected (8,)"
    assert np.all(np.isfinite(vector)), "population_vector contains non-finite values"
    assert rates.shape == (n,), f"spike_rates shape {rates.shape}, expected {(n,)}"
    assert np.all(rates >= 0.0), "spike_rates contains negative rates"


def test_reset_returns_simulator_to_t0(backend_name):
    be = Backend(backend_name, seed=11)
    brain = _brain(be)
    n = brain.cfg.n_neurons
    brain.run(20, drive=be.full((n,), 1.5))
    assert brain._time > 0
    assert brain.total_spikes > 0

    brain.reset()

    assert brain._time == 0, "reset did not restore the simulation clock"
    assert brain.total_spikes == 0, "reset did not clear total_spikes"
    assert brain.total_overflow == 0, "reset did not clear total_overflow"
    assert brain.delay.pending() == pytest.approx(0.0), (
        "reset left synaptic current in the delay buffer"
    )
    np.testing.assert_allclose(
        be.to_numpy(brain.neurons.v_soma),
        np.full(n, brain.neuron_cfg.e_leak, dtype=np.float32),
        atol=1e-6,
        err_msg="reset did not restore the resting membrane potential",
    )


def test_backend_identity_is_reported(backend_name):
    be = Backend(backend_name, seed=11)
    brain = _brain(be)
    assert brain.be.name == backend_name, (
        f"Brain reported backend {brain.be.name!r}, expected {backend_name!r}"
    )


@pytest.mark.skipif(
    not has_mlx(), reason="MLX is not importable on this machine; cross-backend integration skipped"
)
def test_numpy_and_mlx_runs_both_complete_and_agree_qualitatively():
    drive = np.full(64, 1.4, dtype=np.float32)
    fractions = {}
    for name in ("numpy", "mlx"):
        be = Backend(name, seed=5)
        brain = _brain(be, n=64, k=8, seed=5)
        result = brain.run(40, drive=be.array(drive))
        fractions[name] = result["mean_active_frac"]

        assert result["total_spikes"] > 0, f"{name} backend produced no spikes"
        assert 0.0 < fractions[name] < 0.5, (
            f"{name} backend active fraction {fractions[name]} is not sane"
        )
    assert abs(fractions["numpy"] - fractions["mlx"]) < 0.05, (
        f"backends diverge qualitatively: {fractions}"
    )


def test_spike_delivers_signed_weights_to_exact_targets(backend_name):
    """Synaptic delivery end-to-end: exact targets, magnitudes, signs, and delays."""
    be = Backend(backend_name, seed=0)
    n_pre = n_post = 12
    k = 5
    syn = Synapses(
        SynapseConfig(n_pre=n_pre, n_post=n_post, k_out=k, seed=2), be
    )
    buf = DelayBuffer(n_post, syn.cfg.delay_max, be)

    delivered = np.zeros(n_post, dtype=np.float32)
    expected = np.zeros(n_post, dtype=np.float32)
    for pre in range(n_pre):
        idx = be.array([pre], dtype=be.idx_dtype)
        targets = be.take(syn.targets, idx)
        weights = be.take(syn.weights, idx)
        delays = be.take(syn.delays, idx)
        buf.schedule(0, targets, delays, weights)

        target_row = be.to_numpy(targets)[0]
        weight_row = be.to_numpy(weights)[0]
        np.add.at(expected, target_row, weight_row)
        for target, weight in zip(target_row, weight_row):
            if syn.is_exc[pre]:
                assert weight > 0.0, f"excitatory neuron {pre} delivered a non-positive weight"
            else:
                assert weight < 0.0, f"inhibitory neuron {pre} delivered a non-negative weight"

    for t in range(1, syn.cfg.delay_max + 1):
        delivered += be.to_numpy(buf.read_and_clear(t)).astype(np.float32)

    np.testing.assert_allclose(
        delivered,
        expected,
        atol=1e-6,
        err_msg="synaptic current did not reach the exact post-synaptic targets with the "
        "exact signed weights",
    )
    assert np.any(expected > 0.0) and np.any(expected < 0.0), (
        "test is vacuous: no mixed E/I current was delivered"
    )
    assert buf.pending() == pytest.approx(0.0), "delay buffer not empty after delivery"
