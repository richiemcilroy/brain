"""Simulator contract: spike compaction, overflow accounting, determinism.

Spike compaction is the hot path and the one place where an off-by-one silently
corrupts a run, so it is checked against ``np.flatnonzero`` on forced masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.backend import Backend
from brain.connectivity import SynapseConfig
from brain.neurons import NeuronConfig
from brain.plasticity import PlasticityConfig
from brain.simulator import Brain, SimConfig


def _brain(be, *, n=64, k=8, frac=0.10, seed=0, inhibition="none", k_wta=0,
           noise_std=0.0, init_noise=0.0, plas=None):
    return Brain(
        SimConfig(
            n_neurons=n,
            k_out=k,
            seed=seed,
            inhibition=inhibition,
            k_wta=k_wta,
            spike_capacity_frac=frac,
            init_noise=init_noise,
            neu=NeuronConfig(n=n, noise_std=noise_std),
            syn=SynapseConfig(n_pre=n, n_post=n, k_out=k, seed=seed),
            plas=plas or PlasticityConfig(),
        ),
        be,
    )


def test_compaction_is_exact_against_flatnonzero_when_unbounded(backend_name):
    """With capacity == n_neurons the buffer must equal np.flatnonzero exactly."""
    be = Backend(backend_name, seed=0)
    brain = _brain(be, frac=1.0)
    rng = np.random.default_rng(0)
    mask = rng.random(brain.cfg.n_neurons) < 0.25

    buf, overflow = brain._compact(be.array(mask))
    got = be.to_numpy(buf).astype(np.int64)
    expected = np.flatnonzero(mask)

    assert overflow == 0, f"unexpected overflow {overflow} with full capacity"
    assert got.size >= expected.size, "compaction buffer is smaller than the expected index list"
    np.testing.assert_array_equal(
        got[: expected.size],
        expected,
        err_msg="compacted indices are not np.flatnonzero(mask) in order",
    )
    np.testing.assert_array_equal(
        got[expected.size:],
        np.zeros(got.size - expected.size, dtype=np.int64),
        err_msg="compaction left non-zero garbage in unused buffer slots",
    )


def test_compaction_is_exact_when_every_neuron_spikes(backend_name):
    """A full-density mask must reproduce the whole index array exactly."""
    be = Backend(backend_name, seed=0)
    brain = _brain(be, frac=1.0)
    mask = np.ones(brain.cfg.n_neurons, dtype=bool)

    buf, overflow = brain._compact(be.array(mask))
    got = be.to_numpy(buf).astype(np.int64)

    assert overflow == 0, "full-density mask reported overflow despite capacity == n_neurons"
    np.testing.assert_array_equal(
        got,
        np.flatnonzero(mask),
        err_msg="full-density compaction is not exactly np.flatnonzero(mask)",
    )


def test_compaction_truncates_in_order_and_counts_overflow(backend_name):
    be = Backend(backend_name, seed=0)
    brain = _brain(be, frac=0.10)
    rng = np.random.default_rng(3)
    mask = np.zeros(brain.cfg.n_neurons, dtype=bool)
    mask[rng.choice(brain.cfg.n_neurons, 12, replace=False)] = True

    buf, overflow = brain._compact(be.array(mask))
    got = be.to_numpy(buf).astype(np.int64)
    expected = np.flatnonzero(mask)

    assert got.size == brain.capacity, "compaction buffer size != configured capacity"
    assert overflow == 12 - brain.capacity, (
        f"overflow reported {overflow}, expected {12 - brain.capacity}"
    )
    np.testing.assert_array_equal(
        got,
        expected[: brain.capacity],
        err_msg="overflowing compaction did not keep the first capacity indices in order",
    )


def test_spike_overflow_is_counted_when_activity_exceeds_capacity(backend_name):
    be = Backend(backend_name, seed=0)
    brain = _brain(be, k=4, frac=0.10)
    n = brain.cfg.n_neurons

    for _ in range(15):
        brain.step(external_soma=be.full((n,), 3.0))

    assert max(brain.wall_step_ms) > 0.0, "no steps were timed"
    assert brain.total_spikes > 0, "forced drive produced no spikes at all"
    assert brain.total_overflow > 0, (
        f"capacity={brain.capacity} but total_overflow={brain.total_overflow}; "
        "overflow was not counted"
    )


def test_overflow_accounting_matches_per_step_excess(backend_name):
    be = Backend(backend_name, seed=0)
    brain = _brain(be, k=4, frac=0.10)
    n = brain.cfg.n_neurons

    result = brain.run(15, drive=be.full((n,), 3.0))
    excess = sum(max(0, s - brain.capacity) for s in result["spikes_per_step"])

    assert result["overflow"] == excess, (
        f"reported overflow {result['overflow']} != sum of per-step excess {excess}"
    )
    assert excess > 0, "test is vacuous: activity never exceeded capacity"


def test_no_overflow_under_sparse_activity(backend_name):
    be = Backend(backend_name, seed=7)
    brain = _brain(be, n=256, k=8, frac=0.10, seed=7, noise_std=0.02, init_noise=0.1)
    drive = np.zeros(256, dtype=np.float32)
    drive[[3, 17, 40, 99]] = 1.4

    result = brain.run(60, drive=be.array(drive))

    assert result["overflow"] == 0, (
        f"sparse activity overflowed the buffer: {result['overflow']} dropped spikes"
    )
    assert result["mean_active_frac"] < 0.05, (
        f"activity was not sparse: mean_active_frac={result['mean_active_frac']}"
    )
    assert result["total_spikes"] > 0, "test is vacuous: no spikes occurred"


def test_same_seed_is_deterministic(backend_name):
    def run_once(seed):
        be = Backend(backend_name, seed=seed)
        brain = _brain(be, seed=seed, init_noise=0.3, plas=PlasticityConfig())
        result = brain.run(30, drive=be.full((64,), 1.2), neuromod=0.5)
        return result["spikes_per_step"], be.to_numpy(brain.synapses.weights).copy()

    spikes_a, weights_a = run_once(0)
    spikes_b, weights_b = run_once(0)

    assert spikes_a == spikes_b, (
        "same-seed runs produced different spike counts: "
        f"{spikes_a[:8]}... vs {spikes_b[:8]}..."
    )
    np.testing.assert_array_equal(
        weights_a, weights_b, err_msg="same-seed runs produced different synaptic weights"
    )


def test_different_seed_changes_the_run(backend_name):
    def run_once(seed):
        be = Backend(backend_name, seed=seed)
        brain = _brain(be, seed=seed, init_noise=0.3, plas=PlasticityConfig())
        result = brain.run(30, drive=be.full((64,), 1.2), neuromod=0.5)
        return result["spikes_per_step"], be.to_numpy(brain.synapses.weights).copy()

    spikes_a, weights_a = run_once(0)
    spikes_b, weights_b = run_once(1)

    differs = (spikes_a != spikes_b) or (not np.array_equal(weights_a, weights_b))
    assert differs, "different seeds produced identical runs; the seed is not being used"


def test_memory_bytes_reports_plausible_values(backend_name):
    be = Backend(backend_name, seed=0)
    brain = _brain(be)
    brain.run(5, drive=be.full((64,), 1.2))

    report = brain.memory_bytes()
    expected_keys = {
        "synapse_targets",
        "synapse_weights",
        "synapse_delays",
        "neuron_state",
        "eligibility",
        "delay_buffer",
    }
    assert expected_keys <= set(report), f"memory_bytes is missing keys: {expected_keys - set(report)}"
    for key, value in report.items():
        assert isinstance(value, int), f"memory_bytes[{key!r}] is {type(value)}, expected int"
        assert value >= 0, f"memory_bytes[{key!r}] is negative: {value}"
    assert report["synapse_targets"] > 0, "targets array reported as zero bytes"
    assert report["synapse_weights"] > 0, "weights array reported as zero bytes"
    assert report["neuron_state"] > 0, "neuron state reported as zero bytes"
    assert sum(report.values()) > 0, "total reported memory is zero"


def test_measured_throughput_is_populated(backend_name):
    be = Backend(backend_name, seed=0)
    brain = _brain(be, k=8)
    brain.run(10, drive=be.full((64,), 1.2))

    report = brain.measured_throughput()
    assert report["n_neurons"] == 64.0
    assert report["n_synapses"] == 64.0 * 8.0
    assert report["bio_ms_per_wall_s"] > 0.0, "biological-time throughput not measured"
    assert report["synops_executed_per_second"] > 0.0
    assert report["overflow_total"] >= 0.0


@pytest.mark.xfail(
    reason=(
        "known bug (found during this validation run): _compact zero-fills unused buffer "
        "slots, but Brain.step consumes the whole buffer as spike indices, so a quiet network "
        "treats padding zeros as spikes of neuron 0. Observed: 0 spikes in a step yet "
        "x_pre[0]==capacity and a phantom delay-buffer current > 0 is scheduled."
    ),
)
def test_compact_padding_is_not_consumed_as_a_spike_index(backend_name):
    be = Backend(backend_name, seed=0)
    brain = _brain(be, plas=PlasticityConfig())

    stats = brain.step()

    assert stats.spikes == 0, "test setup is wrong: the quiet network spiked"
    x_pre = be.to_numpy(brain.plasticity.x_pre)
    assert x_pre[0] == 0.0, (
        f"neuron 0 never spiked, yet x_pre[0]={x_pre[0]:.3f}; _compact padding "
        f"(capacity={brain.capacity}) is being treated as spike index 0"
    )
    pending = brain.delay.pending()
    assert pending == 0.0, (
        f"no neuron spiked, yet {pending:.3f} of phantom synaptic current was scheduled"
    )


@pytest.mark.xfail(
    reason=(
        "known bug (found during this validation run): _kwta_mask reads v_soma AFTER "
        "NeuronState.update has reset spikers to v_reset=0, so every spike candidate scores "
        "0, the k-th largest cut equals 0, and the mask is returned unchanged. Observed: "
        "k_wta=8 allowed 11 spikes in one step, i.e. k-WTA does not bound sparsity when "
        "v_reset == 0 (the default)."
    ),
)
def test_kwta_enforces_at_most_k_winners(backend_name):
    be = Backend(backend_name, seed=0)
    brain = _brain(be, n=64, k=8, frac=0.5, seed=3, inhibition="kwta", k_wta=8)
    brain.neurons.v_soma = be.zeros((64,))

    raw = np.zeros(64, dtype=bool)
    raw[:11] = True
    mask = brain._kwta_mask(be.array(raw))
    kept = int(be.to_numpy(be.sum(be.astype(mask, be.float_dtype))))

    assert kept <= brain.cfg.k_wta, (
        f"k_wta={brain.cfg.k_wta} but {kept} neurons survived the winner-take-all mask"
    )
