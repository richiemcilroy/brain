"""Connectivity contract: Dale's principle, fan-out shape, and axonal delays.

Dale's principle ("all outgoing synapses of a neuron share one sign") is a hard
constraint of the substrate and must survive plasticity. The delay buffer must
deliver a spike at exactly ``t + delay`` and nowhere else, including across ring
wraparound.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.backend import Backend
from brain.connectivity import DelayBuffer, SynapseConfig, Synapses
from brain.plasticity import PlasticityConfig, ThreeFactorPlasticity


def _syn(be, n_pre=32, n_post=32, k=6, seed=0, excitatory_frac=0.8):
    return Synapses(
        SynapseConfig(
            n_pre=n_pre,
            n_post=n_post,
            k_out=k,
            seed=seed,
            excitatory_frac=excitatory_frac,
        ),
        be,
    )


def _row_signs_ok(syn) -> tuple[bool, str]:
    weights = syn.be.to_numpy(syn.weights)
    is_exc = syn.be.to_numpy(syn.is_exc).astype(bool)
    for i in range(syn.cfg.n_pre):
        row = weights[i]
        if is_exc[i] and np.any(row < 0.0):
            return False, f"excitatory neuron {i} has a negative outgoing synapse"
        if (not is_exc[i]) and np.any(row > 0.0):
            return False, f"inhibitory neuron {i} has a positive outgoing synapse"
    return True, ""


def test_dale_principle_holds_at_initialisation(backend_name):
    be = Backend(backend_name, seed=0)
    syn = _syn(be)
    ok, message = _row_signs_ok(syn)
    assert ok, message
    assert syn.sign_violations() == 0, "sign_violations() reported Dale violations at init"
    weights = be.to_numpy(syn.weights)
    is_exc = be.to_numpy(syn.is_exc).astype(bool)
    assert np.all(weights[is_exc] > 0.0), "excitatory neurons should start with positive weights"
    assert np.all(weights[~is_exc] < 0.0), "inhibitory neurons should start with negative weights"


def test_dale_principle_holds_after_plasticity(backend_name):
    be = Backend(backend_name, seed=0)
    n_pre = n_post = 24
    k = 5
    syn = _syn(be, n_pre=n_pre, n_post=n_post, k=k, seed=3)
    plas = ThreeFactorPlasticity(
        PlasticityConfig(eligibility="off", homeostasis=False), n_pre, n_post, k, be
    )
    rng = np.random.default_rng(0)

    for _ in range(300):
        idx = be.array(rng.integers(0, n_pre, 4), dtype=be.idx_dtype)
        post_mask = be.array(rng.random(n_post) < 0.4)
        plas.apply(idx, be.take(syn.targets, idx), post_mask, syn, float(rng.choice([-1.0, 1.0])))

    ok, message = _row_signs_ok(syn)
    assert ok, f"plasticity broke Dale's principle: {message}"
    assert syn.sign_violations() == 0, "sign_violations() non-zero after plasticity"
    weights = be.to_numpy(syn.weights)
    assert np.any(np.abs(weights) > 0.0), "test is vacuous: no weights changed"
    lo, hi = (be.to_numpy(x) for x in syn.weight_bounds())
    assert np.all(weights >= lo - 1e-6) and np.all(weights <= hi + 1e-6), (
        "plasticity pushed weights outside their configured Dale bounds"
    )


def test_k_out_fan_out_shape_and_target_range(backend_name):
    be = Backend(backend_name, seed=0)
    n_pre, n_post, k = 20, 16, 7
    syn = _syn(be, n_pre=n_pre, n_post=n_post, k=k)

    for name, arr in (
        ("targets", syn.targets),
        ("weights", syn.weights),
        ("delays", syn.delays),
    ):
        shape = tuple(be.to_numpy(arr).shape)
        assert shape == (n_pre, k), f"{name} has shape {shape}, expected {(n_pre, k)}"
    assert syn.n_synapses == n_pre * k, "n_synapses does not match n_pre * k_out"

    targets = be.to_numpy(syn.targets)
    assert targets.min() >= 0 and targets.max() < n_post, (
        f"synapse targets outside [0, {n_post}): min={targets.min()}, max={targets.max()}"
    )
    delays = be.to_numpy(syn.delays)
    assert delays.min() >= syn.cfg.delay_min and delays.max() <= syn.cfg.delay_max, (
        "drawn delays fall outside the configured [delay_min, delay_max] range"
    )


def test_delay_buffer_delivers_at_exactly_t_plus_delay(be):
    n_post, max_delay = 8, 4
    for delay in range(1, max_delay + 1):
        buf = DelayBuffer(n_post, max_delay, be)
        buf.schedule(
            0,
            be.array([3], dtype=be.idx_dtype),
            be.array([delay], dtype=be.idx_dtype),
            be.array([0.5], dtype=be.float_dtype),
        )
        arrivals = []
        for t in range(max_delay + 3):
            arrivals.append(be.to_numpy(buf.read_and_clear(t)).copy())

        nonzero = [(t, np.flatnonzero(vec)) for t, vec in enumerate(arrivals) if np.any(vec)]
        assert len(nonzero) == 1, (
            f"delay={delay} produced arrivals at {nonzero}, expected exactly one timestep"
        )
        assert nonzero[0][0] == delay, (
            f"delay={delay} delivered at t={nonzero[0][0]}, expected t={delay}"
        )
        assert nonzero[0][1].tolist() == [3], (
            f"delay={delay} delivered to targets {nonzero[0][1].tolist()}, expected [3]"
        )
        assert arrivals[delay][3] == pytest.approx(0.5), "delivered current magnitude is wrong"
        others = [vec for t, vec in enumerate(arrivals) if t != delay]
        assert all(np.all(vec == 0.0) for vec in others), (
            f"delay={delay} leaked current into other timesteps"
        )


def test_delay_buffer_read_and_clear_actually_clears(be):
    buf = DelayBuffer(n_post=4, max_delay=3, be=be)
    buf.schedule(
        0,
        be.array([2], dtype=be.idx_dtype),
        be.array([2], dtype=be.idx_dtype),
        be.array([0.25], dtype=be.float_dtype),
    )
    first = be.to_numpy(buf.read_and_clear(2)).copy()
    second = be.to_numpy(buf.read_and_clear(2)).copy()

    assert first[2] == pytest.approx(0.25), "scheduled current did not arrive"
    assert np.count_nonzero(first) == 1, "current arrived at more than one target"
    np.testing.assert_array_equal(
        second, np.zeros(4, dtype=np.float32),
        err_msg="second read at the same timestep returned non-zero current",
    )
    assert buf.pending() == pytest.approx(0.0), "buffer still holds current after read_and_clear"


def test_delay_buffer_sums_duplicate_deliveries(be):
    buf = DelayBuffer(n_post=4, max_delay=3, be=be)
    for weight in (0.1, 0.2):
        buf.schedule(
            0,
            be.array([1], dtype=be.idx_dtype),
            be.array([3], dtype=be.idx_dtype),
            be.array([weight], dtype=be.float_dtype),
        )
    got = be.to_numpy(buf.read_and_clear(3))
    assert got[1] == pytest.approx(0.3), "duplicate deliveries to the same target did not sum"


def test_delay_buffer_survives_ring_wraparound(be):
    n_post, max_delay, steps = 4, 4, 25
    buf = DelayBuffer(n_post, max_delay, be)
    expected: dict[int, np.ndarray] = {}

    for t in range(steps):
        for delay in range(1, max_delay + 1):
            target = (t + delay) % n_post
            weight = np.float32(0.1 * delay)
            buf.schedule(
                t,
                be.array([target], dtype=be.idx_dtype),
                be.array([delay], dtype=be.idx_dtype),
                be.array([weight], dtype=be.float_dtype),
            )
            expected.setdefault(t + delay, np.zeros(n_post, dtype=np.float32))[target] += weight

        got = be.to_numpy(buf.read_and_clear(t)).copy()
        want = expected.pop(t, np.zeros(n_post, dtype=np.float32))
        np.testing.assert_allclose(
            got,
            want,
            atol=1e-6,
            err_msg=f"ring buffer corrupted deliveries at t={t}",
        )

    for t in range(steps, steps + max_delay):
        got = be.to_numpy(buf.read_and_clear(t)).copy()
        want = expected.pop(t, np.zeros(n_post, dtype=np.float32))
        np.testing.assert_allclose(
            got, want, atol=1e-6, err_msg=f"ring buffer lost in-flight deliveries at t={t}"
        )

    assert not expected, f"undelivered current remained for timesteps {sorted(expected)}"


def test_synapse_config_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="k_out cannot exceed n_post"):
        SynapseConfig(n_pre=4, n_post=4, k_out=5)
    with pytest.raises(ValueError, match="excitatory_frac"):
        SynapseConfig(n_pre=4, n_post=4, k_out=1, excitatory_frac=1.5)
    with pytest.raises(ValueError, match="delays must be >= 1"):
        SynapseConfig(n_pre=4, n_post=4, k_out=1, delay_min=0)
    with pytest.raises(ValueError, match="delay_max"):
        SynapseConfig(n_pre=4, n_post=4, k_out=1, delay_min=3, delay_max=2)
    with pytest.raises(ValueError, match="structure"):
        SynapseConfig(n_pre=4, n_post=4, k_out=1, structure="small-world")
