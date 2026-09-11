"""Three-factor plasticity: direction, bounds, and homeostasis.

Direction is the load-bearing property: a coincident pre+post pairing must
potentiate and a post-before-pre pairing must depress. Both rule paths
(``eligibility="off"`` and ``"dense"``) are exercised.

Sign convention for inhibitory synapses: a negative weight is a stronger
inhibitory connection, so Hebbian potentiation of inhibition makes the weight
*more negative*. The tests assert magnitude semantics for inhibition and the
plain weight-increase semantics required for excitatory synapses.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.backend import Backend
from brain.connectivity import SynapseConfig, Synapses
from brain.plasticity import PlasticityConfig, ThreeFactorPlasticity

N_PRE = 12
N_POST = 12
K = 5


def _syn(be, n_pre=N_PRE, n_post=N_POST, k=K, seed=0, excitatory_frac=0.8):
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


def _plas(be, n_pre=N_PRE, n_post=N_POST, k=K, *, eligibility="off",
          homeostasis=False, **kwargs):
    return ThreeFactorPlasticity(
        PlasticityConfig(eligibility=eligibility, homeostasis=homeostasis, **kwargs),
        n_pre,
        n_post,
        k,
        be,
    )


@pytest.mark.parametrize("eligibility", ["off", "dense"])
def test_coincident_pre_and_post_potentiates(be, eligibility):
    syn = _syn(be)
    plas = _plas(be, eligibility=eligibility)
    pre = 3
    idx = be.array([pre], dtype=be.idx_dtype)
    targets = be.take(syn.targets, idx)
    target_row = be.to_numpy(targets)[0]

    post_neuron = int(target_row[0])
    post = np.zeros(N_POST, dtype=bool)
    post[post_neuron] = True

    before = be.to_numpy(syn.weights)[pre].copy()
    plas.apply(idx, targets, be.array(post), syn, 1.0)
    delta = be.to_numpy(syn.weights)[pre] - before

    coincident = np.flatnonzero(target_row == post_neuron)
    assert len(coincident) >= 1, "test setup: chosen post neuron is not a target of the row"
    assert np.all(delta[coincident] > 0.0), (
        f"coincident pre+post did not potentiate (deltas={delta[coincident]}); "
        "LTP is required for a Hebbian rule"
    )
    np.testing.assert_allclose(
        delta[coincident],
        plas.cfg.a_plus,
        rtol=1e-4,
        err_msg="coincident potentiation magnitude is not a_plus",
    )
    non_coincident = np.setdiff1d(np.arange(K), coincident)
    np.testing.assert_allclose(
        delta[non_coincident],
        0.0,
        atol=1e-7,
        err_msg="non-coincident synapses changed during a single coincident pairing",
    )


@pytest.mark.parametrize("eligibility", ["off", "dense"])
def test_post_before_pre_depresses(be, eligibility):
    syn = _syn(be)
    plas = _plas(be, eligibility=eligibility)
    pre = 5
    idx = be.array([pre], dtype=be.idx_dtype)
    targets = be.take(syn.targets, idx)
    target_row = be.to_numpy(targets)[0]
    recent_post = int(target_row[1])

    post = np.zeros(N_POST, dtype=bool)
    post[recent_post] = True
    plas.record_post_spikes(be.array(post))

    before = be.to_numpy(syn.weights)[pre].copy()
    plas.apply(idx, targets, be.array(np.zeros(N_POST, dtype=bool)), syn, 1.0)
    delta = be.to_numpy(syn.weights)[pre] - before

    anti_causal = np.flatnonzero(target_row == recent_post)
    assert np.all(delta[anti_causal] < 0.0), (
        f"post-before-pre did not depress (deltas={delta[anti_causal]}); LTD is "
        "required for the anti-causal half of the STDP window"
    )
    np.testing.assert_allclose(
        delta[anti_causal],
        -plas.cfg.a_minus,
        rtol=1e-4,
        err_msg="depression magnitude is not -a_minus",
    )
    untouched = np.setdiff1d(np.arange(K), anti_causal)
    assert np.all(delta[untouched] <= 1e-7), (
        "a synapse potentiated during a depression-only (anti-causal) pairing"
    )


def test_ltp_and_ltd_have_opposite_signs(be):
    coincident = _syn(be, seed=1)
    anti_causal = _syn(be, seed=1)
    p1 = _plas(be)
    p2 = _plas(be)

    pre = 4
    idx = be.array([pre], dtype=be.idx_dtype)
    targets = be.take(coincident.targets, idx)
    target_row = be.to_numpy(targets)[0]

    post = np.zeros(N_POST, dtype=bool)
    post[int(target_row[0])] = True
    w0 = be.to_numpy(coincident.weights)[pre].copy()
    p1.apply(idx, targets, be.array(post), coincident, 1.0)
    ltp_delta = float((be.to_numpy(coincident.weights)[pre] - w0).max())

    p2.record_post_spikes(be.array(post))
    w1 = be.to_numpy(anti_causal.weights)[pre].copy()
    p2.apply(idx, targets, be.array(np.zeros(N_POST, dtype=bool)), anti_causal, 1.0)
    ltd_delta = float((be.to_numpy(anti_causal.weights)[pre] - w1).min())

    assert ltp_delta > 0.0 > ltd_delta, (
        f"STDP window is not bidirectional: LTP delta={ltp_delta}, LTD delta={ltd_delta}"
    )


def test_inhibitory_synapse_potentiates_by_growing_magnitude(backend_name):
    be = Backend(backend_name, seed=0)
    n_pre = n_post = 10
    k = 4
    syn = Synapses(
        SynapseConfig(n_pre=n_pre, n_post=n_post, k_out=k, excitatory_frac=0.5, seed=0),
        be,
    )
    plas = _plas(be, n_pre=n_pre, n_post=n_post, k=k)
    pre = n_pre - 1
    assert not bool(be.to_numpy(syn.is_exc)[pre]), "test setup: chosen neuron is not inhibitory"
    idx = be.array([pre], dtype=be.idx_dtype)
    targets = be.take(syn.targets, idx)

    post = np.ones(n_post, dtype=bool)
    before = be.to_numpy(syn.weights)[pre].copy()
    plas.apply(idx, targets, be.array(post), syn, 1.0)
    delta = be.to_numpy(syn.weights)[pre] - before

    assert np.all(before < 0.0), "test setup: inhibitory weights should be negative"
    assert np.all(delta < 0.0), (
        f"Hebbian potentiation of inhibition did not grow the weight magnitude: {delta}"
    )


@pytest.mark.parametrize("eligibility", ["off", "dense"])
def test_zero_neuromodulator_produces_no_weight_change(be, eligibility):
    syn = _syn(be)
    plas = _plas(be, eligibility=eligibility)
    idx = be.array([2, 6], dtype=be.idx_dtype)
    post = np.zeros(N_POST, dtype=bool)
    post[[1, 4]] = True

    before = be.to_numpy(syn.weights).copy()
    plas.apply(idx, be.take(syn.targets, idx), be.array(post), syn, 0.0)
    after = be.to_numpy(syn.weights)

    np.testing.assert_array_equal(
        after, before, err_msg="zero neuromodulator changed synaptic weights"
    )
    assert float(be.to_numpy(plas.x_pre)[2]) > 0.0, (
        "eligibility/pre traces should still advance when neuromodulator is zero"
    )


def test_disabled_plasticity_is_inert(be):
    syn = _syn(be)
    plas = _plas(be, enabled=False)
    idx = be.array([1, 2], dtype=be.idx_dtype)
    post = np.zeros(N_POST, dtype=bool)
    post[[0, 3]] = True

    before = be.to_numpy(syn.weights).copy()
    plas.apply(idx, be.take(syn.targets, idx), be.array(post), syn, 1.0)
    after = be.to_numpy(syn.weights)

    np.testing.assert_array_equal(
        after, before, err_msg="apply() changed weights while plasticity was disabled"
    )
    np.testing.assert_array_equal(
        be.to_numpy(plas.x_pre),
        np.zeros(N_PRE, dtype=np.float32),
        err_msg="disabled plasticity still recorded pre-synaptic traces",
    )


def test_weights_stay_within_dale_bounds_after_many_updates(backend_name):
    be = Backend(backend_name, seed=0)
    n_pre = n_post = 10
    k = 4
    syn = Synapses(
        SynapseConfig(n_pre=n_pre, n_post=n_post, k_out=k, excitatory_frac=0.5, seed=0),
        be,
    )
    plas = _plas(be, n_pre=n_pre, n_post=n_post, k=k, homeostasis=False, a_minus=0.0)
    exc_row, inh_row = 0, n_pre - 1
    all_post = np.ones(n_post, dtype=bool)

    for _ in range(400):
        idx = be.array([exc_row], dtype=be.idx_dtype)
        plas.apply(idx, be.take(syn.targets, idx), be.array(all_post), syn, 50.0)

    for _ in range(400):
        idx = be.array([inh_row], dtype=be.idx_dtype)
        plas.apply(idx, be.take(syn.targets, idx), be.array(all_post), syn, 50.0)

    weights = be.to_numpy(syn.weights)
    is_exc = be.to_numpy(syn.is_exc).astype(bool)
    w_max_exc, w_max_inh = syn.cfg.w_max_exc, syn.cfg.w_max_inh

    assert np.all(weights[is_exc] >= 0.0), (
        "an excitatory synapse went negative despite Dale bounds: "
        f"min={weights[is_exc].min()}"
    )
    assert np.all(weights[is_exc] <= w_max_exc + 1e-6), (
        f"an excitatory synapse exceeded w_max_exc={w_max_exc}: max={weights[is_exc].max()}"
    )
    assert np.all(weights[~is_exc] <= 0.0), (
        "an inhibitory synapse became positive despite Dale bounds: "
        f"max={weights[~is_exc].max()}"
    )
    assert np.all(weights[~is_exc] >= -w_max_inh - 1e-6), (
        f"an inhibitory synapse passed -w_max_inh={-w_max_inh}: min={weights[~is_exc].min()}"
    )
    assert syn.sign_violations() == 0, "Dale violations present after saturating updates"
    assert weights[exc_row].max() == pytest.approx(w_max_exc, abs=1e-5), (
        "test is vacuous: the excitatory row never reached its upper bound"
    )
    assert weights[inh_row].min() == pytest.approx(-w_max_inh, abs=1e-5), (
        "test is vacuous: the inhibitory row never reached its lower bound"
    )


@pytest.mark.parametrize("start_factor", [4.0, 0.1])
def test_homeostasis_pulls_mean_abs_weight_toward_target(be, start_factor):
    syn = _syn(be, seed=1)
    target = 0.08
    plas = _plas(
        be, homeostasis=True, homeo_every=1, homeo_rate=0.05, target_mean_w=target
    )
    syn.weights = syn.weights * start_factor
    syn.clamp_weights()

    start = syn.mean_abs_weight()
    start_distance = abs(start - target)
    assert start_distance > 0.01, (
        f"test setup is not discriminating: mean|w| starts at {start:.4f}, target {target}"
    )

    for _ in range(600):
        assert plas.maybe_homeostasis(syn) is True, "homeostasis did not run when due"

    end = syn.mean_abs_weight()
    end_distance = abs(end - target)
    assert end_distance < start_distance, (
        f"homeostasis did not move mean|w| toward the target: {start:.4f} -> {end:.4f}"
    )
    assert end_distance <= 0.1 * target, (
        f"mean|w|={end:.4f} is still more than 10% away from target {target}"
    )
    assert syn.sign_violations() == 0, "homeostatic scaling flipped a synapse's sign"


def test_homeostasis_respects_its_interval(be):
    syn = _syn(be)
    plas = _plas(be, homeo_every=50, homeostasis=True)
    plas._steps = 1
    assert plas.maybe_homeostasis(syn) is False, (
        "homeostasis ran before its interval had elapsed"
    )
    plas._steps = 50
    assert plas.maybe_homeostasis(syn) is True, "homeostasis did not run at its interval"


def test_homeostasis_is_skipped_when_disabled(be):
    syn = _syn(be)
    plas = _plas(be, homeo_every=1, homeostasis=False)
    before = be.to_numpy(syn.weights).copy()
    assert plas.maybe_homeostasis(syn) is False, "disabled homeostasis still ran"
    np.testing.assert_array_equal(
        be.to_numpy(syn.weights), before, err_msg="disabled homeostasis changed weights"
    )
