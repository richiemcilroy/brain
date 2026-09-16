"""Dense causal reference and cache checks for the bounded query-global layer."""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models import llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from query_global_attention import (  # noqa: E402
    LocalQueryGlobalAttention, QueryGlobalCache,
    install_query_global, make_query_global_cache,
)
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402


def tiny_model(n_layers=1):
    args = llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=n_layers,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=97, max_position_embeddings=256,
    )
    return llama.Model(args)


def _softmax(x):
    z = np.exp(x - np.max(x))
    return z / np.sum(z)


def _feature(x):
    return np.concatenate((_softmax(2 * x), _softmax(-2 * x)))


def dense_reference(original, x, *, window, gain):
    """Explicit per-query local softmax and all-older-key feature softmax."""
    batch, length, _ = x.shape
    hq, hkv, dim = (original.n_heads, original.n_kv_heads,
                     original.head_dim)
    q = original.q_proj(x).reshape(
        batch, length, hq, dim).transpose(0, 2, 1, 3)
    k = original.k_proj(x).reshape(
        batch, length, hkv, dim).transpose(0, 2, 1, 3)
    v = original.v_proj(x).reshape(
        batch, length, hkv, dim).transpose(0, 2, 1, 3)
    q = np.asarray(original.rope(q).astype(mx.float32))
    k = np.asarray(original.rope(k).astype(mx.float32))
    v = np.asarray(v.astype(mx.float32))
    mixed = np.zeros((batch, length, hq, dim), dtype=np.float32)
    for b in range(batch):
        for t in range(length):
            old_count = max(0, t - window + 1)
            local_positions = np.arange(max(0, t - window + 1), t + 1)
            for h in range(hq):
                kv_head = h // (hq // hkv)
                local_scores = (
                    k[b, kv_head, local_positions] @ q[b, h, t]
                ) * original.scale
                local = _softmax(local_scores) @ v[b, kv_head, local_positions]
                if old_count and gain:
                    q_feature = _feature(q[b, h, t])
                    key_features = np.stack([
                        _feature(k[b, kv_head, i])
                        for i in range(old_count)
                    ])
                    weights = key_features @ q_feature
                    global_value = (
                        weights / weights.sum()
                    ) @ v[b, kv_head, :old_count]
                    mixed[b, t, h] = (
                        (1 - gain) * local + gain * global_value)
                else:
                    mixed[b, t, h] = local
    flat = mx.array(mixed.reshape(batch, length, -1))
    return np.asarray(original.o_proj(flat).astype(mx.float32))


@pytest.mark.parametrize("gain", [0.0, 0.45])
def test_bounded_state_matches_independent_dense_causal_reference(gain):
    mx.random.seed(11)
    original = tiny_model().layers[0].self_attn
    module = LocalQueryGlobalAttention(
        original, window=8, gain=gain, chunk=4)
    x = mx.array(np.random.default_rng(11).normal(
        size=(2, 30, 32)).astype(np.float32))
    actual = np.asarray(module(x).astype(mx.float32))
    expected = dense_reference(original, x, window=8, gain=gain)
    assert np.max(np.abs(actual - expected)) < 3e-4

    cache = QueryGlobalCache(
        window=8, feature_dim=module.feature_dim, gain=gain)
    parts = []
    start = 0
    for length in (1, 7, 8, 1, 13):
        parts.append(np.asarray(
            module(x[:, start:start + length], cache=cache).astype(mx.float32)))
        start += length
        assert cache.offset == start
        assert cache.keys.shape[2] == min(start, 7)
        assert cache.global_count == (max(0, start - 7) if gain else 0)
    split = np.concatenate(parts, axis=1)
    assert np.max(np.abs(actual - split)) < 3e-4
    if gain:
        assert cache.global_kv.shape == (2, 2, 16, 8)
        assert cache.global_k.shape == (2, 2, 16)
        assert cache.global_kv.dtype == cache.global_k.dtype == mx.float32
    else:
        assert cache.global_kv is cache.global_k is None


def test_cache_bytes_stop_growing_and_bf16_stream_is_preserved():
    mx.random.seed(12)
    original = tiny_model().layers[0].self_attn
    for projection in (original.q_proj, original.k_proj,
                       original.v_proj, original.o_proj):
        projection.weight = projection.weight.astype(mx.bfloat16)
    module = LocalQueryGlobalAttention(
        original, window=8, gain=0.45, chunk=4)
    x = mx.array(np.random.default_rng(12).normal(
        size=(1, 30, 32)).astype(np.float32)).astype(mx.bfloat16)
    cache = QueryGlobalCache(window=8, feature_dim=16, gain=0.45)
    first = module(x[:, :16], cache=cache)
    bytes_at_16 = cache.nbytes
    rest = module(x[:, 16:], cache=cache)
    assert first.dtype == rest.dtype == x.dtype == mx.bfloat16
    assert cache.nbytes == bytes_at_16
    assert cache.keys.dtype == cache.values.dtype == mx.bfloat16
    assert cache.global_kv.dtype == cache.global_k.dtype == mx.float32
    full = np.asarray(module(x).astype(mx.float32))
    split = np.concatenate((
        np.asarray(first.astype(mx.float32)),
        np.asarray(rest.astype(mx.float32))), axis=1)
    assert np.max(np.abs(full - split)) < 0.1


def test_eval_mode_syncs_long_scan_without_changing_causal_output():
    mx.random.seed(14)
    original = tiny_model().layers[0].self_attn
    module = LocalQueryGlobalAttention(
        original, window=8, gain=0.45, chunk=4,
        inference_sync_blocks=2)
    x = mx.array(np.random.default_rng(14).normal(
        size=(1, 30, 32)).astype(np.float32))
    without_sync = np.asarray(module(x).astype(mx.float32))
    module.eval()
    with_sync = np.asarray(module(x).astype(mx.float32))
    assert np.max(np.abs(without_sync - with_sync)) < 3e-4
    cache = QueryGlobalCache(window=8, feature_dim=16, gain=0.45)
    split = np.concatenate((
        np.asarray(module(x[:, :16], cache=cache).astype(mx.float32)),
        np.asarray(module(x[:, 16:], cache=cache).astype(mx.float32))),
        axis=1)
    assert np.max(np.abs(with_sync - split)) < 3e-4
    assert cache.global_count == 23


def test_zero_residual_transfer_starts_exactly_and_freezes_pretrained_weights():
    mx.random.seed(15)
    original = tiny_model().layers[0].self_attn
    fixed = LocalQueryGlobalAttention(
        original, window=8, gain=0.45, chunk=4)
    trainable = TrainableQueryGlobalAttention(
        original, window=8, gain=0.45, chunk=4)
    x = mx.array(np.random.default_rng(15).normal(
        size=(2, 30, 32)).astype(np.float32))
    target = original(x, mask="causal")
    np.testing.assert_array_equal(
        np.asarray(fixed(x)), np.asarray(trainable(x)))
    trainable.freeze()
    trainable.unfreeze(
        keys=["delta_q", "delta_k"], recurse=False, strict=True)
    trainable_keys = sorted(
        key for key, _ in nn.utils.tree_flatten(
            trainable.trainable_parameters()))
    assert trainable_keys == ["delta_k", "delta_q"]

    def loss_fn(module, activations, teacher):
        difference = (
            module(activations)[:, 8:].astype(mx.float32) -
            teacher[:, 8:].astype(mx.float32))
        return mx.mean(difference * difference)

    value_and_grad = nn.value_and_grad(trainable, loss_fn)
    loss, gradients = value_and_grad(trainable, x, target)
    mx.eval(loss, gradients)
    grad_norms = {
        key: float(mx.sqrt(mx.sum(value * value)))
        for key, value in nn.utils.tree_flatten(gradients)
    }
    assert float(loss) > 0
    assert set(grad_norms) == {"delta_q", "delta_k"}
    assert min(grad_norms.values()) > 0


def test_model_cache_factory_restore_and_wrong_cache_fail_loudly():
    mx.random.seed(13)
    model = tiny_model(2)
    original = model.layers[1].self_attn
    installation = install_query_global(
        model, layer=1, window=8, gain=0.45, chunk=4)
    assert installation.conversion["new_trainable_parameters"] == 0
    assert model.layers[1].self_attn is installation.replacement
    cache = make_query_global_cache(model)
    assert isinstance(cache[1], QueryGlobalCache)
    ids = mx.array(np.random.default_rng(13).integers(
        0, 97, size=(1, 30), dtype=np.int32))
    full = np.asarray(model(ids).astype(mx.float32))
    split_cache = make_query_global_cache(model)
    pieces = [
        np.asarray(model(ids[:, i:i + 1], cache=split_cache).astype(mx.float32))
        for i in range(30)
    ]
    split = np.concatenate(pieces, axis=1)
    assert split_cache[1].offset == 30
    assert split_cache[1].global_count == 23
    assert np.max(np.abs(full - split)) < 3e-4
    with pytest.raises(TypeError, match="QueryGlobalCache"):
        installation.replacement(mx.zeros((1, 1, 32)), cache=object())
    with pytest.raises(ValueError, match="gain differs"):
        wrong_gain = QueryGlobalCache(
            window=8, feature_dim=16, gain=0.0)
        installation.replacement(mx.zeros((1, 1, 32)), cache=wrong_gain)
    with pytest.raises(ValueError, match="KV shape"):
        installation.replacement(mx.zeros((2, 1, 32)), cache=split_cache[1])
    installation.restore()
    assert model.layers[1].self_attn is original
