"""The local window, gated trace, and both caches must agree across call shapes."""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import llama
from mlx_lm.models.base import create_causal_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from llm_hybrid import GatedMemoryCarrier  # noqa: E402
from local_window_hybrid import (  # noqa: E402
    HybridAttentionCache, LocalWindowHybridAttention,
    install_local_window_hybrid, make_hybrid_cache,
)


def tiny_model(n_layers=2):
    args = llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=n_layers,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=97, max_position_embeddings=256,
    )
    return llama.Model(args)


def attention_reference(original, x, window=64):
    """One dense full-prefix attention call with an explicit sliding mask."""
    B, T, _ = x.shape
    q = original.q_proj(x).reshape(B, T, original.n_heads, original.head_dim)
    k = original.k_proj(x).reshape(B, T, original.n_kv_heads, original.head_dim)
    v = original.v_proj(x).reshape(B, T, original.n_kv_heads, original.head_dim)
    q = original.rope(q.transpose(0, 2, 1, 3))
    k = original.rope(k.transpose(0, 2, 1, 3))
    v = v.transpose(0, 2, 1, 3)
    mask = create_causal_mask(T, window_size=window)
    y = mx.fast.scaled_dot_product_attention(q, k, v,
                                              scale=original.scale, mask=mask)
    return original.o_proj(y.transpose(0, 2, 1, 3).reshape(B, T, -1))


def as_f32(a):
    return np.asarray(a.astype(mx.float32))


def make_hybrid(*, banks=1, gain=0.05, chunk=64):
    model = tiny_model(1)
    original = model.layers[0].self_attn
    decays = (0.7,) if banks == 1 else (0.7, 0.8, 0.9, 0.99)
    carrier = GatedMemoryCarrier(32, banks=banks, decays=decays,
                                out_gain=0.2)
    return original, LocalWindowHybridAttention(
        original, carrier, window=64, memory_gain=gain, chunk=chunk)


@pytest.mark.parametrize("chunk", [16, 64])
def test_local_attention_is_exact_64_token_window(chunk):
    mx.random.seed(11)
    original, hybrid = make_hybrid(gain=0.0, chunk=chunk)
    rng = np.random.default_rng(11)
    x = mx.array(rng.normal(size=(2, 130, 32)).astype(np.float32))
    expected = as_f32(attention_reference(original, x))
    actual = as_f32(hybrid(x))
    assert np.max(np.abs(expected - actual)) < 3e-4

    # Changing tokens before t-63 cannot affect the local-only path at t.
    edited = np.asarray(x).copy()
    edited[:, :2, :] += 8.0
    changed = as_f32(hybrid(mx.array(edited)))
    assert np.max(np.abs(actual[:, 65:, :] - changed[:, 65:, :])) < 3e-4
    assert np.max(np.abs(actual[:, 1:20, :] - changed[:, 1:20, :])) > 1e-3


@pytest.mark.parametrize("banks", [1, 4])
@pytest.mark.parametrize("gain", [-0.05, 0.05])
def test_cached_chunks_and_single_tokens_match_full_hybrid_prefix(banks, gain):
    mx.random.seed(23)
    original, hybrid = make_hybrid(banks=banks, gain=gain, chunk=32)
    rng = np.random.default_rng(23)
    x = mx.array(rng.normal(size=(2, 130, 32)).astype(np.float32))
    original_q = as_f32(original.q_proj.weight).copy()
    full = as_f32(hybrid(x))
    cache = HybridAttentionCache(64)
    parts = []
    start = 0
    for length in (1, 63, 1, 32, 32, 1):
        part = hybrid(x[:, start:start + length], cache=cache)
        parts.append(as_f32(part))
        start += length
        assert cache.offset == start == cache.memory.offset
        assert cache.keys.shape[2] == cache.values.shape[2] == min(start, 63)
    split = np.concatenate(parts, axis=1)
    assert split.shape == full.shape
    assert cache.memory.state.shape == (2, banks, 32)
    assert cache.nbytes == (cache.keys.nbytes + cache.values.nbytes
                            + cache.memory.state.nbytes)
    assert np.max(np.abs(full - split)) < 4e-4
    np.testing.assert_array_equal(as_f32(original.q_proj.weight), original_q)


def test_pretrained_residual_dtype_is_preserved():
    mx.random.seed(31)
    original, hybrid = make_hybrid(gain=0.05)
    for projection in (original.q_proj, original.k_proj,
                       original.v_proj, original.o_proj):
        projection.weight = projection.weight.astype(mx.bfloat16)
    x = mx.array(np.random.default_rng(31).normal(size=(1, 70, 32))
                 .astype(np.float32)).astype(mx.bfloat16)
    cache = HybridAttentionCache()
    full = hybrid(x)
    a = hybrid(x[:, :69], cache=cache)
    b = hybrid(x[:, 69:], cache=cache)
    assert full.dtype == a.dtype == b.dtype == mx.bfloat16
    assert cache.keys.dtype == cache.values.dtype == mx.bfloat16
    assert cache.memory.state.dtype == mx.float32
    split = np.concatenate((as_f32(a), as_f32(b)), axis=1)
    assert np.max(np.abs(as_f32(full) - split)) < 0.08


def test_model_cache_factory_and_reversible_installer():
    mx.random.seed(41)
    model = tiny_model(2)
    original = model.layers[1].self_attn
    original_dtype = original.q_proj.weight.dtype
    original_values = as_f32(original.v_proj.weight).copy()
    install = install_local_window_hybrid(model, layer=1, decay=0.7,
                                          memory_gain=0.05, chunk=32)
    assert install.conversion["transplanted"]
    assert model.layers[1].self_attn is install.hybrid
    assert install.hybrid.original is original
    cache = make_hybrid_cache(model)
    assert isinstance(cache[1], HybridAttentionCache)

    ids = mx.array(np.random.default_rng(41).integers(0, 97,
                                                      size=(1, 70),
                                                      dtype=np.int32))
    full = as_f32(model(ids))
    one_cache = make_hybrid_cache(model)
    whole_cached = as_f32(model(ids, cache=one_cache))
    split_cache = make_hybrid_cache(model)
    parts = []
    for position in range(70):
        parts.append(as_f32(model(ids[:, position:position + 1],
                                  cache=split_cache)))
    tokenwise = np.concatenate(parts, axis=1)
    assert one_cache[1].offset == split_cache[1].offset == 70
    assert split_cache[1].memory.offset == 70
    assert split_cache[1].keys.shape[2] == 63
    assert np.max(np.abs(full - whole_cached)) < 5e-4
    assert np.max(np.abs(full - tokenwise)) < 5e-4

    install.restore()
    assert model.layers[1].self_attn is original
    assert original.q_proj.weight.dtype == original_dtype
    np.testing.assert_array_equal(as_f32(original.v_proj.weight), original_values)


def test_wrong_cache_type_and_batch_fail_loudly():
    _, hybrid = make_hybrid()
    with pytest.raises(TypeError, match="HybridAttentionCache"):
        hybrid(mx.zeros((1, 1, 32)), cache=object())
    cache = HybridAttentionCache()
    hybrid(mx.zeros((1, 1, 32)), cache=cache)
    with pytest.raises(ValueError, match="batch size"):
        hybrid(mx.zeros((2, 1, 32)), cache=cache)
