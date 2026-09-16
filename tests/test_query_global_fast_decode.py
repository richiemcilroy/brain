"""Behavioral parity of the opt-in one-token state update."""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from query_global_attention import (  # noqa: E402
    LocalQueryGlobalAttention, QueryGlobalCache, make_query_global_cache,
)
from query_global_fast_decode import (  # noqa: E402
    FastDecodeQueryGlobalAttention, FastDecodeTrainableQueryGlobalAttention,
)
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402


def tiny_model():
    args = llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=97,
        max_position_embeddings=256)
    return llama.Model(args)


def run_segments(module, x, *, window, gain):
    cache = QueryGlobalCache(window=window,
                             feature_dim=module.feature_dim, gain=gain)
    parts = [np.asarray(module(x[:, :16], cache=cache).astype(mx.float32))]
    for index in range(16, x.shape[1]):
        parts.append(np.asarray(
            module(x[:, index:index + 1], cache=cache).astype(mx.float32)))
    return np.concatenate(parts, axis=1), cache


@pytest.mark.parametrize("gain", [0.0, 0.45])
def test_fixed_fast_decode_matches_general_path_and_bounded_cache(gain):
    mx.random.seed(31)
    model = tiny_model()
    original = model.layers[0].self_attn
    standard = LocalQueryGlobalAttention(
        original, window=8, gain=gain, chunk=4)
    faster = FastDecodeQueryGlobalAttention(
        original, window=8, gain=gain, chunk=4)
    standard.eval()
    faster.eval()
    x = mx.array(np.random.default_rng(31).normal(
        size=(2, 30, 32)).astype(np.float32))
    slow_output, slow_cache = run_segments(
        standard, x, window=8, gain=gain)
    fast_output, fast_cache = run_segments(
        faster, x, window=8, gain=gain)
    np.testing.assert_array_equal(fast_output, slow_output)
    assert fast_cache.offset == slow_cache.offset == 30
    assert fast_cache.global_count == slow_cache.global_count == (
        23 if gain else 0)
    assert fast_cache.nbytes == slow_cache.nbytes
    for name in ("keys", "values", "global_kv", "global_k"):
        left = getattr(slow_cache, name)
        right = getattr(fast_cache, name)
        if left is None:
            assert right is None
        else:
            np.testing.assert_array_equal(np.asarray(right), np.asarray(left))
    model.layers[0].self_attn = faster
    assert isinstance(make_query_global_cache(model)[0], QueryGlobalCache)


def test_trained_bf16_map_and_external_mask_fallback_match_general_path():
    mx.random.seed(32)
    original = tiny_model().layers[0].self_attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        projection = getattr(original, name)
        projection.weight = projection.weight.astype(mx.bfloat16)
    standard = TrainableQueryGlobalAttention(
        original, window=8, gain=0.45, chunk=4)
    faster = FastDecodeTrainableQueryGlobalAttention(
        original, window=8, gain=0.45, chunk=4)
    rng = np.random.default_rng(32)
    dq = mx.array(rng.normal(0, 0.01, (4, 8, 8)).astype(np.float32))
    dk = mx.array(rng.normal(0, 0.01, (2, 8, 8)).astype(np.float32))
    for module in (standard, faster):
        module.delta_q = dq
        module.delta_k = dk
        module.eval()
    x = mx.array(rng.normal(size=(1, 30, 32)).astype(
        np.float32)).astype(mx.bfloat16)
    slow_output, slow_cache = run_segments(
        standard, x, window=8, gain=0.45)
    fast_output, fast_cache = run_segments(
        faster, x, window=8, gain=0.45)
    np.testing.assert_array_equal(fast_output, slow_output)
    np.testing.assert_array_equal(
        np.asarray(fast_cache.global_kv), np.asarray(slow_cache.global_kv))
    assert fast_cache.global_count == 23

    # A caller-supplied mask takes the general path; it still preserves the
    # exact same output and state in the next one-token decode.
    mask = mx.ones((1, 1, 1, 31), dtype=mx.bool_)
    next_x = x[:, :1]
    slow_next = standard(next_x, mask=mask, cache=slow_cache)
    fast_next = faster(next_x, mask=mask, cache=fast_cache)
    np.testing.assert_array_equal(
        np.asarray(fast_next.astype(mx.float32)),
        np.asarray(slow_next.astype(mx.float32)))
    assert fast_cache.global_count == slow_cache.global_count == 24
