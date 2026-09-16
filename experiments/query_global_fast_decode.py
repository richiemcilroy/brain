"""Optional exact one-token MLX path for the bounded query state.

In decode, the current query reads the already summarized older state. Only
after output is computed does the oldest of its 64 local keys enter the
state for the next query. The general multi-token path builds prefix sums
and gathers to support every query inside a block; one-token decode needs
neither. This specialization stays differentiable but is used only in eval
mode and with no supplied external mask. It makes no novelty claim.
"""
from __future__ import annotations

import mlx.core as mx

from query_global_attention import LocalQueryGlobalAttention, QueryGlobalCache
from query_global_trainable import TrainableQueryGlobalAttention


class FastDecodeMixin:
    def _block(self, x: mx.array, cache: QueryGlobalCache,
               external_mask: mx.array | None, outer_first: int) -> mx.array:
        if (self.training or x.shape[1] != 1 or external_mask is not None):
            return super()._block(x, cache, external_mask, outer_first)
        batch = x.shape[0]
        original = self.original
        previous = 0 if cache.keys is None else cache.keys.shape[2]
        q = original.q_proj(x).reshape(
            batch, 1, original.n_heads, original.head_dim
        ).transpose(0, 2, 1, 3)
        k = original.k_proj(x).reshape(
            batch, 1, original.n_kv_heads, original.head_dim
        ).transpose(0, 2, 1, 3)
        v = original.v_proj(x).reshape(
            batch, 1, original.n_kv_heads, original.head_dim
        ).transpose(0, 2, 1, 3)
        q = original.rope(q, offset=cache.offset)
        k = original.rope(k, offset=cache.offset)
        if previous:
            keys = mx.concatenate((cache.keys, k), axis=2)
            values = mx.concatenate((cache.values, v), axis=2)
        else:
            keys, values = k, v
        if keys.shape[2] > self.window:
            raise AssertionError("one-token decode has more than local-window keys")
        local = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=original.scale, mask=None)

        if self.gain and cache.global_count:
            group = original.n_heads // original.n_kv_heads
            qf = self.feature(q).reshape(
                batch, original.n_kv_heads, group, 1, self.feature_dim)
            qmat = qf.transpose(0, 1, 3, 2, 4)[..., None, :]
            numerator = mx.matmul(
                qmat, cache.global_kv[:, :, None, None, :, :]
            ).squeeze(-2)
            denominator = mx.sum(
                qmat.squeeze(-2) * cache.global_k[:, :, None, None, :],
                axis=-1)
            global_heads = (
                numerator / mx.maximum(denominator[..., None], 1e-8)
            ).transpose(0, 1, 3, 2, 4).reshape(
                batch, original.n_heads, 1, original.head_dim)
            local_f32 = local.astype(mx.float32)
            heads = (
                local_f32 + self.gain * (global_heads - local_f32)
            ).astype(x.dtype)
        else:
            heads = local

        retained = min(self.window - 1, keys.shape[2])
        evicted = keys.shape[2] - retained
        if self.gain and evicted:
            if evicted != 1:
                raise AssertionError("one-token decode evicted more than one key")
            base_kv = cache.global_kv
            base_k = cache.global_k
            if base_kv is None:
                base_kv = mx.zeros(
                    (batch, original.n_kv_heads,
                     self.feature_dim, original.head_dim), mx.float32)
                base_k = mx.zeros(
                    (batch, original.n_kv_heads, self.feature_dim), mx.float32)
            feature = self.feature(keys[..., :1, :]).squeeze(2)
            old_value = values[..., :1, :].astype(mx.float32).squeeze(2)
            cache.global_kv = (
                base_kv + feature[..., None] * old_value[..., None, :])
            cache.global_k = base_k + feature
            cache.global_count += 1
        if retained:
            cache.keys = mx.contiguous(keys[..., -retained:, :])
            cache.values = mx.contiguous(values[..., -retained:, :])
        else:
            cache.keys = cache.values = None
        cache.offset += 1
        flattened = heads.transpose(0, 2, 1, 3).reshape(batch, 1, -1)
        return original.o_proj(flattened)


class FastDecodeQueryGlobalAttention(FastDecodeMixin,
                                     LocalQueryGlobalAttention):
    """Fixed feature map with specialized eval-mode one-token decode."""


class FastDecodeTrainableQueryGlobalAttention(
    FastDecodeMixin, TrainableQueryGlobalAttention
):
    """Selected per-head feature map with specialized one-token decode."""
