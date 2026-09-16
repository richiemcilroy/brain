"""Untrained MLX local + query-addressable global attention baseline.

This is a prior-art-informed functional baseline, not a novel transformer.
It keeps Llama's q/k/v/o projections and RoPE, attends exactly to the latest
64 tokens, and summarizes older keys/values in a constant-size positive
feature state. The identity Hedgehog-style map and fixed local/global gain
come from a separate teacher-output probe. No pretrained tensor is modified.

For cached model calls use make_query_global_cache(model). The default
model.make_cache() returns a plain KV slot for the replaced layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import Attention as LlamaAttention

from llm_hybrid import attention_module


DEFAULT_GAIN = 0.44968487725044576
FEATURE_TEMPERATURE = 2.0


class QueryGlobalCache:
    """Latest local KV plus fixed-size global feature/value and key states."""

    def __init__(self, *, window: int = 64, feature_dim: int = 128,
                 gain: float = DEFAULT_GAIN):
        if window < 1 or feature_dim < 1 or not 0 <= gain <= 1:
            raise ValueError("window, feature_dim and gain are invalid")
        self.window = int(window)
        self.feature_dim = int(feature_dim)
        self.gain = float(gain)
        self.keys = None
        self.values = None
        self.global_kv = None
        self.global_k = None
        self.global_count = 0
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values, self.global_kv, self.global_k

    @property
    def nbytes(self):
        return int(sum(array.nbytes for array in self.state if array is not None))

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0


class LocalQueryGlobalAttention(nn.Module):
    """Exact short softmax window mixed with a bounded query/key feature read."""

    def __init__(self, original: LlamaAttention, *, window: int = 64,
                 gain: float = DEFAULT_GAIN, chunk: int = 64,
                 inference_sync_blocks: int = 8):
        super().__init__()
        if not isinstance(original, LlamaAttention):
            raise TypeError("the baseline requires mlx_lm Llama attention")
        if (window < 1 or chunk < 1 or inference_sync_blocks < 0
                or not 0 <= gain <= 1):
            raise ValueError("window, chunk, sync blocks and gain are invalid")
        if original.n_heads % original.n_kv_heads:
            raise ValueError("GQA heads cannot be grouped evenly")
        self.original = original
        self.window = int(window)
        self.chunk = int(chunk)
        self.gain = float(gain)
        self.feature_dim = 2 * original.head_dim
        self.inference_sync_blocks = int(inference_sync_blocks)

    @staticmethod
    def feature(x: mx.array) -> mx.array:
        y = x.astype(mx.float32) * FEATURE_TEMPERATURE
        return mx.concatenate(
            (mx.softmax(y, axis=-1), mx.softmax(-y, axis=-1)),
            axis=-1,
        )

    def _validate_cache(self, x: mx.array, cache: QueryGlobalCache) -> None:
        if not isinstance(cache, QueryGlobalCache):
            raise TypeError("the replaced layer requires QueryGlobalCache")
        if (cache.window != self.window or cache.feature_dim != self.feature_dim
                or cache.gain != self.gain):
            raise ValueError("cache window, feature map or fixed gain differs")
        if (cache.keys is None) != (cache.values is None):
            raise ValueError("cached keys and values must both be present")
        if (cache.global_kv is None) != (cache.global_k is None):
            raise ValueError("global states must both be present")
        if cache.offset < 0 or cache.global_count < 0:
            raise ValueError("negative cache offset or global count")
        expected_local = min(cache.offset, self.window - 1)
        if expected_local:
            if cache.keys is None:
                raise ValueError("a nonempty cache has no local keys")
            shape = (x.shape[0], self.original.n_kv_heads,
                     expected_local, self.original.head_dim)
            if cache.keys.shape != shape or cache.values.shape != shape:
                raise ValueError(f"cached KV shape differs from {shape}")
        elif cache.keys is not None:
            raise ValueError("an empty local cache has stored KV")
        if self.gain:
            expected_global = max(0, cache.offset - self.window + 1)
            if cache.global_count != expected_global:
                raise ValueError("global count differs from older-key count")
            if expected_global:
                shape_kv = (x.shape[0], self.original.n_kv_heads,
                            self.feature_dim, self.original.head_dim)
                shape_k = (x.shape[0], self.original.n_kv_heads,
                           self.feature_dim)
                if (cache.global_kv is None or
                        cache.global_kv.shape != shape_kv or
                        cache.global_k.shape != shape_k):
                    raise ValueError("global state shape differs from this input")
            elif cache.global_kv is not None:
                raise ValueError("empty global count has allocated state")
        elif cache.global_count or cache.global_kv is not None:
            raise ValueError("local-only cache unexpectedly holds global state")

    def __call__(self, x: mx.array, mask: Any = None,
                 cache: QueryGlobalCache | None = None) -> mx.array:
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError(f"expected nonempty (batch, tokens, width), got {x.shape}")
        work = cache or QueryGlobalCache(
            window=self.window, feature_dim=self.feature_dim, gain=self.gain)
        self._validate_cache(x, work)
        first_offset = work.offset
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"unsupported attention mask {mask!r}")
            mask = None
        elif mask is not None and mask.shape[-2:] != (x.shape[1], first_offset + x.shape[1]):
            raise ValueError(
                f"external mask has dimensions {mask.shape[-2:]}; expected "
                f"{(x.shape[1], first_offset + x.shape[1])}")
        pieces = []
        sync = (
            not self.training and self.inference_sync_blocks > 0
            and x.shape[1] > self.chunk * self.inference_sync_blocks
        )
        for index, first in enumerate(range(0, x.shape[1], self.chunk)):
            part = x[:, first:first + self.chunk]
            pieces.append(self._block(part, work, mask, first))
            if sync and (index + 1) % self.inference_sync_blocks == 0:
                # Long MLX lazy scans otherwise retain every block's
                # feature/value outer products until the final logits eval.
                # Materialize outputs and state together only in eval mode;
                # training must retain its full differentiable graph.
                mx.eval(pieces[-self.inference_sync_blocks:], work.state)
        if sync and len(pieces) % self.inference_sync_blocks:
            mx.eval(pieces[-(len(pieces) % self.inference_sync_blocks):],
                    work.state)
        out = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)
        return out if out.dtype == x.dtype else out.astype(x.dtype)

    def _block(self, x: mx.array, cache: QueryGlobalCache,
               external_mask: mx.array | None, outer_first: int) -> mx.array:
        batch, length, _ = x.shape
        original = self.original
        offset = cache.offset
        previous = 0 if cache.keys is None else cache.keys.shape[2]
        key_start = offset - previous
        q = original.q_proj(x).reshape(
            batch, length, original.n_heads, original.head_dim
        ).transpose(0, 2, 1, 3)
        k = original.k_proj(x).reshape(
            batch, length, original.n_kv_heads, original.head_dim
        ).transpose(0, 2, 1, 3)
        v = original.v_proj(x).reshape(
            batch, length, original.n_kv_heads, original.head_dim
        ).transpose(0, 2, 1, 3)
        q = original.rope(q, offset=offset)
        k = original.rope(k, offset=offset)
        if previous:
            keys = mx.concatenate((cache.keys, k), axis=2)
            values = mx.concatenate((cache.values, v), axis=2)
        else:
            keys, values = k, v
        key_count = keys.shape[2]

        query_positions = mx.arange(offset, offset + length)[:, None]
        key_positions = mx.arange(key_start, offset + length)[None, :]
        local_mask = ((key_positions <= query_positions)
                      & (key_positions > query_positions - self.window))
        if external_mask is not None:
            supplied = external_mask[
                ..., outer_first:outer_first + length, key_start:offset + length]
            if supplied.dtype == mx.bool_:
                local_mask = local_mask & supplied
            else:
                local_mask = mx.where(
                    local_mask, supplied, mx.finfo(supplied.dtype).min)
        elif length == 1 and key_count <= self.window:
            local_mask = None
        local_heads = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=original.scale, mask=local_mask)

        retained = min(self.window - 1, key_count)
        evicted = key_count - retained
        if self.gain:
            base_kv = cache.global_kv
            base_k = cache.global_k
            if base_kv is None:
                base_kv = mx.zeros(
                    (batch, original.n_kv_heads,
                     self.feature_dim, original.head_dim), mx.float32)
                base_k = mx.zeros(
                    (batch, original.n_kv_heads, self.feature_dim), mx.float32)
            if evicted:
                features = self.feature(keys[..., :evicted, :])
                evicted_values = values[..., :evicted, :].astype(mx.float32)
                outer = features[..., None] * evicted_values[..., None, :]
                prefix_kv = mx.cumsum(outer, axis=2)
                prefix_k = mx.cumsum(features, axis=2)
                padded_kv = mx.concatenate((
                    mx.zeros_like(prefix_kv[:, :, :1]), prefix_kv), axis=2)
                padded_k = mx.concatenate((
                    mx.zeros_like(prefix_k[:, :, :1]), prefix_k), axis=2)
            else:
                padded_kv = mx.zeros(
                    (batch, original.n_kv_heads, 1,
                     self.feature_dim, original.head_dim), mx.float32)
                padded_k = mx.zeros(
                    (batch, original.n_kv_heads, 1, self.feature_dim), mx.float32)
            # Query t can use exactly keys <= t-window, never its local keys.
            counts = mx.maximum(
                mx.arange(length) + offset - self.window + 1 - key_start, 0)
            state_kv = base_kv[:, :, None] + mx.take(padded_kv, counts, axis=2)
            state_k = base_k[:, :, None] + mx.take(padded_k, counts, axis=2)
            group = original.n_heads // original.n_kv_heads
            qf = self.feature(q).reshape(
                batch, original.n_kv_heads, group, length, self.feature_dim)
            qmat = qf.transpose(0, 1, 3, 2, 4)[..., None, :]
            numerator = mx.matmul(
                qmat, state_kv[:, :, :, None, :, :]).squeeze(-2)
            denominator = mx.sum(
                qmat.squeeze(-2) * state_k[:, :, :, None, :], axis=-1)
            global_heads = (
                numerator / mx.maximum(denominator[..., None], 1e-8)
            ).transpose(0, 1, 3, 2, 4).reshape(
                batch, original.n_heads, length, original.head_dim)
            valid = (cache.global_count + counts) > 0
            gain = self.gain * valid.astype(mx.float32)
            mixed = local_heads.astype(mx.float32) + gain[None, None, :, None] * (
                global_heads - local_heads.astype(mx.float32))
            heads = mixed.astype(x.dtype)
            if evicted:
                cache.global_kv = base_kv + prefix_kv[:, :, -1]
                cache.global_k = base_k + prefix_k[:, :, -1]
                cache.global_count += evicted
        else:
            heads = local_heads

        if retained:
            cache.keys = mx.contiguous(keys[..., -retained:, :])
            cache.values = mx.contiguous(values[..., -retained:, :])
        else:
            cache.keys = cache.values = None
        cache.offset += length
        flattened = heads.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return original.o_proj(flattened)


@dataclass
class QueryGlobalInstallation:
    layer: Any
    attribute: str
    original: LlamaAttention
    replacement: LocalQueryGlobalAttention
    conversion: dict

    def restore(self) -> None:
        if getattr(self.layer, self.attribute) is not self.replacement:
            raise RuntimeError("attention slot changed since installation")
        setattr(self.layer, self.attribute, self.original)


def install_query_global(model, *, layer: int = 8, gain: float = DEFAULT_GAIN,
                         window: int = 64, chunk: int = 64,
                         inference_sync_blocks: int = 8
                         ) -> QueryGlobalInstallation:
    layer_obj, attribute, original = attention_module(model, layer)
    if not isinstance(original, LlamaAttention):
        raise TypeError("this baseline supports mlx_lm Llama attention only")
    if str(original.q_proj.weight.dtype).endswith("uint32"):
        raise TypeError("this baseline requires unquantized Llama weights")
    replacement = LocalQueryGlobalAttention(
        original, window=window, gain=gain, chunk=chunk,
        inference_sync_blocks=inference_sync_blocks)
    if not model.training:
        replacement.eval()
    setattr(layer_obj, attribute, replacement)
    return QueryGlobalInstallation(
        layer_obj, attribute, original, replacement,
        {
            "mode": "untrained_local_query_global",
            "layer": layer,
            "local_window": window,
            "feature_map": "softmax(+2*RoPE(x)) concat softmax(-2*RoPE(x))",
            "feature_dim": replacement.feature_dim,
            "fixed_gain": gain,
            "inference_sync_blocks": inference_sync_blocks,
            "new_trainable_parameters": 0,
            "original_q_k_v_o_weights_kept": True,
            "prior_art": "LoLCATs local + feature-global attention; this identity map is a baseline",
        },
    )


def make_query_global_cache(model):
    """Standard model cache with a bounded slot at each query-global layer."""
    caches = model.make_cache()
    for index, layer in enumerate(model.layers):
        module = getattr(layer, "self_attn", None)
        if isinstance(module, LocalQueryGlobalAttention):
            caches[index] = QueryGlobalCache(
                window=module.window,
                feature_dim=module.feature_dim,
                gain=module.gain,
            )
    return caches
