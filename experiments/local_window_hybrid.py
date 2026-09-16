"""Opt-in 64-token Llama attention with the repository's gated-memory trace.

The local path uses the *original* q/k/v/o projections and RoPE. It attends to
positions ``max(0, t - 63) ... t`` exactly, including the current position.
The long-range path is the existing value/output transplant and gated scan from
``llm_hybrid.py``; ``memory_gain`` controls its contribution beside local
attention. No pretrained tensor is rewritten or converted to another dtype.

Use ``install_local_window_hybrid(model, layer=8)`` on an already loaded,
unquantized Llama-3.2-1B MLX model. For cached calls, pass a cache made with
``make_hybrid_cache(model)``. The original ``model.make_cache()`` still returns a
plain KV cache at the replaced layer and is not suitable for this wrapper.
The installation handle can restore the original attention module.

This is a functional prototype, not a measured speed or quality improvement.
Replacing one of 16 layers leaves the other 15 full-attention layers intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import Attention as LlamaAttention

from llm_hybrid import attention_module, transplant
from streaming_memory import MemoryCache, StreamingGatedMemoryCarrier


class HybridAttentionCache:
    """One replaced layer's bounded KV cache and persistent gated trace.

    ``offset`` is the absolute RoPE position / number of processed tokens.
    The KV tensors contain only the latest ``window - 1`` positions because
    the next query supplies its own key and value.
    """

    def __init__(self, window: int = 64):
        if window < 1:
            raise ValueError("window must be positive")
        self.window = int(window)
        self.keys = None
        self.values = None
        self.memory = MemoryCache()
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values, self.memory.state

    @property
    def nbytes(self):
        total = 0
        for array in (self.keys, self.values, self.memory.state):
            if array is not None:
                total += array.nbytes
        return int(total)

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0


class LocalWindowHybridAttention(nn.Module):
    """Drop-in MLX Llama attention with exact local attention plus gated memory.

    ``memory_gain=0`` is a local-only control; signed values scale the existing
    normalized gated-memory output. This value is fixed for the entire cache
    lifetime. The module accepts complete uncached prefixes and arbitrary
    cached chunks, including single-token decode.
    """

    def __init__(self, original: LlamaAttention, carrier, *, window: int = 64,
                 memory_gain: float = 0.05, chunk: int = 64):
        super().__init__()
        if not isinstance(original, LlamaAttention):
            raise TypeError("the local path requires mlx_lm.models.llama.Attention")
        if window < 1 or chunk < 1:
            raise ValueError("window and chunk must be positive")
        if not -1.0 <= memory_gain <= 1.0:
            raise ValueError("memory_gain must be in [-1, 1]")
        if carrier.mode == "zero":
            raise ValueError("the hybrid requires a nonzero gated-memory carrier")
        self.original = original
        self.memory = StreamingGatedMemoryCarrier(carrier)
        self.window = int(window)
        self.chunk = int(chunk)
        self.memory_gain = float(memory_gain)

    def __call__(self, x: mx.array, mask: Any = None,
                 cache: HybridAttentionCache | None = None) -> mx.array:
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError(f"expected nonempty (batch, tokens, width), got {x.shape}")
        if cache is not None:
            if not isinstance(cache, HybridAttentionCache):
                raise TypeError("the replaced layer requires HybridAttentionCache")
            if cache.window != self.window:
                raise ValueError("cache window differs from the installed module")
            if cache.offset != cache.memory.offset:
                raise ValueError("KV and gated-memory cache offsets disagree")
            if cache.offset and cache.keys is None and self.window > 1:
                raise ValueError("a nonempty cache has no prior keys")
            if (cache.keys is None) != (cache.values is None):
                raise ValueError("cached keys and values must both be present")
            if cache.keys is not None:
                if cache.keys.shape[0] != x.shape[0]:
                    raise ValueError("cached KV batch size differs from input batch size")
                expected = (x.shape[0], self.original.n_kv_heads,
                            cache.keys.shape[2], self.original.head_dim)
                if cache.keys.shape != expected or cache.values.shape != expected:
                    raise ValueError(
                        f"cached KV shape differs from {expected} for this input")
                if cache.keys.shape[2] > self.window - 1:
                    raise ValueError("cached KV exceeds the local window")
            if cache.memory.h is not None:
                if cache.memory.h.shape[0] != x.shape[0]:
                    raise ValueError("cached memory batch size differs from input batch size")
                expected_memory = (x.shape[0], self.memory.mem.banks,
                                   self.memory.d)
                if cache.memory.h.shape != expected_memory:
                    raise ValueError(
                        f"cached memory shape differs from {expected_memory}")

        local = self._local(x, mask, cache)
        if self.memory_gain:
            memory = self.memory(x, cache=None if cache is None else cache.memory)
            # The carrier returns x.dtype; perform the gain in fp32, then cast
            # back before adding so the pretrained residual stream cannot be
            # promoted by the new branch.
            branch = (memory.astype(mx.float32) * self.memory_gain).astype(x.dtype)
            out = local + branch
        else:
            out = local
            if cache is not None:
                cache.memory.offset += x.shape[1]
        return out if out.dtype == x.dtype else out.astype(x.dtype)

    def _local(self, x: mx.array, mask: Any,
               cache: HybridAttentionCache | None) -> mx.array:
        B, T, _ = x.shape
        original = self.original
        offset = 0 if cache is None else cache.offset
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"unsupported attention mask {mask!r}")
            mask = None
        elif mask is not None and mask.shape[-2:] != (T, offset + T):
            raise ValueError(
                f"external mask has final dimensions {mask.shape[-2:]}; "
                f"expected {(T, offset + T)}")

        q = original.q_proj(x).reshape(B, T, original.n_heads, original.head_dim)
        k = original.k_proj(x).reshape(B, T, original.n_kv_heads, original.head_dim)
        v = original.v_proj(x).reshape(B, T, original.n_kv_heads, original.head_dim)
        q = original.rope(q.transpose(0, 2, 1, 3), offset=offset)
        k = original.rope(k.transpose(0, 2, 1, 3), offset=offset)
        v = v.transpose(0, 2, 1, 3)

        previous = 0 if cache is None or cache.keys is None else cache.keys.shape[2]
        start_position = offset - previous
        if previous:
            keys = mx.concatenate((cache.keys, k), axis=2)
            values = mx.concatenate((cache.values, v), axis=2)
        else:
            keys, values = k, v

        # Each chunk sees at most (window - 1 + chunk) keys. The mask selects
        # exactly the latest window positions for each query, so prefill score
        # work grows linearly with prefix length for fixed window/chunk sizes.
        pieces = []
        for first in range(0, T, self.chunk):
            last = min(first + self.chunk, T)
            query_start = offset + first
            key_start = max(start_position, query_start - self.window + 1)
            key_end = offset + last
            left = key_start - start_position
            right = key_end - start_position
            query_positions = mx.arange(query_start, offset + last)[:, None]
            key_positions = mx.arange(key_start, key_end)[None, :]
            local_mask = ((key_positions <= query_positions)
                          & (key_positions > query_positions - self.window))
            if mask is not None:
                external = mask[..., first:last, key_start:key_end]
                if external.dtype == mx.bool_:
                    local_mask = local_mask & external
                else:
                    local_mask = mx.where(local_mask, external,
                                          mx.finfo(external.dtype).min)
            elif last - first == 1 and key_end - key_start <= self.window:
                # Every key precedes this single query; avoid a needless mask.
                local_mask = None
            piece = mx.fast.scaled_dot_product_attention(
                q[..., first:last, :], keys[..., left:right, :],
                values[..., left:right, :], scale=original.scale,
                mask=local_mask,
            )
            pieces.append(piece)

        attended = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=2)
        attended = attended.transpose(0, 2, 1, 3).reshape(B, T, -1)
        local = original.o_proj(attended)

        if cache is not None:
            retained = min(self.window - 1, keys.shape[2])
            if retained:
                # A compact logical cache; no historic prefix KV is retained.
                cache.keys = mx.contiguous(keys[..., -retained:, :])
                cache.values = mx.contiguous(values[..., -retained:, :])
            else:
                cache.keys = cache.values = None
            cache.offset += T
        return local


@dataclass
class HybridInstallation:
    layer: Any
    attribute: str
    original: LlamaAttention
    hybrid: LocalWindowHybridAttention
    conversion: dict

    def restore(self):
        """Restore the original sublayer without changing its tensors."""
        if getattr(self.layer, self.attribute) is not self.hybrid:
            raise RuntimeError("attention slot changed since installation")
        setattr(self.layer, self.attribute, self.original)


def install_local_window_hybrid(model, layer: int = 8, *, decay: float = 0.7,
                                memory_gain: float = 0.05, chunk: int = 64
                                ) -> HybridInstallation:
    """Install one 64-token hybrid on a loaded MLX Llama, without training.

    ``transplant`` copies W_v/W_o into the existing gated-memory primitive and
    uses an explicit decay; original q/k/v/o/RoPE stay in the local path. An
    explicit decay avoids a hidden recency-fit benchmark during installation.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1); decay=1 disables the normalized trace")
    layer_obj, attribute, original = attention_module(model, layer)
    if not isinstance(original, LlamaAttention):
        raise TypeError("this prototype supports mlx_lm Llama attention only")
    if str(original.q_proj.weight.dtype).endswith("uint32"):
        raise TypeError("this prototype requires an unquantized Llama checkpoint")
    try:
        carrier, conversion = transplant(model, layer, "transfer",
                                         original=original, decay=decay)
        if not conversion.get("transplanted"):
            raise RuntimeError("the gated-memory value/output transplant failed")
        hybrid = LocalWindowHybridAttention(
            original, carrier, window=64, memory_gain=memory_gain, chunk=chunk)
        setattr(layer_obj, attribute, hybrid)
    except Exception:
        setattr(layer_obj, attribute, original)
        raise
    return HybridInstallation(layer_obj, attribute, original, hybrid, conversion)


def make_hybrid_cache(model):
    """Create a standard model cache with HybridAttentionCache at each hybrid."""
    caches = model.make_cache()
    for index, layer in enumerate(model.layers):
        module = getattr(layer, "self_attn", None)
        if isinstance(module, LocalWindowHybridAttention):
            caches[index] = HybridAttentionCache(module.window)
    return caches
