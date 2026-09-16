"""Persistent state for the repository's gated-memory attention replacement.

The older carrier evaluates a complete prefix correctly but starts its trace at
zero on every model call. That makes cached, token-by-token generation a
different function. This module keeps one state vector per batch item and bank.
It changes no weights and uses the original parallel scan for uncached calls.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from llm_efficiency import chunked_gated_scan


class MemoryCache:
    """State passed in the cache slot of one replaced attention layer."""

    def __init__(self):
        self.h = None
        self.offset = 0

    @property
    def state(self):
        return self.h

    def size(self):
        return self.offset


class StreamingGatedMemoryCarrier(nn.Module):
    """The original carrier's weights with a recurrent cache for generation."""

    def __init__(self, carrier):
        super().__init__()
        self.mode = carrier.mode
        self.out_gain = carrier.out_gain
        self.d = carrier.d
        if self.mode != "zero":
            self.mem = carrier.mem

    def __call__(self, x, mask=None, cache=None):
        if self.mode == "zero":
            if cache is not None:
                if not isinstance(cache, MemoryCache):
                    raise TypeError("a replaced layer requires MemoryCache")
                cache.offset += x.shape[1]
            return mx.zeros_like(x)

        if cache is None:
            out = self.mem(x)
        else:
            if not isinstance(cache, MemoryCache):
                raise TypeError("a replaced layer requires MemoryCache")
            out = self._cached(x, cache)
        out = out * self.out_gain
        # Preserve the pretrained residual-stream dtype. Promoting it to fp32
        # changed perplexity even when the memory contribution was zero.
        if out.dtype != x.dtype:
            out = out.astype(x.dtype)
        return out

    def _cached(self, x, cache):
        B, T, D = x.shape
        if D != self.d or T < 1:
            raise ValueError(f"expected (B,T,{self.d}) with T>=1; got {x.shape}")
        banks = self.mem.banks
        v = self.mem.v(x).reshape(B, T, banks, D)
        g = mx.sigmoid(self.mem.gate(x)).reshape(B, T, banks, D)
        v = v.transpose(0, 2, 1, 3).reshape(B * banks, T, D)
        g = g.transpose(0, 2, 1, 3).reshape(B * banks, T, D)

        # The original chunked scan starts with zero state. An initial state
        # contributes h0 * product(g_0 ... g_t) to every prefix. For T=1 the
        # recurrence is a direct vector operation with no scan overhead.
        if T == 1:
            h = v if cache.h is None else v + g * cache.h.reshape(B * banks, 1, D)
        else:
            h = chunked_gated_scan(v, g, self.mem.chunk)
            if cache.h is not None:
                if cache.h.shape != (B, banks, D):
                    raise ValueError(
                        f"cached state has shape {cache.h.shape}; expected "
                        f"{(B, banks, D)}")
                h = h + mx.cumprod(g, axis=1) * cache.h.reshape(B * banks, 1, D)

        cache.h = h[:, -1, :].reshape(B, banks, D)
        cache.offset += T
        h = h.reshape(B, banks, T, D).transpose(0, 2, 1, 3)
        return self.mem.o(h.reshape(B, T, banks * D))
