"""Attention-transfer feature-map parameters for the query-global baseline.

The underlying exact local attention, recurrent feature state and pretrained
q/k/v/o projections are unchanged. Per-head query/KV residual matrices start
at zero, so this module exactly reproduces the fixed identity feature map
before training. It is opt-in and contains only 32+8 maps at Llama-3.2-1B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from llm_hybrid import attention_module
from query_global_attention import (
    DEFAULT_GAIN, LocalQueryGlobalAttention,
)


class TrainableQueryGlobalAttention(LocalQueryGlobalAttention):
    """Identity Hedgehog-style features plus trainable head-wise residuals."""

    def __init__(self, original, *, window: int = 64,
                 gain: float = DEFAULT_GAIN, chunk: int = 64,
                 inference_sync_blocks: int = 8):
        super().__init__(
            original, window=window, gain=gain, chunk=chunk,
            inference_sync_blocks=inference_sync_blocks)
        if original.n_heads == original.n_kv_heads:
            raise ValueError("this first trainer requires distinguishable GQA heads")
        width = original.head_dim
        self.delta_q = mx.zeros((original.n_heads, width, width), mx.float32)
        self.delta_k = mx.zeros((original.n_kv_heads, width, width), mx.float32)

    def feature(self, x: mx.array) -> mx.array:
        if x.shape[1] == self.original.n_heads:
            residual = self.delta_q
        elif x.shape[1] == self.original.n_kv_heads:
            residual = self.delta_k
        else:
            raise ValueError("feature input has an unexpected number of heads")
        x = x.astype(mx.float32)
        mapped = x + mx.matmul(x, residual[None, :, :, :])
        y = mapped * 2.0
        return mx.concatenate(
            (mx.softmax(y, axis=-1), mx.softmax(-y, axis=-1)),
            axis=-1)


@dataclass
class TrainableInstallation:
    layer: Any
    attribute: str
    original: Any
    replacement: TrainableQueryGlobalAttention
    conversion: dict

    def restore(self):
        if getattr(self.layer, self.attribute) is not self.replacement:
            raise RuntimeError("attention slot changed since installation")
        setattr(self.layer, self.attribute, self.original)


def install_trainable_query_global(
    model, *, layer: int = 8, gain: float = DEFAULT_GAIN,
    window: int = 64, chunk: int = 64, inference_sync_blocks: int = 8
) -> TrainableInstallation:
    layer_obj, attribute, original = attention_module(model, layer)
    replacement = TrainableQueryGlobalAttention(
        original, gain=gain, window=window, chunk=chunk,
        inference_sync_blocks=inference_sync_blocks)
    if not model.training:
        replacement.eval()
    setattr(layer_obj, attribute, replacement)
    return TrainableInstallation(
        layer_obj, attribute, original, replacement,
        {
            "mode": "trainable_local_query_global_attention_transfer",
            "layer": layer,
            "gain": gain,
            "window": window,
            "feature_dim": replacement.feature_dim,
            "query_residual_parameters":
                original.n_heads * original.head_dim * original.head_dim,
            "kv_residual_parameters":
                original.n_kv_heads * original.head_dim * original.head_dim,
            "initial_feature_map_matches_fixed_identity_baseline": True,
            "pretrained_q_k_v_o_frozen_during_stage1": True,
        },
    )
