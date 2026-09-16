"""Matched rank-8 q/k/v/o LoRA arms for the one-layer conversion control.

The original pretrained projection modules are shared, frozen references.
Each arm owns separately initialized LoRA matrices, starting with zero B.
The query-state arm also carries the selected, frozen stage-1 feature maps.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.llama import Attention as LlamaAttention
from mlx_lm.tuner.lora import LoRALinear

from query_global_attention import DEFAULT_GAIN, LocalQueryGlobalAttention
from query_global_trainable import TrainableQueryGlobalAttention


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
RANK = 8
LORA_SCALE = 2.0  # alpha=16 divided by rank 8
INIT_SEED = 7
ARMS = ("full_lora", "local_lora", "query_lora")


def adapted_attention(base: LlamaAttention, args, *, seed: int = INIT_SEED
                      ) -> LlamaAttention:
    """Build independent LoRA arrays around identical frozen teacher tensors."""
    if not isinstance(base, LlamaAttention):
        raise TypeError("rank-8 control requires unquantized Llama attention")
    if str(base.q_proj.weight.dtype).endswith("uint32"):
        raise TypeError("this control requires unquantized projection weights")
    adapted = LlamaAttention(args)
    mx.random.seed(seed)
    for name in PROJECTIONS:
        wrapped = LoRALinear.from_base(
            getattr(base, name), r=RANK, dropout=0.0, scale=LORA_SCALE)
        setattr(adapted, name, wrapped)
    adapted.rope = base.rope
    if (adapted.n_heads != base.n_heads or
            adapted.n_kv_heads != base.n_kv_heads or
            adapted.head_dim != base.head_dim or adapted.scale != base.scale):
        raise AssertionError("LoRA attention geometry changed the teacher")
    adapted.eval()
    return adapted


def lora_core(module):
    core = module if isinstance(module, LlamaAttention) else module.original
    if not isinstance(core, LlamaAttention) or any(
            not isinstance(getattr(core, name), LoRALinear)
            for name in PROJECTIONS):
        raise TypeError("arm does not contain four Llama LoRA projections")
    return core


def canonical_lora(module) -> dict[str, mx.array]:
    core = lora_core(module)
    return {
        f"{name}.{part}": getattr(getattr(core, name), part)
        for name in PROJECTIONS for part in ("lora_a", "lora_b")
    }


def canonical_numpy(module) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy()
            for name, value in canonical_lora(module).items()}


def canonical_hashes(module) -> dict[str, str]:
    return {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for name, value in canonical_lora(module).items()}


def load_canonical(module, arrays: Mapping[str, np.ndarray]) -> None:
    expected = canonical_lora(module)
    if set(arrays) != set(expected):
        raise ValueError("LoRA checkpoint keys differ from four projections")
    core = lora_core(module)
    for key, value in arrays.items():
        projection, part = key.split(".")
        if value.shape != expected[key].shape or value.dtype != np.float32:
            raise ValueError(f"LoRA checkpoint {key} shape or dtype differs")
        setattr(getattr(core, projection), part, mx.array(value))
    mx.eval(list(canonical_lora(module).values()))


def trainable_scope(module) -> tuple[list[str], int]:
    module.freeze()
    core = lora_core(module)
    for name in PROJECTIONS:
        getattr(core, name).unfreeze(
            keys=["lora_a", "lora_b"], recurse=False, strict=True)
    names_and_values = nn.utils.tree_flatten(module.trainable_parameters())
    prefix = "" if module is core else "original."
    expected = sorted(
        f"{prefix}{name}.{part}"
        for name in PROJECTIONS for part in ("lora_a", "lora_b"))
    actual = sorted(name for name, _ in names_and_values)
    if actual != expected:
        raise AssertionError(f"trainable LoRA scope differs: {actual}")
    count = sum(int(np.prod(value.shape)) for _, value in names_and_values)
    return actual, count


def build_arm(base: LlamaAttention, args, arm: str,
              *, stage1_q: np.ndarray | None = None,
              stage1_k: np.ndarray | None = None,
              inference_sync_blocks: int = 8):
    if arm not in ARMS:
        raise ValueError(f"unknown rank-8 arm {arm}")
    adapted = adapted_attention(base, args)
    if arm == "full_lora":
        module = adapted
    elif arm == "local_lora":
        module = LocalQueryGlobalAttention(
            adapted, window=64, gain=0.0, chunk=64,
            inference_sync_blocks=inference_sync_blocks)
    else:
        if stage1_q is None or stage1_k is None:
            raise ValueError("query LoRA arm requires selected stage-1 maps")
        if (stage1_q.shape != (adapted.n_heads, adapted.head_dim,
                              adapted.head_dim) or
                stage1_k.shape != (adapted.n_kv_heads, adapted.head_dim,
                                  adapted.head_dim)):
            raise ValueError("stage-1 map shapes do not fit this Llama")
        module = TrainableQueryGlobalAttention(
            adapted, window=64, gain=DEFAULT_GAIN, chunk=64,
            inference_sync_blocks=inference_sync_blocks)
        module.delta_q = mx.array(stage1_q.astype(np.float32))
        module.delta_k = mx.array(stage1_k.astype(np.float32))
        mx.eval(module.delta_q, module.delta_k)
    module.eval()
    names, count = trainable_scope(module)
    return module, {"arm": arm, "rank": RANK,
                    "lora_alpha": LORA_SCALE * RANK,
                    "lora_scale": LORA_SCALE,
                    "lora_init_seed": INIT_SEED,
                    "trainable_keys": names,
                    "trainable_parameters": count,
                    "stage1_feature_maps_frozen": arm == "query_lora",
                    "original_pretrained_projections_shared_and_frozen": True}
