"""Paired 8K scan-block tuning for the selected query-state feature map.

All configurations keep the same teacher q/k/v/o, RoPE, local window,
feature-map checkpoint and gain. Sync occurs every 512 tokens in each arm:
64/8, 128/4 or 256/2. Time genuine teacher layer-8 prefill and 32
teacher-derived single-token decodes, retain raw order and Metal peaks, and
check the complete layer-output numerical difference. This is not a
complete-model or quality-matched speed verdict.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from profile_query_global_layer import layer_repeat  # noqa: E402
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_lora import frozen_prefix  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_chunk_profile_b1.json"
CONTEXT = 8192
DECODE = 32
REPEATS = 5
CONFIGS = {
    "base_64_8": (64, 8),
    "chunk128_sync4": (128, 4),
    "chunk256_sync2": (256, 2),
}


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def main() -> None:
    stage1 = json.loads(STAGE1_RESULT.read_text())
    weight_sha = hashlib.sha256(STAGE1_WEIGHTS.read_bytes()).hexdigest()
    if (stage1["status"] != "complete" or not stage1["meta"]["primary"]
            or stage1["weights"]["sha256"] != weight_sha):
        raise ValueError("complete selected stage-1 map required")
    with np.load(STAGE1_WEIGHTS) as checkpoint:
        delta_q = mx.array(checkpoint["delta_q"].astype(np.float32))
        delta_k = mx.array(checkpoint["delta_k"].astype(np.float32))
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    if len(ids) <= CONTEXT + DECODE:
        raise ValueError("corpus too short for 8K layer profile")
    started = time.perf_counter()
    hidden = frozen_prefix(
        model, mx.array(ids[:CONTEXT + DECODE][None, :], dtype=mx.int32))
    activated = model.layers[8].input_layernorm(hidden)
    mx.eval(activated)
    prefix = activated[:, :CONTEXT]
    future = activated[:, CONTEXT:CONTEXT + DECODE]
    capture_s = time.perf_counter() - started
    original = model.layers[8].self_attn
    modules = {}
    for name, (chunk, sync_blocks) in CONFIGS.items():
        module = TrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=chunk,
            inference_sync_blocks=sync_blocks)
        module.delta_q = delta_q
        module.delta_k = delta_k
        module.eval()
        modules[name] = module
    mx.eval(delta_q, delta_k)
    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "layer_harness_source_sha256": hashlib.sha256(
                (HERE / "profile_query_global_layer.py").read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "trainable_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "stage1_result_sha256": hashlib.sha256(STAGE1_RESULT.read_bytes()).hexdigest(),
            "stage1_weights_sha256": weight_sha,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "context": CONTEXT,
            "decode_tokens": DECODE,
            "repeats": REPEATS,
            "configs": {name: {"chunk": chunk,
                               "inference_sync_blocks": sync,
                               "tokens_per_sync": chunk * sync}
                        for name, (chunk, sync) in CONFIGS.items()},
            "teacher_activation_capture_s": capture_s,
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
            "limitation": "layer-only paired tuning on one busy Apple host; complete-model and quality gate separate",
        },
        "numerical_parity": {},
        "repeats": [],
        "median": {},
    }
    atomic_json(record)
    outputs = {}
    for name, module in modules.items():
        output = module(prefix, mask="causal")
        mx.eval(output)
        outputs[name] = output
    reference = outputs["base_64_8"].astype(mx.float32)
    for name, output in outputs.items():
        difference = output.astype(mx.float32) - reference
        max_abs = mx.max(mx.abs(difference))
        mean_abs = mx.mean(mx.abs(difference))
        mx.eval(max_abs, mean_abs)
        record["numerical_parity"][name] = {
            "max_abs_layer_output_vs_base": float(max_abs),
            "mean_abs_layer_output_vs_base": float(mean_abs),
        }
        if float(max_abs) > 0.1:
            raise AssertionError(f"{name}: chunk changed output beyond bf16 tolerance")
    atomic_json(record)
    del outputs, reference
    for module in modules.values():
        layer_repeat(model, module, prefix, future)
    names = tuple(CONFIGS)
    for repeat in range(REPEATS):
        rotation = repeat % len(names)
        order = names[rotation:] + names[:rotation]
        readings = {
            name: layer_repeat(model, modules[name], prefix, future)
            for name in order
        }
        item = {"index": repeat, "order": list(order),
                "readings": readings}
        for name in names[1:]:
            item[f"{name}_prefill_time_ratio_to_base"] = (
                readings[name]["prefill_s"] /
                readings["base_64_8"]["prefill_s"])
            item[f"{name}_decode_speed_ratio_to_base"] = (
                readings["base_64_8"]["decode_s"] /
                readings[name]["decode_s"])
            item[f"{name}_peak_metal_ratio_to_base"] = (
                readings[name]["final_peak_metal_bytes"] /
                readings["base_64_8"]["final_peak_metal_bytes"])
        record["repeats"].append(item)
        atomic_json(record)
        print(f"repeat={repeat} chunk256/base prefill="
              f"{item['chunk256_sync2_prefill_time_ratio_to_base']:.3f} "
              f"decode speed="
              f"{item['chunk256_sync2_decode_speed_ratio_to_base']:.3f}",
              flush=True)
    keys = [key for key in record["repeats"][0]
            if key.endswith("_to_base")]
    record["median"] = {
        key: statistics.median(item[key] for item in record["repeats"])
        for key in keys
    }
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print("wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
