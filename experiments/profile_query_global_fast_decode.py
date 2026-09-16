"""Paired layer-only speed and exact-state test of one-token decode.

Compare fused full attention, selected query state with the general
multi-token block routine, and the opt-in one-token routine at 8K/32K.
The two query arms use identical 256/2 prefills and pretrained weights;
every one of 32 genuine teacher-derived decode activations must produce
identical layer outputs and final bounded states. This is not a
complete-model or quality-matched speed result.
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
from query_global_attention import DEFAULT_GAIN, QueryGlobalCache  # noqa: E402
from query_global_fast_decode import FastDecodeTrainableQueryGlobalAttention  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_lora import frozen_prefix  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_fast_decode_profile_b1.json"
CONTEXTS = (8192, 32768)
DECODE = 32
REPEATS = {8192: 5, 32768: 3}
ARMS = ("teacher_layer", "general_query_layer", "fast_query_layer")


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def exact_decode(modules, prefix, future) -> dict:
    caches = {}
    for arm in ("general_query_layer", "fast_query_layer"):
        cache = QueryGlobalCache(
            window=64, feature_dim=128, gain=DEFAULT_GAIN)
        output = modules[arm](prefix, mask="causal", cache=cache)
        mx.eval(output, cache.state)
        caches[arm] = cache
    max_abs_output = 0.0
    for index in range(future.shape[1]):
        outputs = {}
        for arm in caches:
            output = modules[arm](
                future[:, index:index + 1], cache=caches[arm])
            mx.eval(output, caches[arm].state)
            outputs[arm] = np.asarray(output.astype(mx.float32))
        max_abs_output = max(
            max_abs_output,
            float(np.max(np.abs(outputs["general_query_layer"] -
                                outputs["fast_query_layer"]))))
    left = caches["general_query_layer"]
    right = caches["fast_query_layer"]
    max_abs_state = max(
        float(np.max(np.abs(np.asarray(left.global_kv) -
                            np.asarray(right.global_kv)))),
        float(np.max(np.abs(np.asarray(left.global_k) -
                            np.asarray(right.global_k)))),
    )
    row = {
        "max_abs_layer_output_across_32_decodes": max_abs_output,
        "max_abs_final_state": max_abs_state,
        "general_global_count": left.global_count,
        "fast_global_count": right.global_count,
        "general_offset": left.offset,
        "fast_offset": right.offset,
        "general_cache_bytes": left.nbytes,
        "fast_cache_bytes": right.nbytes,
    }
    if (row["max_abs_layer_output_across_32_decodes"] != 0.0 or
            row["max_abs_final_state"] != 0.0 or
            row["general_global_count"] != row["fast_global_count"] or
            row["general_offset"] != row["fast_offset"] or
            row["general_cache_bytes"] != row["fast_cache_bytes"]):
        raise AssertionError(f"fast decoder changed selected map: {row}")
    return row


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
    if len(ids) <= max(CONTEXTS) + DECODE:
        raise ValueError("corpus too short for long decode profile")
    original = model.layers[8].self_attn
    modules = {
        "teacher_layer": original,
        "general_query_layer": TrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=256,
            inference_sync_blocks=2),
        "fast_query_layer": FastDecodeTrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=256,
            inference_sync_blocks=2),
    }
    for arm in ARMS[1:]:
        modules[arm].delta_q = delta_q
        modules[arm].delta_k = delta_k
        modules[arm].eval()
    mx.eval(delta_q, delta_k)
    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fast_decode_source_sha256": hashlib.sha256(
                (HERE / "query_global_fast_decode.py").read_bytes()).hexdigest(),
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
            "contexts": list(CONTEXTS),
            "decode_tokens": DECODE,
            "repeats_by_context": {str(k): v for k, v in REPEATS.items()},
            "gain": DEFAULT_GAIN,
            "chunk": 256,
            "inference_sync_blocks": 2,
            "workload": "genuine teacher layer-8 inputs; one-token output and global state checked for bit equality before timings",
            "limitation": "layer-only decode with teacher-derived upstream activations on a busy host; full model and quality gates separate",
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
        },
        "rows": [],
    }
    atomic_json(record)
    for context in CONTEXTS:
        begun = time.perf_counter()
        hidden = frozen_prefix(
            model, mx.array(ids[:context + DECODE][None, :],
                            dtype=mx.int32))
        activated = model.layers[8].input_layernorm(hidden)
        mx.eval(activated)
        prefix = activated[:, :context]
        future = activated[:, context:context + DECODE]
        row = {"context": context,
               "activation_capture_s": time.perf_counter() - begun,
               "exact_decode": exact_decode(modules, prefix, future),
               "repeats": [], "median": {}}
        record["rows"].append(row)
        atomic_json(record)
        for arm in ARMS:
            layer_repeat(model, modules[arm], prefix, future)
        for repeat in range(REPEATS[context]):
            rotation = repeat % len(ARMS)
            order = ARMS[rotation:] + ARMS[:rotation]
            readings = {
                arm: layer_repeat(model, modules[arm], prefix, future)
                for arm in order
            }
            item = {"index": repeat, "order": list(order),
                    "readings": readings,
                    "fast_decode_speed_ratio_to_general": (
                        readings["general_query_layer"]["decode_s"] /
                        readings["fast_query_layer"]["decode_s"]),
                    "fast_decode_speed_ratio_to_teacher": (
                        readings["teacher_layer"]["decode_s"] /
                        readings["fast_query_layer"]["decode_s"]),
                    "fast_prefill_time_ratio_to_general": (
                        readings["fast_query_layer"]["prefill_s"] /
                        readings["general_query_layer"]["prefill_s"]),
                    }
            row["repeats"].append(item)
            atomic_json(record)
            print(f"T={context} repeat={repeat} fast/general decode="
                  f"{item['fast_decode_speed_ratio_to_general']:.3f} "
                  f"fast/teacher decode="
                  f"{item['fast_decode_speed_ratio_to_teacher']:.3f}",
                  flush=True)
        row["median"] = {
            key: statistics.median(item[key] for item in row["repeats"])
            for key in ("fast_decode_speed_ratio_to_general",
                        "fast_decode_speed_ratio_to_teacher",
                        "fast_prefill_time_ratio_to_general")
        }
        atomic_json(record)
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print("wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
