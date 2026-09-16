"""Locate the latency cost of the query state on genuine layer-8 inputs.

Capture the frozen teacher's first-eight-layer activations once per context.
Time the teacher's fused attention, exact local window, fixed query state and
selected stage-1 query map alone, alongside a last-token-head complete-model
teacher call. All arms use persistent caches and 32 genuine teacher-derived
decode activations. This paired Apple MLX diagnostic is not a production-cost
or quality-preserving speed verdict.
"""
from __future__ import annotations

import argparse
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

from benchmark_query_global import one_repeat as whole_model_repeat  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import (  # noqa: E402
    DEFAULT_GAIN, LocalQueryGlobalAttention, QueryGlobalCache,
)
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_lora import frozen_prefix  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "query_global_layer_profile_b1.json"
ARMS = ("teacher_full", "teacher_layer", "local_layer",
        "fixed_query_layer", "trained_query_layer")


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--contexts", default="512,2048,8192")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def layer_cache(model, module):
    if isinstance(module, (LocalQueryGlobalAttention,
                           TrainableQueryGlobalAttention)):
        return QueryGlobalCache(
            window=module.window, feature_dim=module.feature_dim,
            gain=module.gain)
    return model.make_cache()[8]


def active_bytes(cache) -> int:
    if isinstance(cache, QueryGlobalCache):
        return cache.nbytes
    if cache.keys is None:
        return 0
    length = min(cache.offset, cache.keys.shape[2])
    return int(cache.keys[..., :length, :].nbytes +
               cache.values[..., :length, :].nbytes)


def layer_repeat(model, module, prefix, future) -> dict:
    cache = layer_cache(model, module)
    mx.reset_peak_memory()
    begun = time.perf_counter()
    output = module(prefix, mask="causal", cache=cache)
    mx.eval(output, cache.state)
    prefill_s = time.perf_counter() - begun
    prefill_bytes = active_bytes(cache)
    prefill_peak = int(mx.get_peak_memory())
    begun = time.perf_counter()
    for index in range(future.shape[1]):
        output = module(future[:, index:index + 1], cache=cache)
        mx.eval(output, cache.state)
    decode_s = time.perf_counter() - begun
    expected = prefix.shape[1] + future.shape[1]
    if cache.offset != expected:
        raise AssertionError("layer cache offset differs from prefill + decode")
    if isinstance(cache, QueryGlobalCache):
        if (cache.keys.shape[2] > cache.window - 1 or
                cache.global_count != (
                    max(0, expected - cache.window + 1) if cache.gain else 0)):
            raise AssertionError("bounded layer state failed long decode")
    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_ms_per_token": 1000 * decode_s / future.shape[1],
        "prefill_active_logical_cache_bytes": prefill_bytes,
        "final_active_logical_cache_bytes": active_bytes(cache),
        "prefill_peak_metal_bytes": prefill_peak,
        "final_peak_metal_bytes": int(mx.get_peak_memory()),
        "cache_offset": int(cache.offset),
        "host_load_average": list(os.getloadavg()),
    }


def main() -> None:
    opts = options()
    contexts = [int(value) for value in opts.contexts.split(",")]
    if (not (opts.snapshot / "config.json").is_file() or not contexts
            or min(contexts) < 128 or opts.decode_tokens < 1
            or opts.repeats < 1):
        raise ValueError("offline checkpoint, contexts and positive counts required")
    stage1 = json.loads(STAGE1_RESULT.read_text())
    weight_sha = hashlib.sha256(STAGE1_WEIGHTS.read_bytes()).hexdigest()
    if (stage1["status"] != "complete" or not stage1["meta"]["primary"]
            or stage1["weights"]["sha256"] != weight_sha):
        raise ValueError("selected complete stage-1 map required")
    with np.load(STAGE1_WEIGHTS) as checkpoint:
        delta_q = checkpoint["delta_q"].astype(np.float32)
        delta_k = checkpoint["delta_k"].astype(np.float32)
    begun = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    if max(contexts) + opts.decode_tokens >= len(ids):
        raise ValueError("requested context exceeds corpus")
    original = model.layers[8].self_attn
    local = LocalQueryGlobalAttention(original, gain=0.0)
    fixed = LocalQueryGlobalAttention(original, gain=DEFAULT_GAIN)
    trained = TrainableQueryGlobalAttention(original, gain=DEFAULT_GAIN)
    trained.delta_q = mx.array(delta_q)
    trained.delta_k = mx.array(delta_k)
    local.eval()
    fixed.eval()
    trained.eval()
    mx.eval(trained.delta_q, trained.delta_k)
    modules = {"teacher_layer": original, "local_layer": local,
               "fixed_query_layer": fixed, "trained_query_layer": trained}
    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": opts.snapshot.name,
            "config_sha256": hashlib.sha256(
                (opts.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (opts.snapshot / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "trainable_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "prefix_capture_source_sha256": hashlib.sha256(
                (HERE / "train_query_global_lora.py").read_bytes()).hexdigest(),
            "complete_model_timing_source_sha256": hashlib.sha256(
                (HERE / "benchmark_query_global.py").read_bytes()).hexdigest(),
            "stage1_result_sha256": hashlib.sha256(STAGE1_RESULT.read_bytes()).hexdigest(),
            "stage1_weights_sha256": weight_sha,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "contexts": contexts,
            "decode_tokens": opts.decode_tokens,
            "repeats": opts.repeats,
            "batch": 1,
            "attention_layer": 8,
            "workload": "captured genuine teacher layer-8 inputs; q/k/v/o/RoPE plus attention and output projection; whole-model last-token-head prefill and actual next-token decode",
            "limitation": "layer-only decode reuses the teacher's upstream activations, not candidate-generated upstream states; co-resident MLX process",
            "host_load_average_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
            "load_and_tokenize_s": time.perf_counter() - begun,
        },
        "rows": [],
    }
    atomic_json(opts.output, record)
    for context in contexts:
        begun = time.perf_counter()
        input_ids = mx.array(ids[:context + opts.decode_tokens][None, :],
                             dtype=mx.int32)
        hidden = frozen_prefix(model, input_ids)
        activated = model.layers[8].input_layernorm(hidden)
        mx.eval(activated)
        prefix = activated[:, :context]
        future = activated[:, context:context + opts.decode_tokens]
        capture_s = time.perf_counter() - begun
        for arm in ARMS:
            if arm == "teacher_full":
                whole_model_repeat(
                    model, ids, ids[context:context + opts.decode_tokens],
                    context=context, arm="teacher", batch=1)
            else:
                layer_repeat(model, modules[arm], prefix, future)
        row = {"context": context,
               "teacher_activation_capture_s": capture_s,
               "layer_input_shape": list(prefix.shape),
               "layer_input_dtype": str(prefix.dtype),
               "repeats": []}
        record["rows"].append(row)
        atomic_json(opts.output, record)
        for repeat in range(opts.repeats):
            rotation = (context // 512 + repeat) % len(ARMS)
            order = ARMS[rotation:] + ARMS[:rotation]
            readings = {}
            for arm in order:
                if arm == "teacher_full":
                    readings[arm] = whole_model_repeat(
                        model, ids, ids[context:context + opts.decode_tokens],
                        context=context, arm="teacher", batch=1)
                else:
                    readings[arm] = layer_repeat(
                        model, modules[arm], prefix, future)
            item = {"index": repeat, "order": list(order),
                    "readings": readings}
            for arm in ARMS[2:]:
                item[f"{arm}_prefill_ratio_to_teacher_layer"] = (
                    readings[arm]["prefill_s"] /
                    readings["teacher_layer"]["prefill_s"])
                item[f"{arm}_decode_speed_ratio_to_teacher_layer"] = (
                    readings["teacher_layer"]["decode_s"] /
                    readings[arm]["decode_s"])
            item["teacher_layer_prefill_fraction_of_full"] = (
                readings["teacher_layer"]["prefill_s"] /
                readings["teacher_full"]["prefill_s"])
            row["repeats"].append(item)
            atomic_json(opts.output, record)
            print(f"T={context} repeat={repeat} "
                  f"trained/teacher layer prefill="
                  f"{item['trained_query_layer_prefill_ratio_to_teacher_layer']:.3f} "
                  f"decode speed="
                  f"{item['trained_query_layer_decode_speed_ratio_to_teacher_layer']:.3f}",
                  flush=True)
        keys = [key for key in row["repeats"][0] if key.endswith(
            ("_ratio_to_teacher_layer", "_fraction_of_full"))]
        row["median"] = {
            key: statistics.median(item[key] for item in row["repeats"])
            for key in keys}
        atomic_json(opts.output, record)
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(opts.output, record)
    print("wrote", opts.output, flush=True)


if __name__ == "__main__":
    main()
