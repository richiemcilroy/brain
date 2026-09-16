"""Complete-model paired timing for the 64/8 and 256/2 scan settings.

Both query arms use the same selected stage-1 map, teacher q/k/v/o/RoPE,
64-token local window and fixed gain. Each timed prefill projects only the
last position to the vocabulary; 32 teacher-forced single-token decodes use
persistent caches. This is a serving-style engineering comparison, not a
quality-matched or lower-cost model verdict.
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

from benchmark_query_global import one_repeat, parity  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "query_global_opt_benchmark_b1.json"
ARMS = ("teacher", "base_query", "tuned_query")
CONFIGS = {"base_query": (64, 8), "tuned_query": (256, 2)}


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
        raise ValueError("complete selected stage-1 checkpoint required")
    with np.load(STAGE1_WEIGHTS) as checkpoint:
        delta_q = mx.array(checkpoint["delta_q"].astype(np.float32))
        delta_k = mx.array(checkpoint["delta_k"].astype(np.float32))
    begun = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    if max(contexts) + opts.decode_tokens >= len(ids):
        raise ValueError("context plus decode exceeds corpus")
    original = model.layers[8].self_attn
    modules = {}
    for arm, (chunk, sync) in CONFIGS.items():
        module = TrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=chunk,
            inference_sync_blocks=sync)
        module.delta_q = delta_q
        module.delta_k = delta_k
        module.eval()
        modules[arm] = module
    mx.eval(delta_q, delta_k)

    def switch(arm: str) -> None:
        model.layers[8].self_attn = original if arm == "teacher" else modules[arm]

    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": opts.snapshot.name,
            "config_sha256": hashlib.sha256(
                (opts.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (opts.snapshot / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "timing_harness_sha256": hashlib.sha256(
                (HERE / "benchmark_query_global.py").read_bytes()).hexdigest(),
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
            "gain": DEFAULT_GAIN,
            "local_window": 64,
            "configs": {arm: {"chunk": chunk,
                              "inference_sync_blocks": sync,
                              "tokens_per_sync": chunk * sync}
                        for arm, (chunk, sync) in CONFIGS.items()},
            "batch": 1,
            "contexts": contexts,
            "decode_tokens": opts.decode_tokens,
            "repeats": opts.repeats,
            "workload": "complete-model last-token-head prefill and actual next-token teacher-forced decode, evaluated inside timers",
            "memory_note": "all arms in one MLX process; allocator peak is not deployment memory",
            "quality_note": "both converted arms share one selected map, but neither is quality matched to teacher",
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
            "load_and_tokenize_s": time.perf_counter() - begun,
        },
        "parity": {},
        "rows": [],
    }
    atomic_json(opts.output, record)
    try:
        for arm in ARMS:
            switch(arm)
            record["parity"][arm] = parity(model, ids, arm, 1)
            atomic_json(opts.output, record)
        for context in contexts:
            future = ids[context:context + opts.decode_tokens]
            for arm in ARMS:
                switch(arm)
                one_repeat(model, ids, future, context=context,
                           arm=arm, batch=1)
            row = {"context": context, "repeats": [], "median": {}}
            record["rows"].append(row)
            atomic_json(opts.output, record)
            for repeat in range(opts.repeats):
                rotation = (context // 512 + repeat) % len(ARMS)
                order = ARMS[rotation:] + ARMS[:rotation]
                readings = {}
                for arm in order:
                    switch(arm)
                    readings[arm] = one_repeat(
                        model, ids, future, context=context,
                        arm=arm, batch=1)
                item = {"index": repeat, "order": list(order),
                        "readings": readings}
                for arm in ("base_query", "tuned_query"):
                    item[f"{arm}_prefill_time_ratio_to_teacher"] = (
                        readings[arm]["prefill_s"] /
                        readings["teacher"]["prefill_s"])
                    item[f"{arm}_decode_speed_ratio_to_teacher"] = (
                        readings[arm]["decode_tok_s"] /
                        readings["teacher"]["decode_tok_s"])
                item["tuned_prefill_time_ratio_to_base"] = (
                    readings["tuned_query"]["prefill_s"] /
                    readings["base_query"]["prefill_s"])
                item["tuned_decode_speed_ratio_to_base"] = (
                    readings["tuned_query"]["decode_tok_s"] /
                    readings["base_query"]["decode_tok_s"])
                row["repeats"].append(item)
                atomic_json(opts.output, record)
                print(f"T={context} repeat={repeat} tuned/base prefill="
                      f"{item['tuned_prefill_time_ratio_to_base']:.3f} "
                      f"tuned/teacher prefill="
                      f"{item['tuned_query_prefill_time_ratio_to_teacher']:.3f}",
                      flush=True)
            row["median"] = {
                key: statistics.median(item[key] for item in row["repeats"])
                for key in row["repeats"][0] if key.endswith(
                    ("_to_teacher", "_to_base"))
            }
            atomic_json(opts.output, record)
    finally:
        switch("teacher")
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(opts.output, record)
    print("wrote", opts.output, flush=True)


if __name__ == "__main__":
    main()
