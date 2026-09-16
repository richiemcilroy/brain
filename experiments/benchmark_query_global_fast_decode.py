"""Paired complete-model serving test of opt-in one-token query-state decode.

The frozen selected stage-1 feature map, pretrained Llama weights, layer 8,
64-token local window and 256/2 prefill are identical for both query arms.
Only the one-token cache update differs. Teacher fused attention supplies a
reference. Prefill projects the last token only; 32 teacher-forced one-token
decodes use persistent caches. Rotated paired orders and raw repeats expose
host noise. This is not a quality-matched or lower-cost model comparison.
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

from benchmark_local_hybrid import model_call  # noqa: E402
from benchmark_query_global import cache_for, one_repeat, parity  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_fast_decode import FastDecodeTrainableQueryGlobalAttention  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "query_global_fast_decode_benchmark_b1.json"
CONTEXTS = (8192, 32768)
DECODE = 32
REPEATS = {8192: 5, 32768: 3}
ARMS = ("teacher", "general_query", "fast_query")


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def paired_logits(model, ids: np.ndarray, switch) -> dict:
    results = {}
    prefix = mx.array(ids[:128][None, :])
    next_token = mx.array(ids[128:129][None, :])
    for arm in ARMS[1:]:
        switch(arm)
        cache = cache_for(model, arm)
        prefill = model_call(model, prefix, cache)
        decode = model_call(model, next_token, cache)
        results[arm] = (
            np.asarray(prefill.astype(mx.float32)),
            np.asarray(decode.astype(mx.float32)),
            int(cache[8].global_count),
        )
    general, faster = results["general_query"], results["fast_query"]
    row = {
        "prefill_max_abs_logit": float(np.max(np.abs(general[0] - faster[0]))),
        "decode_max_abs_logit": float(np.max(np.abs(general[1] - faster[1]))),
        "prefill_top1_equal": bool(np.array_equal(
            general[0].argmax(-1), faster[0].argmax(-1))),
        "decode_top1_equal": bool(np.array_equal(
            general[1].argmax(-1), faster[1].argmax(-1))),
        "global_counts": [general[2], faster[2]],
    }
    if (not row["prefill_top1_equal"] or not row["decode_top1_equal"]
            or max(row["prefill_max_abs_logit"],
                   row["decode_max_abs_logit"]) >= 0.5
            or row["global_counts"] != [66, 66]):
        raise AssertionError(f"complete-model fast parity failed: {row}")
    return row


def main() -> None:
    opts = options()
    if not (opts.snapshot / "config.json").is_file():
        raise ValueError("offline checkpoint required")
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
    if max(CONTEXTS) + DECODE >= len(ids):
        raise ValueError("context plus decode exceeds corpus")
    original = model.layers[8].self_attn
    modules = {
        "general_query": TrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=256,
            inference_sync_blocks=2),
        "fast_query": FastDecodeTrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=256,
            inference_sync_blocks=2),
    }
    for module in modules.values():
        module.delta_q = delta_q
        module.delta_k = delta_k
        module.eval()
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
            "fast_decode_source_sha256": hashlib.sha256(
                (HERE / "query_global_fast_decode.py").read_bytes()).hexdigest(),
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
            "chunk": 256,
            "inference_sync_blocks": 2,
            "converted_layer": 8,
            "batch": 1,
            "contexts": list(CONTEXTS),
            "decode_tokens": DECODE,
            "repeats_by_context": {str(k): v for k, v in REPEATS.items()},
            "workload": "complete model last-token-head prefill and actual next-token teacher-forced decode evaluated inside timers",
            "memory_note": "all arms co-resident in one MLX process; allocator peak is not deployment memory",
            "quality_note": "query arms are mutually equal; neither is quality-matched to teacher",
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
            "load_and_tokenize_s": time.perf_counter() - begun,
        },
        "parity": {},
        "paired_logits": {},
        "rows": [],
    }
    atomic_json(opts.output, record)
    try:
        for arm in ARMS:
            switch(arm)
            record["parity"][arm] = parity(model, ids, arm, 1)
            atomic_json(opts.output, record)
        record["paired_logits"] = paired_logits(model, ids, switch)
        atomic_json(opts.output, record)
        for context in CONTEXTS:
            future = ids[context:context + DECODE]
            for arm in ARMS:
                switch(arm)
                one_repeat(model, ids, future, context=context,
                           arm=arm, batch=1)
            row = {"context": context, "repeats": [], "median": {}}
            record["rows"].append(row)
            atomic_json(opts.output, record)
            for repeat in range(REPEATS[context]):
                rotation = repeat % len(ARMS)
                order = ARMS[rotation:] + ARMS[:rotation]
                readings = {}
                for arm in order:
                    switch(arm)
                    readings[arm] = one_repeat(
                        model, ids, future, context=context,
                        arm=arm, batch=1)
                item = {"index": repeat, "order": list(order),
                        "readings": readings,
                        "fast_decode_speed_ratio_to_general": (
                            readings["fast_query"]["decode_tok_s"] /
                            readings["general_query"]["decode_tok_s"]),
                        "fast_decode_speed_ratio_to_teacher": (
                            readings["fast_query"]["decode_tok_s"] /
                            readings["teacher"]["decode_tok_s"]),
                        "fast_prefill_time_ratio_to_general": (
                            readings["fast_query"]["prefill_s"] /
                            readings["general_query"]["prefill_s"]),
                        "fast_prefill_time_ratio_to_teacher": (
                            readings["fast_query"]["prefill_s"] /
                            readings["teacher"]["prefill_s"]),
                        }
                row["repeats"].append(item)
                atomic_json(opts.output, record)
                print(f"T={context} repeat={repeat} fast/general decode="
                      f"{item['fast_decode_speed_ratio_to_general']:.3f} "
                      f"fast/teacher decode="
                      f"{item['fast_decode_speed_ratio_to_teacher']:.3f}",
                      flush=True)
            row["median"] = {
                key: statistics.median(item[key] for item in row["repeats"])
                for key in ("fast_decode_speed_ratio_to_general",
                            "fast_decode_speed_ratio_to_teacher",
                            "fast_prefill_time_ratio_to_general",
                            "fast_prefill_time_ratio_to_teacher")
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
