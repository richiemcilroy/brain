"""Paired serving benchmark of the fixed and stage-1-trained query state.

Teacher, zero-residual query state and the selected 200-step feature-map
checkpoint share one offline bf16 Llama. This measures complete-model prefill
and teacher-forced decode at batch 1, with explicit bounded caches and rotating
arm order. It is not a quality-matched or deployment-cost comparison.
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
from query_global_trainable import install_trainable_query_global  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as TRANSFER_RESULT, DEFAULT_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "query_global_trained_benchmark_sync8_b1.json"
ARMS = ("teacher", "fixed_query", "trained_query")


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--contexts", default="512,2048,8192")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sync-blocks", type=int, default=8)
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
            or opts.repeats < 1 or opts.batch < 1 or opts.sync_blocks < 0):
        raise ValueError("offline checkpoint, contexts, counts and sync required")
    transfer = json.loads(TRANSFER_RESULT.read_text())
    weight_sha = hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest()
    if (transfer["status"] != "complete" or not transfer["meta"]["primary"]
            or transfer["weights"]["sha256"] != weight_sha):
        raise ValueError("complete selected primary feature-map checkpoint required")
    with np.load(DEFAULT_WEIGHTS) as checkpoint:
        delta_q = checkpoint["delta_q"].astype(np.float32)
        delta_k = checkpoint["delta_k"].astype(np.float32)
    if delta_q.shape != (32, 64, 64) or delta_k.shape != (8, 64, 64):
        raise ValueError("feature-map checkpoint does not fit Llama-3.2-1B")

    started = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    if max(contexts) + opts.decode_tokens >= len(ids):
        raise ValueError("context and decode exceed corpus")
    installed = install_trainable_query_global(
        model, layer=8, gain=DEFAULT_GAIN,
        inference_sync_blocks=opts.sync_blocks)
    original, replacement = installed.original, installed.replacement
    zero_q = mx.zeros_like(replacement.delta_q)
    zero_k = mx.zeros_like(replacement.delta_k)
    selected_q = mx.array(delta_q)
    selected_k = mx.array(delta_k)
    mx.eval(zero_q, zero_k, selected_q, selected_k)

    def switch(arm: str) -> None:
        if arm == "teacher":
            setattr(installed.layer, installed.attribute, original)
        elif arm in ("fixed_query", "trained_query"):
            replacement.delta_q = zero_q if arm == "fixed_query" else selected_q
            replacement.delta_k = zero_k if arm == "fixed_query" else selected_k
            setattr(installed.layer, installed.attribute, replacement)
        else:
            raise ValueError(arm)

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
            "transfer_result_sha256": hashlib.sha256(
                TRANSFER_RESULT.read_bytes()).hexdigest(),
            "feature_weights_sha256": weight_sha,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "fixed_gain": DEFAULT_GAIN,
            "inference_sync_blocks": opts.sync_blocks,
            "batch": opts.batch,
            "contexts": contexts,
            "decode_tokens": opts.decode_tokens,
            "repeats": opts.repeats,
            "workload": "complete-model last-token prefill and 32 teacher-forced single-token decodes; evaluated inside timers",
            "memory_note": "all arms coexist in one process; allocator peak is not deployment resident memory",
            "host_load_average_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
        },
        "conversion": installed.conversion,
        "parity": {},
        "rows": [],
    }
    atomic_json(opts.output, record)
    for arm in ARMS:
        switch(arm)
        record["parity"][arm] = parity(model, ids, arm, opts.batch)
        atomic_json(opts.output, record)
    for index, context in enumerate(contexts):
        future = ids[context:context + opts.decode_tokens]
        for arm in ARMS:
            switch(arm)
            one_repeat(model, ids, future, context=context,
                       arm=arm, batch=opts.batch)
        row = {"context": context, "batch": opts.batch, "repeats": []}
        record["rows"].append(row)
        atomic_json(opts.output, record)
        for repeat in range(opts.repeats):
            rotation = (index + repeat) % len(ARMS)
            order = ARMS[rotation:] + ARMS[:rotation]
            readings = {}
            for arm in order:
                switch(arm)
                readings[arm] = one_repeat(
                    model, ids, future, context=context,
                    arm=arm, batch=opts.batch)
            item = {
                "index": repeat,
                "order": list(order),
                "readings": readings,
                "fixed_prefill_time_ratio": (
                    readings["fixed_query"]["prefill_s"] /
                    readings["teacher"]["prefill_s"]),
                "trained_prefill_time_ratio": (
                    readings["trained_query"]["prefill_s"] /
                    readings["teacher"]["prefill_s"]),
                "fixed_decode_speed_ratio": (
                    readings["fixed_query"]["decode_tok_s"] /
                    readings["teacher"]["decode_tok_s"]),
                "trained_decode_speed_ratio": (
                    readings["trained_query"]["decode_tok_s"] /
                    readings["teacher"]["decode_tok_s"]),
            }
            row["repeats"].append(item)
            atomic_json(opts.output, record)
            print(
                f"T={context} repeat={repeat} "
                f"trained prefill/teacher={item['trained_prefill_time_ratio']:.3f} "
                f"trained decode/teacher={item['trained_decode_speed_ratio']:.3f}",
                flush=True)
        row["median"] = {
            key: statistics.median(item[key] for item in row["repeats"])
            for key in (
                "fixed_prefill_time_ratio", "trained_prefill_time_ratio",
                "fixed_decode_speed_ratio", "trained_decode_speed_ratio")
        }
        atomic_json(opts.output, record)
    switch("teacher")
    record["status"] = "complete"
    record["meta"]["elapsed_s"] = time.perf_counter() - started
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(opts.output, record)
    print("wrote", opts.output, flush=True)


if __name__ == "__main__":
    main()
