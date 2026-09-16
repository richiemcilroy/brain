"""Paired complete-model serving benchmark of one query-global 1B layer.

Teacher, exact local-only and fixed feature-global arms use one offline bf16
Llama checkpoint. Each timed call projects only the last prefill position to
the vocabulary and evaluates inside the timer. Decode is teacher-forced,
single-token and uses explicit persistent caches. Warmups, rotating arm order
and raw repeat readings are kept; this shared-host run is not a cost proof.
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
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import (  # noqa: E402
    DEFAULT_GAIN, QueryGlobalCache, install_query_global,
    make_query_global_cache,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "query_global_benchmark_b1.json"


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--contexts", default="512,2048,8192")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    parser.add_argument("--sync-blocks", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def cache_for(model, arm: str):
    return model.make_cache() if arm == "teacher" else make_query_global_cache(model)


def cache_bytes(caches, *, storage: bool) -> int:
    total = 0
    for slot in caches:
        if isinstance(slot, QueryGlobalCache):
            total += slot.nbytes
        elif getattr(slot, "keys", None) is not None:
            if storage:
                total += int(slot.keys.nbytes + slot.values.nbytes)
            else:
                active = min(int(slot.offset), int(slot.keys.shape[2]))
                total += int(slot.keys[..., :active, :].nbytes)
                total += int(slot.values[..., :active, :].nbytes)
    return total


def parity(model, ids: np.ndarray, arm: str, batch: int) -> dict:
    prefix = np.tile(ids[:128][None, :], (batch, 1))
    longer = np.tile(ids[:129][None, :], (batch, 1))
    whole_cache = cache_for(model, arm)
    whole = np.asarray(model_call(model, mx.array(prefix), whole_cache).astype(mx.float32))
    split_cache = cache_for(model, arm)
    model_call(model, mx.array(prefix[:, :64]), split_cache)
    split = np.asarray(model_call(
        model, mx.array(prefix[:, 64:]), split_cache).astype(mx.float32))
    longer_whole = np.asarray(model_call(
        model, mx.array(longer), cache_for(model, arm)).astype(mx.float32))
    decode_cache = cache_for(model, arm)
    model_call(model, mx.array(prefix), decode_cache)
    longer_decode = np.asarray(model_call(
        model, mx.array(longer[:, 128:]), decode_cache).astype(mx.float32))
    row = {
        "split_vs_whole_max_abs_logit": float(np.max(np.abs(split - whole))),
        "split_top1_equal": bool(np.array_equal(split.argmax(-1), whole.argmax(-1))),
        "decode_vs_whole_max_abs_logit": float(
            np.max(np.abs(longer_decode - longer_whole))),
        "decode_top1_equal": bool(np.array_equal(
            longer_decode.argmax(-1), longer_whole.argmax(-1))),
        "offsets_128": [int(slot.offset) for slot in split_cache],
        "offsets_129": [int(slot.offset) for slot in decode_cache],
    }
    if arm != "teacher":
        row["layer8_local_old_kv"] = int(decode_cache[8].keys.shape[2])
        row["layer8_global_count"] = int(decode_cache[8].global_count)
        expected = 129 - 63 if decode_cache[8].gain else 0
        if (row["layer8_local_old_kv"] > 63 or
                row["layer8_global_count"] != expected):
            raise AssertionError(f"{arm}: bounded state is wrong: {row}")
    if not (row["split_top1_equal"] and row["decode_top1_equal"]
            and all(x == 128 for x in row["offsets_128"])
            and all(x == 129 for x in row["offsets_129"])
            and max(row["split_vs_whole_max_abs_logit"],
                    row["decode_vs_whole_max_abs_logit"]) < 0.5):
        raise AssertionError(f"{arm}: cache parity failed: {row}")
    return row


def one_repeat(model, ids: np.ndarray, future: np.ndarray, *,
               context: int, arm: str, batch: int) -> dict:
    caches = cache_for(model, arm)
    prefix = np.tile(ids[:context][None, :], (batch, 1))
    mx.reset_peak_memory()
    started = time.perf_counter()
    model_call(model, mx.array(prefix), caches)
    prefill_s = time.perf_counter() - started
    prefill_cache = cache_bytes(caches, storage=False)
    prefill_storage = cache_bytes(caches, storage=True)
    prefill_peak = int(mx.get_peak_memory())
    started = time.perf_counter()
    for token in future:
        model_call(
            model,
            mx.array(np.full((batch, 1), int(token), dtype=np.int32)),
            caches,
        )
    decode_s = time.perf_counter() - started
    return {
        "prefill_s": prefill_s,
        "prefill_tok_s": batch * context / prefill_s,
        "decode_s": decode_s,
        "decode_tok_s": batch * len(future) / decode_s,
        "decode_ms_per_token": 1000 * decode_s / len(future),
        "prefill_active_logical_cache_bytes": prefill_cache,
        "final_active_logical_cache_bytes": cache_bytes(caches, storage=False),
        "prefill_cache_storage_bytes": prefill_storage,
        "final_cache_storage_bytes": cache_bytes(caches, storage=True),
        "prefill_peak_bytes": prefill_peak,
        "final_peak_bytes": int(mx.get_peak_memory()),
        "final_active_metal_bytes": int(mx.get_active_memory()),
        "cache_offsets": [int(slot.offset) for slot in caches],
        "host_load_average": list(os.getloadavg()),
    }


def main() -> None:
    opts = args()
    contexts = [int(x) for x in opts.contexts.split(",")]
    if (not (opts.snapshot / "config.json").is_file() or
            not contexts or min(contexts) < 128 or
            opts.decode_tokens < 1 or opts.repeats < 1 or opts.batch < 1 or
            not 0 < opts.gain <= 1 or opts.sync_blocks < 0):
        raise ValueError("offline checkpoint, contexts, counts and positive gain required")
    started = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    if max(contexts) + opts.decode_tokens >= len(ids):
        raise ValueError("requested context/decode exceeds corpus")
    installed = install_query_global(
        model, layer=8, gain=opts.gain,
        inference_sync_blocks=opts.sync_blocks)
    original = installed.original
    replacement = installed.replacement

    def switch(arm: str) -> None:
        if arm == "teacher":
            setattr(installed.layer, installed.attribute, original)
        elif arm == "local_only":
            replacement.gain = 0.0
            setattr(installed.layer, installed.attribute, replacement)
        elif arm == "query_global":
            replacement.gain = opts.gain
            setattr(installed.layer, installed.attribute, replacement)
        else:
            raise ValueError(arm)

    arms = ("teacher", "local_only", "query_global")
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
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "gain": opts.gain,
            "inference_sync_blocks": opts.sync_blocks,
            "layer": 8,
            "local_window": 64,
            "batch": opts.batch,
            "contexts": contexts,
            "decode_tokens": opts.decode_tokens,
            "repeats": opts.repeats,
            "workload": "serving-style last-token prefill vocabulary head; teacher-forced single-token decode; MLX evaluated inside timers",
            "memory_note": "all arms share one process; allocator peak is not deployment memory",
            "host_load_average_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
        },
        "conversion": installed.conversion,
        "parity": {},
        "rows": [],
    }
    atomic_json(opts.output, record)
    for arm in arms:
        switch(arm)
        record["parity"][arm] = parity(model, ids, arm, opts.batch)
        atomic_json(opts.output, record)
    for index, context in enumerate(contexts):
        future = ids[context:context + opts.decode_tokens]
        for arm in arms:
            switch(arm)
            one_repeat(model, ids, future, context=context,
                       arm=arm, batch=opts.batch)
        row = {"context": context, "batch": opts.batch, "repeats": []}
        record["rows"].append(row)
        atomic_json(opts.output, record)
        for repeat in range(opts.repeats):
            rotation = (index + repeat) % len(arms)
            order = arms[rotation:] + arms[:rotation]
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
                "local_prefill_time_ratio": (
                    readings["local_only"]["prefill_s"] /
                    readings["teacher"]["prefill_s"]),
                "query_prefill_time_ratio": (
                    readings["query_global"]["prefill_s"] /
                    readings["teacher"]["prefill_s"]),
                "local_decode_speed_ratio": (
                    readings["local_only"]["decode_tok_s"] /
                    readings["teacher"]["decode_tok_s"]),
                "query_decode_speed_ratio": (
                    readings["query_global"]["decode_tok_s"] /
                    readings["teacher"]["decode_tok_s"]),
            }
            row["repeats"].append(item)
            atomic_json(opts.output, record)
            print(
                f"T={context} repeat={repeat} "
                f"prefill query/teacher={item['query_prefill_time_ratio']:.3f} "
                f"decode query/teacher={item['query_decode_speed_ratio']:.3f}",
                flush=True)
        row["median"] = {
            key: statistics.median(item[key] for item in row["repeats"])
            for key in (
                "local_prefill_time_ratio", "query_prefill_time_ratio",
                "local_decode_speed_ratio", "query_decode_speed_ratio")
        }
        atomic_json(opts.output, record)
    record["status"] = "complete"
    record["meta"]["elapsed_s"] = time.perf_counter() - started
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(opts.output, record)
    print("wrote", opts.output, flush=True)


if __name__ == "__main__":
    main()
