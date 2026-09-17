"""Exploratory fixed-gain diagnostic on previously inspected 1B windows.

Keep the published layer-8 feature map and pretrained model unchanged. Read
teacher, exact local-only and several fixed local/global output-mixing gains
on six windows already scored in the direct-map experiment. This can diagnose
gain sensitivity; the reused windows cannot serve as a fresh quality gate.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from benchmark_query_global import parity  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_alice_data import (  # noqa: E402
    BODY_SHA256 as ALICE_BODY_SHA256, RAW_SHA256 as ALICE_RAW_SHA256,
    alice_body,
)
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_broad_data import (  # noqa: E402
    MEMBERS, ZIP_SHA256, percentage_starts, wiki_raw, window_rows,
)
from query_global_quality import cached_nll  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as MAP_RESULT, DEFAULT_WEIGHTS as MAP_WEIGHTS,
    atomic_json, mse_after_local, prepare,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_gain_sweep.json"
WINDOW = 3000
OUTPUT_PREFIX = 512
PERCENTAGES = {
    "shakespeare": (52, 76),
    "wikitext2_raw_test": (8, 78),
    "alice_book_body": (10, 80),
}
GAINS = (0.0, 0.2, DEFAULT_GAIN, 0.65, 0.85, 1.0)


def aggregate(rows: dict) -> dict:
    count = sum(row["scored_tokens"] for row in rows.values())
    nll = sum(row["nll"] * row["scored_tokens"]
              for row in rows.values()) / count
    return {"scored_tokens": count, "nll": nll, "ppl": math.exp(nll)}


def main() -> None:
    published = json.loads(MAP_RESULT.read_text())
    weight_sha = hashlib.sha256(MAP_WEIGHTS.read_bytes()).hexdigest()
    if (published["status"] != "complete" or
            not published["meta"]["primary"] or
            published["weights"]["sha256"] != weight_sha):
        raise ValueError("complete published original map required")
    with np.load(MAP_WEIGHTS) as checkpoint:
        q = checkpoint["delta_q"].astype(np.float32)
        k = checkpoint["delta_k"].astype(np.float32)
    if q.shape != (32, 64, 64) or k.shape != (8, 64, 64):
        raise ValueError("original map has wrong 1B shapes")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    model.freeze()
    if (hashlib.sha256((DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest()
            != published["meta"]["config_sha256"] or
            (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name !=
            published["meta"]["weight_blob_id"]):
        raise ValueError("evaluation checkpoint differs from map training")
    datasets = {
        "shakespeare": np.asarray(
            tokenizer.encode(CORPUS.read_text()), dtype=np.int32),
        "wikitext2_raw_test": np.asarray(
            tokenizer.encode(wiki_raw("test")), dtype=np.int32),
        "alice_book_body": np.asarray(
            tokenizer.encode(alice_body()), dtype=np.int32),
    }
    windows, window_meta = {}, {}
    for name, ids in datasets.items():
        starts = percentage_starts(len(ids), PERCENTAGES[name], WINDOW)
        windows[name] = {
            pct: ids[start:start + WINDOW]
            for pct, start in starts.items()}
        window_meta[name] = window_rows(ids, starts, WINDOW)
    original = model.layers[8].self_attn
    prepared = {
        name: {
            pct: prepare(model, original, segment[:OUTPUT_PREFIX])
            for pct, segment in windows[name].items()}
        for name in datasets
    }
    module = TrainableQueryGlobalAttention(
        original, window=64, gain=DEFAULT_GAIN, chunk=64,
        inference_sync_blocks=8)
    module.delta_q = mx.array(q)
    module.delta_k = mx.array(k)
    module.eval()
    mx.eval(module.delta_q, module.delta_k)
    if (not np.array_equal(np.asarray(module.delta_q), q) or
            not np.array_equal(np.asarray(module.delta_k), k)):
        raise AssertionError("gain diagnostic changed selected map")

    labels = [str(gain) for gain in GAINS]
    record = {
        "status": "running",
        "meta": {
            "exploratory": True,
            "previously_inspected_quality_windows": True,
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": published["meta"]["config_sha256"],
            "weight_blob_id": published["meta"]["weight_blob_id"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "trainable_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "quality_harness_source_sha256": hashlib.sha256(
                (HERE / "query_global_quality.py").read_bytes()).hexdigest(),
            "map_result_sha256": hashlib.sha256(MAP_RESULT.read_bytes()).hexdigest(),
            "map_weights_sha256": weight_sha,
            "shakespeare_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "wiki_zip_sha256": ZIP_SHA256,
            "wiki_test_member_sha256": MEMBERS["test"][1],
            "alice_raw_sha256": ALICE_RAW_SHA256,
            "alice_body_sha256": ALICE_BODY_SHA256,
            "token_ids_sha256": {
                name: hashlib.sha256(ids.tobytes()).hexdigest()
                for name, ids in datasets.items()},
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "window_tokens": WINDOW,
            "attention_output_prefix_tokens": OUTPUT_PREFIX,
            "percentages_reused_from_direct_map_holdout": {
                name: list(pcts) for name, pcts in PERCENTAGES.items()},
            "gains": list(GAINS),
            "selected_gain": DEFAULT_GAIN,
            "no_gain_training_or_serving_speed_verdict": True,
            "started_epoch": time.time(),
        },
        "windows": window_meta,
        "cache_parity": {},
        "teacher_rows": {name: {} for name in datasets},
        "rows": {
            name: {label: {} for label in labels}
            for name in datasets},
        "teacher_aggregate": {},
        "aggregate": {name: {} for name in datasets},
        "attention_output_mse": {
            name: {label: {} for label in labels}
            for name in datasets},
        "paired": {name: {} for name in datasets},
        "peak_metal_bytes": None,
    }
    atomic_json(OUTPUT, record)
    mx.reset_peak_memory()
    try:
        record["cache_parity"]["teacher"] = parity(
            model, datasets["shakespeare"], "teacher", 1)
        atomic_json(OUTPUT, record)
        for name in datasets:
            for pct, segment in windows[name].items():
                record["teacher_rows"][name][pct] = cached_nll(
                    model, segment, teacher=True)
                atomic_json(OUTPUT, record)
        model.layers[8].self_attn = module
        for gain in GAINS:
            module.gain = gain
            label = str(gain)
            record["cache_parity"][label] = parity(
                model, datasets["shakespeare"], label, 1)
            atomic_json(OUTPUT, record)
            for name in datasets:
                for pct, (x, target) in prepared[name].items():
                    loss = mse_after_local(module, x, target)
                    mx.eval(loss)
                    record["attention_output_mse"][name][label][pct] = (
                        float(loss))
                atomic_json(OUTPUT, record)
                for pct, segment in windows[name].items():
                    started = time.perf_counter()
                    row = cached_nll(model, segment, teacher=False)
                    row["wall_s"] = time.perf_counter() - started
                    record["rows"][name][label][pct] = row
                    atomic_json(OUTPUT, record)
                    print(name, pct, label, f"ppl={row['ppl']:.5f}",
                          flush=True)
    finally:
        model.layers[8].self_attn = original
    for name in datasets:
        record["teacher_aggregate"][name] = aggregate(
            record["teacher_rows"][name])
        for label in labels:
            record["aggregate"][name][label] = aggregate(
                record["rows"][name][label])
        selected = record["aggregate"][name][str(DEFAULT_GAIN)]["nll"]
        teacher = record["teacher_aggregate"][name]["nll"]
        record["paired"][name] = {
            label: {
                "gain_minus_selected_nll":
                    record["aggregate"][name][label]["nll"] - selected,
                "gain_minus_teacher_nll":
                    record["aggregate"][name][label]["nll"] - teacher,
                "mean_attention_output_mse": sum(
                    record["attention_output_mse"][name][label].values()) /
                    len(windows[name]),
            }
            for label in labels
        }
    record["peak_metal_bytes"] = int(mx.get_peak_memory())
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(OUTPUT, record)
    print("paired", record["paired"], flush=True)


if __name__ == "__main__":
    main()
