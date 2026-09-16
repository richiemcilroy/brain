"""Held-out attention-output and full-model quality after one-layer transfer.

The feature-map checkpoint is selected only by 55%/58% attention-output MSE.
New 70%/72%/74% 3,000-token windows test model quality and a 512-token
attention-output slice each. Previously used 96–99% windows are also reported
separately. Teacher, local-only, fixed untrained query-state and trained
query-state arms share the same original 1B checkpoint and persistent caches.
No LoRA or pretrained projection weights are trained in this stage.
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

from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_quality import cached_nll  # noqa: E402
from query_global_trainable import install_trainable_query_global  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as TRANSFER_RESULT, DEFAULT_WEIGHTS, mse_after_local,
    prepare,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_trained_quality.json"
NEW_PERCENTAGES = (70, 72, 74)
ITERATIVE_PERCENTAGES = (96, 97, 98, 99)
PERCENTAGES = NEW_PERCENTAGES + ITERATIVE_PERCENTAGES
WINDOW = 3000
OUTPUT_PREFIX = 512
ARMS = ("teacher", "local_only", "untrained_query", "trained_query")


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def mean_mse(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate(rows: dict, percentages) -> dict:
    subset = [rows[str(pct)] for pct in percentages]
    count = sum(row["scored_tokens"] for row in subset)
    nll = sum(row["nll"] * row["scored_tokens"] for row in subset) / count
    return {"scored_tokens": count, "nll": nll, "ppl": math.exp(nll)}


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {DEFAULT_SNAPSHOT}")
    transfer = json.loads(TRANSFER_RESULT.read_text())
    if not transfer["status"] == "complete" or not transfer["meta"]["primary"]:
        raise ValueError("complete primary attention-transfer artifact required")
    weights_sha = hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest()
    if weights_sha != transfer["weights"]["sha256"]:
        raise ValueError("selected feature-map checkpoint hash differs")
    with np.load(DEFAULT_WEIGHTS) as checkpoint:
        delta_q = checkpoint["delta_q"].astype(np.float32)
        delta_k = checkpoint["delta_k"].astype(np.float32)
    if delta_q.shape != (32, 64, 64) or delta_k.shape != (8, 64, 64):
        raise ValueError("feature checkpoint has wrong Llama-3.2-1B head shapes")

    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    starts = {str(pct): int(pct * len(ids) / 100) for pct in PERCENTAGES}
    if any(start + WINDOW > len(ids) for start in starts.values()):
        raise ValueError("held-out window exceeds corpus")
    if any(left + WINDOW > right for left, right in zip(
            sorted(starts.values()), sorted(starts.values())[1:])):
        raise ValueError("held-out windows overlap")
    windows = {pct: ids[start:start + WINDOW] for pct, start in starts.items()}
    original = model.layers[8].self_attn
    prepared = {
        str(pct): prepare(model, original, windows[str(pct)][:OUTPUT_PREFIX])
        for pct in NEW_PERCENTAGES
    }
    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "trainable_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "quality_harness_source_sha256": hashlib.sha256(
                (HERE / "query_global_quality.py").read_bytes()).hexdigest(),
            "transfer_harness_source_sha256": hashlib.sha256(
                (HERE / "train_query_global_transfer.py").read_bytes()).hexdigest(),
            "transfer_result_sha256": hashlib.sha256(
                TRANSFER_RESULT.read_bytes()).hexdigest(),
            "feature_weights_sha256": weights_sha,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "new_percentages": list(NEW_PERCENTAGES),
            "previously_inspected_percentages": list(ITERATIVE_PERCENTAGES),
            "window_tokens": WINDOW,
            "scored_tokens_per_window": WINDOW - 1,
            "attention_output_prefix_tokens": OUTPUT_PREFIX,
            "selected_gain": DEFAULT_GAIN,
            "quality_note": "fixed 200-step transfer checkpoint chosen on 55/58% output MSE; 70/72/74% quality windows newly evaluated here; 96-99% are iterative comparisons",
            "started_epoch": time.time(),
        },
        "windows": {
            pct: {
                "token_window": [start, start + WINDOW],
                "token_ids_sha256": hashlib.sha256(windows[pct].tobytes()).hexdigest(),
            }
            for pct, start in starts.items()
        },
        "conversion": None,
        "attention_output_mse": {arm: {} for arm in ARMS if arm != "teacher"},
        "rows": {arm: {} for arm in ARMS},
        "aggregate_new_70_74": {},
        "aggregate_iterative_96_99": {},
        "paired": {},
    }
    atomic_json(record)

    for pct, segment in windows.items():
        record["rows"]["teacher"][pct] = cached_nll(
            model, segment, teacher=True)
        atomic_json(record)
        print("teacher", pct, record["rows"]["teacher"][pct]["ppl"],
              flush=True)

    installed = install_trainable_query_global(
        model, layer=8, gain=DEFAULT_GAIN)
    replacement = installed.replacement
    record["conversion"] = installed.conversion
    zero_q = mx.zeros_like(replacement.delta_q)
    zero_k = mx.zeros_like(replacement.delta_k)
    try:
        for arm in ARMS[1:]:
            replacement.gain = 0.0 if arm == "local_only" else DEFAULT_GAIN
            if arm == "trained_query":
                replacement.delta_q = mx.array(delta_q)
                replacement.delta_k = mx.array(delta_k)
            else:
                replacement.delta_q = zero_q
                replacement.delta_k = zero_k
            mx.eval(replacement.delta_q, replacement.delta_k)
            for pct, (x, target) in prepared.items():
                loss = mse_after_local(replacement, x, target)
                mx.eval(loss)
                record["attention_output_mse"][arm][pct] = float(loss)
            atomic_json(record)
            for pct, segment in windows.items():
                record["rows"][arm][pct] = cached_nll(
                    model, segment, teacher=False)
                atomic_json(record)
                print(arm, pct, record["rows"][arm][pct]["ppl"],
                      flush=True)
    finally:
        installed.restore()

    for arm, rows in record["rows"].items():
        record["aggregate_new_70_74"][arm] = aggregate(
            rows, NEW_PERCENTAGES)
        record["aggregate_iterative_96_99"][arm] = aggregate(
            rows, ITERATIVE_PERCENTAGES)
    untrained = record["attention_output_mse"]["untrained_query"]
    trained = record["attention_output_mse"]["trained_query"]
    record["paired"] = {
        "output_mse_reduction_new_windows": {
            pct: 1.0 - trained[pct] / untrained[pct]
            for pct in prepared
        },
        "mean_output_mse_reduction_new_windows":
            1.0 - mean_mse(list(trained.values())) /
            mean_mse(list(untrained.values())),
        "trained_minus_untrained_nll_new_windows": {
            str(pct): (
                record["rows"]["trained_query"][str(pct)]["nll"] -
                record["rows"]["untrained_query"][str(pct)]["nll"])
            for pct in NEW_PERCENTAGES
        },
        "trained_beats_untrained_new_windows": sum(
            record["rows"]["trained_query"][str(pct)]["nll"] <
            record["rows"]["untrained_query"][str(pct)]["nll"]
            for pct in NEW_PERCENTAGES
        ),
        "trained_minus_teacher_nll_new_windows": {
            str(pct): (
                record["rows"]["trained_query"][str(pct)]["nll"] -
                record["rows"]["teacher"][str(pct)]["nll"])
            for pct in NEW_PERCENTAGES
        },
    }
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print("new aggregate", record["aggregate_new_70_74"],
          "output_mse_recovery", record["paired"][
              "mean_output_mse_reduction_new_windows"],
          flush=True)


if __name__ == "__main__":
    main()
