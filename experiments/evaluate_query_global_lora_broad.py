"""Fresh cached-quality gate for the predeclared larger-data LoRA run.

Score unchanged teacher, selected stage-1 query map, and matched
full/local/query LoRA checkpoints on disjoint new Shakespeare and WikiText-2
raw test windows. No held-out score enters training or checkpoint choice.
This one-layer next-token test is separate from serving/training speed.
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
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_broad_data import (  # noqa: E402
    DATASET_CARD, DATASET_URL, MEMBERS, ZIP_SHA256, percentage_starts,
    wiki_raw, window_rows,
)
from query_global_lora import (  # noqa: E402
    ARMS as LORA_ARMS, build_arm, load_canonical,
)
from query_global_quality import cached_nll  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_lora_broad import DEFAULT_OUTPUT as STAGE2_RESULT  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_lora_broad_holdout.json"
PROTOCOL = HERE.parent / "docs" / "QUERY_GLOBAL_LORA_BROAD_PROTOCOL.md"
WINDOW = 3000
PERCENTAGES = {
    "shakespeare": (77, 88, 93),
    "wikitext2_raw_test": (5, 25, 45, 65, 85),
}
PREVIOUSLY_INSPECTED = {
    "shakespeare": (70, 72, 74, 75, 80, 82, 84, 85, 86,
                    90, 95, 96, 97, 98, 99),
    "wikitext2_raw_test": (10, 15, 30, 35, 50, 55, 70, 75, 90, 95),
}
ARMS = ("teacher", "stage1_query") + LORA_ARMS


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def fixed_windows(ids: np.ndarray, name: str) -> tuple[dict, dict]:
    starts = percentage_starts(len(ids), PERCENTAGES[name], WINDOW)
    old = percentage_starts(len(ids), PREVIOUSLY_INSPECTED[name], WINDOW)
    for start in starts.values():
        if any(start < previous + (
                   8192 if name == "shakespeare" and pct == "90" else WINDOW)
               and previous < start + WINDOW
               for pct, previous in old.items()):
            raise ValueError("new holdout overlaps a previously inspected window")
    arrays = {pct: ids[start:start + WINDOW]
              for pct, start in starts.items()}
    return arrays, window_rows(ids, starts, WINDOW)


def aggregate(rows: dict) -> dict:
    count = sum(row["scored_tokens"] for row in rows.values())
    nll = sum(row["nll"] * row["scored_tokens"]
              for row in rows.values()) / count
    return {"scored_tokens": count, "nll": nll, "ppl": math.exp(nll)}


def decision(record: dict) -> dict:
    full_ok = all(
        record["aggregate"][name]["full_lora"]["nll"] -
        record["aggregate"][name]["teacher"]["nll"] <= 0.01
        for name in PERCENTAGES)
    query_aggregate_ok = all(
        record["aggregate"][name]["query_lora"]["nll"] -
        record["aggregate"][name][comparison]["nll"] <= 0.01
        for name in PERCENTAGES
        for comparison in ("teacher", "full_lora"))
    query_window_ok = all(
        difference <= 0.03
        for name in PERCENTAGES
        for difference in record["paired"][name][
            "query_minus_teacher_nll_by_window"].values())
    return {
        "full_attention_control_within_teacher_plus_0_01_both_texts": full_ok,
        "query_within_teacher_and_full_plus_0_01_both_texts": query_aggregate_ok,
        "query_no_window_worse_than_teacher_plus_0_03": query_window_ok,
        "query_ready_for_multilayer_quality_attempt": (
            full_ok and query_aggregate_ok and query_window_ok),
        "interpretation_if_full_control_fails": (
            "general training-recipe overfit; query-specific failure inconclusive"
            if not full_ok else None),
    }


def main() -> None:
    stage1 = json.loads(STAGE1_RESULT.read_text())
    stage2 = json.loads(STAGE2_RESULT.read_text())
    stage1_weight_sha = hashlib.sha256(STAGE1_WEIGHTS.read_bytes()).hexdigest()
    if (stage1["status"] != "complete" or not stage1["meta"]["primary"]
            or stage1["weights"]["sha256"] != stage1_weight_sha
            or stage2["status"] != "complete" or not stage2["meta"]["primary"]
            or stage2["meta"]["stage1_weights_sha256"] != stage1_weight_sha):
        raise ValueError("complete primary selected stage-1/stage-2 records required")
    if (stage2["meta"]["config_sha256"] != stage1["meta"]["config_sha256"]
            or stage2["meta"]["weight_blob_id"] != stage1["meta"]["weight_blob_id"]
            or stage2["meta"]["source_sha256"] != hashlib.sha256(
                (HERE / "train_query_global_lora_broad.py").read_bytes()).hexdigest()
            or stage2["meta"]["protocol_sha256"] != hashlib.sha256(
                PROTOCOL.read_bytes()).hexdigest()
            or stage2["meta"]["data_source_sha256"] != hashlib.sha256(
                (HERE / "query_global_broad_data.py").read_bytes()).hexdigest()):
        raise ValueError("model or predeclared stage-2 sources changed")
    with np.load(STAGE1_WEIGHTS) as checkpoint:
        stage1_q = checkpoint["delta_q"].astype(np.float32)
        stage1_k = checkpoint["delta_k"].astype(np.float32)
    if stage1_q.shape != (32, 64, 64) or stage1_k.shape != (8, 64, 64):
        raise ValueError("stage-1 map shapes differ from 1B checkpoint")
    lora_arrays = {}
    lora_hashes = {}
    for arm in LORA_ARMS:
        info = stage2["weights"][arm]
        path = Path(info["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise ValueError(f"selected {arm} LoRA checkpoint differs")
        with np.load(path) as checkpoint:
            lora_arrays[arm] = {key: checkpoint[key].astype(np.float32)
                                for key in checkpoint.files}
        lora_hashes[arm] = digest

    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    model.freeze()
    if hashlib.sha256((DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest() != (
            stage2["meta"]["config_sha256"]):
        raise ValueError("evaluation model config differs from training")
    datasets = {
        "shakespeare": np.asarray(tokenizer.encode(CORPUS.read_text()),
                                  dtype=np.int32),
        "wikitext2_raw_test": np.asarray(
            tokenizer.encode(wiki_raw("test")), dtype=np.int32),
    }
    windows = {}
    window_meta = {}
    for name, ids in datasets.items():
        windows[name], window_meta[name] = fixed_windows(ids, name)
    original = model.layers[8].self_attn
    selected_stage1 = TrainableQueryGlobalAttention(
        original, gain=DEFAULT_GAIN)
    selected_stage1.delta_q = mx.array(stage1_q)
    selected_stage1.delta_k = mx.array(stage1_k)
    selected_stage1.eval()
    modules = {"teacher": original, "stage1_query": selected_stage1}
    lora_meta = {}
    for arm in LORA_ARMS:
        modules[arm], lora_meta[arm] = build_arm(
            original, model.args, arm, stage1_q=stage1_q,
            stage1_k=stage1_k)
        load_canonical(modules[arm], lora_arrays[arm])
        modules[arm].eval()

    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": stage2["meta"]["config_sha256"],
            "weight_blob_id": stage2["meta"]["weight_blob_id"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": stage2["meta"]["protocol_sha256"],
            "data_source_sha256": hashlib.sha256(
                (HERE / "query_global_broad_data.py").read_bytes()).hexdigest(),
            "lora_builder_source_sha256": hashlib.sha256(
                (HERE / "query_global_lora.py").read_bytes()).hexdigest(),
            "trainer_source_sha256": stage2["meta"]["source_sha256"],
            "quality_harness_source_sha256": hashlib.sha256(
                (HERE / "query_global_quality.py").read_bytes()).hexdigest(),
            "cache_parity_harness_sha256": hashlib.sha256(
                (HERE / "benchmark_query_global.py").read_bytes()).hexdigest(),
            "stage1_result_sha256": hashlib.sha256(STAGE1_RESULT.read_bytes()).hexdigest(),
            "stage1_weights_sha256": stage1_weight_sha,
            "stage2_result_sha256": hashlib.sha256(STAGE2_RESULT.read_bytes()).hexdigest(),
            "lora_weight_sha256_by_arm": lora_hashes,
            "shakespeare_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "wiki_zip_sha256": ZIP_SHA256,
            "wiki_test_member_sha256": MEMBERS["test"][1],
            "wiki_dataset_url": DATASET_URL,
            "wiki_dataset_card": DATASET_CARD,
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "window_tokens": WINDOW,
            "scored_tokens_per_window": WINDOW - 1,
            "percentages_fixed_before_primary_stage2": {
                name: list(values) for name, values in PERCENTAGES.items()},
            "previously_inspected_percentages": {
                name: list(values) for name, values in PREVIOUSLY_INSPECTED.items()},
            "wiki_test_not_used_for_training_or_selection": True,
            "quality_note": "new windows within a previously inspected WikiText test split; one-layer cached next-token quality only",
            "started_epoch": time.time(),
        },
        "windows": window_meta,
        "conversion": lora_meta,
        "cache_parity": {arm: {} for arm in ARMS},
        "rows": {
            dataset: {arm: {} for arm in ARMS}
            for dataset in datasets},
        "aggregate": {dataset: {} for dataset in datasets},
        "paired": {dataset: {} for dataset in datasets},
        "decision": {},
    }
    atomic_json(record)
    try:
        for arm in ARMS:
            model.layers[8].self_attn = modules[arm]
            parity_arm = "teacher" if arm in ("teacher", "full_lora") else arm
            record["cache_parity"][arm] = parity(
                model, datasets["shakespeare"], parity_arm, 1)
            atomic_json(record)
            for name in datasets:
                for pct, segment in windows[name].items():
                    started = time.perf_counter()
                    row = cached_nll(
                        model, segment,
                        teacher=arm in ("teacher", "full_lora"))
                    row["wall_s"] = time.perf_counter() - started
                    record["rows"][name][arm][pct] = row
                    atomic_json(record)
                    print(name, pct, arm, f"ppl={row['ppl']:.5f}",
                          flush=True)
    finally:
        model.layers[8].self_attn = original
    for name in datasets:
        for arm in ARMS:
            record["aggregate"][name][arm] = aggregate(
                record["rows"][name][arm])
        record["paired"][name] = {
            f"query_minus_{other}_nll_by_window": {
                pct: record["rows"][name]["query_lora"][pct]["nll"] -
                     record["rows"][name][other][pct]["nll"]
                for pct in windows[name]}
            for other in ("teacher", "full_lora", "local_lora",
                          "stage1_query")
        }
    record["decision"] = decision(record)
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(record)
    print("aggregate", record["aggregate"],
          "decision", record["decision"], flush=True)


if __name__ == "__main__":
    main()
