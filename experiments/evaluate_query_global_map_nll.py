"""Fresh prediction-quality gate for direct adjustment of one query map.

Compare unchanged teacher, the published map and the direct-NLL-selected map
on predeclared Shakespeare, WikiText-2 raw test and pinned literary windows.
The last text is outside both adjustment corpora. No scored holdout enters
training or checkpoint choice.
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
    BODY_SHA256 as ALICE_BODY_SHA256, BOOK_PAGE as ALICE_BOOK_PAGE,
    RAW_SHA256 as ALICE_RAW_SHA256, URL as ALICE_URL, alice_body,
)
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_broad_data import (  # noqa: E402
    DATASET_CARD, DATASET_URL, MEMBERS, ZIP_SHA256, percentage_starts,
    wiki_raw, window_rows,
)
from query_global_quality import cached_nll  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_map_nll import (  # noqa: E402
    DEFAULT_RESULT as DIRECT_RESULT, DEFAULT_WEIGHTS as DIRECT_WEIGHTS,
    PROTOCOL, atomic_json,
)
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as OLD_RESULT, DEFAULT_WEIGHTS as OLD_WEIGHTS,
    mse_after_local, prepare,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_map_nll_holdout.json"
WINDOW = 3000
OUTPUT_PREFIX = 512
PERCENTAGES = {
    "shakespeare": (52, 69, 76),
    "wikitext2_raw_test": (8, 18, 38, 58, 78, 98),
    "alice_book_body": (10, 45, 80),
}
PREVIOUSLY_INSPECTED = {
    "shakespeare": (55, 58, 60, 64, 66, 68, 70, 72, 74, 75,
                    77, 80, 82, 84, 85, 86, 88, 90, 93, 95, 96,
                    97, 98, 99),
    "wikitext2_raw_test": (2, 5, 10, 15, 22, 25, 30, 35, 42, 45,
                           50, 55, 62, 65, 70, 75, 82, 85, 90, 95),
    "alice_book_body": (),
}
ARMS = ("teacher", "old_query", "direct_query")


def fixed_windows(ids: np.ndarray, name: str) -> tuple[dict, dict]:
    starts = percentage_starts(len(ids), PERCENTAGES[name], WINDOW)
    if PREVIOUSLY_INSPECTED[name]:
        previous = percentage_starts(
            len(ids), PREVIOUSLY_INSPECTED[name], WINDOW)
        for start in starts.values():
            for pct, prior in previous.items():
                prior_span = (8192 if name == "shakespeare" and pct in
                              ("55", "58", "60", "90") else WINDOW)
                if start < prior + prior_span and prior < start + WINDOW:
                    raise ValueError(
                        f"new {name} holdout overlaps inspected {pct}%")
    arrays = {
        pct: ids[start:start + WINDOW] for pct, start in starts.items()}
    return arrays, window_rows(ids, starts, WINDOW)


def aggregate(rows: dict) -> dict:
    count = sum(row["scored_tokens"] for row in rows.values())
    nll = sum(row["nll"] * row["scored_tokens"]
              for row in rows.values()) / count
    return {"scored_tokens": count, "nll": nll, "ppl": math.exp(nll)}


def decision(record: dict, training: dict) -> dict:
    before = training["baseline_selection"]["domains"]
    after = training["best"]["selection"]["domains"]
    selection_both = all(
        after[name]["nll"] < before[name]["nll"]
        for name in ("shakespeare", "wikitext_valid"))
    quality_all = all(
        record["aggregate"][name]["direct_query"]["nll"] <=
        record["aggregate"][name]["old_query"]["nll"]
        for name in PERCENTAGES)
    old_window_ok = all(
        difference <= 0.02
        for name in PERCENTAGES
        for difference in record["paired"][name][
            "direct_minus_old_nll_by_window"].values())
    teacher_aggregate_ok = all(
        record["aggregate"][name]["direct_query"]["nll"] -
        record["aggregate"][name]["teacher"]["nll"] <= 0.01
        for name in PERCENTAGES)
    teacher_window_ok = all(
        difference <= 0.03
        for name in PERCENTAGES
        for difference in record["paired"][name][
            "direct_minus_teacher_nll_by_window"].values())
    retained = selection_both and quality_all and old_window_ok
    return {
        "selection_nll_improved_both_texts": selection_both,
        "fresh_aggregate_nll_not_worse_than_old_all_three_texts": quality_all,
        "no_fresh_window_worse_than_old_plus_0_02": old_window_ok,
        "direct_map_retained_for_matched_lora_quality_attempt": retained,
        "direct_map_within_teacher_plus_0_01_all_three_texts": (
            teacher_aggregate_ok),
        "no_fresh_window_worse_than_teacher_plus_0_03": teacher_window_ok,
        "direct_map_teacher_quality_gate_pass": (
            retained and teacher_aggregate_ok and teacher_window_ok),
    }


def main() -> None:
    old = json.loads(OLD_RESULT.read_text())
    direct = json.loads(DIRECT_RESULT.read_text())
    old_weights_sha = hashlib.sha256(OLD_WEIGHTS.read_bytes()).hexdigest()
    direct_weights_sha = hashlib.sha256(DIRECT_WEIGHTS.read_bytes()).hexdigest()
    if (old["status"] != "complete" or not old["meta"]["primary"]
            or old["weights"]["sha256"] != old_weights_sha
            or direct["status"] != "complete" or not direct["meta"]["primary"]
            or direct["weights"]["sha256"] != direct_weights_sha
            or direct["meta"]["old_weights_sha256"] != old_weights_sha
            or direct["meta"]["old_result_sha256"] != hashlib.sha256(
                OLD_RESULT.read_bytes()).hexdigest()):
        raise ValueError("complete published old and direct-map runs required")
    if (direct["meta"]["source_sha256"] != hashlib.sha256(
                (HERE / "train_query_global_map_nll.py").read_bytes()).hexdigest()
            or direct["meta"]["protocol_sha256"] != hashlib.sha256(
                PROTOCOL.read_bytes()).hexdigest()
            or direct["meta"]["data_source_sha256"] != hashlib.sha256(
                (HERE / "query_global_broad_data.py").read_bytes()).hexdigest()
            or direct["meta"]["map_source_sha256"] != hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest()
            or direct["meta"]["attention_source_sha256"] != hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest()
            or direct["meta"]["prefix_and_nll_source_sha256"] != hashlib.sha256(
                (HERE / "train_query_global_lora.py").read_bytes()).hexdigest()
            or direct["meta"]["config_sha256"] != old["meta"]["config_sha256"]
            or direct["meta"]["weight_blob_id"] != old["meta"]["weight_blob_id"]):
        raise ValueError("predeclared source, protocol or model identity changed")
    weights = {}
    for arm, path in (("old_query", OLD_WEIGHTS),
                      ("direct_query", DIRECT_WEIGHTS)):
        with np.load(path) as checkpoint:
            q = checkpoint["delta_q"].astype(np.float32)
            k = checkpoint["delta_k"].astype(np.float32)
        if q.shape != (32, 64, 64) or k.shape != (8, 64, 64):
            raise ValueError(f"{arm} map has wrong 1B head shapes")
        weights[arm] = (q, k)

    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    model.freeze()
    if hashlib.sha256((DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest() != (
            direct["meta"]["config_sha256"]):
        raise ValueError("evaluation model differs from training model")
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
        windows[name], window_meta[name] = fixed_windows(ids, name)
    original = model.layers[8].self_attn
    prepared = {
        name: {
            pct: prepare(model, original, segment[:OUTPUT_PREFIX])
            for pct, segment in windows[name].items()}
        for name in datasets
    }
    modules = {"teacher": original}
    for arm in ("old_query", "direct_query"):
        module = TrainableQueryGlobalAttention(
            original, window=64, gain=DEFAULT_GAIN, chunk=64,
            inference_sync_blocks=8)
        module.delta_q = mx.array(weights[arm][0])
        module.delta_k = mx.array(weights[arm][1])
        module.eval()
        mx.eval(module.delta_q, module.delta_k)
        modules[arm] = module

    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": direct["meta"]["config_sha256"],
            "weight_blob_id": direct["meta"]["weight_blob_id"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": direct["meta"]["protocol_sha256"],
            "trainer_source_sha256": direct["meta"]["source_sha256"],
            "data_source_sha256": direct["meta"]["data_source_sha256"],
            "alice_data_source_sha256": hashlib.sha256(
                (HERE / "query_global_alice_data.py").read_bytes()).hexdigest(),
            "quality_harness_source_sha256": hashlib.sha256(
                (HERE / "query_global_quality.py").read_bytes()).hexdigest(),
            "cache_parity_harness_source_sha256": hashlib.sha256(
                (HERE / "benchmark_query_global.py").read_bytes()).hexdigest(),
            "old_result_sha256": hashlib.sha256(OLD_RESULT.read_bytes()).hexdigest(),
            "old_weights_sha256": old_weights_sha,
            "direct_result_sha256": hashlib.sha256(DIRECT_RESULT.read_bytes()).hexdigest(),
            "direct_weights_sha256": direct_weights_sha,
            "shakespeare_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "wiki_zip_sha256": ZIP_SHA256,
            "wiki_test_member_sha256": MEMBERS["test"][1],
            "wiki_dataset_url": DATASET_URL,
            "wiki_dataset_card": DATASET_CARD,
            "alice_raw_sha256": ALICE_RAW_SHA256,
            "alice_body_sha256": ALICE_BODY_SHA256,
            "alice_url": ALICE_URL,
            "alice_book_page": ALICE_BOOK_PAGE,
            "token_counts": {name: len(value) for name, value in datasets.items()},
            "token_ids_sha256": {
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in datasets.items()},
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "window_tokens": WINDOW,
            "scored_tokens_per_window": WINDOW - 1,
            "attention_output_prefix_tokens": OUTPUT_PREFIX,
            "new_percentages": {
                name: list(values) for name, values in PERCENTAGES.items()},
            "previously_inspected_percentages": {
                name: list(values)
                for name, values in PREVIOUSLY_INSPECTED.items()},
            "no_fresh_quality_used_in_adjustment_or_selection": True,
            "quality_note": "one converted layer; Alice is new to adjustment but may be known to pretrained teacher; Shakespeare gaps near earlier inspected windows",
            "started_epoch": time.time(),
        },
        "windows": window_meta,
        "cache_parity": {},
        "attention_output_mse": {
            name: {arm: {} for arm in ARMS[1:]}
            for name in datasets},
        "rows": {
            name: {arm: {} for arm in ARMS}
            for name in datasets},
        "aggregate": {name: {} for name in datasets},
        "paired": {name: {} for name in datasets},
        "decision": {},
    }
    atomic_json(OUTPUT, record)
    try:
        for arm in ARMS:
            model.layers[8].self_attn = modules[arm]
            record["cache_parity"][arm] = parity(
                model, datasets["shakespeare"],
                "teacher" if arm == "teacher" else arm, 1)
            atomic_json(OUTPUT, record)
            if arm != "teacher":
                for name in datasets:
                    for pct, (x, target) in prepared[name].items():
                        loss = mse_after_local(modules[arm], x, target)
                        mx.eval(loss)
                        record["attention_output_mse"][name][arm][pct] = (
                            float(loss))
                atomic_json(OUTPUT, record)
            for name in datasets:
                for pct, segment in windows[name].items():
                    started = time.perf_counter()
                    row = cached_nll(
                        model, segment, teacher=arm == "teacher")
                    row["wall_s"] = time.perf_counter() - started
                    record["rows"][name][arm][pct] = row
                    atomic_json(OUTPUT, record)
                    print(name, pct, arm, f"ppl={row['ppl']:.5f}",
                          flush=True)
    finally:
        model.layers[8].self_attn = original
    for name in datasets:
        for arm in ARMS:
            record["aggregate"][name][arm] = aggregate(
                record["rows"][name][arm])
        record["paired"][name] = {
            f"direct_minus_{label}_nll_by_window": {
                pct: record["rows"][name]["direct_query"][pct]["nll"] -
                     record["rows"][name][arm][pct]["nll"]
                for pct in windows[name]}
            for label, arm in (("teacher", "teacher"),
                               ("old", "old_query"))
        }
        for label, arm in (("teacher", "teacher"),
                           ("old", "old_query")):
            record["paired"][name][f"direct_minus_{label}_aggregate_nll"] = (
                record["aggregate"][name]["direct_query"]["nll"] -
                record["aggregate"][name][arm]["nll"])
    record["decision"] = decision(record, direct)
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(OUTPUT, record)
    print("aggregate", record["aggregate"],
          "decision", record["decision"], flush=True)


if __name__ == "__main__":
    main()
