"""Predeclared fresh cached-quality gate for matched rank-8 LoRA arms.

After stage-2 checkpoint selection on Shakespeare 55/58% windows, score
Shakespeare 80/82/84/86% and WikiText-2 raw test 15/35/55/75/95% windows.
No WikiText text enters either training stage or checkpoint selection.
Teacher, selected stage-1 query map, full-attention LoRA, local-only LoRA and
query-state LoRA use identical persistent-cache next-token scoring. This is
one-layer language-model quality, not a speed or broad-task verdict.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from benchmark_query_global import parity  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_lora import (  # noqa: E402
    ARMS as LORA_ARMS, build_arm, load_canonical,
)
from query_global_quality import cached_nll  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from query_global_wikitext_quality import (  # noqa: E402
    DATASET_URL, MEMBER, MEMBER_SHA256, ZIP, ZIP_SHA256,
)
from train_query_global_lora import (  # noqa: E402
    DEFAULT_OUTPUT as STAGE2_RESULT,
)
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_lora_holdout.json"
WINDOW = 3000
PERCENTAGES = {
    "shakespeare": (80, 82, 84, 86),
    "wikitext2_raw_test": (15, 35, 55, 75, 95),
}
ARMS = ("teacher", "stage1_query") + LORA_ARMS


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def fixed_windows(ids: np.ndarray, percentages) -> tuple[dict, dict]:
    starts = {str(pct): int(pct * len(ids) / 100)
              for pct in percentages}
    ordered = sorted(starts.values())
    if (ordered[-1] + WINDOW > len(ids) or
            any(left + WINDOW > right for left, right in zip(
                ordered, ordered[1:]))):
        raise ValueError("predeclared held-out windows overlap or exceed text")
    arrays = {pct: ids[start:start + WINDOW] for pct, start in starts.items()}
    meta = {pct: {"token_window": [start, start + WINDOW],
                  "token_ids_sha256": hashlib.sha256(
                      arrays[pct].tobytes()).hexdigest()}
            for pct, start in starts.items()}
    return arrays, meta


def aggregate(rows: dict) -> dict:
    count = sum(row["scored_tokens"] for row in rows.values())
    nll = sum(row["nll"] * row["scored_tokens"]
              for row in rows.values()) / count
    return {"scored_tokens": count, "nll": nll, "ppl": math.exp(nll)}


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
            or stage2["meta"]["weight_blob_id"] != stage1["meta"]["weight_blob_id"]):
        raise ValueError("two stages used different pretrained checkpoints")
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
            raise ValueError(f"selected {arm} LoRA weight file differs")
        with np.load(path) as checkpoint:
            lora_arrays[arm] = {key: checkpoint[key].astype(np.float32)
                                for key in checkpoint.files}
        lora_hashes[arm] = digest
    if not ZIP.is_file() or hashlib.sha256(ZIP.read_bytes()).hexdigest() != ZIP_SHA256:
        raise FileNotFoundError(f"frozen WikiText test zip differs: {ZIP}; {DATASET_URL}")
    with zipfile.ZipFile(ZIP) as archive:
        raw = archive.read(MEMBER)
    if hashlib.sha256(raw).hexdigest() != MEMBER_SHA256:
        raise ValueError("frozen WikiText test member differs")

    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    model.freeze()
    if (hashlib.sha256((DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest()
            != stage2["meta"]["config_sha256"]):
        raise ValueError("evaluation config differs from stage-2 model")
    datasets = {
        "shakespeare": np.asarray(tokenizer.encode(CORPUS.read_text()),
                                  dtype=np.int32),
        "wikitext2_raw_test": np.asarray(tokenizer.encode(raw.decode("utf-8")),
                                         dtype=np.int32),
    }
    windows = {}
    window_meta = {}
    for name, ids in datasets.items():
        windows[name], window_meta[name] = fixed_windows(
            ids, PERCENTAGES[name])
    original = model.layers[8].self_attn
    selected_stage1 = TrainableQueryGlobalAttention(original, gain=DEFAULT_GAIN)
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
            "lora_builder_source_sha256": hashlib.sha256(
                (HERE / "query_global_lora.py").read_bytes()).hexdigest(),
            "stage2_trainer_source_sha256": hashlib.sha256(
                (HERE / "train_query_global_lora.py").read_bytes()).hexdigest(),
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
            "wiki_test_member_sha256": MEMBER_SHA256,
            "wiki_dataset_url": DATASET_URL,
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "window_tokens": WINDOW,
            "scored_tokens_per_window": WINDOW - 1,
            "percentages_fixed_before_stage2_primary": {
                name: list(values) for name, values in PERCENTAGES.items()},
            "no_wikitext_training_or_selection": True,
            "quality_note": "fresh windows for stage-2 method; one-layer cached next-token quality only",
            "started_epoch": time.time(),
        },
        "windows": window_meta,
        "conversion": lora_meta,
        "cache_parity": {arm: {} for arm in ARMS},
        "rows": {
            dataset: {arm: {} for arm in ARMS}
            for dataset in datasets
        },
        "aggregate": {dataset: {} for dataset in datasets},
        "paired": {dataset: {} for dataset in datasets},
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
                    record["rows"][name][arm][pct] = cached_nll(
                        model, segment,
                        teacher=arm in ("teacher", "full_lora"))
                    atomic_json(record)
                    print(name, pct, arm,
                          f"ppl={record['rows'][name][arm][pct]['ppl']:.5f}",
                          flush=True)
    finally:
        model.layers[8].self_attn = original
    for name in datasets:
        for arm in ARMS:
            record["aggregate"][name][arm] = aggregate(
                record["rows"][name][arm])
        record["paired"][name] = {
            "query_lora_minus_teacher_nll_by_window": {
                pct: record["rows"][name]["query_lora"][pct]["nll"] -
                     record["rows"][name]["teacher"][pct]["nll"]
                for pct in windows[name]
            },
            "query_lora_minus_full_lora_nll_by_window": {
                pct: record["rows"][name]["query_lora"][pct]["nll"] -
                     record["rows"][name]["full_lora"][pct]["nll"]
                for pct in windows[name]
            },
            "query_lora_minus_local_lora_nll_by_window": {
                pct: record["rows"][name]["query_lora"][pct]["nll"] -
                     record["rows"][name]["local_lora"][pct]["nll"]
                for pct in windows[name]
            },
            "query_lora_minus_stage1_nll_by_window": {
                pct: record["rows"][name]["query_lora"][pct]["nll"] -
                     record["rows"][name]["stage1_query"][pct]["nll"]
                for pct in windows[name]
            },
        }
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(record)
    print("aggregate", record["aggregate"], flush=True)


if __name__ == "__main__":
    main()
