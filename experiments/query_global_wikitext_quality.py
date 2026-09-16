"""Cross-domain WikiText-2 raw test for the Shakespeare-trained feature map.

The official Salesforce WikiText-2 raw test text is read from a frozen zip
mirror used by ggml-org CI. Five disjoint 3,000-token windows at 10/30/50/
70/90% were fixed before inspecting outcomes. No WikiText text is used in
teacher-transfer training or checkpoint selection. Teacher, local-only,
untrained query state and the selected feature-map checkpoint are compared
with identical persistent-cache scoring. This is an independent text domain,
not an LM-eval suite or quality-matched speed result.
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

from oss_model_benchmark import DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_quality import cached_nll  # noqa: E402
from query_global_trainable import install_trainable_query_global  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as TRANSFER_RESULT, DEFAULT_WEIGHTS, mse_after_local,
    prepare,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_wikitext_quality.json"
ZIP = Path("/tmp/human_brain_wikitext2_raw.zip")
ZIP_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
MEMBER = "wikitext-2-raw/wiki.test.raw"
MEMBER_SHA256 = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
DATASET_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip"
DATASET_CARD = "https://huggingface.co/datasets/Salesforce/wikitext"
PERCENTAGES = (10, 30, 50, 70, 90)
WINDOW = 3000
OUTPUT_PREFIX = 512
ARMS = ("teacher", "local_only", "untrained_query", "trained_query")


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(OUTPUT)


def aggregate(rows: dict) -> dict:
    count = sum(x["scored_tokens"] for x in rows.values())
    nll = sum(x["nll"] * x["scored_tokens"] for x in rows.values()) / count
    return {"scored_tokens": count, "nll": nll, "ppl": math.exp(nll)}


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {DEFAULT_SNAPSHOT}")
    if not ZIP.is_file() or hashlib.sha256(ZIP.read_bytes()).hexdigest() != ZIP_SHA256:
        raise FileNotFoundError(
            f"frozen WikiText zip missing or differs: {ZIP}; download {DATASET_URL}")
    with zipfile.ZipFile(ZIP) as archive:
        raw = archive.read(MEMBER)
    if hashlib.sha256(raw).hexdigest() != MEMBER_SHA256:
        raise ValueError("WikiText test member differs from frozen bytes")
    transfer = json.loads(TRANSFER_RESULT.read_text())
    weights_sha = hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest()
    if (transfer["status"] != "complete" or not transfer["meta"]["primary"]
            or transfer["weights"]["sha256"] != weights_sha):
        raise ValueError("complete primary transfer checkpoint hash required")
    with np.load(DEFAULT_WEIGHTS) as weights:
        delta_q = weights["delta_q"].astype(np.float32)
        delta_k = weights["delta_k"].astype(np.float32)
    if delta_q.shape != (32, 64, 64) or delta_k.shape != (8, 64, 64):
        raise ValueError("feature-map checkpoint head shapes differ")

    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    ids = np.asarray(tokenizer.encode(raw.decode("utf-8")), dtype=np.int32)
    starts = {str(pct): int(pct * len(ids) / 100) for pct in PERCENTAGES}
    if any(start + WINDOW > len(ids) for start in starts.values()):
        raise ValueError("WikiText test window exceeds corpus")
    if any(left + WINDOW > right for left, right in zip(
            sorted(starts.values()), sorted(starts.values())[1:])):
        raise ValueError("WikiText test windows overlap")
    windows = {pct: ids[start:start + WINDOW] for pct, start in starts.items()}
    original = model.layers[8].self_attn
    prepared = {
        pct: prepare(model, original, windows[pct][:OUTPUT_PREFIX])
        for pct in windows
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
            "dataset_url": DATASET_URL,
            "dataset_card": DATASET_CARD,
            "wiki_zip_sha256": ZIP_SHA256,
            "wiki_test_member": MEMBER,
            "wiki_test_member_sha256": MEMBER_SHA256,
            "wiki_test_token_count": int(len(ids)),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "percentages_fixed_before_quality_read": list(PERCENTAGES),
            "window_tokens": WINDOW,
            "scored_tokens_per_window": WINDOW - 1,
            "attention_output_prefix_tokens": OUTPUT_PREFIX,
            "fixed_gain": DEFAULT_GAIN,
            "quality_note": "zero WikiText training or checkpoint selection; one raw-test domain, five disjoint windows, no speed verdict",
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
        "aggregate": {},
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
    record["conversion"] = installed.conversion
    module = installed.replacement
    zero_q = mx.zeros_like(module.delta_q)
    zero_k = mx.zeros_like(module.delta_k)
    try:
        for arm in ARMS[1:]:
            module.gain = 0.0 if arm == "local_only" else DEFAULT_GAIN
            if arm == "trained_query":
                module.delta_q = mx.array(delta_q)
                module.delta_k = mx.array(delta_k)
            else:
                module.delta_q = zero_q
                module.delta_k = zero_k
            mx.eval(module.delta_q, module.delta_k)
            for pct, (x, target) in prepared.items():
                loss = mse_after_local(module, x, target)
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
        record["aggregate"][arm] = aggregate(rows)
    record["paired"] = {
        "output_mse_reduction_by_window": {
            pct: 1.0 - record["attention_output_mse"]["trained_query"][pct] /
                 record["attention_output_mse"]["untrained_query"][pct]
            for pct in windows
        },
        "trained_minus_untrained_nll_by_window": {
            pct: record["rows"]["trained_query"][pct]["nll"] -
                 record["rows"]["untrained_query"][pct]["nll"]
            for pct in windows
        },
        "trained_beats_untrained_windows": sum(
            record["rows"]["trained_query"][pct]["nll"] <
            record["rows"]["untrained_query"][pct]["nll"]
            for pct in windows
        ),
        "trained_minus_teacher_nll_by_window": {
            pct: record["rows"]["trained_query"][pct]["nll"] -
                 record["rows"]["teacher"][pct]["nll"]
            for pct in windows
        },
    }
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print("aggregate", record["aggregate"], "paired", record["paired"],
          flush=True)


if __name__ == "__main__":
    main()
