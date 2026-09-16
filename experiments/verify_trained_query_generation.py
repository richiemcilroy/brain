"""Official cached generation and 8,192-token parity for transferred features."""
from __future__ import annotations

import hashlib
import json
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
from query_global_trainable import install_trainable_query_global  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as TRANSFER_RESULT, DEFAULT_WEIGHTS,
)
from verify_query_generation import generation, long_prefix  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "verify_trained_query_generation.json"


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline snapshot missing: {DEFAULT_SNAPSHOT}")
    transfer = json.loads(TRANSFER_RESULT.read_text())
    weights_sha = hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest()
    if (transfer["status"] != "complete" or
            transfer["weights"]["sha256"] != weights_sha):
        raise ValueError("selected feature-map checkpoint hash differs")
    with np.load(DEFAULT_WEIGHTS) as checkpoint:
        delta_q = checkpoint["delta_q"].astype(np.float32)
        delta_k = checkpoint["delta_k"].astype(np.float32)
    if delta_q.shape != (32, 64, 64) or delta_k.shape != (8, 64, 64):
        raise ValueError("checkpoint heads differ from the 1B model")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    corpus = CORPUS.read_text()
    prompt = np.asarray(tokenizer.encode(corpus[:2000])[:129], dtype=np.int32)
    ids = np.asarray(tokenizer.encode(corpus), dtype=np.int32)
    if len(prompt) != 129 or len(ids) < 8193:
        raise ValueError("short prompt or corpus")
    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generation_harness_sha256": hashlib.sha256(
                (HERE / "verify_query_generation.py").read_bytes()).hexdigest(),
            "query_attention_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "query_trainable_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "transfer_result_sha256": hashlib.sha256(
                TRANSFER_RESULT.read_bytes()).hexdigest(),
            "feature_weights_sha256": weights_sha,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "prompt_ids_sha256": hashlib.sha256(prompt.tobytes()).hexdigest(),
            "long_prefix_ids_sha256": hashlib.sha256(ids[:8193].tobytes()).hexdigest(),
            "gain": DEFAULT_GAIN,
            "started_epoch": time.time(),
        },
        "rows": {},
        "conversion": None,
    }
    atomic_json(record)
    record["rows"]["teacher"] = {
        "generation": generation(model, prompt, teacher=True),
        "long_prefix": long_prefix(model, ids, teacher=True),
    }
    atomic_json(record)
    installed = install_trainable_query_global(
        model, layer=8, gain=DEFAULT_GAIN)
    record["conversion"] = installed.conversion
    installed.replacement.delta_q = mx.array(delta_q)
    installed.replacement.delta_k = mx.array(delta_k)
    mx.eval(installed.replacement.delta_q, installed.replacement.delta_k)
    try:
        record["rows"]["trained_query"] = {
            "generation": generation(model, prompt, teacher=False),
            "long_prefix": long_prefix(model, ids, teacher=False),
        }
        atomic_json(record)
    finally:
        installed.restore()
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print(
        "teacher", record["rows"]["teacher"]["generation"]["official_generated_ids"],
        "trained", record["rows"]["trained_query"]["generation"][
            "official_generated_ids"],
        "long_diff", record["rows"]["trained_query"]["long_prefix"][
            "whole_plus_vs_decode_max_abs_logit"],
        flush=True)


if __name__ == "__main__":
    main()
