"""Official generation parity of the specialized one-token query-state path.

The same selected stage-1 map and 256/2 prefill setting are served by the
general decoder and the opt-in one-token decoder. Four official greedy
tokens, a fresh-cache manual loop and an 8,192-token whole/split/decode
prefix must agree. This is functional validation, not speed or quality.
"""
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
from query_global_fast_decode import FastDecodeTrainableQueryGlobalAttention  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)
from verify_query_generation import generation, long_prefix  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "verify_fast_decode_generation.json"


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def main() -> None:
    stage1 = json.loads(STAGE1_RESULT.read_text())
    weight_sha = hashlib.sha256(STAGE1_WEIGHTS.read_bytes()).hexdigest()
    if (stage1["status"] != "complete" or not stage1["meta"]["primary"]
            or stage1["weights"]["sha256"] != weight_sha):
        raise ValueError("complete selected stage-1 map required")
    with np.load(STAGE1_WEIGHTS) as checkpoint:
        delta_q = mx.array(checkpoint["delta_q"].astype(np.float32))
        delta_k = mx.array(checkpoint["delta_k"].astype(np.float32))
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    corpus = CORPUS.read_text()
    prompt = np.asarray(tokenizer.encode(corpus[:2000])[:129], dtype=np.int32)
    ids = np.asarray(tokenizer.encode(corpus), dtype=np.int32)
    if len(prompt) != 129 or len(ids) < 8193:
        raise ValueError("generation and 8K prompts are too short")
    original = model.layers[8].self_attn
    modules = {
        "general": TrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=256,
            inference_sync_blocks=2),
        "fast_one_token": FastDecodeTrainableQueryGlobalAttention(
            original, gain=DEFAULT_GAIN, chunk=256,
            inference_sync_blocks=2),
    }
    for module in modules.values():
        module.delta_q = delta_q
        module.delta_k = delta_k
        module.eval()
    mx.eval(delta_q, delta_k)
    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fast_decode_source_sha256": hashlib.sha256(
                (HERE / "query_global_fast_decode.py").read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "trainable_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "generation_harness_sha256": hashlib.sha256(
                (HERE / "verify_query_generation.py").read_bytes()).hexdigest(),
            "stage1_result_sha256": hashlib.sha256(STAGE1_RESULT.read_bytes()).hexdigest(),
            "stage1_weights_sha256": weight_sha,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "prompt_ids_sha256": hashlib.sha256(prompt.tobytes()).hexdigest(),
            "long_prefix_ids_sha256": hashlib.sha256(ids[:8193].tobytes()).hexdigest(),
            "chunk": 256,
            "inference_sync_blocks": 2,
            "gain": DEFAULT_GAIN,
            "started_epoch": time.time(),
        },
        "rows": {},
        "paired": {},
    }
    atomic_json(record)
    try:
        for arm, module in modules.items():
            model.layers[8].self_attn = module
            record["rows"][arm] = {
                "generation": generation(model, prompt, teacher=False),
                "long_prefix": long_prefix(model, ids, teacher=False),
            }
            atomic_json(record)
    finally:
        model.layers[8].self_attn = original
    standard = record["rows"]["general"]
    faster = record["rows"]["fast_one_token"]
    row = {
        "official_ids_equal": (
            standard["generation"]["official_generated_ids"] ==
            faster["generation"]["official_generated_ids"]),
        "manual_ids_equal": (
            standard["generation"]["manual_ids"] ==
            faster["generation"]["manual_ids"]),
        "bounded_counts_equal": (
            standard["long_prefix"]["layer8_global_count_after_decode"] ==
            faster["long_prefix"]["layer8_global_count_after_decode"]),
    }
    if not all(row.values()):
        raise AssertionError(f"specialized generation disagrees: {row}")
    record["paired"] = row
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print("ids", faster["generation"]["official_generated_ids"],
          "8K global keys", faster["long_prefix"][
              "layer8_global_count_after_decode"], flush=True)


if __name__ == "__main__":
    main()
