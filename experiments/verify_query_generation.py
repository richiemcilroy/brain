"""Real 1B mlx-lm cached-generation and long-prefix parity for query state.

The teacher, local-only and query-global arms use one offline bf16 checkpoint.
Four official greedy tokens must equal a fresh-cache manual greedy loop.
An 8,192-token prefix must preserve top-1 when served whole, split in half,
then followed by one decoded token. This is functional evidence, not speed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402

from benchmark_local_hybrid import model_call  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import (  # noqa: E402
    DEFAULT_GAIN, install_query_global, make_query_global_cache,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "verify_query_generation.json"


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def cache_for(model, teacher: bool):
    return model.make_cache() if teacher else make_query_global_cache(model)


def as_f32(a):
    return np.asarray(a.astype(mx.float32))


def generation(model, prompt: np.ndarray, *, teacher: bool) -> dict:
    direct = as_f32(model(mx.array(prompt[None, :]))[:, -1, :])
    optimized = as_f32(model_call(model, mx.array(prompt[None, :]), None))
    head_max = float(np.max(np.abs(direct - optimized)))
    head_top1 = bool(np.array_equal(direct.argmax(-1), optimized.argmax(-1)))
    cache = cache_for(model, teacher)
    generated = []
    logprobs = []
    for token, token_logprobs in generate_step(
        mx.array(prompt), model, max_tokens=4, prompt_cache=cache,
        prefill_step_size=64, kv_bits=None,
    ):
        generated.append(int(token))
        mx.eval(token_logprobs)
        logprobs.append(float(token_logprobs[int(token)]))
    manual_cache = cache_for(model, teacher)
    logits = model_call(model, mx.array(prompt[None, :]), manual_cache)
    manual = []
    for step in range(4):
        token = int(as_f32(logits).argmax(-1).item())
        manual.append(token)
        if step < 3:
            logits = model_call(
                model, mx.array([[token]], dtype=mx.int32), manual_cache)
    row = {
        "official_generated_ids": generated,
        "manual_ids": manual,
        "ids_equal": generated == manual,
        "sampled_logprobs": logprobs,
        "logprobs_finite": bool(np.isfinite(logprobs).all()),
        "direct_vs_last_head_max_abs_logit": head_max,
        "direct_vs_last_head_top1_equal": head_top1,
        "official_cache_offsets": [int(slot.offset) for slot in cache],
        "manual_cache_offsets": [int(slot.offset) for slot in manual_cache],
    }
    if not teacher:
        row["layer8_local_old_kv_tokens"] = int(cache[8].keys.shape[2])
        row["layer8_global_count"] = int(cache[8].global_count)
        if row["layer8_local_old_kv_tokens"] > 63:
            raise AssertionError("query-state local cache grew beyond 63")
    if not (row["ids_equal"] and row["logprobs_finite"] and head_top1):
        raise AssertionError(f"official generation differs: {row}")
    if len(set(row["official_cache_offsets"])) != 1:
        raise AssertionError(f"official cache offsets disagree: {row}")
    if len(set(row["manual_cache_offsets"])) != 1:
        raise AssertionError(f"manual cache offsets disagree: {row}")
    return row


def long_prefix(model, ids: np.ndarray, *, teacher: bool) -> dict:
    x = mx.array(ids[:8192][None, :])
    x_plus = mx.array(ids[:8193][None, :])
    whole_cache = cache_for(model, teacher)
    whole = as_f32(model_call(model, x, whole_cache))
    split_cache = cache_for(model, teacher)
    model_call(model, x[:, :4096], split_cache)
    split = as_f32(model_call(model, x[:, 4096:], split_cache))
    whole_plus = as_f32(model_call(
        model, x_plus, cache_for(model, teacher)))
    decode_cache = cache_for(model, teacher)
    model_call(model, x, decode_cache)
    decode = as_f32(model_call(
        model, x_plus[:, 8192:], decode_cache))
    row = {
        "prefix_tokens": 8192,
        "whole_vs_split_max_abs_logit": float(np.max(np.abs(whole - split))),
        "whole_vs_split_top1_equal": bool(
            np.array_equal(whole.argmax(-1), split.argmax(-1))),
        "whole_plus_vs_decode_max_abs_logit": float(
            np.max(np.abs(whole_plus - decode))),
        "whole_plus_vs_decode_top1_equal": bool(
            np.array_equal(whole_plus.argmax(-1), decode.argmax(-1))),
        "whole_offsets": [int(slot.offset) for slot in whole_cache],
        "split_offsets": [int(slot.offset) for slot in split_cache],
        "decode_offsets": [int(slot.offset) for slot in decode_cache],
    }
    if not teacher:
        row["layer8_local_old_kv_tokens_after_decode"] = int(
            decode_cache[8].keys.shape[2])
        row["layer8_global_count_after_decode"] = int(
            decode_cache[8].global_count)
        expected_global = (
            8193 - 63 if decode_cache[8].gain else 0)
        if (row["layer8_local_old_kv_tokens_after_decode"] > 63 or
                row["layer8_global_count_after_decode"] != expected_global):
            raise AssertionError("long-prefix bounded state is wrong")
    if not (row["whole_vs_split_top1_equal"]
            and row["whole_plus_vs_decode_top1_equal"]
            and all(x == 8192 for x in row["whole_offsets"])
            and all(x == 8192 for x in row["split_offsets"])
            and all(x == 8193 for x in row["decode_offsets"])):
        raise AssertionError(f"long-prefix cache parity failed: {row}")
    return row


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {DEFAULT_SNAPSHOT}")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    corpus = CORPUS.read_text()
    prompt = np.asarray(tokenizer.encode(corpus[:2000])[:129], dtype=np.int32)
    ids = np.asarray(tokenizer.encode(corpus), dtype=np.int32)
    if len(prompt) != 129 or len(ids) < 8193:
        raise ValueError("prompt or corpus is short")
    record = {
        "status": "running",
        "meta": {
            "revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "prompt_ids_sha256": hashlib.sha256(prompt.tobytes()).hexdigest(),
            "long_prefix_ids_sha256": hashlib.sha256(ids[:8193].tobytes()).hexdigest(),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "generated_tokens": 4,
            "prefill_step_size": 64,
            "greedy": True,
            "kv_bits": None,
            "query_gain": DEFAULT_GAIN,
            "started_epoch": time.time(),
        },
        "conversion": None,
        "rows": {},
    }
    atomic_json(record)
    record["rows"]["teacher"] = {
        "generation": generation(model, prompt, teacher=True),
        "long_prefix": long_prefix(model, ids, teacher=True),
    }
    atomic_json(record)
    installed = install_query_global(model, layer=8, gain=DEFAULT_GAIN)
    record["conversion"] = installed.conversion
    try:
        for arm, gain in (("local_only", 0.0),
                          ("query_global", DEFAULT_GAIN)):
            installed.replacement.gain = gain
            record["rows"][arm] = {
                "generation": generation(model, prompt, teacher=False),
                "long_prefix": long_prefix(model, ids, teacher=False),
            }
            atomic_json(record)
            print(arm, record["rows"][arm]["generation"]["official_generated_ids"],
                  "long-parity", record["rows"][arm]["long_prefix"][
                      "whole_plus_vs_decode_max_abs_logit"],
                  flush=True)
    finally:
        installed.restore()
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)


if __name__ == "__main__":
    main()
