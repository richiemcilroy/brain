"""Fresh, long-context cached quality check for the one-layer 1B hybrid.

Gains were fixed by the earlier 60% selection sweep. Four disjoint 3,000-token
windows at 96–99% of TinyShakespeare were untouched by that sweep and its
95% signed-extension holdout. Each arm processes its whole window in cached
128-token chunks, scoring 2,999 next-token predictions. The comparison is
untimed and uses float32 log-softmax of the same bf16 model logits.

This is one corpus on one checkpoint; a win here cannot establish general
language-model quality or a compute advantage.
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
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from benchmark_local_hybrid import logical_cache_bytes  # noqa: E402
from local_window_hybrid import install_local_window_hybrid, make_hybrid_cache  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "local_hybrid_cached_quality.json"
WINDOW = 3000
CHUNK = 128
START_PERCENTAGES = (96, 97, 98, 99)
GAINS = {"local": 0.0, "signed_selected": -0.1, "positive_control": 0.05}


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(OUTPUT)


def cached_quality(model, tokens: np.ndarray, *, hybrid: bool) -> dict:
    cache = make_hybrid_cache(model) if hybrid else model.make_cache()
    count = len(tokens) - 1
    total = 0.0
    for start in range(0, count, CHUNK):
        length = min(CHUNK, count - start)
        x = mx.array(tokens[start:start + length][None, :], dtype=mx.int32)
        targets = mx.array(
            tokens[start + 1:start + length + 1][None, :], dtype=mx.int32)
        logits = model(x, cache=cache)
        logprobs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        losses = -mx.take_along_axis(
            logprobs, targets[..., None], axis=-1).squeeze(-1)
        mx.eval(losses, [slot.state for slot in cache])
        total += float(mx.sum(losses))
    offsets = [int(slot.offset) for slot in cache]
    if offsets != [count] * len(offsets):
        raise AssertionError(f"cached quality processed wrong positions: {offsets}")
    bounded_kv = None
    if hybrid:
        bounded_kv = int(cache[8].keys.shape[2])
        if bounded_kv > 63 or cache[8].memory.offset != count:
            raise AssertionError("the replaced layer's cache is not bounded/correct")
    nll = total / count
    return {
        "nll": nll,
        "ppl": math.exp(nll),
        "scored_tokens": count,
        "cache_offsets": offsets,
        "active_logical_cache_bytes": logical_cache_bytes(cache),
        "layer8_old_kv_tokens": bounded_kv,
    }


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {DEFAULT_SNAPSHOT}")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    starts = [int(pct * len(ids) / 100) for pct in START_PERCENTAGES]
    if any(end + WINDOW > len(ids) for end in starts):
        raise ValueError("a fresh window exceeds the corpus")
    if any(left + WINDOW > right for left, right in zip(starts, starts[1:])):
        raise ValueError("fresh windows overlap")
    windows = {
        str(pct): ids[start:start + WINDOW]
        for pct, start in zip(START_PERCENTAGES, starts)
    }
    record = {
        "status": "running",
        "meta": {
            "snapshot_revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_window_source_sha256": hashlib.sha256(
                (HERE / "local_window_hybrid.py").read_bytes()).hexdigest(),
            "streaming_memory_source_sha256": hashlib.sha256(
                (HERE / "streaming_memory.py").read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "tokenized_corpus_length": int(len(ids)),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "window_tokens": WINDOW,
            "scored_tokens_per_window": WINDOW - 1,
            "cache_chunk_tokens": CHUNK,
            "start_percentages": list(START_PERCENTAGES),
            "fixed_gains": GAINS,
            "float32_loss_from_bf16_logits": True,
            "quality_note": "first quality read of disjoint 96-99% windows; one contiguous corpus, no training or timing",
            "started_epoch": time.time(),
        },
        "windows": {
            str(pct): {
                "token_window": [start, start + WINDOW],
                "token_ids_sha256": hashlib.sha256(windows[str(pct)].tobytes()).hexdigest(),
            }
            for pct, start in zip(START_PERCENTAGES, starts)
        },
        "conversion": None,
        "rows": {"teacher": {}, **{arm: {} for arm in GAINS}},
        "aggregate": {},
        "paired": {},
    }
    atomic_json(record)
    for pct, segment in windows.items():
        record["rows"]["teacher"][pct] = cached_quality(
            model, segment, hybrid=False)
        atomic_json(record)
        print(f"teacher {pct}% ppl={record['rows']['teacher'][pct]['ppl']:.5f}",
              flush=True)

    mx.random.seed(0)
    installed = install_local_window_hybrid(
        model, layer=8, decay=0.7, memory_gain=0.0)
    record["conversion"] = installed.conversion
    atomic_json(record)
    try:
        for arm, gain in GAINS.items():
            installed.hybrid.memory_gain = gain
            for pct, segment in windows.items():
                record["rows"][arm][pct] = cached_quality(
                    model, segment, hybrid=True)
                atomic_json(record)
                print(f"{arm} {pct}% ppl={record['rows'][arm][pct]['ppl']:.5f}",
                      flush=True)
    finally:
        installed.restore()

    for arm, rows in record["rows"].items():
        nll = sum(row["nll"] * row["scored_tokens"] for row in rows.values())
        count = sum(row["scored_tokens"] for row in rows.values())
        record["aggregate"][arm] = {
            "scored_tokens": count,
            "nll": nll / count,
            "ppl": math.exp(nll / count),
        }
    signed = record["rows"]["signed_selected"]
    record["paired"] = {
        "signed_minus_local_nll_by_window": {
            pct: signed[pct]["nll"] - record["rows"]["local"][pct]["nll"]
            for pct in windows
        },
        "signed_beats_local_windows": sum(
            signed[pct]["nll"] < record["rows"]["local"][pct]["nll"]
            for pct in windows
        ),
        "signed_minus_teacher_nll_by_window": {
            pct: signed[pct]["nll"] - record["rows"]["teacher"][pct]["nll"]
            for pct in windows
        },
    }
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(record)
    print("aggregate", record["aggregate"], "paired", record["paired"],
          flush=True)


if __name__ == "__main__":
    main()
