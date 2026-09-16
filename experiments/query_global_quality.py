"""First complete-1B cached-quality test of the fixed query-global baseline.

The map and scalar gain were selected by attention-output fit on a 60% corpus
window before this model-quality run. Teacher, exact local-only and
local/query-global arms score four disjoint 3,000-token windows at 96–99%.
Cache persists over 128-token model chunks; each arm uses the same bf16
checkpoint and float32 loss from its logits. No weights are trained here.
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

from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import (  # noqa: E402
    DEFAULT_GAIN, QueryGlobalCache, install_query_global,
    make_query_global_cache,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_quality.json"
WINDOW = 3000
CHUNK = 128
PERCENTAGES = (96, 97, 98, 99)


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(OUTPUT)


def active_cache_bytes(caches) -> int:
    total = 0
    for slot in caches:
        if isinstance(slot, QueryGlobalCache):
            total += slot.nbytes
        elif getattr(slot, "keys", None) is not None:
            active = min(int(slot.offset), int(slot.keys.shape[2]))
            total += int(slot.keys[..., :active, :].nbytes)
            total += int(slot.values[..., :active, :].nbytes)
    return total


def cached_nll(model, segment: np.ndarray, *, teacher: bool) -> dict:
    caches = model.make_cache() if teacher else make_query_global_cache(model)
    count = len(segment) - 1
    total = 0.0
    for start in range(0, count, CHUNK):
        length = min(CHUNK, count - start)
        x = mx.array(segment[start:start + length][None, :], dtype=mx.int32)
        labels = mx.array(
            segment[start + 1:start + length + 1][None, :], dtype=mx.int32)
        logits = model(x, cache=caches)
        logprobs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        losses = -mx.take_along_axis(
            logprobs, labels[..., None], axis=-1).squeeze(-1)
        mx.eval(losses, [slot.state for slot in caches])
        total += float(mx.sum(losses))
    offsets = [int(slot.offset) for slot in caches]
    if offsets != [count] * len(offsets):
        raise AssertionError(f"cache offsets disagree with scored input: {offsets}")
    local_kv = None
    global_count = None
    if not teacher:
        custom = caches[8]
        local_kv = int(custom.keys.shape[2])
        global_count = int(custom.global_count)
        if local_kv > 63 or (
            global_count != count - 63 if custom.gain else global_count != 0
        ):
            raise AssertionError("bounded one-layer cache count is wrong")
    nll = total / count
    return {
        "nll": nll,
        "ppl": math.exp(nll),
        "scored_tokens": count,
        "cache_offsets": offsets,
        "active_logical_cache_bytes": active_cache_bytes(caches),
        "layer8_local_old_kv_tokens": local_kv,
        "layer8_global_older_keys_summarized": global_count,
    }


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {DEFAULT_SNAPSHOT}")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    starts = [int(pct * len(ids) / 100) for pct in PERCENTAGES]
    if any(start + WINDOW > len(ids) for start in starts):
        raise ValueError("quality window exceeds corpus")
    if any(left + WINDOW > right for left, right in zip(starts, starts[1:])):
        raise ValueError("quality windows overlap")
    windows = {
        str(pct): ids[start:start + WINDOW]
        for pct, start in zip(PERCENTAGES, starts)
    }
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
            "query_probe_source_sha256": hashlib.sha256(
                (HERE / "query_global_probe.py").read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "scored_tokens_per_window": WINDOW - 1,
            "cache_chunk_tokens": CHUNK,
            "percentages": list(PERCENTAGES),
            "selected_gain_from_output_probe": DEFAULT_GAIN,
            "query_global_layer": 8,
            "local_window": 64,
            "float32_loss_from_bf16_logits": True,
            "quality_note": "iterative research holdouts; map/gain fixed before this first query-global next-token evaluation; no training or timing",
            "started_epoch": time.time(),
        },
        "windows": {
            str(pct): {
                "token_window": [start, start + WINDOW],
                "token_ids_sha256": hashlib.sha256(windows[str(pct)].tobytes()).hexdigest(),
            }
            for pct, start in zip(PERCENTAGES, starts)
        },
        "conversion": None,
        "rows": {"teacher": {}, "local_only": {}, "query_global": {}},
        "fresh_aggregate": {},
        "paired": {},
    }
    atomic_json(record)
    for pct, segment in windows.items():
        record["rows"]["teacher"][pct] = cached_nll(
            model, segment, teacher=True)
        atomic_json(record)
        print("teacher", pct, record["rows"]["teacher"][pct]["ppl"],
              flush=True)

    installed = install_query_global(model, layer=8, gain=DEFAULT_GAIN)
    record["conversion"] = installed.conversion
    atomic_json(record)
    try:
        for arm, gain in (("local_only", 0.0),
                          ("query_global", DEFAULT_GAIN)):
            installed.replacement.gain = gain
            for pct, segment in windows.items():
                record["rows"][arm][pct] = cached_nll(
                    model, segment, teacher=False)
                atomic_json(record)
                print(arm, pct, record["rows"][arm][pct]["ppl"],
                      flush=True)
    finally:
        installed.restore()
    for arm, rows in record["rows"].items():
        weighted = sum(row["nll"] * row["scored_tokens"]
                       for row in rows.values())
        count = sum(row["scored_tokens"] for row in rows.values())
        nll = weighted / count
        record["fresh_aggregate"][arm] = {
            "scored_tokens": count, "nll": nll, "ppl": math.exp(nll)}
    record["paired"] = {
        "query_minus_local_nll_by_window": {
            pct: record["rows"]["query_global"][pct]["nll"] -
                 record["rows"]["local_only"][pct]["nll"]
            for pct in windows
        },
        "query_beats_local_windows": sum(
            record["rows"]["query_global"][pct]["nll"] <
            record["rows"]["local_only"][pct]["nll"]
            for pct in windows
        ),
        "query_minus_teacher_nll_by_window": {
            pct: record["rows"]["query_global"][pct]["nll"] -
                 record["rows"]["teacher"][pct]["nll"]
            for pct in windows
        },
    }
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print("aggregate", record["fresh_aggregate"],
          "paired", record["paired"], flush=True)


if __name__ == "__main__":
    main()
