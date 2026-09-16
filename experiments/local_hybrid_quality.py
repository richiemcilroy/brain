"""Exploratory signed-gain sweep for one Llama local-window hybrid layer.

The same locally cached bf16 checkpoint, original projections, seeded gate,
tokenizer and text windows are used for every arm. A layer-output probe showed
the trace was negatively aligned with the missing attention output on the
already-inspected 90% window. The symmetric signed grid is fixed before this
run; 60% selects a gain, 90% is the earlier comparison window, and a disjoint
95% window is a fresh holdout for the signed extension. This is exploratory,
not a preregistered confirmatory experiment. All gains are saved.

Run after other Metal experiments finish:
    ~/zbrain/venv/bin/python experiments/local_hybrid_quality.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from benchmark_local_hybrid import model_call  # noqa: E402
from llm_hybrid import perplexity  # noqa: E402
from local_window_hybrid import install_local_window_hybrid, make_hybrid_cache  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "local_hybrid_quality.json"
GAINS = (-1.0, -0.5, -0.25, -0.1, -0.05, -0.01,
         0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
WINDOW = 3000


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def long_prefix_parity(model, ids: np.ndarray, *, hybrid: bool) -> dict:
    """Check 8,192 tokens plus decode, beyond the 128-token benchmark guard."""
    def cache():
        return make_hybrid_cache(model) if hybrid else model.make_cache()

    x = mx.array(ids[:8192][None, :])
    x_plus = mx.array(ids[:8193][None, :])
    whole_cache = cache()
    whole = np.asarray(model_call(model, x, whole_cache).astype(mx.float32))
    split_cache = cache()
    model_call(model, x[:, :4096], split_cache)
    split = np.asarray(model_call(model, x[:, 4096:], split_cache).astype(mx.float32))
    whole_plus_cache = cache()
    whole_plus = np.asarray(model_call(model, x_plus, whole_plus_cache).astype(mx.float32))
    decode_cache = cache()
    model_call(model, x, decode_cache)
    decode = np.asarray(model_call(model, x_plus[:, 8192:], decode_cache).astype(mx.float32))
    return {
        "prefix_tokens": 8192,
        "whole_vs_split_max_abs_logit": float(np.abs(whole - split).max()),
        "whole_vs_split_top1_equal": bool(np.array_equal(whole.argmax(-1), split.argmax(-1))),
        "whole_plus_vs_decode_max_abs_logit": float(np.abs(whole_plus - decode).max()),
        "whole_plus_vs_decode_top1_equal": bool(np.array_equal(whole_plus.argmax(-1), decode.argmax(-1))),
        "whole_offsets_correct": all(c.offset == 8192 for c in whole_cache),
        "split_offsets_correct": all(c.offset == 8192 for c in split_cache),
        "decode_offsets_correct": all(c.offset == 8193 for c in decode_cache),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.7)
    args = parser.parse_args()
    if not (args.snapshot / "config.json").is_file():
        raise FileNotFoundError(f"offline local checkpoint missing: {args.snapshot}")

    model, tokenizer = load(str(args.snapshot))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    select_start = int(0.6 * len(ids))
    val_start = int(0.9 * len(ids))
    fresh_start = int(0.95 * len(ids))
    select = ids[select_start:select_start + WINDOW]
    val = ids[val_start:val_start + WINDOW]
    fresh = ids[fresh_start:fresh_start + WINDOW]
    if (len(select) != WINDOW or len(val) != WINDOW or len(fresh) != WINDOW
            or select_start + WINDOW > val_start
            or val_start + WINDOW > fresh_start):
        raise ValueError("the fixed selection and evaluation windows overlap or are short")

    record = {
        "status": "running",
        "meta": {
            "checkpoint": str(args.snapshot),
            "revision": args.snapshot.name,
            "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (args.snapshot / "model.safetensors").resolve().name,
            "tokenizer_blob_id": (args.snapshot / "tokenizer.json").resolve().name,
            "quality_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_window_source_sha256": hashlib.sha256(
                (HERE / "local_window_hybrid.py").read_bytes()
            ).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "selection_window": [select_start, select_start + WINDOW],
            "validation_window": [val_start, val_start + WINDOW],
            "fresh_holdout_window": [fresh_start, fresh_start + WINDOW],
            "selection_tokens_sha256": hashlib.sha256(select.tobytes()).hexdigest(),
            "validation_tokens_sha256": hashlib.sha256(val.tobytes()).hexdigest(),
            "fresh_holdout_tokens_sha256": hashlib.sha256(fresh.tobytes()).hexdigest(),
            "tokens_scored_per_window": WINDOW - 1,
            "gains_predeclared": list(GAINS),
            "layer": args.layer,
            "window": 64,
            "decay": args.decay,
            "gate_mlx_seed": 0,
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "hardware": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "started_epoch": time.time(),
            "quality_note": "exploratory signed extension after inspecting the 90% window's output-fit probe; 95% is a fresh disjoint holdout; independent 512-token scoring chunks; no training",
        },
        "teacher": {},
        "long_prefix_parity": {},
        "conversion": None,
        "rows": [],
        "selection_choice": None,
    }
    atomic_json(args.output, record)

    windows = (("selection", select), ("validation", val),
               ("fresh_holdout", fresh))
    for name, tokens in windows:
        record["teacher"][name] = perplexity(model, tokens)
    record["long_prefix_parity"]["teacher"] = long_prefix_parity(model, ids, hybrid=False)
    atomic_json(args.output, record)

    mx.random.seed(0)
    installed = install_local_window_hybrid(
        model, layer=args.layer, decay=args.decay, memory_gain=0.0)
    record["conversion"] = installed.conversion
    for name, gain in (("local", 0.0), ("hybrid", 0.05),
                       ("signed", -0.25)):
        installed.hybrid.memory_gain = gain
        record["long_prefix_parity"][name] = long_prefix_parity(model, ids, hybrid=True)
    atomic_json(args.output, record)

    for gain in GAINS:
        installed.hybrid.memory_gain = gain
        row = {"gain": gain}
        for name, tokens in windows:
            row[name] = perplexity(model, tokens)
        record["rows"].append(row)
        atomic_json(args.output, record)
        print(f"gain={gain:g} select={row['selection']['ppl']:.4f} "
              f"val={row['validation']['ppl']:.4f} "
              f"fresh={row['fresh_holdout']['ppl']:.4f}", flush=True)

    best = min(record["rows"], key=lambda row: row["selection"]["nll"])
    record["selection_choice"] = {
        "gain": best["gain"],
        "selection_ppl": best["selection"]["ppl"],
        "validation_ppl": best["validation"]["ppl"],
        "fresh_holdout_ppl": best["fresh_holdout"]["ppl"],
        "picked_by": "lowest selection NLL; neither evaluation window used to select",
    }
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(args.output, record)
    print(f"selected gain={best['gain']:g}; fresh held-out ppl="
          f"{best['fresh_holdout']['ppl']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
