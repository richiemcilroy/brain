"""End-to-end Llama baseline versus one cached gated-memory replacement.

This is a measurement harness, not an efficiency claim. It loads the same local
bf16 OSS checkpoint in a fresh process per arm, uses identical token inputs,
forces MLX evaluation inside each timer, and checks chunked decode against an
unsplit cached prefix before reporting throughput. The candidate is the repo's
untrained attention transplant at one layer; its quality loss is reported too.

Example:
  ~/zbrain/venv/bin/python experiments/oss_model_benchmark.py --arm teacher
  ~/zbrain/venv/bin/python experiments/oss_model_benchmark.py --arm transfer
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

# Local snapshot only: a benchmark must not silently download a different model.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from llm_hybrid import attention_module, perplexity, transplant  # noqa: E402
from streaming_memory import MemoryCache, StreamingGatedMemoryCarrier  # noqa: E402

DEFAULT_SNAPSHOT = Path(
    "/Users/richie/zbrain/hf/hub/models--unsloth--Llama-3.2-1B/snapshots/"
    "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
)
CORPUS = ROOT / "data" / "tinyshakespeare.txt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("teacher", "transfer"), required=True)
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--decay", type=float, default=0.7)
    p.add_argument("--contexts", default="128,512,2048")
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def make_cache(model, arm, layer):
    cache = model.make_cache()
    if arm == "transfer":
        cache[layer] = MemoryCache()
    return cache


def cache_bytes(cache):
    """Logical bytes for active KV positions and the recurrent state."""
    total = 0
    for c in cache:
        if isinstance(c, MemoryCache):
            if c.state is not None:
                total += c.state.nbytes
        else:
            for name in ("keys", "values"):
                a = getattr(c, name, None)
                if a is not None:
                    active = min(int(c.offset), int(a.shape[2]))
                    total += a[..., :active, :].nbytes
    return int(total)


def cache_storage_bytes(cache):
    """Allocated KV tensor capacity, including MLX's spare positions."""
    total = 0
    for c in cache:
        if isinstance(c, MemoryCache):
            if c.state is not None:
                total += c.state.nbytes
        else:
            for name in ("keys", "values"):
                a = getattr(c, name, None)
                if a is not None:
                    total += a.nbytes
    return int(total)


def model_call(model, inputs, cache):
    # Serving needs only the final prefill prediction. Calling Model.__call__
    # would project all T hidden states into a 128k-token vocabulary and make
    # the prefill benchmark mostly a measurement of the output head.
    hidden = model.model(inputs, cache=cache)
    last_hidden = hidden[:, -1:, :]
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(last_hidden)
    else:
        logits = model.lm_head(last_hidden)
    last = logits[:, -1, :]
    mx.eval(last)
    return last


def prefix_equivalence(model, ids, arm, layer, batch):
    """Cached chunks and a decode step must agree with one cached model call."""
    x = np.tile(np.asarray(ids[:128], np.int32)[None, :], (batch, 1))
    one_cache = make_cache(model, arm, layer)
    one = model_call(model, mx.array(x), one_cache)
    split_cache = make_cache(model, arm, layer)
    start = 0
    for length in (64, 32, 32):
        split = model_call(model, mx.array(x[:, start : start + length]), split_cache)
        start += length
    a = np.asarray(one.astype(mx.float32))
    b = np.asarray(split.astype(mx.float32))
    diff = np.abs(a - b)
    # A one-token call exercises the state update used for generation. Checking
    # only a segmented prefill would miss a carrier that resets on decode.
    x129 = np.tile(np.asarray(ids[:129], np.int32)[None, :], (batch, 1))
    decode_one = model_call(model, mx.array(x129), make_cache(model, arm, layer))
    decode_cache = make_cache(model, arm, layer)
    model_call(model, mx.array(x129[:, :128]), decode_cache)
    decode_split = model_call(model, mx.array(x129[:, 128:]), decode_cache)
    da = np.asarray(decode_one.astype(mx.float32))
    db = np.asarray(decode_split.astype(mx.float32))
    return {
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "top1_equal": bool(np.array_equal(a.argmax(axis=-1), b.argmax(axis=-1))),
        "decode_max_abs_logit_diff": float(np.abs(da - db).max()),
        "decode_top1_equal": bool(np.array_equal(da.argmax(axis=-1), db.argmax(axis=-1))),
        "one_cache_bytes": cache_bytes(one_cache),
        "split_cache_bytes": cache_bytes(split_cache),
        "one_cache_storage_bytes": cache_storage_bytes(one_cache),
        "split_cache_storage_bytes": cache_storage_bytes(split_cache),
        "offsets_equal": bool(all(c1.offset == c2.offset == 128
                                  for c1, c2 in zip(one_cache, split_cache))),
        "decode_offset_correct": bool(all(c.offset == 129 for c in decode_cache)),
    }


def one_repeat(model, ids, *, context, future, arm, layer, batch):
    cache = make_cache(model, arm, layer)
    prefix = np.tile(np.asarray(ids[:context], np.int32)[None, :], (batch, 1))
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    last = model_call(model, mx.array(prefix), cache)
    prefill_s = time.perf_counter() - t0
    prefill_cache_bytes = cache_bytes(cache)
    prefill_cache_storage_bytes = cache_storage_bytes(cache)
    prefill_peak_bytes = mx.get_peak_memory()

    t0 = time.perf_counter()
    for token in future:
        step = np.full((batch, 1), int(token), dtype=np.int32)
        last = model_call(model, mx.array(step), cache)
    decode_s = time.perf_counter() - t0
    mx.eval(last)
    return {
        "prefill_s": prefill_s,
        "prefill_tok_s": batch * context / prefill_s,
        "decode_s": decode_s,
        "decode_tok_s": batch * len(future) / decode_s,
        "decode_ms_per_token": 1000 * decode_s / len(future),
        "prefill_cache_bytes": prefill_cache_bytes,
        "final_cache_bytes": cache_bytes(cache),
        "prefill_cache_storage_bytes": prefill_cache_storage_bytes,
        "final_cache_storage_bytes": cache_storage_bytes(cache),
        "prefill_peak_bytes": int(prefill_peak_bytes),
        "final_peak_bytes": int(mx.get_peak_memory()),
        "final_active_bytes": int(mx.get_active_memory()),
        "cache_offsets": [int(c.offset) for c in cache],
        "load_average": list(os.getloadavg()),
    }


def main():
    args = parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    if not args.snapshot.is_dir() or not (args.snapshot / "config.json").is_file():
        raise FileNotFoundError(f"local model snapshot missing: {args.snapshot}")
    if args.decode_tokens < 1 or args.repeats < 1 or args.batch < 1:
        raise ValueError("decode-tokens, repeats, and batch must be positive")
    if min(contexts) < 128:
        raise ValueError("minimum context is 128 for the cache-equivalence guard")

    print(f"loading local {args.snapshot} ...", flush=True)
    load_start = time.perf_counter()
    model, tokenizer = load(str(args.snapshot))
    model.eval()
    load_s = time.perf_counter() - load_start
    text = CORPUS.read_text()
    ids = np.asarray(tokenizer.encode(text), dtype=np.int32)
    n = len(ids)
    val = ids[int(0.9 * n) : int(0.9 * n) + 3000]
    if len(val) != 3000 or max(contexts) + args.decode_tokens >= int(0.5 * n):
        raise ValueError("the fixed corpus is too short for this benchmark")

    conversion = None
    if args.arm == "transfer":
        _, _, original = attention_module(model, args.layer)
        mx.random.seed(0)  # gate weights have no pretrained counterpart
        carrier, conversion = transplant(
            model, args.layer, "transfer", original=original, decay=args.decay)
        model.layers[args.layer].self_attn = StreamingGatedMemoryCarrier(carrier)
        del carrier, original
        gc.collect()
        mx.clear_cache()

    print("checking cached-prefix equivalence ...", flush=True)
    parity = prefix_equivalence(model, ids, args.arm, args.layer, args.batch)
    if not parity["offsets_equal"] or not parity["decode_offset_correct"]:
        raise AssertionError("a cache did not advance by the input length")
    if not parity["top1_equal"] or not parity["decode_top1_equal"]:
        raise AssertionError("chunked cached prefix changed the top prediction")
    if max(parity["max_abs_logit_diff"], parity["decode_max_abs_logit_diff"]) > 0.5:
        raise AssertionError(
            f"cached split differs from one-pass by "
            f"{parity['max_abs_logit_diff']:.3f} logits")

    quality = perplexity(model, val)
    print(f"held-out ppl={quality['ppl']:.4f} | parity max="
          f"{parity['max_abs_logit_diff']:.5f}", flush=True)
    out = {
        "status": "running",
        "meta": {
            "arm": args.arm,
            "snapshot": str(args.snapshot),
            "snapshot_revision": args.snapshot.name,
            "config_sha256": hashlib.sha256(
                (args.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (args.snapshot / "model.safetensors").resolve().name,
            "tokenizer_blob_id": (args.snapshot / "tokenizer.json").resolve().name,
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "streaming_memory_source_sha256": hashlib.sha256(
                (HERE / "streaming_memory.py").read_bytes()).hexdigest(),
            "llm_hybrid_source_sha256": hashlib.sha256(
                (HERE / "llm_hybrid.py").read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "corpus_tokens": n,
            "model_type": type(model).__name__,
            "model_args": {
                "hidden_size": int(model.args.hidden_size),
                "num_hidden_layers": int(model.args.num_hidden_layers),
                "vocab_size": int(model.args.vocab_size),
            },
            "layer": args.layer if args.arm == "transfer" else None,
            "decay": args.decay if args.arm == "transfer" else None,
            "conversion_mlx_seed": 0 if args.arm == "transfer" else None,
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "hardware": platform.platform(),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "load_s": load_s,
            "baseline_active_memory_bytes": int(mx.get_active_memory()),
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "workload": "cached prefill with last-token vocabulary head and "
                        "teacher-forced single-token decode; same corpus tokens "
                        "and precision in both arms",
            "cache_bytes": "active logical KV positions plus trace state",
            "cache_storage_bytes": "whole allocated KV tensors including spare capacity",
            "batch": args.batch,
            "contexts": contexts,
            "decode_tokens": args.decode_tokens,
            "repeats": args.repeats,
            "load_average_at_start": list(os.getloadavg()),
        },
        "conversion": conversion,
        "parity": parity,
        "quality": quality,
        "rows": [],
    }
    output = args.output or HERE / "results" / f"oss_benchmark_{args.arm}_b{args.batch}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for context in contexts:
        future = ids[context : context + args.decode_tokens]
        # One warmup with the same shapes, excluded from timing summaries.
        one_repeat(model, ids, context=context, future=future, arm=args.arm,
                   layer=args.layer, batch=args.batch)
        reps = []
        for _ in range(args.repeats):
            reps.append(one_repeat(model, ids, context=context, future=future,
                                   arm=args.arm, layer=args.layer,
                                   batch=args.batch))
        row = {
            "context": context,
            "batch": args.batch,
            "decode_tokens": args.decode_tokens,
            "repeats": reps,
            "median_prefill_s": statistics.median(r["prefill_s"] for r in reps),
            "median_decode_s": statistics.median(r["decode_s"] for r in reps),
            "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in reps),
            "median_final_cache_bytes": statistics.median(
                r["final_cache_bytes"] for r in reps),
        }
        out["rows"].append(row)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2) + "\n")
        tmp.replace(output)
        print(f"{args.arm} B={args.batch} T={context}: prefill "
              f"{row['median_prefill_s']:.3f}s, decode "
              f"{row['median_decode_tok_s']:.1f} tok/s, cache "
              f"{row['median_final_cache_bytes']/1e6:.1f} MB", flush=True)
    out["status"] = "complete"
    out["meta"]["finished_epoch"] = time.time()
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    tmp.replace(output)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
