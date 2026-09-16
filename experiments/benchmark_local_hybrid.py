"""Interleaved serving benchmark of Llama teacher, local, and local+trace.

One locally cached bf16 Llama-3.2-1B checkpoint is used for all arms. The local
arms share the same original attention projections and transplanted gated
memory weights; ``memory_gain=0`` skips the trace computation for the local-only
control. Each timed call projects only the last hidden prefill position to the
vocabulary, then decodes one teacher-forced token per call. Every call forces
MLX evaluation inside the timer. A 128/129-token cache-equivalence check runs
before timing and each repeat retains its raw readings in JSON.

Default workload: batch 1, contexts 512/2048/8192, decode 32, three repeats.
This is a single-layer intervention; the other Llama attention layers remain
full-context. Both original and candidate modules coexist in this process, so
absolute allocator memory is not a deployment-memory comparison. Cache bytes
are active logical tensor bytes, independent of KVCache's 256-token capacity.

Example:
    ~/zbrain/venv/bin/python experiments/benchmark_local_hybrid.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

# A benchmark must never silently download a different checkpoint.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from local_window_hybrid import (  # noqa: E402
    HybridAttentionCache, install_local_window_hybrid, make_hybrid_cache,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = Path(
    "/Users/richie/zbrain/hf/hub/models--unsloth--Llama-3.2-1B/snapshots/"
    "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
)
CORPUS = ROOT / "data" / "tinyshakespeare.txt"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "local_hybrid_benchmark_b1.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--memory-gain", type=float, default=0.05)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--contexts", default="512,2048,8192")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_cache(model, arm):
    return model.make_cache() if arm == "teacher" else make_hybrid_cache(model)


def logical_cache_bytes(caches):
    """Only active KV tokens plus active trace state, excluding spare capacity."""
    total = 0
    for cache in caches:
        if isinstance(cache, HybridAttentionCache):
            total += cache.nbytes
            continue
        if getattr(cache, "keys", None) is not None:
            active = min(int(cache.offset), int(cache.keys.shape[2]))
            total += int(cache.keys[..., :active, :].nbytes)
            total += int(cache.values[..., :active, :].nbytes)
    return int(total)


def storage_cache_bytes(caches):
    """Current cache tensor capacities, useful beside logical active bytes."""
    total = 0
    for cache in caches:
        if isinstance(cache, HybridAttentionCache):
            total += cache.nbytes
            continue
        for name in ("keys", "values"):
            array = getattr(cache, name, None)
            if array is not None:
                total += array.nbytes
    return int(total)


def model_call(model, inputs, cache):
    """Serve the last prediction without a T-by-vocabulary prefill head."""
    hidden = model.model(inputs, cache=cache)
    last_hidden = hidden[:, -1:, :]
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(last_hidden)
    else:
        logits = model.lm_head(last_hidden)
    last = logits[:, -1, :]
    mx.eval(last)
    return last


def as_f32(a):
    return np.asarray(a.astype(mx.float32))


def cache_parity(model, ids, arm, batch):
    """Uncached/full cached/chunked/decode calls must predict alike."""
    prefix = np.tile(ids[:128][None, :], (batch, 1))
    prefix_plus = np.tile(ids[:129][None, :], (batch, 1))
    whole_cache = make_cache(model, arm)
    whole = as_f32(model_call(model, mx.array(prefix), whole_cache))
    uncached = as_f32(model_call(model, mx.array(prefix), None))

    split_cache = make_cache(model, arm)
    position = 0
    for length in (64, 32, 32):
        split = as_f32(model_call(
            model, mx.array(prefix[:, position:position + length]), split_cache))
        position += length

    decode_whole = as_f32(model_call(
        model, mx.array(prefix_plus), make_cache(model, arm)))
    decode_cache = make_cache(model, arm)
    model_call(model, mx.array(prefix_plus[:, :128]), decode_cache)
    decode_split = as_f32(model_call(
        model, mx.array(prefix_plus[:, 128:]), decode_cache))
    offsets_128 = [int(c.offset) for c in split_cache]
    offsets_129 = [int(c.offset) for c in decode_cache]
    if arm != "teacher":
        hybrid_caches = [c for c in decode_cache
                         if isinstance(c, HybridAttentionCache)]
        bounded = all(c.keys.shape[2] <= 63 and c.memory.offset == 129
                      for c in hybrid_caches)
    else:
        bounded = None

    checks = {
        "uncached_vs_cached_max_abs_logit": float(np.abs(uncached - whole).max()),
        "split_vs_whole_max_abs_logit": float(np.abs(split - whole).max()),
        "decode_vs_whole_max_abs_logit":
            float(np.abs(decode_split - decode_whole).max()),
        "uncached_top1_equal": bool(np.array_equal(uncached.argmax(-1),
                                                    whole.argmax(-1))),
        "split_top1_equal": bool(np.array_equal(split.argmax(-1),
                                                 whole.argmax(-1))),
        "decode_top1_equal": bool(np.array_equal(decode_split.argmax(-1),
                                                  decode_whole.argmax(-1))),
        "offsets_128": offsets_128,
        "offsets_129": offsets_129,
        "hybrid_kv_bounded_and_memory_offset_correct": bounded,
        "logical_cache_bytes_128": logical_cache_bytes(split_cache),
        "logical_cache_bytes_129": logical_cache_bytes(decode_cache),
    }
    if not (checks["uncached_top1_equal"] and checks["split_top1_equal"]
            and checks["decode_top1_equal"]):
        raise AssertionError(f"{arm}: cached calls changed top-1 prediction: {checks}")
    if max(checks["uncached_vs_cached_max_abs_logit"],
           checks["split_vs_whole_max_abs_logit"],
           checks["decode_vs_whole_max_abs_logit"]) > 0.5:
        raise AssertionError(f"{arm}: cache parity exceeds 0.5 logits: {checks}")
    if not (all(x == 128 for x in offsets_128)
            and all(x == 129 for x in offsets_129)):
        raise AssertionError(f"{arm}: cache offsets are wrong: {checks}")
    if bounded is False:
        raise AssertionError(f"{arm}: local cache is not bounded: {checks}")
    return checks


def one_repeat(model, ids, *, context, future, arm, batch):
    cache = make_cache(model, arm)
    prefix = np.tile(ids[:context][None, :], (batch, 1))
    mx.reset_peak_memory()
    start = time.perf_counter()
    last = model_call(model, mx.array(prefix), cache)
    prefill_s = time.perf_counter() - start
    prefill_logical = logical_cache_bytes(cache)
    prefill_storage = storage_cache_bytes(cache)
    prefill_peak = int(mx.get_peak_memory())

    start = time.perf_counter()
    for token in future:
        step = np.full((batch, 1), int(token), dtype=np.int32)
        last = model_call(model, mx.array(step), cache)
    decode_s = time.perf_counter() - start
    mx.eval(last)
    return {
        "prefill_s": prefill_s,
        "prefill_tok_s": batch * context / prefill_s,
        "decode_s": decode_s,
        "decode_tok_s": batch * len(future) / decode_s,
        "decode_ms_per_token": 1000 * decode_s / len(future),
        "prefill_logical_cache_bytes": prefill_logical,
        "final_logical_cache_bytes": logical_cache_bytes(cache),
        "prefill_cache_storage_bytes": prefill_storage,
        "final_cache_storage_bytes": storage_cache_bytes(cache),
        "prefill_peak_bytes": prefill_peak,
        "final_peak_bytes": int(mx.get_peak_memory()),
        "final_active_bytes": int(mx.get_active_memory()),
        "cache_offsets": [int(c.offset) for c in cache],
        "load_average": list(os.getloadavg()),
    }


def atomic_write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def main():
    args = parse_args()
    contexts = [int(part) for part in args.contexts.split(",")]
    if (not args.snapshot.is_dir() or
            not (args.snapshot / "config.json").is_file()):
        raise FileNotFoundError(f"local snapshot is missing: {args.snapshot}")
    if (not contexts or min(contexts) < 128 or args.decode_tokens < 1
            or args.repeats < 1 or args.batch < 1):
        raise ValueError("contexts >=128, decode tokens, repeats, batch required")
    if not -1.0 <= args.memory_gain <= 1.0 or args.memory_gain == 0.0:
        raise ValueError("memory-gain must be in [-1, 0) or (0, 1]")
    print(f"loading local checkpoint {args.snapshot}", flush=True)
    started = time.perf_counter()
    model, tokenizer = load(str(args.snapshot))
    model.eval()
    load_s = time.perf_counter() - started
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    if max(contexts) + args.decode_tokens >= len(ids) // 2:
        raise ValueError("corpus is too short for the requested context")

    mx.random.seed(0)
    installation = install_local_window_hybrid(
        model, layer=args.layer, decay=args.decay,
        memory_gain=args.memory_gain, chunk=args.chunk)
    hybrid = installation.hybrid
    original = installation.original

    def switch(arm):
        if arm == "teacher":
            setattr(installation.layer, installation.attribute, original)
        elif arm == "local":
            hybrid.memory_gain = 0.0
            setattr(installation.layer, installation.attribute, hybrid)
        elif arm == "hybrid":
            hybrid.memory_gain = args.memory_gain
            setattr(installation.layer, installation.attribute, hybrid)
        else:
            raise ValueError(arm)

    arms = ("teacher", "local", "hybrid")
    record = {
        "status": "running",
        "meta": {
            "snapshot": str(args.snapshot),
            "snapshot_revision": args.snapshot.name,
            "config_sha256": hashlib.sha256(
                (args.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (args.snapshot / "model.safetensors").resolve().name,
            "tokenizer_blob_id": (args.snapshot / "tokenizer.json").resolve().name,
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_window_source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parent / "local_window_hybrid.py").read_bytes()
            ).hexdigest(),
            "streaming_memory_source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parent / "streaming_memory.py").read_bytes()
            ).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "corpus_tokens": int(len(ids)),
            "layer": args.layer,
            "local_window_including_current": 64,
            "decay": args.decay,
            "hybrid_memory_gain": args.memory_gain,
            "chunk": args.chunk,
            "conversion_mlx_seed": 0,
            "batch": args.batch,
            "contexts": contexts,
            "decode_tokens": args.decode_tokens,
            "repeats": args.repeats,
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "workload": "same 1B checkpoint; last-token prefill vocabulary head; "
                        "teacher-forced single-token decode; each call evaluated",
            "cache_bytes": "active logical KV tokens plus gated trace state",
            "memory_note": "teacher and candidate weights coexist; allocator memory "
                           "is not a deployment comparison",
            "hardware": platform.platform(),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "load_s": load_s,
            "baseline_active_memory_bytes": int(mx.get_active_memory()),
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "load_average_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
        },
        "conversion": installation.conversion,
        "parity": {},
        "rows": [],
    }
    atomic_write(args.output, record)
    for arm in arms:
        switch(arm)
        record["parity"][arm] = cache_parity(model, ids, arm, args.batch)
        atomic_write(args.output, record)
        print(f"{arm} cache parity max="
              f"{record['parity'][arm]['split_vs_whole_max_abs_logit']:.4f}",
              flush=True)

    for context_index, context in enumerate(contexts):
        future = ids[context:context + args.decode_tokens]
        # Warm the same shapes for each arm, excluded from measured repeats.
        for arm in arms:
            switch(arm)
            one_repeat(model, ids, context=context, future=future,
                       arm=arm, batch=args.batch)
        row = {"context": context, "batch": args.batch, "repeats": []}
        record["rows"].append(row)
        atomic_write(args.output, record)
        for repeat in range(args.repeats):
            rotation = (context_index + repeat) % len(arms)
            order = arms[rotation:] + arms[:rotation]
            readings = {}
            for arm in order:
                switch(arm)
                readings[arm] = one_repeat(
                    model, ids, context=context, future=future,
                    arm=arm, batch=args.batch)
            row["repeats"].append({
                "index": repeat,
                "order": list(order),
                "readings": readings,
                "prefill_ratio_local_over_teacher":
                    readings["local"]["prefill_s"] / readings["teacher"]["prefill_s"],
                "prefill_ratio_hybrid_over_teacher":
                    readings["hybrid"]["prefill_s"] / readings["teacher"]["prefill_s"],
                "decode_speedup_local_over_teacher":
                    readings["local"]["decode_tok_s"] / readings["teacher"]["decode_tok_s"],
                "decode_speedup_hybrid_over_teacher":
                    readings["hybrid"]["decode_tok_s"] / readings["teacher"]["decode_tok_s"],
            })
            atomic_write(args.output, record)
            print(f"T={context} repeat={repeat} order={','.join(order)} "
                  f"decode local/teacher="
                  f"{row['repeats'][-1]['decode_speedup_local_over_teacher']:.2f} "
                  f"hybrid/teacher="
                  f"{row['repeats'][-1]['decode_speedup_hybrid_over_teacher']:.2f}",
                  flush=True)
        row["median"] = {
            "prefill_ratio_local_over_teacher": statistics.median(
                p["prefill_ratio_local_over_teacher"] for p in row["repeats"]),
            "prefill_ratio_hybrid_over_teacher": statistics.median(
                p["prefill_ratio_hybrid_over_teacher"] for p in row["repeats"]),
            "decode_speedup_local_over_teacher": statistics.median(
                p["decode_speedup_local_over_teacher"] for p in row["repeats"]),
            "decode_speedup_hybrid_over_teacher": statistics.median(
                p["decode_speedup_hybrid_over_teacher"] for p in row["repeats"]),
        }
        atomic_write(args.output, record)
    record["meta"]["elapsed_s"] = time.perf_counter() - started
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_write(args.output, record)
    print(f"wrote raw benchmark {args.output}", flush=True)


if __name__ == "__main__":
    main()
