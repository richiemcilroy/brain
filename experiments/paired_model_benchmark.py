"""Interleave teacher and converted-Llama runs to reduce machine-load drift.

Both arms use one loaded checkpoint and the same inputs. The original attention
and its replacement are both retained, so allocator peak memory is not a fair
deployment-memory comparison; logical cache bytes are comparable. Use
oss_model_benchmark.py's separate-process runs for memory context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from oss_model_benchmark import (  # noqa: E402
    CORPUS, DEFAULT_SNAPSHOT, attention_module, one_repeat, perplexity,
    prefix_equivalence, transplant,
)
from streaming_memory import StreamingGatedMemoryCarrier  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--contexts", default="128,512,2048,4096")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    if min(contexts) < 128 or args.decode_tokens < 1 or args.repeats < 1:
        raise ValueError("invalid contexts/repeats/decode-tokens")

    model, tokenizer = load(str(args.snapshot))
    model.eval()
    text = CORPUS.read_text()
    ids = np.asarray(tokenizer.encode(text), dtype=np.int32)
    n = len(ids)
    val = ids[int(0.9 * n) : int(0.9 * n) + 3000]
    layer_obj, attr, original = attention_module(model, args.layer)
    mx.random.seed(0)  # the transplanted carrier's gate is otherwise ambient RNG
    carrier, conversion = transplant(model, args.layer, "transfer",
                                     original=original, decay=args.decay)
    candidate = StreamingGatedMemoryCarrier(carrier)

    def switch(arm):
        setattr(layer_obj, attr, original if arm == "teacher" else candidate)

    controls = {}
    for arm in ("teacher", "transfer"):
        switch(arm)
        parity = prefix_equivalence(model, ids, arm, args.layer, args.batch)
        if (not parity["offsets_equal"] or not parity["decode_offset_correct"]
                or not parity["top1_equal"] or not parity["decode_top1_equal"]
                or max(parity["max_abs_logit_diff"],
                       parity["decode_max_abs_logit_diff"]) > 0.5):
            raise AssertionError(f"cache-equivalence guard failed: {arm}: {parity}")
        controls[arm] = {"parity": parity, "quality": perplexity(model, val)}
    print("quality", {a: round(v["quality"]["ppl"], 4)
                      for a, v in controls.items()}, flush=True)

    out = {
        "status": "running",
        "meta": {
            "snapshot": str(args.snapshot),
            "snapshot_revision": args.snapshot.name,
            "config_sha256": hashlib.sha256(
                (args.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (args.snapshot / "model.safetensors").resolve().name,
            "tokenizer_blob_id": (args.snapshot / "tokenizer.json").resolve().name,
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "oss_benchmark_source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parent / "oss_model_benchmark.py").read_bytes()
            ).hexdigest(),
            "streaming_memory_source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parent / "streaming_memory.py").read_bytes()
            ).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "layer": args.layer,
            "decay": args.decay,
            "conversion_mlx_seed": 0,
            "batch": args.batch,
            "contexts": contexts,
            "decode_tokens": args.decode_tokens,
            "repeats": args.repeats,
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "workload": "cached prefill with last-token vocabulary head and "
                        "teacher-forced single-token decode",
            "memory_note": "both original and replacement modules are retained; "
                           "compare logical cache bytes, not allocator peak",
            "cache_bytes": "active logical KV positions plus trace state",
            "cache_storage_bytes": "whole allocated KV tensors including spare capacity",
            "started_epoch": time.time(),
            "load_average_at_start": list(os.getloadavg()),
        },
        "conversion": conversion,
        "controls": controls,
        "rows": [],
    }
    output = args.output or Path(__file__).resolve().parent / "results" / f"oss_benchmark_paired_b{args.batch}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for context in contexts:
        future = ids[context : context + args.decode_tokens]
        for arm in ("teacher", "transfer"):
            switch(arm)
            one_repeat(model, ids, context=context, future=future, arm=arm,
                       layer=args.layer, batch=args.batch)
        pairs = []
        for i in range(args.repeats):
            order = ("teacher", "transfer") if i % 2 == 0 else ("transfer", "teacher")
            readings = {}
            for arm in order:
                switch(arm)
                readings[arm] = one_repeat(
                    model, ids, context=context, future=future, arm=arm,
                    layer=args.layer, batch=args.batch)
            pairs.append({
                "order": list(order),
                "teacher": readings["teacher"],
                "transfer": readings["transfer"],
                "prefill_time_ratio_transfer_over_teacher":
                    readings["transfer"]["prefill_s"] / readings["teacher"]["prefill_s"],
                "decode_speed_ratio_transfer_over_teacher":
                    readings["transfer"]["decode_tok_s"] / readings["teacher"]["decode_tok_s"],
                "cache_bytes_saved": readings["teacher"]["final_cache_bytes"] -
                                     readings["transfer"]["final_cache_bytes"],
                "cache_storage_bytes_saved":
                    readings["teacher"]["final_cache_storage_bytes"] -
                    readings["transfer"]["final_cache_storage_bytes"],
            })
        row = {
            "context": context,
            "batch": args.batch,
            "pairs": pairs,
            "median_prefill_time_ratio_transfer_over_teacher": statistics.median(
                p["prefill_time_ratio_transfer_over_teacher"] for p in pairs),
            "median_decode_speed_ratio_transfer_over_teacher": statistics.median(
                p["decode_speed_ratio_transfer_over_teacher"] for p in pairs),
            "median_cache_bytes_saved": statistics.median(
                p["cache_bytes_saved"] for p in pairs),
            "median_cache_storage_bytes_saved": statistics.median(
                p["cache_storage_bytes_saved"] for p in pairs),
        }
        out["rows"].append(row)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2) + "\n")
        tmp.replace(output)
        print(f"B={args.batch} T={context}: prefill time x"
              f"{row['median_prefill_time_ratio_transfer_over_teacher']:.2f}; "
              f"decode speed x{row['median_decode_speed_ratio_transfer_over_teacher']:.2f}; "
              f"cache saved {row['median_cache_bytes_saved']/1e6:.1f} MB",
              flush=True)
    out["status"] = "complete"
    out["meta"]["finished_epoch"] = time.time()
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    tmp.replace(output)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
