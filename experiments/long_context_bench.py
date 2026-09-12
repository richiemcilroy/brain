"""Does the FLOP advantage become a WALL-CLOCK advantage, and at what context?

WHY THIS EXISTS
---------------
The cost model in cost_model.py says gated memory is 1.14x cheaper than causal
attention at d=1024, T=512 and 4.0x cheaper at T=32768. Those are FLOP counts.
FLOPs are not wall clock, and the two operations have opposite execution
characters:

  * attention's context term is a DENSE MATMUL, which runs near peak FLOPs.
  * the scan's context term is ELEMENTWISE and bandwidth-bound, and the
    Hillis-Steele formulation stores O(log C) intermediate levels.

A FLOP advantage for a bandwidth-bound op over a matmul can easily be a
wall-clock LOSS. So the structural claim is only worth anything if it is
measured. That is what this file does: pure forward pass, no training, so the
number is a property of the architecture and not of the optimizer.

WHAT IS MEASURED
----------------
Per-token wall-clock for both arms at fixed d across a context sweep, plus the
crossover T. Warmup iterations are discarded and mx.eval is called every step,
because MLX is lazy and an unevaluated graph times at dispatch speed only.

PREDICTION STATED BEFORE THE RUN: memory should be slower than attention at
short T (where attention is a few large matmuls and the scan has log C
sequential levels of bandwidth traffic) and should cross over somewhere in the
thousands. If it NEVER crosses over in wall clock, the FLOP advantage is not
realisable on this hardware and the efficiency claim has no wall-clock content.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import (  # noqa: E402
    chunked_gated_scan, AttentionMemory, GatedMemory,
)

OUT = os.environ.get(
    "BRAIN_LCB_OUT", "/Volumes/T9/human-brain/scratch/long_context_bench.json"
)


def time_fn(fn, warmup=3, iters=8):
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.time()
    for _ in range(iters):
        mx.eval(fn())
    dt = (time.time() - t0) / iters
    return dt


def bench_attention(B, T, D, n_head):
    x = mx.random.normal((B, T, D))
    m = AttentionMemory(D, n_head)
    mx.eval(m.parameters())
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
    dt = time_fn(lambda: m(x, mask))
    return dt, B * T


def bench_memory(B, T, D, banks, chunk):
    x = mx.random.normal((B, T, D))
    m = GatedMemory(D, banks=banks, chunk=chunk)
    mx.eval(m.parameters())
    dt = time_fn(lambda: m(x, None))
    return dt, B * T


def bench_scan_only(B, T, D, chunk):
    """The scan in isolation, to separate it from the projections."""
    v = mx.random.normal((B, T, D))
    g = mx.sigmoid(mx.random.normal((B, T, D)))
    dt = time_fn(lambda: chunked_gated_scan(v, g, chunk))
    return dt, B * T


if __name__ == "__main__":
    D = int(os.environ.get("BRAIN_D", "512"))
    n_head = 8
    ts = [int(x) for x in os.environ.get(
        "BRAIN_TS", "512,1024,2048,4096,8192,16384").split(",")]
    out = []
    print(f"d={D}  heads={n_head}  forward pass only, warmup discarded")
    print(f"{'T':>7}{'B':>4}{'attn ms':>10}{'mem1 ms':>10}{'mem4 ms':>10}"
          f"{'scan ms':>10}"
          f"{'attn us/tok':>13}{'mem1 us/tok':>13}{'A/M1':>8}{'A/M4':>8}")
    for T in ts:
        # size the batch to keep total tokens roughly constant and the GPU fed
        B = max(1, min(16, 16384 // T))
        try:
            ta, n = bench_attention(B, T, D, n_head)
        except Exception as e:
            print(f"{T:>7}{B:>4}  attention failed: {type(e).__name__}: {e}")
            ta = None
        tm1 = tm4 = ts_ = None
        try:
            tm1, _ = bench_memory(B, T, D, 1, 64)
        except Exception as e:
            print(f"  mem1 failed at T={T}: {type(e).__name__}: {e}")
        try:
            tm4, _ = bench_memory(B, T, D, 4, 64)
        except Exception as e:
            print(f"  mem4 failed at T={T}: {type(e).__name__}: {e}")
        try:
            ts_, _ = bench_scan_only(B, T, D, 64)
        except Exception as e:
            pass

        def us(dt):
            return f"{dt/n*1e6:>13.3f}" if dt is not None else f"{'--':>13}"

        def rat(a, b):
            return f"{a/b:>7.2f}x" if (a and b) else f"{'--':>8}"

        print(f"{T:>7}{B:>4}"
              f"{(ta*1e3 if ta else float('nan')):>10.2f}"
              f"{(tm1*1e3 if tm1 else float('nan')):>10.2f}"
              f"{(tm4*1e3 if tm4 else float('nan')):>10.2f}"
              f"{(ts_*1e3 if ts_ else float('nan')):>10.2f}"
              f"{us(ta)}{us(tm1)}{rat(ta, tm1)}{rat(ta, tm4)}", flush=True)
        out.append(dict(T=T, B=B, d=D, n_head=n_head,
                        attn_ms=ta, mem1_ms=tm1, mem4_ms=tm4, scan_ms=ts_,
                        attn_us_per_tok=(ta / n * 1e6) if ta else None,
                        mem1_us_per_tok=(tm1 / n * 1e6) if tm1 else None,
                        ratio_A_M1=(ta / tm1) if (ta and tm1) else None,
                        ratio_A_M4=(ta / tm4) if (ta and tm4) else None))
        json.dump(out, open(OUT, "w"), indent=1)

    print()
    xs = [r for r in out if r["ratio_A_M1"]]
    if xs:
        print(f"attention is faster at T<={[r['T'] for r in xs if r['ratio_A_M1']>1][-1] if any(r['ratio_A_M1']>1 for r in xs) else 'n/a'}")
        cross = [r["T"] for r in xs if r["ratio_A_M1"] < 1.0]
        if cross:
            print(f"WALL-CLOCK CROSSOVER: memory becomes cheaper at T>={cross[0]}")
        else:
            print("NO WALL-CLOCK CROSSOVER in this sweep: attention is faster "
                  "everywhere measured, so the FLOP advantage is not realisable.")
    print(f"\nwrote {OUT}")
