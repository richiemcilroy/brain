"""WHERE IS THE CROSSOVER AT REALISTIC WIDTH?

THE ISSUE WITH THE EXISTING BENCHMARK
-------------------------------------
scan_bench.py measured the memory-vs-attention wall-clock ratio at d=128, and
found memory only becomes consistently faster above T~256. But d=128 is a toy
width; Llama-3.2-1B uses d=2048 and Llama-3.1-8B uses d=4096.

The cost model says the crossover MOVES WITH d:

    attention (causal): 4*d^2  +  2*T*d     MACs/token
    gated memory:       3*d^2  +  O(T*d)    MACs/token

The T-dependent term in attention is 2*T*d; memory's is a scan that touches
T*d values once. So attention's share grows with T while memory's is flat at
3*d^2, and the ratio at large T is roughly 2*T*d / 3*d^2 = 2T/(3d).

That means the crossover context length scales with d:
  - at d=128 you need T ~ 128-256 to break even (what scan_bench found)
  - at d=4096 you would need T ~ 4096

If true, the honest efficiency claim is NOT "memory is faster" but "memory is
faster above a context length that scales with model width", which is a much
narrower and more testable statement -- and it predicts that at realistic width
the crossover is around the model's own training context, i.e. the advantage is
marginal exactly where transformers actually operate.

This script measures it directly. Both blocks are timed with mx.eval INSIDE the
timing loop, min-of-N, at matched batch size.

WHAT WOULD FALSIFY THE COST MODEL: if the measured crossover does NOT move with
d (stays near T~256 at every width), then the d^2 projection terms are not what
dominates and the model above is wrong.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import GatedMemory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("WC_OUT", os.path.join(HERE, "results", "width_crossover.json"))

WIDTHS = [int(x) for x in os.environ.get("WC_WIDTHS", "128,512,2048,4096").split(",")]
TS = [int(x) for x in os.environ.get("WC_TS", "128,256,512,1024,2048,4096").split(",")]
B = int(os.environ.get("WC_B", "4"))
REPS = int(os.environ.get("WC_REPS", "9"))


class AttnBlock(nn.Module):
    """GPT-2-style causal self-attention sublayer at width d."""

    def __init__(self, d, n_head):
        super().__init__()
        self.d, self.n_head = d, n_head
        hd = d // n_head
        self.q = nn.Linear(d, d, bias=True)
        self.k = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, d, bias=True)
        self.o = nn.Linear(d, d, bias=True)
        self.scale = hd ** -0.5

    def __call__(self, x):
        B, T, _ = x.shape
        hd = self.d // self.n_head
        q = self.q(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = mx.triu(mx.full((T, T), -1e30, dtype=scores.dtype), k=1)
        scores = scores + mask
        w = mx.softmax(scores, axis=-1)
        y = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
        return self.o(y)


class MemBlock(nn.Module):
    """Gated multi-timescale memory sublayer at width d, same shape contract."""

    def __init__(self, d, chunk=64):
        super().__init__()
        self.mem = GatedMemory(d, banks=1, chunk=chunk)

    def __call__(self, x):
        return self.mem(x)


def timeit(fn, x, reps):
    """min-of-N with mx.eval inside the loop; returns seconds."""
    best = float("inf")
    for _ in range(3):          # warmup, discarded
        y = fn(x)
        mx.eval(y)
    for _ in range(reps):
        t0 = time.perf_counter()
        y = fn(x)
        mx.eval(y)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    print(f"device {mx.default_device()} | B={B} reps={REPS}", flush=True)
    out = dict(
        meta=dict(host=platform.platform(), machine=platform.machine(),
                  driver="experiments/width_crossover.py", B=B, reps=REPS,
                  widths=WIDTHS, Ts=TS,
                  cost_model=dict(
                      attention_causal="4*d^2 + 2*T*d MACs/token",
                      memory="3*d^2 + O(T*d) MACs/token",
                      prediction="crossover T scales with d"),
                  elapsed_s=None),
        rows=[])

    for d in WIDTHS:
        n_head = max(1, d // 64)          # head_dim 64, as in both Llama sizes
        for T in TS:
            x = mx.random.normal((B, T, d)).astype(mx.float32)
            mx.eval(x)
            attn = AttnBlock(d, n_head)
            mem = MemBlock(d)
            mx.eval(attn.parameters(), mem.parameters())
            ta = timeit(attn, x, REPS)
            tm = timeit(mem, x, REPS)
            # MAC bookkeeping (per token)
            mac_attn = 4 * d * d + 2 * T * d
            mac_mem = 3 * d * d
            row = dict(d=d, T=T, n_head=n_head, B=B,
                       attn_ms=ta * 1e3, mem_ms=tm * 1e3,
                       mem_over_attn=tm / ta,
                       mem_faster=bool(tm < ta),
                       mac_attn_per_token=mac_attn, mac_mem_per_token=mac_mem,
                       mac_ratio=mac_attn / mac_mem)
            out["rows"].append(row)
            print(f"  d={d:5d} T={T:5d}  attn {ta*1e3:8.3f} ms  mem {tm*1e3:8.3f} ms  "
                  f"ratio {tm/ta:5.3f}  {'MEM faster' if tm<ta else 'attn faster'}  "
                  f"(mac ratio {mac_attn/mac_mem:.2f}x)", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)

    # crossover per width: smallest T with ratio < 1
    cross = {}
    for d in WIDTHS:
        rs = [r for r in out["rows"] if r["d"] == d]
        hit = [r["T"] for r in rs if r["mem_over_attn"] < 1.0]
        cross[str(d)] = dict(first_T_mem_faster=min(hit) if hit else None,
                             predicted_T_approx=d)
    out["crossover"] = cross
    out["meta"]["elapsed_s"] = None
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nCROSSOVER SUMMARY (predicted: crossover T ~ d)", flush=True)
    for d, v in cross.items():
        print(f"  d={d:>5}: first T where memory is faster = "
              f"{v['first_T_mem_faster']}  (predicted ~{v['predicted_T_approx']})", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
