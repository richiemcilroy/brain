"""PATH A: a rigorous, self-checking kernel benchmark.

WHAT WENT WRONG LAST TIME -- three failure modes, all producing clean numbers:

1. WRONG BASELINE. Attention was hand-rolled (matmul -> triu mask -> softmax ->
   matmul) instead of mx.fast.scaled_dot_product_attention. The fused kernel is
   ~2x faster, so the comparison flattered our block.
2. MEASURED UNDER LOAD. Wall-clock taken while other MLX jobs ran; this project
   already knows that produces junk (docs/WALLCLOCK.md).
3. NO SANITY CHECK. A row where hand-rolled attention beat the fused kernel was
   reported instead of flagged as impossible.

DESIGN
------
- Arms: attn_fused (the fair baseline), mem (ours), attn_naive (kept only to
  quantify how much the old baseline was understated).
- Correctness: the two attention arms must compute the SAME function (shared
  projections, only the attention op differs) and the chunked scan must match
  the sequential recurrence. The run ABORTS on disagreement.
- Timing: min-of-N, mx.eval inside the loop, load average recorded per row.
- Impossible-row detection: if attn_naive beats attn_fused by >15% the row is
  flagged, because a fused kernel cannot be slower than an unfused
  implementation of the same computation.

FALSIFIER, stated before running: if memory is slower than attn_fused at EVERY
(T, d), the efficiency direction is dead.
"""
from __future__ import annotations
import json, os, platform, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import GatedMemory, chunked_gated_scan, sequential_scan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FB_OUT", os.path.join(HERE, "results", "fair_bench.json"))
WIDTHS = [int(x) for x in os.environ.get("FB_WIDTHS", "512,2048,4096").split(",")]
TS = [int(x) for x in os.environ.get("FB_TS", "128,512,1024,2048,4096").split(",")]
B = int(os.environ.get("FB_B", "4"))
REPS = int(os.environ.get("FB_REPS", "15"))


def load_avg():
    try:
        return round(float(os.getloadavg()[0]), 2)
    except Exception:
        return None


class _Attn(nn.Module):
    """Shared projections; `fused` selects only the attention op."""

    def __init__(self, d, n_head, fused):
        super().__init__()
        self.d, self.n_head, self.fused = d, n_head, fused
        self.q = nn.Linear(d, d, bias=True)
        self.k = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, d, bias=True)
        self.o = nn.Linear(d, d, bias=True)
        self.scale = (d // n_head) ** -0.5

    def __call__(self, x):
        B, T, _ = x.shape
        hd = self.d // self.n_head
        q = self.q(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        if self.fused:
            # mask="causal" is the fused causal path. Verified against the
            # explicit-mask arm in verify_implementations(): with mask=None the
            # fused op is UNMASKED (attends to the future) while the naive arm
            # below is causal, so timing those two would compare two different
            # functions. That mismatch is exactly what the self-check caught.
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                     mask="causal")
        else:
            s = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            mask = mx.triu(mx.full((T, T), -1e30, dtype=s.dtype), k=1)
            y = mx.softmax(s + mask, axis=-1) @ v
        return self.o(y.transpose(0, 2, 1, 3).reshape(B, T, self.d))


class Mem(nn.Module):
    def __init__(self, d, chunk=64):
        super().__init__()
        self.mem = GatedMemory(d, banks=1, chunk=chunk)

    def __call__(self, x):
        return self.mem(x)


def verify_implementations():
    """Both attention arms must compute the same function. Abort if not."""
    mx.random.seed(0)
    d, T, nh = 256, 65, 4
    x = mx.random.normal((2, T, d)).astype(mx.float32)
    a, f = _Attn(d, nh, False), _Attn(d, nh, True)
    for name in ("q", "k", "v", "o"):
        setattr(f, name, getattr(a, name))
    mx.eval(a.parameters(), f.parameters())
    ya, yf = a(x), f(x)
    mx.eval(ya, yf)
    err = float(mx.max(mx.abs(ya - yf)))
    rel = err / (float(mx.max(mx.abs(ya))) + 1e-9)
    print(f"  VERIFY naive-vs-fused attention: max abs diff {err:.3e} (rel {rel:.3e})",
          flush=True)
    if rel > 1e-3:
        raise RuntimeError(
            f"the two attention implementations disagree (rel {rel:.3e}), so "
            f"timing them measures different functions and any ratio is meaningless")

    v = mx.random.normal((1, 97, 32)).astype(mx.float32)
    g = mx.sigmoid(mx.random.normal((1, 97, 32)))
    mx.eval(v, g)
    e1 = float(mx.max(mx.abs(chunked_gated_scan(v, g, 16) - sequential_scan(v, g))))
    print(f"  VERIFY chunked scan vs sequential recurrence: max abs diff {e1:.3e}",
          flush=True)
    if e1 > 1e-3:
        raise RuntimeError(f"chunked scan disagrees with the recurrence ({e1:.3e})")


def timeit(fn, x, reps=REPS):
    for _ in range(3):                      # warmup, discarded
        mx.eval(fn(x))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(x))
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    la0 = load_avg()
    print(f"device {mx.default_device()} | B={B} reps={REPS} | load avg {la0}", flush=True)
    print("verifying implementations before timing anything:", flush=True)
    verify_implementations()
    if la0 is not None and la0 > 8:
        print(f"  *** WARNING: load average {la0} before starting. Wall-clock "
              f"numbers from a loaded machine are not trustworthy; every row "
              f"records its load so this can be audited later.", flush=True)

    out = dict(
        meta=dict(host=platform.platform(), machine=platform.machine(),
                  driver="experiments/fair_bench.py", B=B, reps=REPS,
                  widths=WIDTHS, Ts=TS, load_at_start=la0,
                  baseline="mx.fast.scaled_dot_product_attention (FUSED)",
                  correctness="verified in-run; aborts on disagreement",
                  falsifier="memory slower than fused at every (T,d) kills it"),
        rows=[])

    for d in WIDTHS:
        n_head = max(1, d // 64)
        for T in TS:
            x = mx.random.normal((B, T, d)).astype(mx.float32)
            mx.eval(x)
            an, af, mm = _Attn(d, n_head, False), _Attn(d, n_head, True), Mem(d)
            mx.eval(an.parameters(), af.parameters(), mm.parameters())
            tn, tf, tm = timeit(an, x), timeit(af, x), timeit(mm, x)
            impossible = bool(tn < tf * 0.85)
            row = dict(d=d, T=T, n_head=n_head, B=B, load=load_avg(),
                       attn_naive_ms=tn * 1e3, attn_fused_ms=tf * 1e3,
                       mem_ms=tm * 1e3,
                       mem_over_fused=tm / tf, mem_over_naive=tm / tn,
                       fused_speedup_vs_naive=tn / tf,
                       mem_faster_than_fused=bool(tm < tf),
                       flagged_impossible=impossible)
            out["rows"].append(row)
            flag = "  <-- FLAGGED: naive beat fused, not physically sensible" if impossible else ""
            print(f"  d={d:5d} T={T:5d} load {row['load']:>6} | naive {tn*1e3:9.3f} "
                  f"fused {tf*1e3:9.3f} mem {tm*1e3:9.3f} | mem/fused {tm/tf:6.3f} "
                  f"{'MEM' if tm<tf else 'ATTN'}{flag}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)

    rows = out["rows"]
    clean = [r for r in rows if not r["flagged_impossible"]]
    out["verdict"] = dict(
        n_points=len(rows),
        n_clean=len(clean),
        n_flagged_impossible=len(rows) - len(clean),
        n_mem_faster_than_fused=sum(1 for r in clean if r["mem_faster_than_fused"]),
        n_mem_faster_than_naive=sum(1 for r in clean if r["mem_over_naive"] < 1),
        median_fused_speedup_vs_naive=float(np.median([r["fused_speedup_vs_naive"] for r in clean])),
        claim_survives=bool(clean and all(r["mem_faster_than_fused"] for r in clean)),
        load_max=max(r["load"] for r in rows if r["load"] is not None),
    )
    json.dump(out, open(OUT, "w"), indent=1)

    v = out["verdict"]
    print(f"\n=== VERDICT ===", flush=True)
    print(f"points: {v['n_points']} ({v['n_clean']} clean, {v['n_flagged_impossible']} flagged)", flush=True)
    print(f"memory faster than FUSED attention: {v['n_mem_faster_than_fused']}/{v['n_clean']} clean points", flush=True)
    print(f"memory faster than NAIVE attention: {v['n_mem_faster_than_naive']}/{v['n_clean']}", flush=True)
    print(f"fused kernel median speedup over naive: {v['median_fused_speedup_vs_naive']:.2f}x", flush=True)
    print(f"max load during run: {v['load_max']}", flush=True)
    print(f"EFFICIENCY CLAIM SURVIVES: {v['claim_survives']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
