"""FAIR CROSSOVER: is the memory advantage real once attention gets its fused kernel?

THE FLAW IN THE PREVIOUS BENCHMARK
----------------------------------
width_crossover.py built attention as explicit matmul -> mask -> softmax -> matmul,
and (worse) constructed an (T, T) causal mask with mx.triu before every call. MLX
ships `mx.fast.scaled_dot_product_attention`, a fused kernel that applies causal
masking internally and never materialises the T x T matrix.

So the previous benchmark may have been comparing our memory against a
hand-rolled, deliberately slow attention. That is exactly the class of error
this project has already made twice (the FLOP/MAC unit error, the bigram floor),
and if it holds here then the entire "memory is faster at every width" result is
an artifact.

This script re-measures with THREE attention arms:
  attn_naive   matmul/softmax/matmul + explicit mask construction  (previous)
  attn_fused   mx.fast.scaled_dot_product_attention               (the fair baseline)
  mem          our gated memory block

and reports the memory/attn ratio against EACH. The claim only survives if the
ratio is still below 1 against the FUSED baseline.

FALSIFIER, stated before running: if memory is slower than attn_fused at every
(T, d) tested, the efficiency claim is dead and must be retracted.
"""
from __future__ import annotations
import json, os, platform, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import GatedMemory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("WCF_OUT", os.path.join(HERE, "results", "width_crossover_fair.json"))
WIDTHS = [int(x) for x in os.environ.get("WCF_WIDTHS", "512,2048,4096").split(",")]
TS = [int(x) for x in os.environ.get("WCF_TS", "128,512,1024,2048,4096").split(",")]
B = int(os.environ.get("WCF_B", "4"))
REPS = int(os.environ.get("WCF_REPS", "9"))


class AttnNaive(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.d, self.n_head = d, n_head
        self.q = nn.Linear(d, d, bias=True); self.k = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, d, bias=True); self.o = nn.Linear(d, d, bias=True)
        self.scale = (d // n_head) ** -0.5

    def __call__(self, x):
        B, T, _ = x.shape
        hd = self.d // self.n_head
        q = self.q(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = mx.triu(mx.full((T, T), -1e30, dtype=s.dtype), k=1)
        w = mx.softmax(s + mask, axis=-1)
        y = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
        return self.o(y)


class AttnFused(nn.Module):
    """Same projections, but the attention op is MLX's fused kernel."""

    def __init__(self, d, n_head):
        super().__init__()
        self.d, self.n_head = d, n_head
        self.q = nn.Linear(d, d, bias=True); self.k = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, d, bias=True); self.o = nn.Linear(d, d, bias=True)
        self.scale = (d // n_head) ** -0.5

    def __call__(self, x):
        B, T, _ = x.shape
        hd = self.d // self.n_head
        q = self.q(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        # mask="causal" is the FUSED CAUSAL path. mask=None is UNMASKED: it
        # attends to future tokens. This was a real defect -- the previous
        # version passed mask=None while the memory block is causal by
        # construction, so the "fair" baseline was solving an EASIER problem
        # (it could see the answer). Prefix-perturbation check: with mask=None,
        # perturbing the last key moves position 0's output by 0.103; with
        # mask="causal", by exactly 0.0. verify_causality() now asserts this.
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        return self.o(y)


class Mem(nn.Module):
    def __init__(self, d, chunk=64):
        super().__init__()
        self.mem = GatedMemory(d, banks=1, chunk=chunk)

    def __call__(self, x):
        return self.mem(x)


def verify_causality():
    """Hard self-check: the timing arms must compute the SAME function.

    A previous version of this file passed mask=None to the fused kernel, which
    in MLX is UNMASKED -- it attends to future tokens. The memory block is
    causal by construction, so that "fair" baseline was solving an easier
    problem and any ratio was meaningless. This is the same failure class as the
    bigram-floor baseline and the MAC/FLOP unit error, so it is checked in-run
    and the run aborts rather than publishing a number.
    """
    mx.random.seed(0)
    T, d, nh = 32, 64, 4
    x = mx.random.normal((2, T, d)).astype(mx.float32)
    a, f = AttnNaive(d, nh), AttnFused(d, nh)
    for nm in ("q", "k", "v", "o"):
        setattr(f, nm, getattr(a, nm))
    mx.eval(a.parameters(), f.parameters())
    ya, yf = a(x), f(x)
    mx.eval(ya, yf)
    rel = float(mx.max(mx.abs(ya - yf))) / (float(mx.max(mx.abs(ya))) + 1e-9)
    # prefix-perturbation: changing the LAST token must not move position 0
    xp = x.at[:, T - 1, :].add(7.0)
    leak_n = float(mx.max(mx.abs(a(x)[:, 0, :] - a(xp)[:, 0, :])))
    leak_f = float(mx.max(mx.abs(f(x)[:, 0, :] - f(xp)[:, 0, :])))
    print(f"  VERIFY naive-vs-fused: rel {rel:.3e} | "
          f"causal leak naive {leak_n:.3e} fused {leak_f:.3e}", flush=True)
    if rel > 1e-3:
        raise RuntimeError(
            f"timing arms disagree (rel {rel:.3e}); a ratio across different "
            f"functions is meaningless")
    if leak_f > 1e-6 or leak_n > 1e-6:
        raise RuntimeError(
            f"an arm is NON-CAUSAL (leak naive {leak_n:.3e}, fused {leak_f:.3e}); "
            f"mask=None in MLX is unmasked and attends to the future")


def timeit(fn, x, reps=REPS):
    best = float("inf")
    for _ in range(3):
        mx.eval(fn(x))
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn(x)); best = min(best, time.perf_counter() - t0)
    return best


def main():
    print(f"device {mx.default_device()} B={B} reps={REPS}", flush=True)
    verify_causality()
    out = dict(meta=dict(host=platform.platform(), machine=platform.machine(),
                         driver="experiments/width_crossover_fair.py", B=B, reps=REPS,
                         widths=WIDTHS, Ts=TS,
                         note="attn_fused uses mx.fast.scaled_dot_product_attention; "
                              "attn_naive is the previous hand-rolled baseline"),
               rows=[])
    for d in WIDTHS:
        n_head = max(1, d // 64)
        for T in TS:
            x = mx.random.normal((B, T, d)).astype(mx.float32); mx.eval(x)
            an, af, mm = AttnNaive(d, n_head), AttnFused(d, n_head), Mem(d)
            mx.eval(an.parameters(), af.parameters(), mm.parameters())
            tn, tf, tm = timeit(an, x), timeit(af, x), timeit(mm, x)
            row = dict(d=d, T=T, n_head=n_head, B=B,
                       attn_naive_ms=tn*1e3, attn_fused_ms=tf*1e3, mem_ms=tm*1e3,
                       mem_over_naive=tm/tn, mem_over_fused=tm/tf,
                       fused_speedup_vs_naive=tn/tf,
                       mem_faster_than_fused=bool(tm < tf),
                       mac_ratio_vs_naive=(4*d*d+2*T*d)/(3*d*d),
                       mac_ratio_vs_fused=(4*d*d+2*T*d)/(3*d*d))
            out["rows"].append(row)
            print(f"  d={d:5d} T={T:5d} naive {tn*1e3:8.3f} fused {tf*1e3:8.3f} "
                  f"mem {tm*1e3:8.3f} | mem/fused {tm/tf:5.3f} "
                  f"{'MEM faster' if tm<tf else 'ATTN FASTER'}  (fused is {tn/tf:.1f}x faster than naive)", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]
    out["verdict"] = dict(
        n_points=len(rows),
        n_mem_faster_than_naive=int(sum(r["mem_faster_than_naive"] if "mem_faster_than_naive" in r
                                        else r["mem_over_naive"] < 1 for r in rows)),
        n_mem_faster_than_fused=int(sum(1 for r in rows if r["mem_over_fused"] < 1)),
        median_fused_speedup_vs_naive=float(np.median([r["fused_speedup_vs_naive"] for r in rows])),
        claim_survives=bool(all(r["mem_over_fused"] < 1 for r in rows)))
    json.dump(out, open(OUT, "w"), indent=1)
    v = out["verdict"]
    print(f"\nmemory faster than NAIVE  at {v['n_mem_faster_than_naive']}/{v['n_points']} points", flush=True)
    print(f"memory faster than FUSED  at {v['n_mem_faster_than_fused']}/{v['n_points']} points", flush=True)
    print(f"fused kernel is {v['median_fused_speedup_vs_naive']:.2f}x faster than the hand-rolled baseline", flush=True)
    print(f"CLAIM SURVIVES AGAINST THE FAIR BASELINE: {v['claim_survives']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
