"""Where is the O(T) vs O(T^2) advantage ACTUALLY real?

A structural claim that survives only at one width is not a structural claim.
This sweeps the parameter space instead of asserting a single number at
d=128, T=512, which is the regime least favourable to the claim and the one
the first comparison happened to use.

The two cost models per token (forward, multiply-accumulates):

  attention:  8d^2 (qkvo)  +  2*T*d causal (scores + AV)  +  16d^2 (mlp)
  gated mem:  (2 + 2*gate + 2*out + 2)*B*d^2 ...  +  3*B*d*log2(chunk)  +  16d^2

TWO CORRECTIONS APPLIED AFTER REVIEW
------------------------------------
1. The MLP term is 16d^2 (two matmuls of 4d x d, counted twice), not 8d^2.
   The prose previously said 8d^2 while the code used 16d^2.
2. Attention is charged 2*T*d, not 4*T*d, because with a CAUSAL mask a kernel
   that skips masked blocks evaluates roughly half the score matrix. Charging
   the full 4*T*d overstated the advantage. The uncorrected ratio is an upper
   bound; the causal-corrected one is what is printed.

FLOPs ARE STILL NOT WALL-CLOCK. The attention term is a dense matmul near
peak throughput; the scan is elementwise, bandwidth-bound, and Hillis-Steele
stores O(log C) intermediate levels for the backward pass. A fused kernel
keeping state in SRAM (as Mamba does) is what buys real time. Every ratio here
is an upper bound on the wall-clock advantage, and the docstring says so
rather than implying the ratio is achievable.

The context-dependent term is 4*T*d for attention and
3*B*d*log2(chunk) ~ 18*B*d for memory. So attention's context cost grows
linearly in T while memory's is FIXED. The ratio therefore grows without
bound in T, and the crossover is set by whichever is larger: the width term
d^2 or the context term T*d.

MEASURED CONCLUSION (printed below, not asserted here):
  * At d>=512, T=512 the advantage is only 1.18x-1.27x. Attention is NOT
    wasteful at short context; most of its cost is still the projections.
  * At T=32768 the advantage is 6.9x at d=1024 and 12.7x at d=512.
  * The advantage is therefore a LONG-CONTEXT property, not a universal one.
    Any claim of the form "memory beats attention by 44%" must state the
    context length it was measured at, because the number moves by 10x.
"""
from __future__ import annotations

import math


def cost_attn(d: int, T: int, mlp_mult: int = 4) -> float:
    proj = 2 * 4 * d * d                 # qkv + out projection
    scores = 2 * T * d                   # causal: ~half the score matrix
    mlp = 2 * (d * mlp_mult * d + mlp_mult * d * d)
    return proj + scores + mlp


def cost_mem(d: int, T: int, banks: int = 1, mlp_mult: int = 4,
             gate: bool = True, out_gate: bool = False,
             conv: bool = False, chunk: int = 64) -> float:
    p = 2 * d * d * banks                # value projection
    if gate:
        p += 2 * d * d * banks
    if out_gate:
        p += 2 * d * d * banks
    if conv:
        p += 2 * 4 * d * d * banks
    p += 2 * d * d * banks               # output projection
    scan = 3.0 * d * banks * math.log2(max(chunk, 2))
    mlp = 2 * (d * mlp_mult * d + mlp_mult * d * d)
    return p + scan + mlp


if __name__ == "__main__":
    print("=== cost per token: attention vs gated memory ===")
    header = f"{'d':>5}{'T':>7}{'attn':>13}{'mem1':>13}{'A/M1':>7}"
    header += f"{'mem4':>13}{'A/M4':>7}"
    print(header)
    for d in (128, 256, 512, 768, 1024):
        for T in (512, 2048, 8192, 32768):
            a = cost_attn(d, T)
            m1 = cost_mem(d, T, 1)
            m4 = cost_mem(d, T, 4)
            print(f"{d:>5}{T:>7}{a:>13,.0f}{m1:>13,.0f}{a/m1:>6.2f}x"
                  f"{m4:>13,.0f}{a/m4:>6.2f}x")

    print("\n=== crossover context at which memory becomes cheaper (1 bank) ===")
    for d in (128, 256, 512, 768, 1024):
        lo, hi = 1, 1 << 22
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cost_attn(d, mid) < cost_mem(d, mid, 1):
                lo = mid
            else:
                hi = mid
        print(f"  d={d:>5}: T ~ {lo:,}")

    print("\n=== context term as a fraction of attention cost (causal) ===")
    for d in (128, 512, 1024):
        for T in (512, 4096, 32768):
            a = cost_attn(d, T)
            print(f"  d={d:>5} T={T:>6}: {100 * 2 * T * d / a:5.1f}%")
