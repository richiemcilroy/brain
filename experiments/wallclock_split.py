"""Where does wall-clock ACTUALLY go in a memory block? (It is not where the FLOP model says.)

WHY THIS EXISTS
---------------
Every efficiency argument in this repo until now has been made in MACs or FLOPs.
A FLOP count is a claim about arithmetic; wall clock is a claim about the
machine. Those can disagree by an order of magnitude when the work has the
wrong execution character, and this file measures the disagreement rather than
assuming it away.

The specific worry: the gated-memory arm's context term is an ELEMENTWISE
recurrence and the attention arm's is a DENSE MATMUL. A matmul runs near peak;
an elementwise scan does not. So the memory arm's 2.9x MAC advantage could
easily fail to appear in wall clock, and the only way to know is to time it.

WHAT IS MEASURED
----------------
Per-component min-of-N wall clock for a single Block at fixed d, plus the MAC
count for the same component, so the two can be compared directly. Backward
pass is NOT included: this is a property of the forward architecture, not of
the optimizer, and it keeps the measurement stable.

MLX IS LAZY. An unevaluated graph times at dispatch speed and produces
meaningless numbers, so mx.eval is called inside every timed region and warmup
iterations are discarded. Min is reported rather than mean because the
distribution is right-skewed by scheduler noise.

RESULT (measured, M4 Max, MLX 0.31, d=128, B=16, T=512)
------------------------------------------------------
The primitive that owns 72.7% of the MACs (the MLP) owns less of the wall clock
than the primitive that owns 27.3% (the scan), so the scan is the wall-clock
bottleneck. Component times each include a fixed dispatch cost and are not
additive; the dispatch floor is measured and reported. Attention is charged a
TRUE MAC count (4d^2 + 2Td, i.e. the full masked T x T matrix). An earlier
version charged attention the 2x FLOP figure used elsewhere in this repo while
charging memory and the MLP a true 1x, overstating the MAC advantage by 2x.

CONSEQUENCE: optimising FLOPs in this architecture is close to useless. The
next real lever is the scan's execution character, not its arithmetic.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import GatedMemory, AttentionMemory, Block  # noqa: E402

OUT = os.environ.get(
    "BRAIN_WC_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                 "wallclock_split.json"),
)
D, B, T = 128, 16, 512
REPS = 30


def bench(fn, reps=REPS):
    """min/median ms per call. mx.eval every step: MLX is lazy."""
    for _ in range(4):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    return float(ts.min()), float(np.median(ts))


def main():
    g = globals()
    mx.random.seed(0)
    d = g["D"]
    B = int(os.environ.get("BRAIN_WC_B", str(g["B"])))
    T = int(os.environ.get("BRAIN_WC_T", str(g["T"])))
    x = mx.random.normal((B, T, d))
    mx.eval(x)

    gm = GatedMemory(d, 1)
    am = AttentionMemory(d, 4)
    ln = nn.LayerNorm(d)
    mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    blk = Block(d, 4, "mem", 1, 64)
    ablk = Block(d, 4, "attn", 1, 64)
    for m in (gm, am, ln, mlp, blk, ablk):
        mx.eval(m.parameters())

    mem_macs = 3 * d * d
    mlp_macs = 2 * d * 4 * d
    # TRUE MACs. flops_per_token() in llm_efficiency charges 2x for every arm;
    # an earlier version of this file charged attention that 2x while charging
    # memory and the MLP a true 1x, overstating attention by exactly 2x.
    attn_macs = 4 * d * d + 2 * T * d          # full masked T x T scores + AV
    attn_macs_causal = 4 * d * d + T * d       # if the kernel skipped masked work
    block_macs = mem_macs + mlp_macs
    ablock_macs = attn_macs + mlp_macs
    ablock_macs_causal = attn_macs_causal + mlp_macs

    # Dispatch floor: time a trivial op on a tensor of the same size class.
    # Every separately-timed component pays this, so component times are NOT
    # additive and the floor must be reported to read the table honestly.
    tiny = mx.zeros((4,))
    mx.eval(tiny)
    n_call = 60
    t0 = time.perf_counter()
    for _ in range(n_call):
        mx.eval(tiny + 1.0)
    dispatch_floor_ms = (time.perf_counter() - t0) / n_call * 1e3

    rows = []
    t_ln, _ = bench(lambda: ln(x))
    t_mem, _ = bench(lambda: gm(x))
    t_mlp, _ = bench(lambda: mlp(ln(x)))
    t_blk, t_blk_med = bench(lambda: blk(x))
    t_ablk, t_ablk_med = bench(lambda: ablk(x))

    for label, t, macs, tot in [
        ("LayerNorm", t_ln, 0, t_blk),
        ("GatedMemory(banks=1)", t_mem, mem_macs, t_blk),
        ("MLP(d->4d->d)+LN", t_mlp, mlp_macs, t_blk),
        ("Block(mem) total", t_blk, block_macs, t_blk),
        ("Block(attn) total", t_ablk, ablock_macs, t_blk),
    ]:
        rows.append(dict(
            component=label, macs_per_token=macs, min_ms=round(t, 4),
            share_of_mem_block_wall_pct=round(100.0 * t / t_blk, 1),
            mac_share_of_mem_block_pct=round(100.0 * macs / block_macs, 1),
        ))

    out = dict(
        host=dict(platform=sys.platform, machine=os.uname().machine),
        config=dict(d=d, batch=B, ctx=T, reps=REPS,
                    mlx=mx.__version__ if hasattr(mx, "__version__") else None),
        note=("Shares sum above 100% because components overlap on the GPU. "
              "min-of-N with warmup discarded and mx.eval every step."),
        rows=rows,
        headline=dict(
            memory_prime_wall_share_pct=round(100.0 * t_mem / t_blk, 1),
            memory_prime_mac_share_pct=round(100.0 * mem_macs / block_macs, 1),
            mlp_wall_share_pct=round(100.0 * t_mlp / t_blk, 1),
            mlp_mac_share_pct=round(100.0 * mlp_macs / block_macs, 1),
            mem_over_attn_wall_ratio=round(t_blk / t_ablk, 3),
            mem_over_attn_mac_ratio=round(ablock_macs / block_macs, 3),
            mem_over_attn_mac_ratio_causal=round(ablock_macs_causal / block_macs, 3),
            attention_macs_full_masked=attn_macs,
            attention_macs_causal_skipping=attn_macs_causal,
            dispatch_floor_ms=round(dispatch_floor_ms, 4),
            block_ms_min=round(t_blk, 4), block_ms_median=round(t_blk_med, 4),
            attn_block_ms_min=round(t_ablk, 4),
            attn_block_ms_median=round(t_ablk_med, 4),
        ),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    print(f"d={d} B={B} T={T}  MLX {out['config']['mlx']}  min-of-{REPS}")
    print(f"dispatch floor for a trivial op: {dispatch_floor_ms:.4f} ms "
          f"(component times below include this; they are not additive)\n")
    print(f"{'component':<24}{'MACs/tok':>11}{'min ms':>9}{'wall share':>12}{'MAC share':>11}")
    for r in rows:
        print(f"{r['component']:<24}{r['macs_per_token']:>11,}{r['min_ms']:>9.3f}"
              f"{r['share_of_mem_block_wall_pct']:>11.1f}%"
              f"{r['mac_share_of_mem_block_pct']:>10.1f}%")
    h = out["headline"]
    print()
    print(f"memory primitive: {h['memory_prime_mac_share_pct']:.1f}% of MACs "
          f"but {h['memory_prime_wall_share_pct']:.1f}% of wall")
    print(f"MLP:              {h['mlp_mac_share_pct']:.1f}% of MACs "
          f"but {h['mlp_wall_share_pct']:.1f}% of wall")
    print(f"mem/attn wall ratio {h['mem_over_attn_wall_ratio']} "
          f"vs MAC ratio {h['mem_over_attn_mac_ratio']} (full masked) "
          f"/ {h['mem_over_attn_mac_ratio_causal']} (causal-skipping)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
