"""Does the memory advantage grow with context? This is the structural test.

WHY THIS EXPERIMENT EXISTS
--------------------------
A memory arm and an attention arm can be tuned to tie at one context length.
That tells you almost nothing, because you can spend parameters to buy back
what you lose. The interesting question is structural:

    attention pays 2*2*T*d FLOPs per token for scores + AV   -> grows with T
    gated memory pays 3*d*banks*log2(chunk) per token        -> flat in T

So the two arms MUST diverge as T grows, and the only questions are (a) at
what T the divergence becomes measurable, and (b) how fast it compounds. This
file measures bpc and wall-clock for both arms across a context sweep at
MATCHED parameter count and MATCHED training tokens.

A prediction stated before the run: if the mechanism is real, the bpc gap
should be non-decreasing in T and the wall-clock ratio should fall. If the
gap CLOSES at long T, the mechanism is not what we think it is and the claim
in docs/RESULTS.md must be retracted.
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
import mlx.optimizers as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import (  # noqa: E402
    build, n_params, flops_per_token, load_corpus, make_batcher, verify_scan,
    CORPUS,
)

OUT = os.environ.get(
    "BRAIN_SCALE_OUT", "/Volumes/T9/human-brain/scratch/llm_context_scaling.json"
)


def run(arm, data, vocab, ctx, steps, bs=16, d=128, lr=3e-3, seed=0):
    m, spec = build(arm, vocab, d, ctx)
    mx.eval(m.parameters())
    P = n_params(m)
    n_layer = spec.get("n_layer", spec.get("depth", 1) * spec.get("n_col", 1))
    mult = spec.get("loops", 1) if spec.get("looped") else 1
    FL = flops_per_token(
        vocab, d, n_layer, ctx,
        "attn" if spec["kind"] == "attn" else "mem",
        banks=spec.get("banks", 1), mult=mult, n_col=spec.get("n_col", 1),
    )
    batch = make_batcher(data, ctx)
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)

    # warm up so compile/first-touch cost does not land in the measured window
    x, y = batch(bs, rng)
    l, g = lg(m, x, y)
    opt.update(m, g)
    mx.eval(m.parameters(), opt.state)
    t0 = time.time()
    ntok = x.size

    for s in range(1, steps + 1):
        x, y = batch(bs, rng)
        l, g = lg(m, x, y)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
    wall = time.time() - t0
    xv, yv = batch(64, rng, "val")
    vl = float(loss_fn(m, xv, yv))
    return dict(
        arm=arm, ctx=ctx, d=d, steps=steps, bs=bs, seed=seed,
        params=P, flops_per_token=FL, bpc=vl / math.log(2), val_loss=vl,
        wall_s=wall, tok_s=ntok / wall, tokens=ntok,
        cum_flops=ntok * FL, spec=spec,
    )


if __name__ == "__main__":
    err = verify_scan()
    print(f"scan verification max err={err:.3e}")
    if err > 1e-4:
        raise SystemExit("scan mismatch")

    data, vocab = load_corpus(CORPUS)
    ctxs = [int(c) for c in os.environ.get(
        "BRAIN_CTXS", "256,512,1024,2048,4096").split(",")]
    # matched training tokens per arm: hold tokens (not steps) constant
    tokens_target = int(os.environ.get("BRAIN_TOKENS", "4096000"))
    out = []
    for ctx in ctxs:
        steps = max(20, tokens_target // (16 * ctx))
        for arm in ("A_attention", "C_gated_banks4"):
            print(f"\n=== ctx={ctx} steps={steps} {arm} ===", flush=True)
            try:
                r = run(arm, data, vocab, ctx, steps)
            except Exception as e:  # OOM at long ctx is a real result, record it
                print(f"  FAILED at ctx={ctx}: {type(e).__name__}: {e}")
                out.append(dict(arm=arm, ctx=ctx, error=f"{type(e).__name__}: {e}"))
                continue
            out.append(r)
            print(
                f"  params={r['params']:,} fl/tok={r['flops_per_token']:,.0f} "
                f"bpc={r['bpc']:.4f} tok/s={r['tok_s']:,.0f} "
                f"wall={r['wall_s']:.1f}s", flush=True
            )
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"  [checkpoint written]")

    # summary
    print("\n\n=== SUMMARY: bpc and wall-clock ratio vs context ===")
    print(f"{'ctx':>6} {'attn bpc':>10} {'mem bpc':>10} {'bpc gain':>9} "
          f"{'attn t/s':>11} {'mem t/s':>11} {'speedup':>8} {'flop ratio':>11}")
    for ctx in ctxs:
        a = next((r for r in out if r["arm"] == "A_attention"
                  and r["ctx"] == ctx and "error" not in r), None)
        b = next((r for r in out if r["arm"] == "C_gated_banks4"
                  and r["ctx"] == ctx and "error" not in r), None)
        if not a or not b:
            print(f"{ctx:>6} {'(incomplete)':>10}")
            continue
        print(f"{ctx:>6} {a['bpc']:>10.4f} {b['bpc']:>10.4f} "
              f"{a['bpc']-b['bpc']:>+9.4f} {a['tok_s']:>11,.0f} "
              f"{b['tok_s']:>11,.0f} {b['tok_s']/a['tok_s']:>7.2f}x "
              f"{a['flops_per_token']/b['flops_per_token']:>10.2f}x")
    print(f"\nwrote {OUT}")
