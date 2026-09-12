"""Is "simultaneous thinking" a WIN, or just an expensive ensemble?

THE IDEA UNDER TEST
-------------------
A deep stack of N blocks evaluates one hypothesis with a dependency chain of
length N. K parallel columns of depth 1 evaluate K hypotheses with a
dependency chain of length 1, and a learned mixer decides how much to trust
each. If the bottleneck is depth-serialisation rather than representational
capacity, columns should help per unit of WALL CLOCK.

WHY THIS NEEDS A CONTROL
------------------------
K columns of depth 1, concatenated and passed through a single linear mixer, is
structurally a WIDE SINGLE LAYER with block-diagonal projections. The mixer can
only reweight entire columns, not mix their internal features, so the function
class is a strict subset of one dense layer of width K*d. If a dense layer of
the matched width does just as well, then "columns" buys nothing
representationally and the ONLY thing it changes is the dependency chain.

So four arms, and the comparisons that matter:

  cols2   : 2 columns x depth 1, d=128          (the idea)
  dense1  : 1 block, width matched to cols2     (rules out "it's just width")
  depth2  : 2 blocks in series, d=128           (the serial alternative)
  cols2A  : 2 ATTENTION columns x depth 1       (does the mix change the answer?)

PRE-REGISTERED PREDICTION (stated before running, per an independent review):
  - depth2 should BEAT cols2 on bpc, because induction-head-style computation
    needs at least two composed layers, while a single-layer column cannot
    condition its memory readout nonlinearly on its own earlier output.
  - dense1 should MATCH cols2 at matched width, consistent with columns being
    a restricted wide layer.
  - cols2 should win on WALL CLOCK, because its dependency chain is one block
    deep and this M4 Max is underutilised at these sizes.
  If cols2 beats depth2 on bpc, that REFUTES the prediction and is the result.
  If columns win on wall clock only, the honest conclusion is that "simultaneous
  thinking" buys latency, not quality, and should be described that way.
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
    ColumnLM, LM, n_params, load_corpus, make_batcher, CORPUS,
)
from llm_tuned import floors  # noqa: E402

OUT = os.environ.get(
    "BRAIN_COLS_OUT", "/Volumes/T9/human-brain/scratch/parallel_columns.json"
)


def make(arm, vocab, ctx):
    """Arms are built explicitly so the width match is visible, not implied."""
    if arm == "cols2_mem":
        return ColumnLM(vocab, 128, ctx, "mem", 2, 1, 4, 1, 64)
    if arm == "cols2_attn":
        return ColumnLM(vocab, 128, ctx, "attn", 2, 1, 4, 1, 64)
    if arm == "dense1_mem":
        return LM(vocab, 192, ctx, "mem", 1, 4, 1, 64)
    if arm == "depth2_mem":
        return LM(vocab, 128, ctx, "mem", 2, 4, 1, 64)
    if arm == "dense1_attn":
        return LM(vocab, 192, ctx, "attn", 1, 4, 1, 64)
    if arm == "depth2_attn":
        return LM(vocab, 128, ctx, "attn", 2, 4, 1, 64)
    raise ValueError(arm)


def run(arm, data, vocab, ctx=512, steps=1500, bs=16, lr=1e-3, seed=0,
        log=print):
    m = make(arm, vocab, ctx)
    mx.eval(m.parameters())
    P = n_params(m)
    batch = make_batcher(data, ctx)
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)
    t0 = time.time()
    ntok = 0
    for s in range(1, steps + 1):
        opt.learning_rate = lr * min(1.0, s / 100)
        x, y = batch(bs, rng)
        l, g = lg(m, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
    wall = time.time() - t0
    tot, n = 0.0, 0
    for _ in range(16):
        xv, yv = batch(64, rng, "val")
        tot += float(loss_fn(m, xv, yv)) * yv.size
        n += yv.size
    b = (tot / n) / math.log(2)

    # dependency-chain latency: time a single forward pass
    xf, _ = batch(16, rng, "val")
    for _ in range(3):
        mx.eval(m(xf))
    t1 = time.time()
    for _ in range(10):
        mx.eval(m(xf))
    fwd = (time.time() - t1) / 10
    return dict(arm=arm, params=P, bpc=b, wall_s=wall, tok_s=ntok / wall,
                fwd_ms=fwd * 1e3, seed=seed, steps=steps)


if __name__ == "__main__":
    data, vocab = load_corpus(CORPUS)
    bg, ug = floors(data)
    print(f"unigram {ug:.4f}  BIGRAM {bg:.4f} bpc")
    print("  an arm at/above the bigram floor has not learned context\n")
    arms = sys.argv[1:] or [
        "cols2_mem", "dense1_mem", "depth2_mem", "cols2_attn", "depth2_attn",
    ]
    seeds = [int(s) for s in os.environ.get("BRAIN_SEEDS", "0,1").split(",")]
    out = []
    for arm in arms:
        for seed in seeds:
            r = run(arm, data, vocab, seed=seed)
            out.append(r)
            print(f"  {arm:<13} seed={seed} params={r['params']:>9,} "
                  f"bpc={r['bpc']:.4f} ({bg - r['bpc']:+.4f}) "
                  f"fwd={r['fwd_ms']:6.2f}ms tok/s={r['tok_s']:>8,.0f}",
                  flush=True)
            json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== SUMMARY (mean over seeds) ===")
    print(f"{'arm':<13}{'params':>10}{'bpc':>9}{'vs floor':>10}"
          f"{'fwd ms':>9}{'tok/s':>10}")
    for arm in arms:
        rs = [r for r in out if r["arm"] == arm]
        if not rs:
            continue
        print(f"{arm:<13}{rs[0]['params']:>10,}"
              f"{np.mean([r['bpc'] for r in rs]):>9.4f}"
              f"{bg - np.mean([r['bpc'] for r in rs]):>+10.4f}"
              f"{np.mean([r['fwd_ms'] for r in rs]):>9.2f}"
              f"{np.mean([r['tok_s'] for r in rs]):>10,.0f}")
    print(f"\nbigram floor = {bg:.4f} bpc")
    print(f"wrote {OUT}")
