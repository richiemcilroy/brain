"""Confirmation run for the one result that survived falsification.

WHAT IS BEING CONFIRMED
-----------------------
docs/MATCHED_COMPARISON.md reports that at matched depth (2 layers), matched
width (d=128) and matched seeds, a gated-memory stack scores 2.2127 bpc against
2.3376 for an attention stack. That is the project's only claim that has survived
every control tried so far. An independent review listed three things that still
stood between it and being trustworthy. This script fixes all three:

  1. SEEDS. n=2 is not enough to quantify anything; at n=2 the 95% t-multiplier
     on the standard error is 12.7, versus 4.3 at n=3 and 2.8 at n=5. This runs
     FIVE seeds per arm and reports PAIRED differences (the seeds are paired, and
     an earlier run showed seed k beats seed k-1 in every arm, i.e. a shared
     effect worth cancelling).

  2. VALIDATION WAS SAMPLED, AND OVERSAMPLED. Sixteen batches of 64 at ctx=512
     draws 1,024 windows from an ~111K-character split, covering it about five
     times over, and the windows differ between seeds. This evaluates
     DETERMINISTICALLY on every non-overlapping ctx-length window of the split
     (~195 windows), so every arm is scored on exactly the same data and the
     number is reproducible.

  3. TRAIN BPC WAS NOT REPORTED. At ~12 passes with no dropout, overfitting and
     underfitting cannot be told apart from validation alone. Both are logged.

ARMS
----
  depth2_mem   445,440 params   2 gated-memory layers at d=128
  depth2_attn  478,976 params   2 attention layers at d=128   (only the
                                primitive differs - the clean comparison)
  A_attention  875,520 params   4 attention layers             (the baseline)

REPORTED: per-seed values, paired mean differences with 95% t-intervals, and the
number of seeds where the sign agrees. A paired interval that straddles zero is
reported as a null, not as a win.
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
from llm_efficiency import LM, load_corpus, CORPUS  # noqa: E402
from llm_tuned import floors  # noqa: E402

OUT = "/Volumes/T9/human-brain/scratch/confirm_headline.json"
# 95% two-sided t quantiles by degrees of freedom, for paired intervals
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262}


def build(arm, vocab, ctx):
    if arm == "depth2_mem":
        return LM(vocab, 128, ctx, "mem", 2, 4, 1, 64)
    if arm == "depth2_attn":
        return LM(vocab, 128, ctx, "attn", 2, 4, 1, 64)
    if arm == "A_attention":
        return LM(vocab, 128, ctx, "attn", 4, 4, 1, 64)
    raise ValueError(arm)


def deterministic_val_batches(data, ctx, val_frac=0.1):
    """Every non-overlapping ctx-length window of the val split, as arrays."""
    n = int((1.0 - val_frac) * len(data))
    va = data[n:]
    n_win = (len(va) - 1) // ctx
    x = np.stack([va[i * ctx:i * ctx + ctx] for i in range(n_win)])
    y = np.stack([va[i * ctx + 1:i * ctx + 1 + ctx] for i in range(n_win)])
    return mx.array(x), mx.array(y), n_win


def evaluate(m, xv, yv, vocab, bs=32):
    """Full deterministic validation, in batches, weighted by token count."""
    tot, n = 0.0, 0
    for i in range(0, xv.shape[0], bs):
        lo = m(xv[i:i + bs])
        l = nn.losses.cross_entropy(
            lo.reshape(-1, vocab), yv[i:i + bs].reshape(-1), reduction="sum"
        )
        mx.eval(l)
        tot += float(l)
        n += int(yv[i:i + bs].size)
    return tot / n


def run(arm, data, vocab, *, ctx=512, steps=1500, bs=16, lr=1e-3, seed=0,
        log=print):
    mx.random.seed(seed)
    m = build(arm, vocab, ctx)
    mx.eval(m.parameters())
    P = sum(int(np.prod(v.shape))
            for _, v in nn.utils.tree_flatten(m.parameters()))
    n = int((1.0 - 0.1) * len(data))
    tr = data[:n]
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)
    xv, yv, n_win = deterministic_val_batches(data, ctx)
    t0 = time.time()
    ntok = 0
    last_train = float("nan")
    for s in range(1, steps + 1):
        opt.learning_rate = lr * min(1.0, s / 100)
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + ctx] for i in ix]))
        y = mx.array(np.stack([tr[i + 1:i + 1 + ctx] for i in ix]))
        l, g = lg(m, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
        last_train = float(l)
    wall = time.time() - t0
    val_nats = evaluate(m, xv, yv, vocab)
    return dict(arm=arm, seed=seed, params=P, steps=steps,
                val_bpc=val_nats / math.log(2),
                train_bpc=last_train / math.log(2),
                n_val_windows=n_win, tok_s=ntok / wall, wall_s=wall)


def paired_report(results, a, b):
    """Paired difference (a - b) across shared seeds, with a 95% t-interval."""
    sa = {r["seed"]: r["val_bpc"] for r in results if r["arm"] == a}
    sb = {r["seed"]: r["val_bpc"] for r in results if r["arm"] == b}
    seeds = sorted(set(sa) & set(sb))
    d = np.array([sa[s] - sb[s] for s in seeds])
    n = len(d)
    mean = d.mean() if n else float("nan")
    if n < 2:
        return dict(comparison=f"{a} - {b}", n_seeds=n, paired_mean_diff=mean,
                    ci95=None, sign_agreement=None, verdict="insufficient seeds")
    se = d.std(ddof=1) / math.sqrt(n)
    crit = T95.get(n - 1, 1.96)
    lo, hi = mean - crit * se, mean + crit * se
    agree = int(sum(1 for v in d if v < 0)) if mean < 0 else int(sum(1 for v in d if v > 0))
    if lo < 0 < hi:
        verdict = "NULL: interval straddles zero"
    else:
        verdict = f"{'memory' if mean < 0 else 'attention'} better, interval excludes 0"
    return dict(comparison=f"{a} - {b}", n_seeds=n,
                per_seed_diff=[round(float(v), 5) for v in d],
                paired_mean_diff=round(float(mean), 5),
                ci95=[round(float(lo), 5), round(float(hi), 5)],
                sign_agreement=f"{agree}/{n}", verdict=verdict)


if __name__ == "__main__":
    data, vocab = load_corpus(CORPUS)
    bg, ug = floors(data)
    print(f"unigram {ug:.4f}  BIGRAM {bg:.4f} bpc")
    ctx = 512
    xv, yv, nw = deterministic_val_batches(data, ctx)
    print(f"deterministic validation: {nw} non-overlapping windows of {ctx} "
          f"= {nw*ctx:,} tokens (every arm scored on identical data)\n")

    seeds = [int(s) for s in os.environ.get("BRAIN_SEEDS", "0,1,2,3,4").split(",")]
    arms = os.environ.get("BRAIN_ARMS",
                          "depth2_mem,depth2_attn,A_attention").split(",")
    steps = int(os.environ.get("BRAIN_STEPS", "1500"))
    out = []
    for arm in arms:
        for seed in seeds:
            r = run(arm, data, vocab, ctx=ctx, steps=steps, seed=seed)
            out.append(r)
            print(f"  {arm:<13} seed={seed} params={r['params']:>8,} "
                  f"val={r['val_bpc']:.4f} train={r['train_bpc']:.4f} "
                  f"({bg-r['val_bpc']:+.4f} vs floor) "
                  f"tok/s={r['tok_s']:>8,.0f}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== PER-ARM (mean +/- sd over seeds) ===")
    for arm in arms:
        rs = [r for r in out if r["arm"] == arm]
        v = [r["val_bpc"] for r in rs]
        print(f"  {arm:<13} params={rs[0]['params']:>8,}  "
              f"val={np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}  "
              f"min={min(v):.4f} max={max(v):.4f}")

    print("\n=== PAIRED DIFFERENCES (negative = the first arm has lower bpc) ===")
    report = []
    for a, b in (("depth2_mem", "depth2_attn"), ("depth2_mem", "A_attention"),
                 ("depth2_attn", "A_attention")):
        pr = paired_report(out, a, b)
        report.append(pr)
        print(f"  {pr['comparison']:<26} n={pr['n_seeds']} "
              f"mean={pr['paired_mean_diff']} "
              f"CI={pr['ci95']} sign={pr['sign_agreement']} "
              f"-> {pr['verdict']}")
    json.dump(dict(runs=out, paired=report, bigram_floor=bg),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
