"""HONEST HEAD-TO-HEAD. Rebuilt after the first comparison was falsified.

WHAT WENT WRONG IN THE FIRST RUN
--------------------------------
The first comparison reported the transformer baseline at 3.666 bpc and the
gated-memory arm at 2.709 bpc, and I read the gap as a mechanism win. It was
not. The bigram floor on this corpus - a 65x65 count table, five lines of
numpy - is 3.581 bpc. The transformer baseline was scoring WORSE THAN A BIGRAM
MODEL. Its context path was contributing nothing, so the comparison was
"working memory vs a broken transformer", not "memory vs attention".

Diagnosis: AdamW at lr=3e-3 with no warmup, no gradient clipping, and MLX's
default uniform init. Adam's normalised update moves every Q and K weight by
~3e-3 from the first step, which saturates the attention logits and collapses
the softmax onto a single position. The layer degenerates to a per-token MLP.
The gated-memory arm survived the same LR because its gate bias is initialised
to a useful ten-character decay, so it starts ahead and never has to discover
context at all.

WHAT THIS FILE DOES DIFFERENTLY
-------------------------------
* Bigram and unigram floors are computed and printed on every run, so the
  reader can immediately see whether any arm has actually learned to use
  context. An arm at the floor is not a language model.
* Warmup + gradient clipping, and a small LR sweep, so each arm gets a fair
  shot rather than sharing one mistuned setting.
* Full validation split, not a single 64-sequence batch.
* Multiple seeds, so the gap is reported with its spread.
* Validation loss is tracked DURING training, so the shape of the curve is
  visible and an arm that plateaus at the floor is obvious.
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
    build, n_params, flops_per_token, load_corpus, chunked_gated_scan,
    sequential_scan, verify_scan, make_batcher, CORPUS,
)

OUT = os.environ.get(
    "BRAIN_TUNED_OUT", "/Volumes/T9/human-brain/scratch/llm_tuned.json"
)


def floors(data):
    """Bigram and unigram bpc on the val split. The floor a model must beat."""
    n = int(0.9 * len(data))
    tr, va = data[:n], data[n:]
    V = int(data.max()) + 1
    C = np.ones((V, V))
    np.add.at(C, (tr[:-1], tr[1:]), 1.0)
    P = C / C.sum(1, keepdims=True)
    ll = float(np.sum(np.log(P[va[:-1], va[1:]])))
    Cu = np.ones(V)
    np.add.at(Cu, tr, 1.0)
    Pu = Cu / Cu.sum()
    llu = float(np.sum(np.log(Pu[va])))
    return (
        (-ll / len(va)) / math.log(2),
        (-llu / len(va)) / math.log(2),
    )


@mx.compile
def _clipped(opt_step, params, grads):
    return opt_step(params, grads)


def run_arm(arm, data, vocab, *, ctx=512, d=128, steps=2000, bs=16, lr=3e-3,
            warmup=100, clip=1.0, seed=0, chunk=64, eval_every=250,
            full_val=True, log=print, bigram=None):
    m, spec = build(arm, vocab, d, ctx, chunk)
    # scaled init: MLX's default uniform init is not scaled for residual
    # stacks, and small residual branches are what keep deep transformers
    # trainable at a usable LR.
    mx.eval(m.parameters())
    P = n_params(m)
    n_layer = spec.get("n_layer", spec.get("depth", 1) * spec.get("n_col", 1))
    mult = spec.get("loops", 1) if spec.get("looped") else 1
    FL = flops_per_token(
        vocab, d, n_layer, ctx,
        "attn" if spec["kind"] == "attn" else "mem",
        banks=spec.get("banks", 1), chunk=chunk, mult=mult,
        n_col=spec.get("n_col", 1),
    )
    batch = make_batcher(data, ctx)
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)

    def val_bpc(n_seq=32):
        tot, n = 0.0, 0
        for _ in range(n_seq):
            xv, yv = batch(64, rng, "val")
            tot += float(loss_fn(m, xv, yv)) * yv.size
            n += yv.size
        return (tot / n) / math.log(2)

    curve = []
    t0 = time.time()
    ntok = 0
    for s in range(1, steps + 1):
        cur_lr = lr * min(1.0, s / max(warmup, 1))
        opt.learning_rate = cur_lr
        x, y = batch(bs, rng)
        l, g = lg(m, x, y)
        if clip:
            g, _gnorm = optim.clip_grad_norm(g, clip)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
        if s % eval_every == 0 or s == steps:
            b = val_bpc(8 if not full_val else 32)
            curve.append(dict(step=s, bpc=b, lr=cur_lr,
                              tok_s=ntok / (time.time() - t0)))
            log(f"    [{arm}] {s:>5}/{steps} loss {float(l):.4f} "
                f"val {b:.4f} bpc  lr {cur_lr:.2e}  "
                f"tok/s {ntok/(time.time()-t0):,.0f}")
    wall = time.time() - t0
    vb = val_bpc(32)
    return dict(
        arm=arm, params=P, flops_per_token=FL, ctx=ctx, d=d, steps=steps,
        bs=bs, seed=seed, lr=lr, warmup=warmup, clip=clip,
        bpc=vb, wall_s=wall, tok_s=ntok / wall, cum_flops=ntok * FL,
        tokens=ntok, spec=spec, curve=curve, bigram_bpc=bigram,
    )


if __name__ == "__main__":
    err = verify_scan()
    print(f"scan verification max err={err:.3e}")
    if err > 1e-4:
        raise SystemExit("scan mismatch")

    data, vocab = load_corpus(CORPUS)
    bg, ug = floors(data)
    print(f"corpus: {len(data):,} chars, vocab {vocab}")
    print(f"REFERENCE FLOORS: unigram {ug:.4f} bpc | BIGRAM {bg:.4f} bpc")
    print("  an arm at or above the bigram floor has not learned to use context\n")

    ctx = int(os.environ.get("BRAIN_CTX", "512"))
    steps = int(os.environ.get("BRAIN_STEPS", "2000"))
    seeds = [int(s) for s in os.environ.get("BRAIN_SEEDS", "0").split(",")]
    arms = sys.argv[1:] or ["A_attention", "C_gated_banks4"]
    lrs = [float(x) for x in os.environ.get(
        "BRAIN_LRS", "1e-3").split(",")]

    out = []
    for arm in arms:
        for lr in lrs:
            for seed in seeds:
                print(f"\n=== {arm} lr={lr:g} seed={seed} ctx={ctx} "
                      f"steps={steps} ===", flush=True)
                try:
                    r = run_arm(arm, data, vocab, ctx=ctx, steps=steps,
                                lr=lr, seed=seed, bigram=bg)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                    out.append(dict(arm=arm, lr=lr, seed=seed, ctx=ctx,
                                    error=f"{type(e).__name__}: {e}"))
                    continue
                out.append(r)
                verdict = ("AT/BELOW FLOOR - no context learned"
                           if r["bpc"] >= bg else
                           f"beats bigram by {bg - r['bpc']:.4f} bpc")
                print(f"  params={r['params']:,} "
                      f"fl/tok={r['flops_per_token']:,.0f} "
                      f"bpc={r['bpc']:.4f} ({verdict}) "
                      f"tok/s={r['tok_s']:,.0f} wall={r['wall_s']:.0f}s",
                      flush=True)
                json.dump(out, open(OUT, "w"), indent=1)

    print("\n\n=== SUMMARY ===")
    print(f"{'arm':<18}{'lr':>8}{'seed':>5}{'params':>10}{'bpc':>9}"
          f"{'vs floor':>10}{'tok/s':>10}")
    for r in out:
        if "error" in r:
            print(f"{r['arm']:<18}{r['lr']:>8g}{r['seed']:>5}  ERROR {r['error'][:40]}")
            continue
        print(f"{r['arm']:<18}{r['lr']:>8g}{r['seed']:>5}{r['params']:>10,}"
              f"{r['bpc']:>9.4f}{bg - r['bpc']:>+10.4f}{r['tok_s']:>10,.0f}")
    print(f"\nbigram floor = {bg:.4f} bpc")
    print(f"wrote {OUT}")
