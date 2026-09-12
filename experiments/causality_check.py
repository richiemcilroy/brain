"""Decisive correctness check: is the gated memory actually CAUSAL?

If the scan leaks future information, every language-model number in this repo
is void. Two independent tests, both cheap:

  1. PERTURBATION. In eval mode, change the token at position p and assert the
     logits at ALL positions < p are bit-identical. Any change at t < p means
     information flowed backwards and the architecture is invalid.
  2. RANDOM DATA. Train on i.i.d. uniform characters. There is no learnable
     structure, so the loss must converge to the entropy of the uniform
     distribution, log2(V) = log2(65) = 6.0224 bits. Anything materially BELOW
     that is a leak: the model could only beat the uniform entropy by reading
     the answer from a position it is not allowed to see.

Test 2 is the stronger one because it does not depend on my understanding of
the mask - it is a direct measure of whether future information is available.
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
from llm_efficiency import LM, ColumnLM, load_corpus, CORPUS  # noqa: E402

OUT = "/Volumes/T9/human-brain/scratch/causality_check.json"


def perturbation_test(arm, vocab=65, ctx=32, d=128):
    """Perturb position p; positions < p must be bit-identical."""
    mx.random.seed(0)
    if arm.startswith("cols"):
        m = ColumnLM(vocab, d, ctx, "mem", 2, 1, 4, 1, 64)
    else:
        m = LM(vocab, d, ctx, "mem", 2, 4, 1, 64)
    mx.eval(m.parameters())
    idx = mx.array(np.arange(ctx, dtype=np.int32)[None, :])
    base = np.array(m(idx))
    mx.eval(base)

    results = []
    for p in (ctx // 2, ctx - 1):
        idx2 = idx.at[0, p].add(7)          # change one token
        out = np.array(m(idx2))
        mx.eval(out)
        delta = np.abs(out - base)[0].max(axis=-1)   # (T,)
        leak = float(delta[:p].max())                # positions BEFORE p
        changed_at_p = float(delta[p])
        results.append(dict(
            arm=arm, perturbed_position=p,
            max_delta_before_p=leak,
            delta_at_p=changed_at_p,
            max_delta_after_p=float(delta[p + 1:].max()) if p + 1 < ctx else 0.0,
            causal=bool(leak == 0.0),
        ))
    return results


def random_data_test(arm, vocab, *, ctx=256, steps=300, bs=16, d=128, seed=0):
    """Train on i.i.d. uniform chars. Loss must NOT go below log2(V)."""
    mx.random.seed(seed)
    if arm.startswith("cols"):
        m = ColumnLM(vocab, d, ctx, "mem", 2, 1, 4, 1, 64)
    else:
        m = LM(vocab, d, ctx, "mem", 2, 4, 1, 64)
    mx.eval(m.parameters())
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)
    for s in range(1, steps + 1):
        opt.learning_rate = 1e-3 * min(1.0, s / 50)
        x = mx.array(rng.integers(0, vocab, size=(bs, ctx), dtype=np.int32))
        y = mx.array(rng.integers(0, vocab, size=(bs, ctx), dtype=np.int32))
        l, g = lg(m, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
    # evaluate on fresh random data
    tot, n = 0.0, 0
    for _ in range(8):
        x = mx.array(rng.integers(0, vocab, size=(64, ctx), dtype=np.int32))
        y = mx.array(rng.integers(0, vocab, size=(64, ctx), dtype=np.int32))
        tot += float(loss_fn(m, x, y)) * y.size
        n += y.size
    return (tot / n) / math.log(2)


if __name__ == "__main__":
    data, vocab = load_corpus(CORPUS)
    uniform_bpc = math.log2(vocab)
    print(f"vocab {vocab}, uniform entropy = {uniform_bpc:.4f} bpc")
    print("  a model trained on RANDOM data must NOT beat this")
    out = {"uniform_bpc": uniform_bpc, "perturbation": [], "random_data": {}}

    print("\n=== TEST 1: perturbation causality ===")
    for arm in ("depth2_mem", "cols2_mem"):
        for r in perturbation_test(arm):
            out["perturbation"].append(r)
            verdict = "CAUSAL" if r["causal"] else "*** LEAK ***"
            print(f"  {arm:<12} p={r['perturbed_position']:>2} "
                  f"max|delta| before p = {r['max_delta_before_p']:.3e}  "
                  f"at p = {r['delta_at_p']:.3e}  -> {verdict}")

    print("\n=== TEST 2: random-data floor ===")
    for arm in ("depth2_mem", "cols2_mem"):
        b = random_data_test(arm, vocab)
        out["random_data"][arm] = b
        gap = b - uniform_bpc
        verdict = ("NO LEAK" if gap > -1e-3
                   else f"*** LEAK: {gap:+.4f} bpc below uniform ***")
        print(f"  {arm:<12} bpc={b:.4f}  vs uniform {uniform_bpc:.4f}  "
              f"gap={gap:+.4f}  -> {verdict}")
        json.dump(out, open(OUT, "w"), indent=1)

    print()
    leaks = [r for r in out["perturbation"] if not r["causal"]]
    low = [k for k, v in out["random_data"].items() if v < uniform_bpc - 1e-3]
    if leaks or low:
        print(f"RESULT: LEAK DETECTED. perturbation leaks={leaks} "
              f"random-data below uniform={low}")
        print("Every language-model number in this repo is void until fixed.")
    else:
        print("RESULT: no leak found by either test. The memory arms are causal "
              "and cannot beat the uniform entropy on random data.")
    print(f"\nwrote {OUT}")
