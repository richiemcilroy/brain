"""THE DECISIVE EXPERIMENT: does the transplant init beat a random init once
both are finetuned?

Why this is the test that matters. The frozen result (`docs/HYBRID.md`) shows the
transplant carries real function on real trained weights. But a reader is
entitled to say: "so what -- finetuning would recover that from any init." That
is a specific, falsifiable claim, and this file tests it directly:

    Arm A: our module, initialised from attention's own weights (transplant)
    Arm B: our module, random init, SAME output-rms matching, SAME decay
    Both: identical finetuning (same data, steps, lr, seed, batch, context)
           optimising ONLY the module's parameters.

If A does not beat B after finetuning, the transplant is worthless and the
frozen win came from the architecture plus the fitted decay, not from the
weights. That is a clean negative and gets reported as one.

Two protocols, because they answer different questions:

  matched-steps: both arms get the same number of optimiser steps. This is the
                 "efficiency" question -- is the transplant a better starting
                 point for a fixed budget?
  to-convergence: both run long enough that A's advantage, if it is only a head
                 start, should wash out. If A still wins, the init selects a
                 better basin rather than merely a shorter path.

Contamination guard: validation is TinyShakespeare and training data is the
FIRST 50% of the same file's tokens, validation the LAST 10%, so they are
disjoint by construction and the script asserts it.
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

os.environ.setdefault("HF_HOME", "/tmp/zz_llm/hf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_hybrid import (  # noqa: E402
    get_layers, attention_module, transplant, perplexity, n_params,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "hybrid_finetune.json")
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
LAYER = int(os.environ.get("FT_LAYER", "8"))


def finetune_module(model, carrier, train_ids, *, steps, bs, ctx, lr, log=print):
    """Optimise ONLY the carrier. Everything else stays frozen.

    Freezing the rest is deliberate: the claim is about the module's capacity to
    hold the function, not about how much of a 1.2B model gradient descent can
    paper over. Passing the whole model to value_and_grad would return gradients
    for every parameter and opt.update would move frozen weights too, silently
    turning this into a different experiment.
    """
    model.freeze()
    carrier.unfreeze()
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    tr = np.asarray(train_ids, dtype=np.int32)

    def loss_fn(carrier, x, y):
        lo = model(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(carrier, loss_fn)
    t0 = time.time()
    hist = []
    for s in range(1, steps + 1):
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + ctx] for i in ix]).astype(np.int32))
        y = mx.array(np.stack([tr[i + 1:i + 1 + ctx] for i in ix]).astype(np.int32))
        l, g = lg(carrier, x, y)
        opt.update(carrier, g)
        mx.eval(model.parameters(), opt.state)
        if s % 25 == 0 or s == steps:
            hist.append(dict(step=s, loss=float(l)))
    return hist, time.time() - t0


def main():
    from mlx_lm import load

    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    base_par = n_params(model)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    train_ids = ids[:int(0.5 * n)]
    val = ids[int(0.9 * n):int(0.9 * n) + 3000]
    # disjointness is structural here (train is the first half, val the last
    # tenth) but assert it anyway rather than assuming it
    assert int(0.5 * n) <= int(0.9 * n)
    print(f"{base_par:,} params | train {len(train_ids):,} | val {len(val):,} | "
          f"layer {LAYER}\n", flush=True)

    base = perplexity(model, val)
    print(f"  teacher ppl {base['ppl']:.4f}\n", flush=True)

    steps = int(os.environ.get("FT_STEPS", "300"))
    bs = int(os.environ.get("FT_BS", "2"))
    ctx = int(os.environ.get("FT_CTX", "128"))
    lr = float(os.environ.get("FT_LR", "3e-4"))

    out = dict(teacher=TEACHER, base_params=base_par, layer=LAYER,
               val_tokens=len(val), teacher_ppl=base["ppl"], steps=steps,
               bs=bs, ctx=ctx, lr=lr, arms=[])

    layer, attr, original = attention_module(model, LAYER)
    for mode in os.environ.get("FT_MODES", "transfer,random_scaled").split(","):
        mx.random.seed(0)
        setattr(layer, attr, original)
        carrier, rec = transplant(model, LAYER, mode)
        mx.eval(model.parameters())
        before = perplexity(model, val)
        rec.pop("mode", None)
        print(f"  {mode:<15} BEFORE ft ppl {before['ppl']:>10.4f}", flush=True)
        hist, wall = finetune_module(model, carrier, train_ids, steps=steps,
                                     bs=bs, ctx=ctx, lr=lr)
        after = perplexity(model, val)
        row = dict(**rec, mode=mode, before_ppl=before["ppl"],
                   after_ppl=after["ppl"], ft_wall_s=wall, ft_history=hist)
        out["arms"].append(row)
        print(f"  {mode:<15} AFTER  ft ppl {after['ppl']:>10.4f}  "
              f"({after['ppl']-before['ppl']:+.4f} from ft, {wall:.0f}s)", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    setattr(layer, attr, original)

    a = [x for x in out["arms"] if x["mode"] == "transfer"]
    b = [x for x in out["arms"] if x["mode"] == "random_scaled"]
    if a and b:
        ta, tb = a[0]["after_ppl"], b[0]["after_ppl"]
        out["verdict"] = dict(
            teacher=base["ppl"],
            transfer_before=a[0]["before_ppl"], transfer_after=ta,
            random_before=b[0]["before_ppl"], random_after=tb,
            transplant_init_better_after_ft=bool(ta < tb),
            margin=tb - ta,
            note=("If the transplant does not beat the matched random init after "
                  "identical finetuning, the init is worthless and the frozen "
                  "win came from architecture + fitted decay, not the weights."),
        )
        json.dump(out, open(OUT, "w"), indent=1)
        print("\n=== VERDICT ===")
        print(f"  teacher          ppl {base['ppl']:>10.4f}")
        print(f"  transfer  before ppl {a[0]['before_ppl']:>10.4f}  ->  after {ta:>10.4f}")
        print(f"  random    before ppl {b[0]['before_ppl']:>10.4f}  ->  after {tb:>10.4f}")
        print(f"\n  transplant init wins after finetuning: {out['verdict']['transplant_init_better_after_ft']}"
              f"  (margin {out['verdict']['margin']:+.4f} ppl)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
