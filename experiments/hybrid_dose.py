"""DOSE-RESPONSE: is the frozen "transfer beats zero" result a property of the
weights, or of one unvalidated scalar?

THE PROBLEM THIS EXISTS TO SETTLE
---------------------------------
The committed headline (`docs/HYBRID.md`) is:

    teacher  20.3756 | zero 24.1044 | transfer 23.7228   -> transfer beats zero

`transfer`'s output is scaled by a constant that was never fitted or swept. It
comes from the identity sum_k (1-g) g^k = 1 with the gate decay g=0.99, giving a
factor of 0.01. That number has a principled derivation but NO validation, and
the module's true output rms on real layer-8 activations is 0.0505 against our
0.0032 at that setting -- i.e. the committed arm runs at about 0.44x attention's
output scale (the figure already recorded in this repo).

Raising the gain to the rms-matched value (0.01564) gives 24.3233, which is
WORSE than deleting attention outright (24.1044). So the sign of the headline
comparison flips with a scalar that was set by a formula nothing checked.

WHAT THIS RUNS
--------------
A frozen sweep over the module's output scale for each arm, expressed as a
multiple of the attention rms it replaces, so the axis is comparable across
arms:

    ratio 1.0  = our module contributes exactly as much output power as the
                 attention sublayer it replaced
    ratio 0.5  = half that power, etc.

Each arm is scored at its OWN best ratio, which is the fair version of the
comparison: "the best transfer can do" against "the best matched-random can do".
If those two optima coincide, the weights carry no information and the
committed result was a scale artifact. If transfer's optimum is meaningfully
lower, the weights carry something.

In every arm the gate decay is fitted from the SAME measured recency profile, so
the only difference between arms is whose W_v/W_o they hold.

This is cheap (no training) and deterministic (fixed 3000-token window, no
sampling), so the numbers are exact.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import mlx.core as mx

os.environ.setdefault("HF_HOME", os.path.expanduser("~/zbrain/hf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_hybrid import (  # noqa: E402
    attention_module, transplant, perplexity, n_params, GatedMemoryCarrier,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "hybrid_dose.json")
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
LAYER = int(os.environ.get("DOSE_LAYER", "8"))
RATIOS = [float(x) for x in
          os.environ.get("DOSE_RATIOS",
                         "0.25,0.44,0.71,1.0,1.41,2.0,2.83").split(",")]


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


def main():
    from mlx_lm import load

    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    train_ids = ids[:int(0.5 * n)]
    val = ids[int(0.9 * n):int(0.9 * n) + 3000]
    print(f"{n_params(model):,} params | val {len(val):,} | layer {LAYER}\n", flush=True)

    base = perplexity(model, val)["ppl"]
    print(f"  teacher ppl {base:.4f}", flush=True)

    # real activations at this layer
    layer, attr, original = attention_module(model, LAYER)
    cap = {}

    class Capture:
        def __call__(self, x, *a, **k):
            cap["h"] = x
            return original(x, *a, **k)

    setattr(layer, attr, Capture())
    model(mx.array(train_ids[:1024][None, :].astype(np.int32)))
    probe_h = cap["h"]
    ref_out = original(probe_h, None, None)
    mx.eval(probe_h, ref_out)
    setattr(layer, attr, original)
    r_ref = rms(ref_out)
    d = int(probe_h.shape[-1])
    print(f"  probe {tuple(probe_h.shape)} rms {rms(probe_h):.4f} | "
          f"attention out rms {r_ref:.5f}\n", flush=True)

    # zero arm: attention deleted
    setattr(layer, attr, GatedMemoryCarrier(d, mode="zero"))
    mx.eval(model.parameters())
    zero_ppl = perplexity(model, val)["ppl"]
    setattr(layer, attr, original)
    print(f"  zero (attn deleted) ppl {zero_ppl:.4f}\n", flush=True)

    out = dict(teacher=TEACHER, layer=LAYER, n_layers=len(model.model.layers),
               teacher_ppl=base, zero_ppl=zero_ppl, val_tokens=len(val),
               ratios=RATIOS, arms={}, protocol=dict(
                   gain="fixed scalar on the module output; never trained",
                   axis="output rms as a multiple of the replaced attention's rms",
                   decay="fitted from the measured recency profile, same for all arms",
                   eval="frozen, fixed 3000-token window, no sampling -> exact"))

    for arm in os.environ.get("DOSE_ARMS", "transfer,random_matched").split(","):
        rows = []
        for ratio in RATIOS:
            setattr(layer, attr, original)
            carrier, rec = transplant(model, LAYER, "transfer",
                                      original=original, probe_h=probe_h)
            if arm == "random_matched":
                rng = np.random.default_rng(0)
                s_native = float(np.array(carrier.mem.v.weight).std())
                carrier.mem.v.weight = mx.array(
                    rng.normal(0, s_native, (d, d)).astype(np.float32))
                carrier.mem.o.weight = mx.array(
                    rng.normal(0, s_native, (d, d)).astype(np.float32))
            elif arm == "random_fanin":
                rng = np.random.default_rng(0)
                s_ = 1.0 / math.sqrt(d)
                carrier.mem.v.weight = mx.array(
                    rng.normal(0, s_, (d, d)).astype(np.float32))
                carrier.mem.o.weight = mx.array(
                    rng.normal(0, s_, (d, d)).astype(np.float32))
            # set the gain so the output rms is `ratio` x attention's
            mx.eval(model.parameters())
            carrier.out_gain = 1.0
            o1 = carrier(probe_h, None, None)
            mx.eval(o1)
            g_match = r_ref / rms(o1)
            carrier.out_gain = float(g_match * ratio)
            o2 = carrier(probe_h, None, None)
            mx.eval(o2)
            p = perplexity(model, val)["ppl"]
            rows.append(dict(ratio=ratio, gain=float(carrier.out_gain),
                             out_rms=rms(o2), ppl=p,
                             beats_teacher=bool(p < base),
                             beats_zero=bool(p < zero_ppl)))
            print(f"  {arm:<15} ratio {ratio:>5.2f}  gain {carrier.out_gain:.6f}  "
                  f"ppl {p:>9.4f}  "
                  f"{'BEATS zero' if p < zero_ppl else '      worse'}", flush=True)
            setattr(layer, attr, original)
        best = min(rows, key=lambda r: r["ppl"])
        out["arms"][arm] = dict(rows=rows, best=best)
        print(f"  -> {arm} best: ratio {best['ratio']:.2f} ppl {best['ppl']:.4f}\n",
              flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

    setattr(layer, attr, original)
    tb = out["arms"].get("transfer", {}).get("best", {})
    rb = out["arms"].get("random_matched", {}).get("best", {})
    rf = out["arms"].get("random_fanin", {}).get("best", {})
    out["verdict"] = dict(
        teacher=base, zero=zero_ppl,
        transfer_best=tb.get("ppl"), transfer_best_ratio=tb.get("ratio"),
        random_matched_best=rb.get("ppl"), random_matched_best_ratio=rb.get("ratio"),
        random_fanin_best=rf.get("ppl"), random_fanin_best_ratio=rf.get("ratio"),
        transfer_beats_best_control=(
            bool(tb["ppl"] < min(rb.get("ppl", np.inf), rf.get("ppl", np.inf)))
            if tb else None),
        transfer_beats_zero_at_its_own_best=(
            bool(tb["ppl"] < zero_ppl) if tb else None),
        note=("Each arm is scored at its OWN best scale, so this is the fair "
              "version of the comparison. If the optima coincide the weights "
              "carry no information and the committed result was a scale "
              "artifact; the committed arm ran at ratio 0.44, not at its optimum."))
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== VERDICT (each arm at its own best output scale) ===")
    print(f"  teacher               ppl {base:>9.4f}")
    print(f"  zero (attn deleted)   ppl {zero_ppl:>9.4f}")
    for nm, b in (("transfer", tb), ("random_matched", rb), ("random_fanin", rf)):
        if b:
            print(f"  {nm:<21} ppl {b['ppl']:>9.4f}  at ratio {b['ratio']:.2f}")
    print(f"\n  transfer beats the best control at its own best scale: "
          f"{out['verdict']['transfer_beats_best_control']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
