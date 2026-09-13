"""DECAY SWEEP: is the "transfer beats zero" result about the WEIGHTS, or about
one scalar that was never validated?

WHERE THIS CAME FROM
--------------------
Two problems were found in the committed headline, in this order.

1. THE FITTED DECAY WAS NEVER FITTED. `fit_decay_from_attention` called
   `inner(x, None, None, return_weights=True)` inside a try/except and fell back
   to 0.99. MLX's `Attention.__call__` does not accept `return_weights` at all,
   so it raised on EVERY layer and every layer got 0.99 by default. The docs
   claimed the gate was initialised from the MEASURED recency profile; it was
   not. The tell, visible in every committed result file, is
   `recency_profile_first8: null` and `fit_cdf_l1: null`.

2. WHEN ACTUALLY MEASURED, THE FIT GOES TO THE EDGE. The profile is now computed
   from the layer's own q/k projections with a causal softmax and a histogram by
   past-token distance. For every layer tested (0, 4, 8, 12, 15 of 16) the best
   single exponential is 0.99999, the top of the grid, with a CDF L1 residual of
   16.8 to 44.1. A well-specified fit returns an interior optimum with a small
   residual. Pinning at the boundary with a large residual means the model class
   is wrong: real attention recency is not one exponential. The measured profiles
   have a recency spike on top of a broad background, which is the shape you
   expect from attention sinks plus a diffuse long-range component.

CONSEQUENCE, AND WHY THIS EXPERIMENT EXISTS
-------------------------------------------
A single exponential at 0.99999 has an effective horizon of 1/(1-g) = 1e5
tokens, far beyond the 3000-token evaluation, so the trace degenerates into an
unweighted running mean that carries almost no positional information. That is a
plausible explanation for why the module only marginally beats deleting the
attention it replaces -- and it means the committed comparison may be measuring
the decay constant rather than the transplanted weights.

This sweeps the decay DIRECTLY, for the transplant and for both random controls,
and adds a multi-timescale arm (`banks`), since a mixture of exponentials is the
natural response to a profile that is not one exponential. If all arms share the
same optimum, the weights carry no information about the decay and the committed
result is a scalar artifact. If the transplant has a distinct and better
optimum, the weights carry something.

Frozen, deterministic, no training: a fixed 3000-token window with no sampling,
so these are exact numbers and not estimates.
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
OUT = os.path.join(HERE, "results", "hybrid_decay.json")
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
LAYER = int(os.environ.get("DECAY_LAYER", "8"))
DECAYS = [float(x) for x in os.environ.get(
    "DECAYS", "0.5,0.9,0.95,0.99,0.999,0.9999,0.99999,1.0").split(",")]


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
    # SELECT is where the decay is chosen; VAL is only ever reported.
    # Choosing the decay on val would make the reported optimum an artefact of
    # the evaluation set, which is the same contamination the lr protocol
    # already guards against.
    select = ids[int(0.6 * n):int(0.6 * n) + 3000]
    val = ids[int(0.9 * n):int(0.9 * n) + 3000]
    assert int(0.5 * n) <= int(0.6 * n) < int(0.9 * n)
    print(f"{n_params(model):,} params | val {len(val):,} | layer {LAYER}\n", flush=True)

    base = perplexity(model, val)["ppl"]
    print(f"  teacher ppl {base:.4f}", flush=True)

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
    print(f"  probe rms {rms(probe_h):.4f} | attention out rms {r_ref:.5f}\n", flush=True)

    setattr(layer, attr, GatedMemoryCarrier(d, mode="zero"))
    mx.eval(model.parameters())
    zero_ppl = perplexity(model, val)["ppl"]
    setattr(layer, attr, original)
    print(f"  zero (attn deleted) ppl {zero_ppl:.4f}\n", flush=True)

    out = dict(teacher=TEACHER, layer=LAYER, n_layers=len(model.model.layers),
               teacher_ppl=base, zero_ppl=zero_ppl, val_tokens=len(val),
               decays=DECAYS, arms={}, protocol=dict(
                   selection="decay chosen on SELECT split, reported on VAL",
                   eval="frozen, fixed 3000-token window, no sampling -> exact",
                   norm="every arm scaled to attention's output rms before scoring",
                   banks="multi-timescale arm keeps one trace per decay constant"))

    for arm in os.environ.get("DECAY_ARMS",
                              "transfer,random_matched,banks_multi").split(","):
        rows = []
        for decay in DECAYS:
            setattr(layer, attr, original)
            banks, span = 1, None
            if arm == "banks_multi":
                # A MIXTURE of exponentials, the natural fix for a profile that
                # no single exponential fits. For each swept centre decay the
                # banks are spread geometrically around it, so the arm sees a
                # short, a medium and a long horizon at once -- a single decay
                # per bank would just repeat the single-bank arm.
                banks = 3
                span = (max(decay * decay, 1e-6),
                        decay,
                        min(math.sqrt(decay), 1.0 - 1e-9))
            carrier, rec = transplant(model, LAYER, "transfer",
                                      original=original, probe_h=probe_h,
                                      banks=banks,
                                      decays=span if span else (decay,) * banks)
            if arm in ("random_matched", "random_fanin"):
                rng = np.random.default_rng(0)
                s_native = float(np.array(carrier.mem.v.weight).std())
                s_ = s_native if arm == "random_matched" else 1.0 / math.sqrt(d)
                carrier.mem.v.weight = mx.array(
                    rng.normal(0, s_, (d * banks, d)).astype(np.float32))
                carrier.mem.o.weight = mx.array(
                    rng.normal(0, s_, (d, d * banks)).astype(np.float32))
            # normalise to attention's output rms using the SAME rule for all
            # arms, so the comparison is not a scale comparison
            mx.eval(model.parameters())
            carrier.out_gain = 1.0
            o1 = carrier(probe_h, None, None)
            mx.eval(o1)
            carrier.out_gain = float(r_ref / rms(o1))
            p_sel = perplexity(model, select)["ppl"]
            p = perplexity(model, val)["ppl"]
            rows.append(dict(decay=decay, banks=banks, ppl=p, select_ppl=p_sel,
                             horizon=float("inf") if decay >= 1 else 1.0 / (1 - decay),
                             beats_teacher=bool(p < base),
                             beats_zero=bool(p < zero_ppl)))
            print(f"  {arm:<16} decay {decay:<8.5f} horizon {rows[-1]['horizon']:>9.0f}  "
                  f"ppl {p:>9.4f}  "
                  f"{'BEATS zero' if p < zero_ppl else '      worse'}", flush=True)
            setattr(layer, attr, original)
        # the decay is chosen on SELECT only; the printed val ppl at that decay
        # is the honest number
        best = min(rows, key=lambda r: r["select_ppl"])
        out["arms"][arm] = dict(rows=rows, best=best, best_selection="select")
        print(f"  -> {arm} best on SELECT: decay {best['decay']} "
              f"(select {best['select_ppl']:.4f}) -> val {best['ppl']:.4f}\n",
              flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

    setattr(layer, attr, original)
    tb = out["arms"].get("transfer", {}).get("best")
    ctrl = {a: v.get("best") for a, v in out["arms"].items() if a != "transfer"}
    best_ctrl = min((c for c in ctrl.values() if c), key=lambda c: c["ppl"],
                    default=None)
    out["verdict"] = dict(
        teacher=base, zero=zero_ppl,
        transfer_best=tb, best_control=best_ctrl,
        transfer_beats_best_control=bool(
            tb and best_ctrl and tb["ppl"] < best_ctrl["ppl"]),
        transfer_beats_zero_at_best=bool(tb and tb["ppl"] < zero_ppl),
        note=("Each arm is swept over the same decay grid and scored at its own "
              "best setting, so this separates 'the weights know the right time "
              "constant' from 'the right time constant helps everyone'. If the "
              "optima coincide the transplant carries no information about it."))
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== VERDICT (each arm at its own best decay) ===")
    print(f"  teacher              ppl {base:>9.4f}")
    print(f"  zero (attn deleted)  ppl {zero_ppl:>9.4f}")
    for a, v in out["arms"].items():
        b = v["best"]
        print(f"  {a:<20} ppl {b['ppl']:>9.4f}  at decay {b['decay']}")
    print(f"\n  transfer beats the best control at its own best decay: "
          f"{out['verdict']['transfer_beats_best_control']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
