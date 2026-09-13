"""8B-scale evidence: does the carrier beat a matched-random carrier when it
REPLACES attention in Llama-3.1-8B?

This is the same protocol as verify_transplant.py, run on a model 6.5x larger and
quantized (4-bit affine), which exercises a different code path from the bf16 1B
model the rest of the project was developed on.

Three arms, each scaled so its output rms equals the attention it replaces:
  transfer        W_v/W_o copied from this layer's attention
  random_matched  random W_v/W_o at matched weight scale
  zero            the carrier returns zeros (attention deleted entirely)

`zero` is the reference point: it is what you get with no memory at all. An arm
that cannot beat `zero` has negative value, not merely missing value.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_hybrid import (  # noqa: E402
    attention_module, transplant, perplexity, n_params, GatedMemoryCarrier,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "data", "tinyshakespeare.txt")
TEACHER = os.environ.get("B8_TEACHER", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
LAYERS = [int(x) for x in os.environ.get("B8_LAYERS", "16").split(",")]
DECAYS = [float(x) for x in os.environ.get("B8_DECAYS", "0.5,0.9,0.99,1.0").split(",")]
SEEDS = [int(x) for x in os.environ.get("B8_SEEDS", "0,1,2").split(",")]
OUT = os.environ.get("B8_OUT", os.path.join(HERE, "results", "hybrid_8b.json"))
WINDOW = 3000
SELECT_FRAC, VAL_FRAC = 0.6, 0.9


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


def evaluate(model, li, layer, attr, original, probe_h, *, decay, seed, mode,
             rms_target, select, val):
    mx.random.seed(0)
    carrier, rec = transplant(model, li, "transfer", original=original,
                              probe_h=probe_h, banks=1, decays=(decay,))
    d = int(rec["d"])
    if mode == "random_matched":
        rng = np.random.default_rng(seed)
        s_native = float(np.array(carrier.mem.v.weight).std())
        carrier.mem.v.weight = mx.array(rng.normal(0, s_native, (d, d)).astype(np.float32))
        carrier.mem.o.weight = mx.array(rng.normal(0, s_native, (d, d)).astype(np.float32))
        rec["control_seed"] = int(seed)
    mx.eval(model.parameters())
    carrier.out_gain = 1.0
    o1 = carrier(probe_h, None, None)
    mx.eval(o1)
    pre = rms(o1)
    carrier.out_gain = float(rms_target / pre)
    o2 = carrier(probe_h, None, None)
    mx.eval(o2)
    p_sel = perplexity(model, select)["ppl"]
    p_val = perplexity(model, val)["ppl"]
    setattr(layer, attr, original)
    return dict(decay=float(decay), ppl=float(p_val), select_ppl=float(p_sel),
                rms_pre_norm=float(pre), rms_post_norm=float(rms(o2)),
                rms_target=float(rms_target), out_gain=float(carrier.out_gain),
                control_seed=rec.get("control_seed"),
                value_expand_reps=rec.get("value_expand_reps"),
                quantized=rec.get("quantized"))


def main():
    from mlx_lm import load
    t0 = time.time()
    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    s0 = int(SELECT_FRAC * n)
    v0 = int(VAL_FRAC * n)
    select, val = ids[s0:s0 + WINDOW], ids[v0:v0 + WINDOW]

    out = dict(
        meta=dict(teacher=TEACHER, n_params=n_params(model),
                  n_layers=len(model.model.layers), d=int(model.args.hidden_size),
                  corpus_tokens=int(n), layers=LAYERS, decays=DECAYS, seeds=SEEDS,
                  host=platform.platform(), driver="experiments/hybrid_8b.py",
                  protocol="same as verify_transplant.py: rms-matched arms, "
                           "frozen select/val windows, paired per decay",
                  val_sha1=hashlib.sha1(val.tobytes()).hexdigest(),
                  elapsed_s=None),
        teacher_ppl=None, layers={})

    base = perplexity(model, val)["ppl"]
    out["teacher_ppl"] = float(base)
    print(f"  teacher val ppl {base:.4f}", flush=True)

    for li in LAYERS:
        print(f"\n=== LAYER {li} ===", flush=True)
        layer, attr, original = attention_module(model, li)
        cap = {}

        class Capture:
            def __call__(self, x, *a, **k):
                cap["h"] = x
                return original(x, *a, **k)

        setattr(layer, attr, Capture())
        model(mx.array(ids[:1024][None, :].astype(np.int32)))
        probe_h = cap["h"]
        mx.eval(probe_h)
        setattr(layer, attr, original)
        ref = original(probe_h, None, None)
        mx.eval(ref)
        rms_target = rms(ref)

        setattr(layer, attr, GatedMemoryCarrier(int(probe_h.shape[-1]), mode="zero"))
        mx.eval(model.parameters())
        zero_ppl = perplexity(model, val)["ppl"]
        setattr(layer, attr, original)
        print(f"  probe {tuple(probe_h.shape)} rms_in {rms(probe_h):.4f} "
              f"attn_out_rms {rms_target:.5f} | zero ppl {zero_ppl:.4f}", flush=True)

        rec = dict(probe=dict(shape=[int(x) for x in probe_h.shape],
                              rms_in=rms(probe_h), rms_attn_out=rms_target),
                   zero_ppl=float(zero_ppl), paired=[], per_arm={})
        arms = [("transfer", None, "transfer"),
                ("random_matched", SEEDS[0], "random_matched")]
        for decay in DECAYS:
            row = dict(decay=float(decay))
            for arm, seed, mode in arms:
                r = evaluate(model, li, layer, attr, original, probe_h, decay=decay,
                             seed=seed, mode=mode, rms_target=rms_target,
                             select=select, val=val)
                r["wall_s"] = None
                row[arm] = r
                print(f"    d={decay:<6} {arm:<15} sel {r['select_ppl']:>8.4f} "
                      f"val {r['ppl']:>8.4f}", flush=True)
            row["ppl_gap_transfer_minus_control"] = float(
                row["transfer"]["ppl"] - row["random_matched"]["ppl"])
            rec["paired"].append(row)
            out["layers"][str(li)] = rec
            json.dump(out, open(OUT, "w"), indent=1)

        # seed sweep on the arm that wins on SELECT
        # select_ppl lives INSIDE each arm's record, not on the row. Reading the
        # row key raised KeyError after the whole sweep had already run.
        def _select_score(row):
            return min(row["transfer"]["select_ppl"],
                       row["random_matched"]["select_ppl"])
        best_decay = min((r["decay"] for r in rec["paired"]), key=lambda d_: _select_score(
            next(r for r in rec["paired"] if r["decay"] == d_)))
        print(f"  best SELECT decay {best_decay}; sweeping control seeds", flush=True)
        rows = []
        for s in SEEDS:
            r = evaluate(model, li, layer, attr, original, probe_h,
                         decay=best_decay, seed=s, mode="random_matched",
                         rms_target=rms_target, select=select, val=val)
            rows.append(dict(seed=int(s), ppl=float(r["ppl"]),
                             select_ppl=float(r["select_ppl"])))
            print(f"    seed {s}: val {r['ppl']:.4f}", flush=True)
        tr_ppl = [r["transfer"]["ppl"] for r in rec["paired"]
                  if r["decay"] == best_decay][0]
        rec["control_seed_sweep"] = dict(
            decay=float(best_decay), seeds=rows, transfer_ppl=float(tr_ppl),
            n_seeds_beating_transfer=int(sum(1 for r in rows
                                             if r["ppl"] < tr_ppl)))

        for arm in ("transfer", "random_matched"):
            rr = [dict(decay=r["decay"], ppl=r[arm]["ppl"],
                       select_ppl=r[arm]["select_ppl"]) for r in rec["paired"]]
            rec["per_arm"][arm] = dict(
                best_on_select=min(rr, key=lambda x: x["select_ppl"]),
                best_val=min(rr, key=lambda x: x["ppl"]),
                val_curve=[dict(decay=x["decay"], ppl=x["ppl"]) for x in rr])

    out["meta"]["elapsed_s"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT} in {out['meta']['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
