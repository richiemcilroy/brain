"""DOES A BETTER RECENCY FIT ACTUALLY IMPROVE PERPLEXITY?

docs/IMPROVE.md shows the carrier's default decay placement fits attention's
recency profile badly (cdf_l1 111 vs 7.5 at the optimal placement). But fit
quality is a PROXY. This measures the thing that matters.

Arms, all at the SAME bank count and therefore the same parameter count, so the
comparison cannot repeat the feature-count confound that refuted the earlier
multiscale claim (docs/MULTISCALE.md):

  default_geo   (0.90, 0.99, 0.999, 0.9999)   -- what the library ships
  near1         (0.98, 0.99, 0.999, 0.99999)  -- clustered near 1
  optimal       (0.99245, 0.99496, 0.99748, 0.99999) -- top-NNLS placement

Each arm is built with the SAME transplant code path and rms-normalised, so the
only difference is where the decay banks sit.

FALSIFIER, stated before running: if the arms are indistinguishable, the kernel
misspecification is not the bottleneck and the diagnosis in docs/IMPROVE.md is
wrong. That is the outcome this experiment is designed to be able to produce.
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
from llm_hybrid import attention_module, transplant, perplexity, n_params  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
TEACHER = os.environ.get("BT_TEACHER", "unsloth/Llama-3.2-1B")
LAYER = int(os.environ.get("BT_LAYER", "8"))
OUT = os.environ.get("BT_OUT", os.path.join(HERE, "results", "banks_test.json"))
WINDOW = 3000

# THE DECISIVE CONTROL, added after the first run: a SINGLE bank at the decay
# verify_transplant.py already measured as best (0.8 -> 22.7305). If multi-bank
# is worse than this, the "more timescales" hypothesis is dead regardless of how
# well the banks fit the recency profile.
SCHEMES = {
    "single_0.8": (0.8,),
    "single_0.9": (0.9,),
    "default_geo": (0.90, 0.99, 0.999, 0.9999),
    "near1": (0.98, 0.99, 0.999, 0.99999),
    "optimal": (0.99245, 0.99496, 0.99748, 0.99999),
}


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


def main():
    from mlx_lm import load
    t0 = time.time()
    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    select = ids[int(0.6 * n):int(0.6 * n) + WINDOW]
    val = ids[int(0.9 * n):int(0.9 * n) + WINDOW]

    out = dict(
        meta=dict(teacher=TEACHER, n_params=n_params(model), layer=LAYER,
                  n_layers=len(model.model.layers), corpus_tokens=int(n),
                  host=platform.platform(), driver="experiments/banks_test.py",
                  val_sha1=hashlib.sha1(val.tobytes()).hexdigest(),
                  preregistered_falsifier=(
                      "if fit quality does not translate into perplexity, the "
                      "kernel-misspecification diagnosis is WRONG"),
                  elapsed_s=None),
        teacher_ppl=None, arms={})

    base = perplexity(model, val)["ppl"]
    out["teacher_ppl"] = float(base)
    print(f"  teacher val ppl {base:.4f}", flush=True)

    layer, attr, original = attention_module(model, LAYER)
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
    print(f"  probe {tuple(probe_h.shape)} attn_out_rms {rms_target:.5f}", flush=True)

    for name, decays in SCHEMES.items():
        mx.random.seed(0)
        carrier, rec = transplant(model, LAYER, "transfer", original=original,
                                  probe_h=probe_h, banks=len(decays),
                                  decays=decays)
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
        entry = dict(decays=[float(x) for x in decays], banks=len(decays),
                     ppl=float(p_val), select_ppl=float(p_sel),
                     rms_pre_norm=float(pre), rms_post_norm=float(rms(o2)),
                     rms_target=float(rms_target), out_gain=float(carrier.out_gain),
                     value_expand_reps=rec.get("value_expand_reps"),
                     quantized=rec.get("quantized"))
        out["arms"][name] = entry
        print(f"  {name:16s} decays {[round(float(x),5) for x in decays]} "
              f"-> val {p_val:.4f}  select {p_sel:.4f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

    # paired comparison against the shipped default
    d = out["arms"]["default_geo"]["ppl"]
    for name, e in out["arms"].items():
        if name == "default_geo":
            continue
        e["delta_vs_default_ppl"] = float(d - e["ppl"])
        e["better_than_default"] = bool(e["ppl"] < d)
    best = min(out["arms"].items(), key=lambda kv: kv[1]["ppl"])
    out["verdict"] = dict(
        best_arm=best[0], best_ppl=float(best[1]["ppl"]),
        default_ppl=float(d),
        improvement_ppl=float(d - best[1]["ppl"]),
        fit_quality_translates=bool(best[0] != "default_geo" and best[1]["ppl"] < d))
    out["meta"]["elapsed_s"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nVERDICT: best {best[0]} at {best[1]['ppl']:.4f} vs default {d:.4f} "
          f"({'+' if d>best[1]['ppl'] else ''}{d-best[1]['ppl']:.4f} ppl)", flush=True)
    print(f"wrote {OUT} in {out['meta']['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
