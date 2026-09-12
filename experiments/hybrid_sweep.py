"""Layer sweep + the control that decides whether the weight transplant mattered.

The single-layer result (layer 8 of 16, Llama-3.2-1B, 3000 held-out tokens):

    teacher     ppl 20.3756      unmodified
    zero        ppl 24.1044      attention deleted entirely
    transfer    ppl 23.7154      our gated memory, attention's weights transplanted

`transfer` beats `zero`, which is the first of the two gates. But that result
alone is NOT evidence for our primitive, because replacing a sublayer with ANY
trainable module might beat deleting it. Two controls decide it:

  random   same architecture, SAME fitted gate decay, random v/o weights.
           If random also beats zero by a similar margin, then the transplant
           bought nothing and the win belongs to "having a recurrence with the
           right time constant", not to attention's weights.
  identity the sublayer returns its own input. A second trivial intervention.

And robustness: if the effect exists it should not be a property of layer 8.
This sweeps several layers in ONE process (the model loads once), restoring the
original attention between arms, so state cannot leak.

Each arm is deterministic: the evaluation is a fixed 3000-token window and
there is no sampling, so these are exact numbers, not samples. Seed variation
enters only through the random arm's init, which is seeded.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

os.environ.setdefault("HF_HOME", "/tmp/zz_llm/hf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_hybrid import (  # noqa: E402
    get_layers, attention_module, transplant, perplexity, n_params,
    GatedMemoryCarrier,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SWEEP_OUT", os.path.join(HERE, "results", "hybrid_sweep.json"))
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")


def main():
    from mlx_lm import load

    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    base_par = n_params(model)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    val = ids[int(0.9 * len(ids)):int(0.9 * len(ids)) + 3000]
    layers, _ = get_layers(model)
    n_layer = len(layers)
    print(f"{base_par:,} params, {n_layer} layers, val {len(val)} tokens\n",
          flush=True)

    base = perplexity(model, val)
    print(f"  teacher ppl {base['ppl']:.4f}\n", flush=True)

    out = dict(teacher=TEACHER, base_params=base_par, n_layers=n_layer,
               val_tokens=len(val), teacher_ppl=base["ppl"], arms=[])

    # save and restore the original attention so the model is pristine per arm
    layer_idx = int(os.environ.get("SWEEP_LAYER", "8"))
    layer, attr, original = attention_module(model, layer_idx)

    modes = os.environ.get("SWEEP_MODES", "zero,random,transfer").split(",")
    seeds = [int(s) for s in os.environ.get("SWEEP_SEEDS", "0").split(",")]

    def restore():
        setattr(layer, attr, original)

    for mode in modes:
        for seed in seeds:
            mx.random.seed(seed)
            restore()
            carrier, rec = transplant(model, layer_idx, mode)
            mx.eval(model.parameters())
            r = perplexity(model, val)
            rec.pop("mode", None)
            row = dict(layer=layer_idx, mode=mode, seed=seed, ppl=r["ppl"],
                       nll=r["nll"], **rec)
            out["arms"].append(row)
            print(f"  layer {layer_idx:>2} {mode:<9} seed={seed} "
                  f"ppl={r['ppl']:>9.4f}  (+{r['ppl']-base['ppl']:+.4f} vs teacher)",
                  flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    restore()

    # verdict for this layer
    def ppl_of(m):
        v = [a["ppl"] for a in out["arms"] if a["mode"] == m]
        return float(np.mean(v)) if v else float("nan")

    t, z, rnd, tr = base["ppl"], ppl_of("zero"), ppl_of("random"), ppl_of("transfer")
    recov = float((z - tr) / (z - t)) if z > t else None
    recov_rnd = float((z - rnd) / (z - t)) if (z > t and rnd == rnd) else None
    out["verdict"] = dict(
        teacher=t, zero=z, random_init=rnd, transfer=tr,
        transfer_beats_zero=bool(tr < z) if (tr == tr and z == z) else None,
        random_beats_zero=bool(rnd < z) if (rnd == rnd and z == z) else None,
        transfer_beats_random=bool(tr < rnd) if (tr == tr and rnd == rnd) else None,
        zero_gap=t and float(z - t),
        transfer_recovers_fraction=recov,
        random_recovers_fraction=recov_rnd,
        note=("transfer_beats_random is the decisive test: it separates 'the "
              "transplanted weights carry the function' from 'any module with a "
              "sensible decay helps'."),
    )
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== VERDICT (layer %d) ===" % layer_idx)
    print(f"  teacher       ppl {t:>9.4f}")
    print(f"  zero          ppl {z:>9.4f}  ({z-t:+.4f} vs teacher)  <- delete attention")
    print(f"  random init   ppl {rnd:>9.4f}  ({rnd-t:+.4f})  <- same module, random weights")
    print(f"  transfer      ppl {tr:>9.4f}  ({tr-t:+.4f})  <- attention's weights")
    print(f"\n  transfer beats zero:   {out['verdict']['transfer_beats_zero']}")
    print(f"  transfer beats random: {out['verdict']['transfer_beats_random']}")
    if recov is not None:
        print(f"\n  transfer recovers {100*recov:.1f}% of the damage done by deleting attention")
    if recov_rnd is not None:
        print(f"  random   recovers {100*recov_rnd:.1f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
