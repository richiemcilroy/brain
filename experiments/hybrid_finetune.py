"""THE DECISIVE EXPERIMENT, v2: does a transplanted attention layer finetune
better than a scale-matched random init?

READ THIS FIRST -- v1 AND WHAT WAS WRONG WITH IT
------------------------------------------------
v1 of this experiment ran 300 Adam steps at lr=3e-4 on both arms and drove them
to 1392.79 ppl (transfer) and 1554.12 ppl (random) from 23.72 and 74.63. Both
arms were destroyed, which is a protocol failure, not a result. The cause is
worth stating precisely because it is easy to repeat:

    Adam's step is ABSOLUTE (lr), not relative to the weight it updates.

`transplant` normalises the gated trace by scaling W_v by (1-g)=0.01, because
h = g*h + v is an unnormalised SUM where attention's output is a CONVEX
combination. At d=2048 the transplanted W_v has per-element magnitude ~0.002,
so an lr of 3e-4 moves each element by ~15% of its own magnitude every step.
The run measured how fast Adam can destroy a precise init, nothing else.

v2 fixes this two ways, and both matter:

  1. NORMALISATION MOVES TO A FIXED OUTPUT GAIN. Scaling the output by (1-g)
     and scaling W_v by (1-g) are mathematically identical (verified: max abs
     diff 7.6e-10 on a random module). But the output gain is not a parameter,
     so no optimizer can touch it. W_v keeps its native trained magnitude and
     Adam's absolute step becomes a normal relative step again.

  2. BOTH ARMS ARE OUTPUT-RMS MATCHED TO THE ATTENTION THEY REPLACE, by the
     same rule, via that same fixed gain. In v1 only the random arm was matched
     (to 1.019x) while the transplanted arm sat at 0.44x, so the two arms
     differed in output SCALE as well as in weight provenance. That is a
     confound: a reader could not tell whether the random arm lost because its
     weights are meaningless or because its contribution was 2.3x too large.

After both fixes the arms differ in exactly one thing: whose weights they are.
Transfer's weights came from training; random's are fan-in draws. Same
architecture, same fitted decay, same gate treatment, same output scale, same
data, same steps, same lr.

THE ARMS
--------
  zero            attention deleted entirely. The reference every arm is
                  judged against -- any swap destroys information, so the
                  question is whether ours destroys LESS than nothing.
  transfer        gated memory with W_v/W_o transplanted from attention.
  random_matched  gated memory with random W_v/W_o at fan-in scale, calibrated
                  to the same output rms. THE CONTROL THAT DECIDES IT.

WHAT WOULD FALSIFY IT
---------------------
* If `transfer` does not beat `random_matched` after IDENTICAL finetuning, the
  transplant bought nothing and the frozen win came from the architecture plus
  the fitted decay. That is a clean negative and gets reported as one.
* If `transfer` does not beat `zero`, the primitive does not carry the replaced
  function better than deleting it.

CONTAMINATION CONTROL
---------------------
train = first 50% of the corpus. select = 60%..80%, used ONLY to read the lr
sweep. val = last 10%, the reported number. They are disjoint by construction
and the script asserts it. lr is never chosen on val.
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

os.environ.setdefault("HF_HOME", os.path.expanduser("~/zbrain/hf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_hybrid import (  # noqa: E402
    attention_module, transplant, perplexity, n_params, to_f32,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "hybrid_finetune.json")
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
LAYER = int(os.environ.get("FT_LAYER", "8"))


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


def build_arm(model, layer_idx, arm, probe_h, ref_out, original):
    """Construct a carrier for `arm`, calibrating its gain to attention's rms.

    `original` MUST be the attention module that was removed. Re-deriving it
    from the model is a trap: once a carrier is installed the slot no longer
    holds a parameterised attention module, so the inferred width becomes None
    and the next call fails two frames later. This bit an earlier version.
    """
    layer, attr, _ = attention_module(model, layer_idx)
    setattr(layer, attr, original)
    carrier, rec = transplant(model, layer_idx, "transfer", original=original)
    rec["arm"] = arm

    if arm == "random_matched":
        d = rec["d"]
        rng = np.random.default_rng(0)
        v_rand = rng.normal(0, 1.0 / math.sqrt(d), size=(d, d)).astype(np.float32)
        o_rand = rng.normal(0, 1.0 / math.sqrt(d), size=(d, d)).astype(np.float32)
        carrier.mem.v.weight = mx.array(v_rand)
        carrier.mem.o.weight = mx.array(o_rand)
        rec["transplanted"] = False
        rec["control"] = "random W_v/W_o at fan-in scale, output-rms matched"

    # calibrate the fixed gain so the module's output rms equals attention's,
    # for EVERY arm, including transfer
    mx.eval(model.parameters())
    carrier.out_gain = 1.0
    out_new = carrier(probe_h, None, None)
    mx.eval(out_new)
    r_new = rms(out_new)
    r_ref = rms(ref_out)
    gain = (r_ref / r_new) if r_new > 0 else 1.0
    carrier.out_gain = float(gain)
    rec["rms_before_gain"] = r_new
    rec["rms_target"] = r_ref
    rec["out_gain_calibrated"] = float(gain)
    rec["wv_per_element_mag"] = float(
        mx.mean(mx.abs(carrier.mem.v.weight)).item())
    return carrier, rec


def finetune(model, carrier, train_ids, *, steps, bs, ctx, lr, select_ids,
             eval_every=50, seed=0, log=print):
    """Optimise ONLY the carrier. Everything else stays frozen.

    Freezing the rest is deliberate: the claim is about whether the module can
    hold the function, not how much of a 1.2B model gradient descent can paper
    over. Letting the whole model move would turn this into a different and much
    easier experiment.
    """
    model.freeze()
    carrier.unfreeze()
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_ids, dtype=np.int32)

    def loss_fn(car, x, y):
        lo = model(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(carrier, loss_fn)
    t0 = time.time()
    curve = []
    for s in range(1, steps + 1):
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + ctx] for i in ix]).astype(np.int32))
        y = mx.array(np.stack([tr[i + 1:i + 1 + ctx] for i in ix]).astype(np.int32))
        loss, grads = lg(carrier, x, y)
        opt.update(carrier, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if s % eval_every == 0 or s == steps:
            row = dict(step=s, loss=float(loss))
            if select_ids is not None:
                row["select_ppl"] = perplexity(model, select_ids)["ppl"]
            curve.append(row)
            log(f"      step {s:>5}  loss {float(loss):7.4f}"
                + (f"  select_ppl {row['select_ppl']:.4f}" if "select_ppl" in row else ""))
    return curve, time.time() - t0


def main():
    from mlx_lm import load

    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    train_ids = ids[:int(0.5 * n)]
    select_ids = ids[int(0.6 * n):int(0.6 * n) + 3000]
    val = ids[int(0.9 * n):int(0.9 * n) + 3000]
    assert int(0.5 * n) <= int(0.6 * n) < int(0.9 * n)
    print(f"{n_params(model):,} params | train {len(train_ids):,} | "
          f"select {len(select_ids):,} | val {len(val):,} | layer {LAYER}\n", flush=True)

    base = perplexity(model, val)
    print(f"  teacher ppl {base['ppl']:.4f}\n", flush=True)
    zero_ppl = float("nan")

    steps = int(os.environ.get("FT_STEPS", "300"))
    bs = int(os.environ.get("FT_BS", "2"))
    ctx = int(os.environ.get("FT_CTX", "128"))
    lrs = [float(x) for x in
           os.environ.get("FT_LRS", "1e-5,3e-5").split(",")]
    arms = os.environ.get("FT_ARMS", "transfer,random_matched").split(",")

    # REAL layer-LAYER activations for the rms calibration. Passing the
    # embedding through layer-0's norm (the obvious shortcut) measures the wrong
    # distribution entirely -- layer 8 has been through 8 blocks of residual
    # mixing and its rms is not the embedding's. So capture the actual input the
    # replaced module receives, and its actual output, by running a capture
    # module in its place. Probe tokens come from TRAIN, so calibration cannot
    # leak val information.
    layer0, attr0, original0 = attention_module(model, LAYER)
    ATTENTION_ORIGINAL = original0
    captured = {}

    class Capture:
        def __call__(self, x, *a, **k):
            captured["h"] = x
            return original0(x, *a, **k)

    setattr(layer0, attr0, Capture())
    probe_ids = mx.array(train_ids[:1024][None, :].astype(np.int32))
    model(probe_ids)
    probe_h = captured["h"]
    ref_out = original0(probe_h, None, None)
    mx.eval(probe_h, ref_out)
    setattr(layer0, attr0, original0)
    d_model = int(probe_h.shape[-1])
    print(f"  calibration probe: real layer-{LAYER} activations "
          f"{tuple(probe_h.shape)}, rms {rms(probe_h):.4f}, "
          f"attention out rms {rms(ref_out):.4f}", flush=True)

    out = dict(teacher=TEACHER, base_params=n_params(model), layer=LAYER,
               n_layers=len(model.model.layers), val_tokens=len(val),
               select_tokens=len(select_ids), teacher_ppl=base["ppl"],
               steps=steps, bs=bs, ctx=ctx, lrs=lrs, arms=[],
               protocol=dict(
                   normalisation="fixed output gain (not a parameter)",
                   rms_matching="every arm matched to attention output rms, same rule",
                   frozen="only the carrier is trainable",
                   lr_selection="selection split only; val never used to choose lr",
                   naive_v1_artifact="results/hybrid_finetune_naive.json"))

    # zero arm: attention deleted, the reference
    layer, attr, original = attention_module(model, LAYER)
    from llm_hybrid import GatedMemoryCarrier
    setattr(layer, attr, GatedMemoryCarrier(d_model, mode="zero"))
    mx.eval(model.parameters())
    zero_ppl = perplexity(model, val)["ppl"]
    setattr(layer, attr, original)
    out["zero_ppl"] = zero_ppl
    print(f"  zero (attention deleted) ppl {zero_ppl:.4f}\n", flush=True)

    for lr in lrs:
        for arm in arms:
            layer, attr, original = attention_module(model, LAYER)
            setattr(layer, attr, original)
            carrier, rec = build_arm(model, LAYER, arm, probe_h, ref_out,
                                     ATTENTION_ORIGINAL)
            mx.eval(model.parameters())
            before = perplexity(model, val)
            print(f"  lr={lr:g} {arm:<15} BEFORE ppl {before['ppl']:>9.4f}  "
                  f"(gain {rec['out_gain_calibrated']:.5f})", flush=True)
            curve, wall = finetune(model, carrier, train_ids, steps=steps, bs=bs,
                                   ctx=ctx, lr=lr, select_ids=select_ids, log=print)
            after = perplexity(model, val)
            row = dict(**rec, lr=lr, before_ppl=before["ppl"],
                       after_ppl=after["ppl"], wall_s=wall, curve=curve)
            out["arms"].append(row)
            print(f"  lr={lr:g} {arm:<15} AFTER  ppl {after['ppl']:>9.4f}  "
                  f"({after['ppl']-before['ppl']:+.4f}, {wall:.0f}s)\n", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
            setattr(layer, attr, original)

    # verdict at each lr: read on val, having never used it for selection
    verdicts = []
    for lr in lrs:
        rows = {r["arm"]: r for r in out["arms"] if r["lr"] == lr}
        if "transfer" in rows and "random_matched" in rows:
            t, r = rows["transfer"], rows["random_matched"]
            denom = zero_ppl - base["ppl"]
            verdicts.append(dict(
                lr=lr,
                transfer_after=t["after_ppl"], random_after=r["after_ppl"],
                zero=zero_ppl, teacher=base["ppl"],
                transfer_beats_random=bool(t["after_ppl"] < r["after_ppl"]),
                margin_random_minus_transfer=r["after_ppl"] - t["after_ppl"],
                transfer_beats_zero=bool(t["after_ppl"] < zero_ppl),
                random_beats_zero=bool(r["after_ppl"] < zero_ppl),
                transfer_recovery_frac=(zero_ppl - t["after_ppl"]) / denom if denom > 0 else None,
                random_recovery_frac=(zero_ppl - r["after_ppl"]) / denom if denom > 0 else None))
    out["verdicts"] = verdicts
    out["verdict"] = verdicts[0] if verdicts else None
    out["interpretation"] = (
        "transfer_beats_random is the decisive test: it separates 'the "
        "transplanted weights carry the function' from 'any module with the "
        "right decay and scale helps'. A negative here is a clean falsification "
        "of the transplant claim.")
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== VERDICT (val, lr never selected on val) ===")
    print(f"  teacher                ppl {base['ppl']:>9.4f}")
    print(f"  zero (attn deleted)    ppl {zero_ppl:>9.4f}")
    for v in verdicts:
        print(f"  --- lr {v['lr']:g} ---")
        print(f"  transfer    after      ppl {v['transfer_after']:>9.4f}  "
              f"recovers {v['transfer_recovery_frac']*100:5.1f}% of the zero gap")
        print(f"  random      after      ppl {v['random_after']:>9.4f}  "
              f"recovers {v['random_recovery_frac']*100:5.1f}%")
        print(f"  transfer beats random after ft: {v['transfer_beats_random']} "
              f"(margin {v['margin_random_minus_transfer']:+.4f} ppl)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
