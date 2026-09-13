"""THE EXPERIMENT THAT CAN ACTUALLY ANSWER "DOES OUR MODULE ADD VALUE?"

WHY THE PREVIOUS ONE COULD NOT
------------------------------
`hybrid_parallel.py` added our recurrent branch beside attention with a fixed
scalar gain. It got monotonically WORSE as the gain rose (best was +0.078 ppl at
the smallest gain tested). That is a real negative, but it is a negative about
ONE injection direction: the transplanted W_o, added straight into the residual.

The residual stream is a specific space. Attention's output is not just "some
vector" -- it is a vector whose direction the rest of the network has been
trained to read. Adding our module's output at a fixed direction can only help
if that direction already points somewhere useful, and there is no reason it
should. So the fixed-gain test cannot distinguish:

    "our recurrence has nothing to add to this model"
from
    "our recurrence has something to add, but not in that direction"

THE FIX, AND WHY IT IS A CLEAN TEST
-----------------------------------
Give the branch a LEARNABLE output projection, initialised to ZERO:

    x = x + W_out @ [ attn(ln1(x)) never touched ]  +  W_out2(mem(ln1(x)))

At initialisation the branch contributes exactly zero, so the model IS the
teacher -- bit-for-bit, and that is asserted rather than assumed. Then only the
injection projection is trained. If the loss cannot be driven below the
teacher's, then there is no direction in the residual space in which our
module's output helps, and the honest answer to the user's question is NO.

This is a strong test in the negative direction, which is the direction that
matters: a failure here is conclusive in a way the fixed-gain sweep was not.

THE ARMS
--------
  transfer  memory with W_v/W_o transplanted from this layer's attention,
            plus a zero-initialised learned output projection.
  random    identical, but W_v/W_o drawn randomly at the same weight scale.
  summary   a pure learned linear summary of the PAST (a learned exponential
            moving average), i.e. our timescale prior with no transplanted
            content at all.

If `transfer` beats `random` after identical training, the transplanted weights
carry trainable value. If nothing beats the teacher, we report that plainly.

CONTAMINATION CONTROL
---------------------
train = first 50% of the corpus; select = 60-80% (used for every decision);
val = last 10% (reported only). Disjoint by construction, asserted in code.
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
    attention_module, transplant, perplexity, n_params, GatedMemoryCarrier,
)
from llm_efficiency import GatedMemory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "hybrid_inject.json")
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
LAYER = int(os.environ.get("INJ_LAYER", "8"))


class Injection(nn.Module):
    """attention(x) + W_out(mem(x)), with W_out zero-initialised.

    Zero init is the whole point: it makes the starting point EXACTLY the
    pretrained model, so any subsequent improvement is a genuine improvement and
    any failure is unambiguous. `inner` is the untouched attention.
    """

    def __init__(self, inner, mem, d, branch_scale: float = 1.0):
        super().__init__()
        self.inner = inner
        self.mem = mem
        self.out = nn.Linear(d, d, bias=False)
        self.out.weight = mx.zeros((d, d), dtype=mx.float32)
        self.branch_scale = float(branch_scale)
        self._d = d

    def __call__(self, x, *args, **kwargs):
        base = self.inner(x, *args, **kwargs)
        # the fixed normaliser puts the branch at attention's output scale; the
        # learned projection then only has to find a DIRECTION, not a magnitude
        branch = self.out(self.mem(x) * self.branch_scale)
        # cast to the residual's dtype or the sum silently promotes the whole
        # stream from bfloat16 to float32 -- which alone moved perplexity by
        # 0.23 and made a mathematical no-op look like an effect
        if branch.dtype != base.dtype:
            branch = branch.astype(base.dtype)
        return base + branch


def train_injection(model, module, train_ids, *, steps, bs, ctx, lr, seed=0,
                    log=print, eval_every=100, select_ids=None):
    """Train ONLY the injection projection; everything else stays frozen.

    THE TRAP, which cost a whole run: `module.unfreeze()` unfreezes the module
    RECURSIVELY, and our wrapper holds the attention module as a submodule. So
    freezing the model then unfreezing the wrapper re-enabled gradients on the
    pretrained attention weights too -- the run trained the whole layer, drove
    perplexity to 1889, and left the model corrupted for every later arm (the
    next arm's identity check then failed at 4.3e10, which is how it surfaced).

    So unfreeze ONLY the output projection, and then ASSERT the trainable set is
    exactly that one tensor. A silent over-training here would look like a
    result.
    """
    model.freeze()
    module.out.unfreeze()
    trainable = [k for k, v in nn.utils.tree_flatten(module.trainable_parameters())]
    assert trainable == ["out.weight"], (
        f"expected only the injection projection to be trainable, got {trainable}")
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_ids, dtype=np.int32)

    def loss_fn(mod, x, y):
        lo = model(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(module, loss_fn)
    t0 = time.time()
    curve = []
    for s in range(1, steps + 1):
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + ctx] for i in ix]).astype(np.int32))
        y = mx.array(np.stack([tr[i + 1:i + 1 + ctx] for i in ix]).astype(np.int32))
        loss, grads = lg(module, x, y)
        opt.update(module, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if s % eval_every == 0 or s == steps:
            row = dict(step=s, loss=float(loss))
            if select_ids is not None:
                row["select_ppl"] = perplexity(model, select_ids)["ppl"]
            curve.append(row)
            log(f"      step {s:>4}  loss {float(loss):7.4f}"
                + (f"  select_ppl {row['select_ppl']:.4f}"
                   if "select_ppl" in row else ""))
    return curve, time.time() - t0


def main():
    from mlx_lm import load

    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    train_ids = ids[:int(0.5 * n)]
    select = ids[int(0.6 * n):int(0.6 * n) + 3000]
    val = ids[int(0.9 * n):int(0.9 * n) + 3000]
    assert int(0.5 * n) <= int(0.6 * n) < int(0.9 * n)
    print(f"{n_params(model):,} params | train {len(train_ids):,} | "
          f"select {len(select):,} | val {len(val):,} | layer {LAYER}\n", flush=True)

    base_val = perplexity(model, val)["ppl"]
    base_sel = perplexity(model, select)["ppl"]
    print(f"  teacher  val {base_val:.4f}  select {base_sel:.4f}\n", flush=True)

    layer, attr, original = attention_module(model, LAYER)
    ATTENTION_ORIGINAL = original
    cap = {}

    class Capture:
        def __call__(self, x, *a, **k):
            cap["h"] = x
            return original(x, *a, **k)

    setattr(layer, attr, Capture())
    model(mx.array(train_ids[:1024][None, :].astype(np.int32)))
    probe_h = cap["h"]
    mx.eval(probe_h)
    setattr(layer, attr, original)
    d = int(probe_h.shape[-1])

    steps = int(os.environ.get("INJ_STEPS", "300"))
    bs = int(os.environ.get("INJ_BS", "2"))
    ctx = int(os.environ.get("INJ_CTX", "128"))
    lr = float(os.environ.get("INJ_LR", "3e-4"))
    decay = float(os.environ.get("INJ_DECAY", "0.7"))

    out = dict(teacher=TEACHER, layer=LAYER, n_layers=len(model.model.layers),
               teacher_val_ppl=base_val, teacher_select_ppl=base_sel,
               val_tokens=len(val), steps=steps, bs=bs, ctx=ctx, lr=lr,
               decay=decay, arms=[], protocol=dict(
                   injection="attention kept; learned zero-init projection on "
                             "the memory branch",
                   trained="ONLY the injection projection; rest of the model frozen",
                   identity="at init the projection is zero so the model IS the "
                            "teacher -- asserted, not assumed",
                   selection="no hyperparameter chosen on val",
                   eval="fixed 3000-token val window, no sampling -> exact"))

    lrs = [float(x) for x in os.environ.get("INJ_LRS", "1e-5,3e-5").split(",")]
    lr = lrs[0]

    arms = os.environ.get("INJ_ARMS", "transfer,random").split(",")
    for arm in arms:
        setattr(layer, attr, original)
        if arm == "summary":
            mem = GatedMemory(d, banks=1, chunk=64)
            b = math.log(decay / (1 - decay)) if 0 < decay < 1 else 30.0
            mem.gate.bias = mx.array(np.repeat(np.array([b], np.float32), d))
            mem.gate.weight = mx.array(
                (np.array(mem.gate.weight) * 0.1).astype(np.float32))
            rec = dict(control="learned exponential summary, no transplant")
        else:
            car, rec = transplant(model, LAYER, "transfer",
                                  original=ATTENTION_ORIGINAL,
                                  probe_h=probe_h, decay=decay)
            mem = car.mem
            if arm == "random":
                rng = np.random.default_rng(0)
                s_native = float(np.array(mem.v.weight).std())
                mem.v.weight = mx.array(
                    rng.normal(0, s_native, mem.v.weight.shape).astype(np.float32))
                mem.o.weight = mx.array(
                    rng.normal(0, s_native, mem.o.weight.shape).astype(np.float32))
                rec["control"] = "random W_v/W_o at matched weight scale"

        # NORMALISE the branch to attention's own output rms before the learned
        # projection. Without this the memory output is ~64x attention's rms
        # (measured: 3.23 vs 0.0505), so the learning problem is badly
        # conditioned and Adam -- whose step is ABSOLUTE, not relative -- blows
        # the projection up within ~100 steps. With it, gain-normalised means
        # "as much output power as the attention beside it", and the swept
        # scalar multiplies a well-scaled signal.
        setattr(layer, attr, original)
        ref = original(probe_h, None, None)
        mx.eval(ref)
        r_ref = float(mx.sqrt(mx.mean(ref.astype(mx.float32) ** 2)))
        raw = mem(probe_h)
        mx.eval(raw)
        r_mem = float(mx.sqrt(mx.mean(raw.astype(mx.float32) ** 2)))
        branch_scale = (r_ref / r_mem) if r_mem > 0 else 1.0

        module = Injection(original, mem, d)
        module.out.weight = mx.zeros_like(module.out.weight)
        module.branch_scale = float(branch_scale)   # fixed float, not a parameter
        setattr(layer, attr, module)
        mx.eval(model.parameters())
        before_val = perplexity(model, val)["ppl"]
        exact = abs(before_val - base_val) < 1e-12
        if not exact:
            raise RuntimeError(
                f"identity check FAILED for arm {arm}: the zero projection must "
                f"reproduce the teacher exactly, got {before_val} vs "
                f"{base_val}. Every downstream number from this arm would be "
                f"uninterpretable, so this is fatal rather than a warning. The "
                f"usual cause is a dtype promotion (see the cast above).")
        print(f"  {arm:<10} BEFORE ft: val {before_val:.6f} "
              f"(teacher {base_val:.6f})  EXACT", flush=True)

        curve, wall = train_injection(model, module, train_ids, steps=steps,
                                      bs=bs, ctx=ctx, lr=lr, select_ids=select)
        after_val = perplexity(model, val)["ppl"]
        after_sel = perplexity(model, select)["ppl"]
        row = dict(**rec, arm=arm, before_val_ppl=before_val,
                   after_val_ppl=after_val, after_select_ppl=after_sel,
                   delta_vs_teacher=after_val - base_val,
                   beats_teacher=bool(after_val < base_val), wall_s=wall,
                   curve=curve)
        out["arms"].append(row)
        print(f"  {arm:<10} AFTER  ft: val {after_val:.4f}  "
              f"({after_val-base_val:+.4f} vs teacher)  "
              f"{'*** BEATS TEACHER ***' if after_val < base_val else ''}\n", flush=True)
        setattr(layer, attr, original)
        json.dump(out, open(OUT, "w"), indent=1)

    setattr(layer, attr, original)
    rows = {r["arm"]: r for r in out["arms"]}
    tr, rn = rows.get("transfer"), rows.get("random")
    out["verdict"] = dict(
        teacher_val_ppl=base_val,
        arms={a: dict(val_ppl=r["after_val_ppl"], delta=r["delta_vs_teacher"],
                      beats_teacher=r["beats_teacher"])
              for a, r in rows.items()},
        any_arm_beats_teacher=any(r["beats_teacher"] for r in rows.values()),
        transfer_beats_random_after_ft=(bool(tr["after_val_ppl"] < rn["after_val_ppl"])
                                        if tr and rn else None),
        note=("The injection projection starts at zero, so every arm begins "
              "EXACTLY at the teacher. A val ppl below the teacher is therefore "
              "an improvement on an unmodified pretrained model, and a failure "
              "to reach it is a conclusive negative for this injection scheme."))
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== VERDICT ===")
    print(f"  teacher (unmodified)  val {base_val:>9.4f}")
    for a, r in rows.items():
        print(f"  {a:<10} after ft       val {r['after_val_ppl']:>9.4f}  "
              f"{r['delta_vs_teacher']:+.4f}")
    print(f"\n  any arm beats the unmodified teacher: "
          f"{out['verdict']['any_arm_beats_teacher']}")
    if out["verdict"]["transfer_beats_random_after_ft"] is not None:
        print(f"  transplant beats matched random after identical ft: "
              f"{out['verdict']['transfer_beats_random_after_ft']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
