"""CAN OUR MODULE MAKE A PRETRAINED LLM ACTUALLY BETTER?

THE QUESTION, AND WHY EVERY EARLIER VERSION COULD NOT ANSWER IT
---------------------------------------------------------------
Every previous transplant experiment REPLACED an attention sublayer. That is a
destructive test: it can only ever show how much damage the swap does, so the
best available outcome is a partial recovery, and the ceiling is the teacher.
Those runs answer "how much can our module carry", which is a real question, but
it is not the question "can our brain method improve an LLM".

This adds a PARALLEL path instead. The layer keeps its attention untouched, and
our gated memory is summed alongside it:

    x = x + [ attn(ln1(x)) + gain * mem(ln1(x)) ]

Nothing is removed, so the teacher is a strict lower bound on what is reachable
and any improvement is a real improvement rather than a recovered fraction of
self-inflicted damage. The gain starts at 0, so at t=0 the model is EXACTLY the
teacher -- which is asserted, not assumed -- and the sweep moves it up.

THE THREE ARMS, and what each rules out:

  mem_transfer   gated memory whose W_v/W_o are transplanted from the attention
                 weights in the same layer. Tests: does attention's own value
                 subspace, re-expressed as a recurrence, add anything?
  mem_random     the same module with random W_v/W_o at the same weight scale
                 and the same output scale. Tests: is any gain just "more
                 capacity / a second branch", independent of whose weights?
  mem_frozen     the module with NO transplanted weights and NO training, i.e.
                 a fixed random recurrence. The trivial-intervention control.

If mem_transfer does not beat mem_random at matched scale, the answer is
"a second branch helps, our weights do not". If neither beats the teacher, the
answer is "this does not improve a pretrained LLM", which is a clean negative.

SCALE, AND WHY IT IS SWEPT RATHER THAN SET
------------------------------------------
The single biggest confound found in this project is that the module's output
scale was never validated: the committed headline ran at 0.44x the attention
rms it replaced, and that one constant moved the result by more than a
perplexity point. So the gain is swept here, chosen on the SELECT split, and
reported on VAL. Every arm is normalised by the same rule.

Frozen, deterministic: a fixed 3000-token window with no sampling, so these are
exact numbers and not estimates.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

os.environ.setdefault("HF_HOME", os.path.expanduser("~/zbrain/hf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_hybrid import (  # noqa: E402
    attention_module, transplant, perplexity, n_params, GatedMemoryCarrier,
)
from llm_efficiency import GatedMemory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "hybrid_parallel.json")
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
LAYER = int(os.environ.get("PAR_LAYER", "8"))
GAINS = [float(x) for x in os.environ.get(
    "PAR_GAINS", "0,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5").split(",")]
DECAYS = [float(x) for x in os.environ.get(
    "PAR_DECAYS", "0.5,0.7,0.8,0.9,0.999").split(",")]


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


class ParallelCarrier(nn.Module):
    """attention(x) + gain * memory(x), a drop-in for the attention sublayer.

    The attention is kept and called normally, so the model at gain=0 is the
    untouched teacher. `gain` is a plain float, not a parameter, so it cannot be
    moved by an optimizer; it is a swept constant and is reported per row.
    """

    def __init__(self, inner, mem, gain=0.0):
        super().__init__()
        self.inner = inner
        self.mem = mem
        self.gain = float(gain)

    def __call__(self, x, *args, **kwargs):
        base = self.inner(x, *args, **kwargs)
        if self.gain == 0.0:
            return base
        branch = self.gain * self.mem(x)
        # dtype promotion here would silently switch the residual stream from
        # bfloat16 to float32 and move perplexity by ~0.23 on its own -- see
        # docs/DTYPE_BUG.md
        if branch.dtype != base.dtype:
            branch = branch.astype(base.dtype)
        return base + branch


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
    print(f"{n_params(model):,} params | select {len(select):,} | "
          f"val {len(val):,} | layer {LAYER}\n", flush=True)

    base_val = perplexity(model, val)["ppl"]
    base_sel = perplexity(model, select)["ppl"]
    print(f"  teacher  val {base_val:.4f}  select {base_sel:.4f}", flush=True)

    layer, attr, original = attention_module(model, LAYER)
    ATTENTION_ORIGINAL = original

    # capture REAL activations at this layer for the scale probe
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
    print(f"  probe rms {rms(probe_h):.4f} | attention out rms {r_ref:.5f}", flush=True)

    # the teacher must be bit-identical at gain 0 -- this makes "we improved on
    # the teacher" a meaningful statement rather than a comparison across
    # different code paths
    zero_mem = GatedMemoryCarrier(d, mode="zero")
    carrier0 = ParallelCarrier(original, zero_mem, gain=0.0)
    setattr(layer, attr, carrier0)
    mx.eval(model.parameters())
    ident = perplexity(model, val)["ppl"]
    setattr(layer, attr, original)
    if abs(ident - base_val) > 1e-12:
        raise RuntimeError(
            f"gain=0 identity check FAILED: {ident} vs {base_val}. The branch "
            f"contributes nothing at gain 0, so this must be exact; a mismatch "
            f"means the comparison is contaminated (dtype promotion is the "
            f"usual cause -- see docs/DTYPE_BUG.md), and every teacher-relative "
            f"delta below would be measuring the bug rather than the module.")
    print(f"  gain=0 identity check: val {ident:.6f} vs teacher {base_val:.6f}"
          f"  -> EXACT\n", flush=True)

    out = dict(teacher=TEACHER, layer=LAYER, n_layers=len(model.model.layers),
               teacher_val_ppl=base_val, teacher_select_ppl=base_sel,
               identity_check_val=ident, identity_exact=bool(abs(ident-base_val) < 1e-9),
               val_tokens=len(val), select_tokens=len(select),
               gains=GAINS, decays=DECAYS, arms={}, protocol=dict(
                   mode="PARALLEL: attention is kept, memory is added alongside",
                   gain="fixed swept scalar, not a parameter, never trained",
                   selection="decay and gain chosen on SELECT; VAL only reported",
                   eval="frozen, fixed 3000-token windows, no sampling -> exact",
                   note="gain=0 reproduces the teacher exactly, so any ppl below "
                        "the teacher is an improvement over an unmodified model"))

    for arm in os.environ.get("PAR_ARMS", "mem_transfer,mem_random").split(","):
        rows = []
        for decay in DECAYS:
            # build the memory module once per decay
            if arm in ("mem_transfer", "mem_random"):
                setattr(layer, attr, original)
                car, rec = transplant(model, LAYER, "transfer",
                                      original=ATTENTION_ORIGINAL,
                                      probe_h=probe_h, decay=decay)
                mem = car.mem
                if arm == "mem_random":
                    rng = np.random.default_rng(0)
                    s_native = float(np.array(mem.v.weight).std())
                    mem.v.weight = mx.array(
                        rng.normal(0, s_native, mem.v.weight.shape).astype(np.float32))
                    mem.o.weight = mx.array(
                        rng.normal(0, s_native, mem.o.weight.shape).astype(np.float32))
            else:
                setattr(layer, attr, original)
                mem = GatedMemory(d, banks=1, chunk=64)
                b = math.log(decay / (1 - decay)) if 0 < decay < 1 else 30.0
                mem.gate.bias = mx.array(np.repeat(np.array([b], np.float32), d))
                mem.gate.weight = mx.array(
                    (np.array(mem.gate.weight) * 0.1).astype(np.float32))

            # normalise the memory branch to attention's output rms ONCE, so
            # gain=1.0 means "as much output power as the attention it sits
            # beside" and the sweep axis is interpretable across arms
            pc = ParallelCarrier(original, mem, gain=1.0)
            setattr(layer, attr, pc)
            mx.eval(model.parameters())
            o1 = mem(probe_h)
            mx.eval(o1)
            r_mem = rms(o1)
            normaliser = (r_ref / r_mem) if r_mem > 0 else 1.0

            for gain in GAINS:
                pc.gain = float(gain * normaliser)
                mx.eval(model.parameters())
                p_sel = perplexity(model, select)["ppl"]
                p_val = perplexity(model, val)["ppl"]
                rows.append(dict(decay=decay, sweep_gain=gain,
                                 effective_gain=float(pc.gain),
                                 select_ppl=p_sel, val_ppl=p_val,
                                 delta_vs_teacher=p_val - base_val,
                                 beats_teacher=bool(p_val < base_val)))
                print(f"  {arm:<13} decay {decay:<6.3f} gain {gain:<5.2f}  "
                      f"select {p_sel:>9.4f}  val {p_val:>9.4f}  "
                      f"{'  <-- BEATS TEACHER' if p_val < base_val else ''}",
                      flush=True)
                setattr(layer, attr, original)
                setattr(layer, attr, pc)
            setattr(layer, attr, original)

        # pick decay+gain on SELECT only
        best = min(rows, key=lambda r: r["select_ppl"])
        out["arms"][arm] = dict(rows=rows, best=best)
        print(f"  -> {arm} best on SELECT: decay {best['decay']} "
              f"gain {best['sweep_gain']} -> val {best['val_ppl']:.4f} "
              f"({best['delta_vs_teacher']:+.4f} vs teacher)\n", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

    setattr(layer, attr, original)
    bests = {a: v["best"] for a, v in out["arms"].items()}
    improving = {a: b for a, b in bests.items() if b["val_ppl"] < base_val}
    out["verdict"] = dict(
        teacher_val_ppl=base_val,
        identity_check_exact=out["identity_exact"],
        arms_at_their_best={a: dict(val_ppl=b["val_ppl"], decay=b["decay"],
                                    gain=b["sweep_gain"],
                                    delta=b["delta_vs_teacher"])
                            for a, b in bests.items()},
        any_arm_beats_teacher=bool(improving),
        arms_beating_teacher=sorted(improving),
        best_arm=min(bests, key=lambda a: bests[a]["val_ppl"]) if bests else None,
        best_delta=(min(b["val_ppl"] for b in bests.values()) - base_val) if bests else None,
        note=("Because gain=0 reproduces the teacher exactly (asserted above), a "
              "val ppl below the teacher is an improvement over an UNMODIFIED "
              "pretrained model. Arms are chosen on SELECT and reported on VAL."))
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== VERDICT (chosen on SELECT, reported on VAL) ===")
    print(f"  teacher (unmodified)  val {base_val:>9.4f}")
    for a, b in bests.items():
        print(f"  {a:<22} val {b['val_ppl']:>9.4f}  "
              f"{b['delta_vs_teacher']:+.4f}  at decay {b['decay']} gain {b['sweep_gain']}")
    print(f"\n  any arm beats the unmodified teacher: {out['verdict']['any_arm_beats_teacher']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
