"""IS THE INJECTION WIN ATTRIBUTABLE TO THE MEMORY, OR TO ANY TRAINED LOW-RANK BRANCH?

THE CLAIM UNDER TEST
--------------------
`hybrid_inject.py` reports that adding a gated-memory branch beside layer 8's
attention improves `unsloth/Llama-3.2-1B` from 20.3756 to 19.8922 val ppl
(-0.4834). The repo's own decomposition already flags the problem: a
scale-matched RANDOM-WEIGHT memory already reaches 19.9987 (-0.3769), i.e. 78%
of the gain. But `random` is still a memory module -- it still integrates the
past with a gate, its output is still a time-dependent function of the whole
history. So `transfer vs random` cannot separate two very different stories:

    (i)  the RECURRENT MEMORY contributes something a static map cannot;
    (ii) ANY trained rank-64 branch with 262k parameters improves this model by
         this much, and the recurrence adds nothing measurable.

`hybrid_inject.py` never tested (ii). This file does. The discriminating arms
drop the memory entirely and keep everything else as close as possible.

ARMS -- all layer 8, all rank 64, all zero-init `up`, all 262,144 trainable
--------------------------------------------------------------------------
  T   mem_transfer  the existing memory branch: W_v/W_o transplanted from the
                    attention beside it, fixed decay 0.7, plus the zero-init
                    rank-64 projection. Repro of the +0.4834 headline.
  R   mem_random    identical module, W_v/W_o random at the same weight scale.
  L1  lora_out      rank-64 zero-init low-rank branch on the ATTENTION OUTPUT.
                    NO memory, NO recurrent state: a static linear map of
                    `attn(ln1(x))`. "Any trained low-rank branch" in its
                    cheapest and most standard form (a LoRA on the sublayer
                    output).
  L2  lora_in       rank-64 zero-init low-rank branch on the layer INPUT
                    ln1(x). Static, no time dependence. Checks a second
                    readout point so L1 cannot win by reading position.

MATCHING, AND WHAT IS ACTUALLY MATCHED
--------------------------------------
* TRAINABLE parameters are exactly matched: rank 64 at d=2048 gives
  64*2048 + 2048*64 = 262,144 scalars in all four arms -- asserted per arm, not
  assumed.
* The branch INPUT is scale-matched: every arm's projection sees a signal whose
  rms has been set to the attention output rms measured on the same real probe.
  T and L1 get this for free (T's normaliser is the existing one; L1's input IS
  the attention output); L2 is normalised explicitly. Without this, Adam's
  absolute step size would make the arms differ in conditioning, not mechanism.
* TOTAL parameters are NOT matched between T/R and L1/L2, and this is worth
  stating rather than hiding: T and R carry a frozen carrier (W_v/W_o/gate,
  ~12.6M non-trainable scalars) that defines the feature the projection reads.
  T vs R are total-matched with each other; L1/L2 are total-matched with each
  other and are the smaller models. Only 262,144 scalars receive gradients in
  every arm, so the learned capacity -- the thing the claim is about -- is
  matched. A result where L1 matches T is therefore not explained by L1 having
  more (or less) trainable capacity; it is explained by the memory being
  unnecessary.
* Corpus split, steps, batch, ctx, lr, decay, rank, seed schedule and the
  3000-token val window are identical across arms, in the same protocol as the
  published run (train = first 50%, select = 60-80%, val = last 10%).
* Identity: `up` is zero, so at step 0 every arm is bit-identical to the
  teacher. Asserted on held-out perplexity AND on raw logits (max abs diff must
  be exactly 0.0) -- a dtype promotion is how this silently broke before
  (`docs/DTYPE_BUG.md`), and it produced a plausible-looking 0.23 ppl "effect".

SEEDS AND THE INTERVAL
----------------------
Every seed runs ALL arms with (a) an identical pre-drawn batch sequence and
(b) an identical `down` initialisation, so arms are paired within a seed and
the seed-to-seed cultural variation cancels in the paired difference. The
T-minus-L1 delta is reported per seed, with a paired bootstrap 95% interval
over seeds, plus each arm's own across-seed sd (the "seed noise" the verdict is
compared against). n is small by construction: the interval is indicative and
is labelled as such rather than presented as an exact test.

WHAT EACH OUTCOME MEANS
-----------------------
* T beats L1 beyond seed noise  -> the time-dependent memory carries something a
  static rank-64 branch does not, and the headline survives its strongest
  control.
* T does NOT beat L1            -> the headline effect is NOT attributable to the
  memory mechanism; "any trained low-rank branch helps" explains it, and
  `docs/LORA_CONTROL.md` says so in plain words. That is a clean negative and it
  is the most valuable thing this file can produce.
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
    attention_module, transplant, perplexity, n_params,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("LORA_CTL_OUT",
                      os.path.join(HERE, "results", "hybrid_lora_control.json"))
TEACHER = os.environ.get("HYBRID_TEACHER", "unsloth/Llama-3.2-1B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")
LAYER = int(os.environ.get("LORA_CTL_LAYER", "8"))

ARMS = tuple(os.environ.get("LORA_CTL_ARMS", "T,R,L1,L2").split(","))
SEEDS = [int(x) for x in os.environ.get("LORA_CTL_SEEDS", "0,1,2").split(",")]
STEPS = int(os.environ.get("LORA_CTL_STEPS", "400"))
BS = int(os.environ.get("LORA_CTL_BS", "2"))
CTX = int(os.environ.get("LORA_CTL_CTX", "128"))
LRS = [float(x) for x in os.environ.get(
    "LORA_CTL_LRS", os.environ.get("LORA_CTL_LR", "1e-4")).split(",")]
DECAY = float(os.environ.get("LORA_CTL_DECAY", "0.7"))
RANK = int(os.environ.get("LORA_CTL_RANK", "64"))
VAL_TOKENS = int(os.environ.get("LORA_CTL_VAL_TOKENS", "3000"))
FRESH = bool(int(os.environ.get("LORA_CTL_FRESH", "0")))
LOGIT_PROBE = 128

ARM_DESC = {
    "T": ("rank-64 zero-init projection on the gated-memory branch, W_v/W_o "
          "transplanted from this layer's attention (repro of the headline)"),
    "R": ("identical module, W_v/W_o random at the same per-matrix weight scale"),
    "L1": ("rank-64 zero-init low-rank branch on the attention output -- no "
           "memory, no recurrent state"),
    "L2": ("rank-64 zero-init low-rank branch on the layer input ln1(x) -- no "
           "memory, no recurrent state"),
}
MEMORY_ARMS = ("T", "R")


class InjectionBranch(nn.Module):
    """attention(x) + up(down(SIGNAL)), with up zero so the start is the teacher.

    SIGNAL is the only thing that differs between arms:

      source="mem"      the gated-memory trace of the sublayer input (T, R)
      source="attn_out" the attention output itself (L1)
      source="layer_in" the sublayer input ln1(x) (L2)

    `input_scale` is a fixed float, not a parameter, and it normalises the
    branch's input rms to the attention output rms measured on a real probe, so
    all arms present the projection with the same signal magnitude. Only ONE of
    down/up may be zero-initialised: with up=0 the branch is exactly zero at init
    (identity, asserted) but down's gradient is also zero, so down must start
    non-zero for anything to train at all. That is the standard LoRA
    factorisation and it is why `down` is drawn at 1/sqrt(d).
    """

    def __init__(self, inner, d: int, rank: int, source: str,
                 init_seed: int, mem=None, input_scale: float = 1.0):
        super().__init__()
        if source not in ("mem", "attn_out", "layer_in"):
            raise ValueError(f"unknown branch source {source!r}")
        if source == "mem" and mem is None:
            raise ValueError("source='mem' needs a memory module")
        self.inner = inner
        self.d = int(d)
        self.rank = int(rank)
        self.source = source
        self.mem = mem
        self.input_scale = float(input_scale)
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        # seeded so the initialisation is identical across arms at a given seed
        rng = np.random.default_rng(init_seed)
        self.down.weight = mx.array(
            (rng.standard_normal((rank, d)) / math.sqrt(d)).astype(np.float32))
        self.up.weight = mx.zeros((d, rank), dtype=mx.float32)

    def branch_input(self, base, x):
        if self.source == "mem":
            return self.mem(x)
        if self.source == "attn_out":
            return base
        return x

    def forward_branch(self, base, x):
        inp = self.branch_input(base, x) * self.input_scale
        out = self.up(self.down(inp))
        # cast to the residual's dtype or the sum silently promotes the whole
        # stream to float32 -- a mathematical no-op that moves perplexity by
        # 0.23 on its own (docs/DTYPE_BUG.md)
        if out.dtype != base.dtype:
            out = out.astype(base.dtype)
        return out

    def __call__(self, x, *args, **kwargs):
        base = self.inner(x, *args, **kwargs)
        return base + self.forward_branch(base, x)


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


def grad_norm_at_init(model, module, batch):
    """Gradient norms on `down` and `up` at step 0, before any update.

    A FAIRNESS DIAGNOSTIC, and it is needed. If one arm started with a much
    smaller gradient than the others, a worse final loss would be explained by
    the arm being effectively undertrained rather than by its mechanism, and the
    comparison would be unfair in whichever direction happens to be
    better-conditioned. Recording |g| at init makes that checkable instead of
    assumed. The arms are expected to differ somewhat -- the memory branch feeds
    a different signal -- but a collapse to ~0 would be a red flag.
    """
def grad_norm_at_init(model, module, batch):
    model.freeze()
    module.down.unfreeze()
    module.up.unfreeze()
    mx.eval(model.parameters())

    def loss_fn(mod, x, y):
        lo = model(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(module, loss_fn)
    x, y = mx.array(batch[0]), mx.array(batch[1])
    _, grads = lg(module, x, y)
    flat = dict(nn.utils.tree_flatten(grads))
    gd = np.asarray(flat["down.weight"].astype(mx.float32), dtype=np.float64)
    gu = np.asarray(flat["up.weight"].astype(mx.float32), dtype=np.float64)
    return dict(grad_norm_down_init=float(np.linalg.norm(gd)),
                grad_norm_up_init=float(np.linalg.norm(gu)))


def train_branch(model, module, batches, *, lr, log=print, eval_every=100,
                 select_ids=None):
    """Train ONLY the rank-64 projection; everything else stays frozen.

    `module.unfreeze()` would unfreeze RECURSIVELY and re-enable gradients on the
    wrapped pretrained attention (this happened: ppl went to 1889 and the model
    was corrupted for later arms). So unfreeze the two matrices by name and
    assert the trainable set is exactly those two.
    """
    model.freeze()
    module.down.unfreeze()
    module.up.unfreeze()
    trainable = sorted(k for k, _ in nn.utils.tree_flatten(
        module.trainable_parameters()))
    assert trainable == ["down.weight", "up.weight"], (
        f"expected only the rank-{module.rank} projection to be trainable, "
        f"got {trainable}")
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.0)

    def loss_fn(mod, x, y):
        lo = model(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(module, loss_fn)
    t0 = time.time()
    curve = []
    for step, (x_np, y_np) in enumerate(batches, start=1):
        x, y = mx.array(x_np), mx.array(y_np)
        loss, grads = lg(module, x, y)
        opt.update(module, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step % eval_every == 0 or step == len(batches):
            row = dict(step=step, loss=float(loss))
            if select_ids is not None:
                row["select_ppl"] = perplexity(model, select_ids)["ppl"]
            curve.append(row)
            log(f"      step {step:>4}  loss {float(loss):7.4f}"
                + (f"  select_ppl {row['select_ppl']:.4f}"
                   if "select_ppl" in row else ""))
    return curve, time.time() - t0


def draw_batches(train_ids, seed, steps, bs, ctx):
    """Pre-draw the batch sequence so every arm at this seed sees it identically.

    Drawing inside the training loop would let two arms diverge if any code path
    consumed a different number of random numbers -- and a batch-order
    difference is exactly the kind of confound that would make a small ppl
    margin uninterpretable.
    """
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_ids, dtype=np.int32)
    out = []
    for _ in range(steps):
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = np.stack([tr[i:i + ctx] for i in ix]).astype(np.int32)
        y = np.stack([tr[i + 1:i + 1 + ctx] for i in ix]).astype(np.int32)
        out.append((x, y))
    return out


def perplexity_hp(model, ids_arr, window=512):
    """Perplexity with float32 logits AND float32 accumulation of the loss sum.

    WHY THIS EXISTS, and it is not fussiness. The repo's `perplexity()` computes
    `loss` in the model's bfloat16 dtype and calls `loss.sum()` on it, so the
    per-window sum is accumulated in bf16 -- 8 mantissa bits. Measured on this
    exact val window and teacher: the six window sums come out as exact
    multiples of 8 nats (1560, 1328, 1424, 1560, 1704, 1464) against true
    float32 sums of 1558.1, 1334.1, 1428.6, 1562.3, 1705.1, 1462.8 -- i.e. up
    to 6.1 nats of error on a 1500-nat sum, ~0.4% per window. The reported ppl
    therefore sits on a grid of ~0.009 ppl and the ABSOLUTE value is biased
    -0.0754 ppl against the float32 computation (teacher 20.3756 vs 20.4510).

    Consequence, stated precisely because it decides what this file can claim:
    DELTAS VS THE TEACHER inherit ~0.1 ppl of metric-dependent bias (measured:
    the four arms' repo-metric deltas differ from their float32 deltas by
    0.098-0.125 ppl), while the PAIRED T-vs-L1 delta is stable to 0.016 ppl
    across the two metrics because both arms carry almost the same loss
    sequence and the quantisation largely cancels. So the verdict below is
    stated on the paired comparison, which is the metric-robust quantity, and
    both metrics are reported for every arm so a reader can see the spread.

    This is a measurement defect in the shared helper, not in the arms; it is
    recorded in docs/LORA_CONTROL.md rather than silently patched here, because
    it also affects the published 20.3756/19.8922 numbers.
    """
    ids_arr = np.asarray(ids_arr, dtype=np.int32)
    T = len(ids_arr) - 1
    tot, n = 0.0, 0
    for i in range(0, T, window):
        seg = ids_arr[i:i + window + 1]
        if len(seg) < 2:
            break
        x = mx.array(seg[:-1].astype(np.int32)[None, :])
        y = mx.array(seg[1:].astype(np.int32)[None, :])
        logits = model(x).astype(mx.float32)
        mx.eval(logits)
        logp = nn.log_softmax(logits, axis=-1)
        loss = -mx.take_along_axis(logp, y[..., None], axis=-1).squeeze(-1)
        loss = loss.astype(mx.float32)
        mx.eval(loss)
        tot += float(loss.sum())
        n += int(loss.size)
    nll = tot / n
    return dict(nll=nll, ppl=math.exp(nll), n_tokens=n)


def build_arm(arm, model, layer, attr, original, probe_h, d, seed):
    """Install one arm's branch at layer `LAYER`; return (module, record)."""
    init_seed = 1_000_000 + seed
    rec = dict(arm=arm, description=ARM_DESC[arm], source=None, rank=RANK)
    setattr(layer, attr, original)
    mx.eval(model.parameters())

    if arm in MEMORY_ARMS:
        carrier, trec = transplant(model, LAYER, "transfer", original=original,
                                   probe_h=probe_h, decay=DECAY)
        mem = carrier.mem
        for k in ("transplanted", "value_expand_reps", "wv_norm", "wo_norm",
                  "gate_bias", "fitted_decay", "decay_source"):
            if k in trec:
                rec[k] = trec[k]
        if arm == "R":
            rng = np.random.default_rng(2_000_000 + seed)
            sv = float(np.asarray(mem.v.weight).std())
            so = float(np.asarray(mem.o.weight).std())
            mem.v.weight = mx.array(
                rng.normal(0.0, sv, mem.v.weight.shape).astype(np.float32))
            mem.o.weight = mx.array(
                rng.normal(0.0, so, mem.o.weight.shape).astype(np.float32))
            rec["random_w_std_v"] = sv
            rec["random_w_std_o"] = so
            rec["control"] = ("random W_v/W_o, each drawn at its own "
                              "transplanted counterpart's std")
            rec["note_previous_impl"] = (
                "hybrid_inject.py drew BOTH matrices at std(W_v); the two stds "
                "differ by design here and per-matrix matching is the tighter "
                "control on weight scale")
        setattr(layer, attr, original)
        base = original(probe_h, None, None)
        mx.eval(base)
        r_ref = rms(base)
        raw = mem(probe_h)
        mx.eval(raw)
        r_sig = rms(raw)
        scale = (r_ref / r_sig) if r_sig > 0 else 1.0
        rec["rms_branch_input_before_scale"] = r_sig
        rec["rms_target_attention_out"] = r_ref
        module = InjectionBranch(original, d, RANK, "mem", init_seed, mem=mem,
                                 input_scale=scale)
        rec["source"] = "mem(x) (gated trace of ln1(x))"
        rec["time_dependent"] = True
    else:
        source = "attn_out" if arm == "L1" else "layer_in"
        raw = original(probe_h, None, None) if source == "attn_out" else probe_h
        mx.eval(raw)
        r_sig = rms(raw)
        r_ref = rms(original(probe_h, None, None))
        mx.eval(original(probe_h, None, None))
        scale = (r_ref / r_sig) if r_sig > 0 else 1.0
        module = InjectionBranch(original, d, RANK, source, init_seed,
                                 input_scale=scale)
        rec["source"] = ("attn(ln1(x)) -- sublayer output" if source == "attn_out"
                         else "ln1(x) -- sublayer input")
        rec["time_dependent"] = False
        rec["rms_branch_input_before_scale"] = r_sig
        rec["rms_target_attention_out"] = r_ref
        rec["control"] = "no memory, no recurrent state"

    rec["input_scale"] = float(scale)

    # VERIFY the scale-matching claim instead of asserting it in prose: the
    # branch input must arrive at the projection with the same rms as the
    # attention output it sits beside. If this drifts, the arms differ in
    # conditioning and Adam's absolute step size compares mechanisms unfairly.
    if rec["source"].startswith("mem"):
        sig = mem(probe_h)
    elif rec["source"].startswith("attn"):
        sig = original(probe_h, None, None)
    else:
        sig = probe_h
    mx.eval(sig)
    rec["rms_branch_input_after_scale"] = rms(sig) * float(scale)
    rec["input_scale_max_rel_dev"] = abs(
        rec["rms_branch_input_after_scale"] - r_ref) / r_ref
    assert rec["input_scale_max_rel_dev"] < 0.01, (
        f"arm {arm}: branch input rms {rec['rms_branch_input_after_scale']:.6f} "
        f"deviates {rec['input_scale_max_rel_dev']:.2%} from the attention "
        f"output rms {r_ref:.6f}; the arms would not be scale-matched")

    # COUNTING MUST HAPPEN AFTER FREEZING. `trainable_parameters()` reflects the
    # freeze flags, not the module topology, so counting before `model.freeze()`
    # reports the memory carrier's frozen W_v/W_o/gate as trainable and the
    # parameter-match assertion fails on a module that is in fact matched
    # (measured: 23,332,864 "trainable" instead of 262,144). Freeze first, then
    # unfreeze exactly the two projection matrices, then count what will really
    # receive a gradient.
    setattr(layer, attr, module)
    model.freeze()
    module.down.unfreeze()
    module.up.unfreeze()
    mx.eval(model.parameters())

    trainable = sorted(k for k, _ in nn.utils.tree_flatten(
        module.trainable_parameters()))
    assert trainable == ["down.weight", "up.weight"], (
        f"arm {arm}: expected only down/up to be trainable after freeze, got "
        f"{trainable}. A recursively unfrozen module is how a previous run "
        f"trained the pretrained attention and corrupted the model.")
    rec["trainable_params"] = sum(
        int(np.prod(v.shape))
        for _, v in nn.utils.tree_flatten(module.trainable_parameters()))
    rec["trainable_names"] = trainable
    # `n_params(module)` includes the wrapped pretrained attention, which every
    # arm carries and none of them changes. Counting it as "added" would report
    # 23.3M params added by an arm that adds 262k trainable + (T/R only) the
    # frozen carrier. So count only what is genuinely on top of the teacher.
    rec["frozen_params_added"] = (n_params(module) - rec["trainable_params"]
                                 - n_params(original))
    rec["total_params_this_arm"] = n_params(model) + rec["frozen_params_added"] \
        if arm in MEMORY_ARMS else n_params(model)
    expected = 2 * RANK * d
    assert rec["trainable_params"] == expected, (
        f"arm {arm}: {rec['trainable_params']} trainable != {expected} "
        f"(rank {RANK} at d={d}); the arms must be parameter-matched")
    return module, rec


def main():
    from mlx_lm import load

    t_start = time.time()
    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    train_ids = ids[:int(0.5 * n)]
    select = ids[int(0.6 * n):int(0.6 * n) + VAL_TOKENS]
    val = ids[int(0.9 * n):int(0.9 * n) + VAL_TOKENS]
    assert int(0.5 * n) <= int(0.6 * n) < int(0.9 * n), "split overlap"
    layer, attr, original = attention_module(model, LAYER)

    cap = {}

    class Capture:
        def __call__(self, x, *a, **k):
            cap["h"] = x
            return original(x, *a, **k)

    setattr(layer, attr, Capture())
    probe_ids = mx.array(train_ids[:1024][None, :].astype(np.int32))
    teacher_logits = model(probe_ids[..., :LOGIT_PROBE])
    mx.eval(teacher_logits)
    probe_h = cap["h"][..., :1024, :] if cap["h"].ndim == 3 else cap["h"]
    mx.eval(probe_h)
    setattr(layer, attr, original)
    mx.eval(model.parameters())
    d = int(probe_h.shape[-1])

    base_val = perplexity(model, val)["ppl"]
    base_val_hp = perplexity_hp(model, val)["ppl"]
    base_sel = perplexity(model, select)["ppl"]
    print(f"{n_params(model):,} params | d={d} | layer {LAYER} | "
          f"train {len(train_ids):,} | select {len(select):,} | "
          f"val {len(val):,}\n  teacher val {base_val:.4f}  "
          f"select {base_sel:.4f}\n", flush=True)

    cfg = dict(teacher=TEACHER, layer=LAYER, n_layers=len(model.model.layers),
               d=d, rank=RANK, steps=STEPS, bs=BS, ctx=CTX, lr=LRS[0],
               decay=DECAY, seeds=SEEDS, arms=list(ARMS), lrs=LRS,
               config_lr0=LRS[0],
               val_tokens=len(val), select_tokens=len(select),
               teacher_val_ppl=base_val, teacher_select_ppl=base_sel,
               teacher_val_ppl_hp=base_val_hp,
               protocol=dict(
                   split="train=first 50%, select=60-80%, val=last 10%, disjoint",
                   trained="ONLY the rank-64 projection; model frozen; "
                           "trainable set asserted",
                   identity="up=0 at init -> bit-identical to the teacher; "
                            "asserted on val ppl and on max abs logit diff",
                   matching="trainable params matched exactly (2*rank*d); "
                            "branch input rms matched to attention output rms",
                   total_params="T/R carry a frozen ~12.6M-param carrier; "
                                "L1/L2 do not -- trainable capacity is matched, "
                                "total is not; stated in docs/LORA_CONTROL.md",
                   eval="fixed 3000-token held-out window, no sampling"),
               runs=[])

    if os.path.exists(OUT) and not FRESH:
        # cfg IS the top-level object written to OUT (there is no nested
        # "config"), so the compatibility check must read the top level or it
        # silently never matches and every rerun retrains everything.
        prev = json.load(open(OUT))
        same = (prev.get("rank") == RANK and prev.get("steps") == STEPS
                and prev.get("teacher") == TEACHER and prev.get("d") == d
                and prev.get("decay") == DECAY and prev.get("bs") == BS
                and prev.get("ctx") == CTX
                and prev.get("val_tokens") == len(val))
        if same:
            cfg["runs"] = prev.get("runs", [])
            print(f"  resuming: {len(cfg['runs'])} runs already in {OUT}\n",
                  flush=True)
        else:
            print(f"  {OUT} exists but the config differs "
                  f"(rank/steps/decay/bs/ctx/val/d/teacher); starting a fresh "
                  f"run rather than mixing incompatible arms\n", flush=True)

    done = {(r["arm"], r["seed"], r.get("lr", cfg["lr"]))
            for r in cfg["runs"]}

    def save():
        json.dump(cfg, open(OUT, "w"), indent=1)

    save()
    batches = {s: draw_batches(train_ids, s, STEPS, BS, CTX) for s in SEEDS}

    for lr in LRS:
      for seed in SEEDS:
        for arm in ARMS:
            if (arm, seed, lr) in done:
                print(f"  [skip] {arm} seed {seed} lr {lr} already measured",
                      flush=True)
                continue
            print(f"\n=== arm {arm} (seed {seed}, lr {lr}) ===", flush=True)
            module, rec = build_arm(arm, model, layer, attr, original, probe_h,
                                    d, seed)
            before_val = perplexity(model, val)
            before_val_hp = perplexity_hp(model, val)
            after_logits = model(probe_ids[..., :LOGIT_PROBE])
            mx.eval(after_logits)
            logit_diff = float(mx.max(mx.abs(
                after_logits.astype(mx.float32)
                - teacher_logits.astype(mx.float32))))
            rec["init_val_ppl"] = before_val["ppl"]
            rec["init_val_ppl_hp"] = before_val_hp["ppl"]
            rec["teacher_val_ppl_hp"] = base_val_hp
            rec["init_max_abs_logit_diff_vs_teacher"] = logit_diff
            if not (abs(before_val["ppl"] - base_val) < 1e-12 and logit_diff == 0.0):
                raise RuntimeError(
                    f"identity check FAILED for arm {arm} seed {seed}: the "
                    f"zero-initialised projection must reproduce the teacher "
                    f"exactly. val ppl {before_val['ppl']!r} vs teacher "
                    f"{base_val!r}, max abs logit diff {logit_diff}. Every "
                    f"downstream number from this arm would be uninterpretable, "
                    f"so this is fatal; the usual cause is a dtype promotion.")
            print(f"  at init: val {before_val['ppl']:.6f} == teacher "
                  f"{base_val:.6f}  max|dlogit| {logit_diff}  EXACT", flush=True)
            rec.update(grad_norm_at_init(model, module, batches[seed][0]))
            print(f"  |grad| at init: down {rec['grad_norm_down_init']:.6g}  "
                  f"up {rec['grad_norm_up_init']:.6g}", flush=True)

            curve, wall = train_branch(model, module, batches[seed], lr=lr,
                                       select_ids=select,
                                       eval_every=max(STEPS // 4, 1))
            after_val = perplexity(model, val)
            after_val_hp = perplexity_hp(model, val)
            after_sel = perplexity(model, select)

            def branch_out_stats():
                """rms of what the branch actually adds, vs the attention beside it.

                The probe must go through the MODEL (the module is handed a
                hidden state, not token ids -- calling `module.inner(ids)` is a
                shape error, and silently passing the wrong tensor here would
                make the reported scales meaningless).
                """
                got = {}

                class Grab:
                    def __call__(self, h, *a, **k):
                        got["h"] = h
                        return module(h, *a, **k)

                setattr(layer, attr, Grab())
                model(mx.array(batches[seed][0][0]))
                x = got["h"]
                mx.eval(x)
                setattr(layer, attr, original)
                base = module.inner(x)
                b = module.forward_branch(base, x)
                mx.eval(b, base)
                return dict(branch_out_rms=rms(b), attn_out_rms=rms(base))

            rec.update(seed=seed, lr=lr, steps=STEPS,
                       before_val_ppl=before_val["ppl"],
                       after_val_ppl=after_val["ppl"],
                       after_select_ppl=after_sel["ppl"],
                       before_select_ppl=base_sel,
                       delta_vs_teacher=after_val["ppl"] - base_val,
                       rel_delta_pct=100.0 * (after_val["ppl"] - base_val) / base_val,
                       beats_teacher=bool(after_val["ppl"] < base_val),
                       after_val_ppl_hp=after_val_hp["ppl"],
                       delta_vs_teacher_hp=after_val_hp["ppl"] - base_val_hp,
                       beats_teacher_hp=bool(after_val_hp["ppl"] < base_val_hp),
                       wall_s=wall, curve=curve)
            rec.update(branch_out_stats())
            setattr(layer, attr, original)
            mx.eval(model.parameters())
            cfg["runs"].append(rec)
            done.add((arm, seed, lr))
            save()
            print(f"  AFTER: val {after_val['ppl']:.4f} "
                  f"({rec['delta_vs_teacher']:+.4f} vs teacher)  "
                  f"branch rms {rec['branch_out_rms']:.4f}  "
                  f"{'*** BEATS TEACHER ***' if rec['beats_teacher'] else ''}\n",
                  flush=True)
            del module

    analyse(cfg)
    print(f"\ntotal wall {time.time() - t_start:.0f}s -> {OUT}")


def analyse(cfg):
    """Per-lr verdicts. The two lr banks are NOT pooled.

    Pooling would be wrong twice over: (a) the arms' gains differ by a factor of
    ~1.5 between lr=1e-5 and lr=1e-4, so a pooled mean describes no operating
    point that was actually run; (b) the whole question is whether the ORDERING
    of T and L1 is stable, and an lr-averaged ordering can be produced by a
    crossover that neither operating point exhibits.
    """
    runs = cfg["runs"]
    seeds_all = sorted({r["seed"] for r in runs})
    base = cfg["teacher_val_ppl"]
    base_hp = cfg["teacher_val_ppl_hp"]
    lrs = sorted({r.get("lr") for r in runs})

    def bootstrap_ci(deltas, n_boot=4000, alpha=0.05, seed=0):
        a = np.asarray(deltas, dtype=np.float64)
        if a.size < 2:
            return None
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, a.size, size=(n_boot, a.size))
        means = a[idx].mean(axis=1)
        lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return dict(mean=float(a.mean()), lo=float(lo), hi=float(hi), n=a.size,
                    sd=float(a.std(ddof=1)), all_same_sign=bool(
                        np.all(a > 0) or np.all(a < 0)),
                    direction="T better" if a.mean() > 0 else "T worse",
                    method="percentile bootstrap of the paired per-seed mean "
                           f"({n_boot} resamples); n is the seed count")

    def direction_verdict(ci, a, b):
        """Three-way, because "{a} does not win" splits into two very different
        findings: a tie (the mechanism adds nothing but costs nothing) and a
        LOSS (the mechanism is actively worse than the arm without it).

        `a`/`b` are substituted into the label rather than hardcoded -- an
        earlier version returned "T beats L1 ..." for the T-vs-R comparison and
        printed a verdict about arms that were not being compared.
        """
        if ci is None:
            return f"underdetermined (n<2) for {a} vs {b}"
        if ci["lo"] > 0:
            return f"{a} beats {b} beyond seed noise"
        if ci["hi"] < 0:
            return f"{a} LOSES to {b} beyond seed noise"
        return f"{a} and {b} indistinguishable at this n"

    out_by_lr = {}
    for lr in lrs:
        rr = [r for r in runs if r.get("lr") == lr]
        seeds = sorted({r["seed"] for r in rr})
        by = {}
        for r in rr:
            by.setdefault(r["arm"], {})[r["seed"]] = r

        def deltas(arm, key="delta_vs_teacher"):
            sr = by.get(arm, {})
            return np.array([sr[x][key] for x in seeds if x in sr],
                            dtype=np.float64)

        arms = {}
        for arm in cfg["arms"]:
            dv = deltas(arm)
            dvh = deltas(arm, "delta_vs_teacher_hp")
            if dv.size == 0:
                continue
            arms[arm] = dict(
                n=int(dv.size),
                mean_after_val_ppl=float(np.mean(
                    [by[arm][x]["after_val_ppl"] for x in seeds if x in by[arm]])),
                mean_after_val_ppl_hp=float(np.mean(
                    [by[arm][x]["after_val_ppl_hp"]
                     for x in seeds if x in by[arm]])),
                mean_delta_vs_teacher=float(dv.mean()),
                mean_delta_vs_teacher_hp=float(dvh.mean()),
                per_seed_delta={int(x): float(by[arm][x]["delta_vs_teacher"])
                                for x in seeds if x in by[arm]},
                per_seed_delta_hp={int(x): float(by[arm][x]["delta_vs_teacher_hp"])
                                   for x in seeds if x in by[arm]},
                gain_ppl=float(-dv.mean()), gain_ppl_hp=float(-dvh.mean()),
                across_seed_sd_of_delta=(float(dv.std(ddof=1))
                                         if dv.size > 1 else None),
                beats_teacher_all_seeds=bool(np.all(dv < 0)),
                beats_teacher_every_seed_hp=bool(np.all(dvh < 0)),
                trainable_params=int(by[arm][seeds[0]]["trainable_params"]),
                frozen_params_added=int(by[arm][seeds[0]]["frozen_params_added"]),
            )

        entry = dict(lr=lr, n_seeds=len(seeds), arms=arms)

        # seed noise: the largest across-seed sd of any arm's own delta. This is
        # the yardstick the T-vs-L1 delta is compared against.
        sds = [arms[a]["across_seed_sd_of_delta"] for a in arms
               if arms[a]["across_seed_sd_of_delta"] is not None]
        entry["seed_noise_sd_max_arm"] = float(max(sds)) if sds else None

        for a, b, label in (("T", "L1", "T_vs_L1"), ("T", "R", "T_vs_R")):
            if a not in arms or b not in arms:
                continue
            pairs = [x for x in seeds if x in by[a] and x in by[b]]
            ps = {int(x): float(by[b][x]["delta_vs_teacher"]
                                - by[a][x]["delta_vs_teacher"]) for x in pairs}
            psh = {int(x): float(by[b][x]["delta_vs_teacher_hp"]
                                 - by[a][x]["delta_vs_teacher_hp"])
                   for x in pairs}
            ci = bootstrap_ci([ps[int(x)] for x in pairs])
            cih = bootstrap_ci([psh[int(x)] for x in pairs], seed=1)
            entry[label] = dict(
                definition=f"per-seed ({b} delta - {a} delta) vs teacher; "
                           f"positive means {a} improved val ppl more than {b}",
                per_seed=ps, paired_delta_ppl=float(np.mean(list(ps.values()))),
                paired_ci95=ci, per_seed_hp=psh,
                paired_delta_ppl_hp=float(np.mean(list(psh.values()))),
                paired_ci95_hp=cih,
                direction_repo=direction_verdict(ci, a, b),
                direction_hp=direction_verdict(cih, a, b),
                metrics_agree=(direction_verdict(ci, a, b)
                               == direction_verdict(cih, a, b))
                if (ci and cih) else None,
                metric_note="repo metric = bf16 window sums (the published "
                            "metric); _hp = float32 logits and float32 "
                            "accumulation. The verdict requires both to agree.",
            )

        if "T" in arms and "L1" in arms:
            tg, lg = arms["T"]["gain_ppl_hp"], arms["L1"]["gain_ppl_hp"]
            entry["memory_attribution"] = dict(
                T_gain_ppl=tg, L1_gain_ppl=lg,
                fraction_of_T_gain_from_a_memoryless_branch=(lg / tg if tg else None),
                T_gain_ppl_repo_metric=arms["T"]["gain_ppl"],
                L1_gain_ppl_repo_metric=arms["L1"]["gain_ppl"],
                note="a fraction >= 1 means a memoryless rank-64 branch already "
                     "accounts for all of T's gain and more")
            tv = entry["T_vs_L1"]
            entry["verdict"] = dict(
                T_mean_gain_ppl=tg, L1_mean_gain_ppl=lg,
                T_mean_gain_ppl_repo_metric=arms["T"]["gain_ppl"],
                L1_mean_gain_ppl_repo_metric=arms["L1"]["gain_ppl"],
                T_minus_L1_ppl=tv["paired_delta_ppl"],
                T_minus_L1_ppl_hp=tv["paired_delta_ppl_hp"],
                paired_ci95_hp=tv["paired_ci95_hp"],
                direction=tv["direction_hp"],
                T_beats_L1_beyond_seed_noise=(tv["direction_hp"]
                                              == "T beats L1 beyond seed noise"),
                T_loses_to_L1_beyond_seed_noise=(tv["direction_hp"]
                                                 == "T LOSES to L1 beyond seed noise"),
                attribution_to_memory=(tv["direction_hp"]
                                      == "T beats L1 beyond seed noise"),
                fraction_of_T_gain_from_memoryless_L1=(lg / tg if tg else None),
                seed_noise_sd_max_arm=entry["seed_noise_sd_max_arm"],
            )
        out_by_lr[f"{lr:g}"] = entry

    cfg["result_by_lr"] = out_by_lr
    json.dump(cfg, open(OUT, "w"), indent=1)

    for key, e in out_by_lr.items():
        print(f"\n=== LORA CONTROL (lr {key}) ===")
        print(f"  teacher val {base:.4f}  (float32 metric {base_hp:.4f})")
        for a in cfg["arms"]:
            if a in e["arms"]:
                r = e["arms"][a]
                sd = r["across_seed_sd_of_delta"]
                print(f"  {a:<3} after {r['mean_after_val_ppl']:>9.4f}  "
                      f"delta {r['mean_delta_vs_teacher']:>+8.4f} ppl  "
                      f"hp delta {r['mean_delta_vs_teacher_hp']:>+8.4f}  "
                      f"(n={r['n']}"
                      + (f", seed sd {sd:.4f}" if sd is not None else "") + ")")
        for lab in ("T_vs_L1", "T_vs_R"):
            if lab in e:
                v = e[lab]
                ci = v["paired_ci95_hp"]
                print(f"  paired {lab}: {v['paired_delta_ppl_hp']:+.4f} ppl "
                      f"(float32)"
                      + (f"  95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]" if ci else "")
                      + f"  per-seed { {k: round(x, 4) for k, x in v['per_seed_hp'].items()} }")
                print(f"     -> {v['direction_hp']}"
                      + ("" if v["metrics_agree"] else
                         f"   [METRICS DISAGREE: repo metric says "
                         f"'{v['direction_repo']}']"))
        if "verdict" in e:
            v = e["verdict"]
            print(f"\n  VERDICT (lr {key}): T gain {v['T_mean_gain_ppl']:.4f} ppl "
                  f"vs L1 {v['L1_mean_gain_ppl']:.4f} -> "
                  f"{100 * v['fraction_of_T_gain_from_memoryless_L1']:.1f}% of "
                  f"the win comes from a memoryless low-rank branch.")
            print(f"     {v['direction']}.")
            if not v["attribution_to_memory"]:
                print("     The headline effect is NOT attributable to the memory "
                      "mechanism at this operating point.")
        print(f"  wrote {OUT}")


def grad_check():
    """Gradient magnitudes at init for every arm and seed.

    Deliberately a SEPARATE, cheap pass (one forward+backward per arm, no
    training): it answers "was any arm unfairly starved of gradient?" without
    invalidating or re-running the 24 trained runs. It also measures the
    branch-input rms each arm presents to its projection, which is the other
    half of the conditioning question.
    """
    from mlx_lm import load

    print("=== GRADIENT / CONDITIONING CHECK AT INIT ===")
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    train_ids = ids[:int(0.5 * len(ids))]
    layer, attr, original = attention_module(model, LAYER)
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

    rows = []
    for seed in SEEDS:
        batches = draw_batches(train_ids, seed, 1, BS, CTX)
        for arm in ARMS:
            module, rec = build_arm(arm, model, layer, attr, original, probe_h,
                                    d, seed)
            g = grad_norm_at_init(model, module, batches[0])
            row = dict(arm=arm, seed=seed, **g,
                       rms_branch_input=rec["rms_branch_input_after_scale"],
                       rms_target_attention_out=rec["rms_target_attention_out"],
                       input_scale=rec["input_scale"],
                       trainable_params=rec["trainable_params"])
            rows.append(row)
            print(f"  seed {seed} {arm:<3} |g_down| {row['grad_norm_down_init']:.6g}"
                  f"  |g_up| {row['grad_norm_up_init']:.6g}"
                  f"  branch rms {row['rms_branch_input']:.5f}")
            setattr(layer, attr, original)
            mx.eval(model.parameters())

    summary = {}
    for arm in ARMS:
        gu = np.array([r["grad_norm_up_init"] for r in rows if r["arm"] == arm])
        gd = np.array([r["grad_norm_down_init"] for r in rows if r["arm"] == arm])
        rb = np.array([r["rms_branch_input"] for r in rows if r["arm"] == arm])
        summary[arm] = dict(n=int(gu.size), mean_grad_norm_up=float(gu.mean()),
                            mean_grad_norm_down=float(gd.mean()),
                            min_grad_norm_up=float(gu.min()),
                            mean_branch_input_rms=float(rb.mean()))
    # quantify the fairness question: ratio of the weakest arm's gradient to the
    # strongest, so "undertrained" can be ruled in or out from the record
    gu = {a: summary[a]["mean_grad_norm_up"] for a in summary}
    if gu:
        lo, hi = min(gu.values()), max(gu.values())
        summary["gradient_spread"] = dict(
            min_mean_grad_norm_up=lo, max_mean_grad_norm_up=hi,
            max_over_min_ratio=(hi / lo if lo else None),
            note="if one arm's init gradient were orders of magnitude smaller, "
                 "a worse final loss could be undertraining rather than "
                 "mechanism; a ratio near 1 removes that explanation")
    json.dump(dict(teacher=TEACHER, layer=LAYER, d=d, seeds=SEEDS, arms=list(ARMS),
                   rows=rows, summary=summary),
              open(os.path.join(HERE, "results",
                                "hybrid_lora_control_gradcheck.json"), "w"),
              indent=1)
    print("\n  gradient spread across arms: "
          f"{summary['gradient_spread']['max_over_min_ratio']:.3f}x "
          "(max |g_up| / min |g_up|)")


def selftest():
    """Checks that the arms are what the JSON says they are.

    THE FIRST VERSION OF THIS TEST WAS WRONG and it is worth recording why,
    because the wrong version looked reasonable. It asserted that a
    "time-dependent" arm's branch input changes when the input sequence is
    reversed. Reversal changes the input itself, so EVERY arm changes, including
    the position-wise one -- the test reported L2 as time-dependent and failed.
    Reversal cannot separate "recurrent state" from "just different data".

    What actually distinguishes the arms is what the branch input is a function
    of, and for T/R there is an exact algebraic signature: the gated trace
    satisfies h_t = g_t * h_{t-1} + v_t. So the recurrence is verified by
    independently recomputing the trace with an explicit Python-style sequential
    recurrence and requiring it to match the module's parallel scan. That check
    cannot be passed by accident by a static map.
    """
    from mlx_lm import load
    import hashlib

    print("=== SELFTEST: arm identity, structure, determinism ===")
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    train_ids = ids[:int(0.5 * len(ids))]
    layer, attr, original = attention_module(model, LAYER)
    cap = {}

    class Capture:
        def __call__(self, x, *a, **k):
            cap["h"] = x
            return original(x, *a, **k)

    setattr(layer, attr, Capture())
    probe_ids = mx.array(train_ids[:512][None, :].astype(np.int32))
    teacher_logits = model(probe_ids)
    mx.eval(teacher_logits)
    probe_h = cap["h"]
    mx.eval(probe_h)
    setattr(layer, attr, original)
    d = int(probe_h.shape[-1])
    fails = []

    def branch_inputs(module, h):
        """The branch input the projection will actually see, on hidden state h."""
        base = module.inner(h)
        mx.eval(base)
        return base, module.branch_input(base, h)

    # (1) identity at init for every arm
    # (2) the branch input is exactly the labelled signal
    # (3) position-wise vs prefix-dependent, tested by perturbing ONE early token
    # numpy cannot buffer bf16 (PEP 3118 "oat16"), so go through mx.float32
    # explicitly rather than np.array(hidden_state)
    np_h = np.array(probe_h.astype(mx.float32), dtype=np.float32)
    np_h[:, 0, :] += 3.0                     # perturb the FIRST position only
    h_pert = mx.array(np_h)
    mx.eval(h_pert)

    for arm in ARMS:
        module, rec = build_arm(arm, model, layer, attr, original, probe_h, d, 0)
        lg = model(probe_ids)
        mx.eval(lg)
        diff = float(mx.max(mx.abs(lg.astype(mx.float32)
                                   - teacher_logits.astype(mx.float32))))
        base, bi = branch_inputs(module, probe_h)
        _, bi_p = branch_inputs(module, h_pert)
        # how much does a perturbation at position 0 move the branch input at
        # positions >= 1? zero => position-wise; nonzero => depends on the prefix
        early = float(mx.max(mx.abs(
            bi.astype(mx.float32)[:, 1:, :] - bi_p.astype(mx.float32)[:, 1:, :])))
        rec["selftest_identity_max_abs_logit_diff"] = diff
        rec["selftest_prefix_dependence"] = early

        note = ""
        if arm == "L1":
            m = float(mx.max(mx.abs(bi.astype(mx.float32)
                                    - base.astype(mx.float32))))
            rec["selftest_matches_attention_output"] = m
            note = f"== attn out? {m}"
            if m != 0.0:
                fails.append(f"L1 branch input is not the attention output ({m})")
            if early <= 0.0:
                fails.append("L1 shows no prefix dependence, so it is not the "
                             "attention output")
        elif arm == "L2":
            m = float(mx.max(mx.abs(bi.astype(mx.float32)
                                    - probe_h.astype(mx.float32))))
            rec["selftest_matches_layer_input"] = m
            note = f"== layer input? {m}"
            if m != 0.0:
                fails.append(f"L2 branch input is not ln1(x) ({m})")
            if early != 0.0:
                fails.append(
                    f"L2 is supposed to be position-wise but a perturbation at "
                    f"position 0 moved later positions by {early}")
        else:
            # verify the gated recurrence h_t = g_t * h_{t-1} + v_t exactly, by
            # recomputing the trace sequentially and comparing to the scan
            gm = module.mem
            B, T, _ = probe_h.shape
            v = gm.v(probe_h).reshape(B, T, gm.banks, gm.d)
            g = mx.sigmoid(gm.gate(probe_h)).reshape(B, T, gm.banks, gm.d)
            mx.eval(v, g)
            v_np = np.array(v.astype(mx.float32))
            g_np = np.array(g.astype(mx.float32))
            h_seq = np.zeros((B, gm.banks, gm.d), dtype=np.float32)
            seq = []
            for t in range(T):
                h_seq = g_np[:, t] * h_seq + v_np[:, t]
                seq.append(h_seq.copy())
            seq = np.stack(seq, axis=1)                      # (B, T, banks, d)
            h_scan = np.array(mx.array(seq))
            from llm_efficiency import chunked_gated_scan
            scan = chunked_gated_scan(
                v.transpose(0, 2, 1, 3).reshape(B * gm.banks, T, gm.d),
                g.transpose(0, 2, 1, 3).reshape(B * gm.banks, T, gm.d),
                gm.chunk)
            mx.eval(scan)
            scan_np = np.array(scan.astype(mx.float32)).reshape(
                B, gm.banks, T, gm.d).transpose(0, 2, 1, 3)
            err = float(np.max(np.abs(scan_np - seq)))
            rec["selftest_recurrence_max_abs_error"] = err
            note = f"recurrence ||scan-seq||inf {err:.3e}"
            if err > 1e-4:
                fails.append(
                    f"{arm} does not satisfy h_t = g*h_(t-1) + v_t (max err "
                    f"{err}); the branch is not the gated recurrence")
            if arm == "T":
                # T's memory must be the attention transplant, not random
                if not rec.get("transplanted"):
                    fails.append("T is not flagged as transplanted")
        print(f"  {arm:<3} identity max|dlogit| {diff}  prefix-dep {early:.4g}  "
              f"{note}  trainable {rec['trainable_params']}  "
              f"frozen {rec['frozen_params_added']}")
        setattr(layer, attr, original)
        mx.eval(model.parameters())

    # (4) determinism: same seed twice -> identical trained parameters
    batches = draw_batches(train_ids, 0, 3, 2, 128)
    digs = []
    for rep in (0, 1):
        module, _ = build_arm("L1", model, layer, attr, original, probe_h, d, 0)
        train_branch(model, module, batches, lr=1e-4,
                     log=lambda *a, **k: None, eval_every=99)
        arr = np.ascontiguousarray(np.asarray(module.down.weight))
        digs.append(hashlib.sha256(arr.tobytes()).hexdigest())
        setattr(layer, attr, original)
        mx.eval(model.parameters())
    print(f"  determinism: down.weight sha256 {digs[0][:16]} vs {digs[1][:16]}  "
          f"{'MATCH' if digs[0] == digs[1] else 'MISMATCH'}")
    if digs[0] != digs[1]:
        fails.append("non-deterministic: two identical L1 runs gave different "
                     "down.weight after identical training")

    # (5) a missed freeze must be impossible
    module, _ = build_arm("T", model, layer, attr, original, probe_h, d, 0)
    names = sorted(k for k, _ in nn.utils.tree_flatten(
        module.trainable_parameters()))
    print(f"  trainable tensors in the T arm: {names}")
    if names != ["down.weight", "up.weight"]:
        fails.append(f"freeze broken: trainable set is {names}")
    setattr(layer, attr, original)

    json.dump(dict(teacher=TEACHER, layer=LAYER, d=d, arms=list(ARMS),
                   fails=fails, determinism_sha256=digs),
              open(os.path.join(HERE, "results",
                                "hybrid_lora_control_selftest.json"), "w"),
              indent=1)
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        raise SystemExit(f"selftest FAILED ({len(fails)} problem(s))")
    print("  all selftest checks passed")


if __name__ == "__main__":
    if os.environ.get("LORA_CTL_SELFTEST") == "1":
        selftest()
    elif os.environ.get("LORA_CTL_GRADS") == "1":
        grad_check()
    else:
        main()
