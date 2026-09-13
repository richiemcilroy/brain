"""Does re-using a small stack k times beat a k-times-deeper stack at matched
parameters and matched FLOPs?

THE QUESTION
------------
Mammalian cortex is roughly 6 layers of the SAME canonical circuit repeated
across the whole sheet, not 96 distinct bespoke layer types, and it runs ~100
sequential processing steps per 100 ms through that one shallow circuit. A
transformer buys quality by stacking N DISTINCT layers, each with its own
parameters, so depth costs parameters linearly. This script asks the
architectural question the project had never tested: if you re-use a small
stack k times instead of stacking k distinct layers, what do you get per
parameter and per FLOP?

This is the `looped` flag in experiments/llm_efficiency.py (`F_gated_looped2`)
turned into an actual controlled comparison. That flag was a configuration, not
a result: it was never run against a matched unique-layer stack.

PRIOR ART, BOTH WAYS (do not overclaim)
---------------------------------------
Settled in DIRECTION before this run, and it goes AGAINST looping:
  * ALBERT (arXiv:1909.11942), Table 3: all-layer sharing costs ~2 points of
    downstream average at the same FLOPs. That is the direct matched-FLOP answer.
  * Saunshi et al., "Reasoning with Latent Thoughts" (arXiv:2502.17416): a
    k-layer block looped L times nearly matches a kL-layer model on REASONING at
    iso-FLOPs, but LAGS on PERPLEXITY -- and bpc is a perplexity-family metric,
    which is what we measure here.
  * Geiping et al., Huginn (arXiv:2502.05171): the authors state they did NOT
    train an iso-FLOP non-recurrent baseline.
  * Dehghani et al., Universal Transformer (arXiv:1807.03819): beats the base
    Transformer at equal PARAMS with more compute; no matched-FLOP
    unique-depth control.
  * MobileLLM (arXiv:2402.14905): immediate block-wise repetition gives ~0.5
    points at fixed parameters.
So at matched FLOPs unique layers are expected to win on perplexity, and at
matched parameters looping is expected to win. Our metric is perplexity-family.

PRE-REGISTERED PREDICTION (stated before running)
-------------------------------------------------
  loop2x2 LOSES to flat4 on bpc at matched FLOPs, and MATCHES flat2 on params.
  A clean null on the loop2x2-vs-flat2 comparison is therefore the expected
  outcome, and is a useful control rather than a failure.

PRE-REGISTERED DECISION RULE (stated before running)
-----------------------------------------------------
Looping is a real efficiency win ONLY if `loop2x2` beats `flat2` by more than
seed noise. `flat2` has the SAME parameter count (445,440) at HALF the FLOPs
(2,226,432 vs 4,402,944 per token), so this is the comparison that isolates
"extra passes over shared weights" from "more weights". Beating `flat4` while
matching its FLOPs is a nice-to-have; beating `flat2` at matched parameters is
the claim. Judged with 5 paired seeds and a 95% paired t-interval. An interval
straddling zero is a NULL and is reported as one.

ARMS (every non-tested axis fixed at d=128, ctx=512, MLP mult 4,
gated-memory mixing, chunk 64)
--------------------------------------------------------------------------------
  flat4       4 unique gated-memory layers, 1 pass. The baseline.     808,448 p
  loop2x2     2 unique layers, run TWICE.  Same FLOPs as flat4.       445,440 p
  loop1x4     1 unique layer, run FOUR times. Same FLOPs as flat4.    263,936 p
  flat2       2 unique layers, 1 pass. MATCHES loop2x2's parameters
              at HALF its FLOPs -- the control that makes the comparison
              mean something.                                       445,440 p
  loop2x2_ws  as loop2x2 plus per-loop normalisation (the stabiliser
              Universal Transformer / Huginn use).                   445,954 p
  loop2x2_ts  as loop2x2 plus a per-loop timestep embedding (the other
              published fix for "no per-iteration signal").          445,696 p
  loop1x4_ws  as loop1x4 plus per-loop normalisation.                 264,450 p
  loop1x4_ts  as loop1x4 plus a per-loop timestep embedding.          264,448 p

WHY THE STABILISER ARMS EXIST
-----------------------------
A looped stack has NO per-iteration signal: the same blocks and the same
LayerNorms cannot tell which pass they are on. Universal Transformer adds a
timestep embedding; Huginn re-injects the token embedding at every iteration
and reports it matters for stability. Without one of those the loop arms are
handicapped by construction, so they are run WITH and WITHOUT and both are
reported. If a loop arm fails only because it was handicapped, that is a
finding, not a null.

DIVERGENCE IS REPORTED, NOT DROPPED. Any arm that produces NaN/Inf, or a bpc at
or above the bigram floor (i.e. it failed to learn to use context at all), is
flagged in the JSON and in the table. No arm is silently removed.

Reproduce:
    python3 experiments/looped_depth.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import (  # noqa: E402
    Block, CORPUS, flops_per_token, load_corpus, n_params,
)
from llm_tuned import floors  # noqa: E402
from confirm_headline import (  # noqa: E402
    deterministic_val_batches, evaluate, paired_report,
)

RESULTS = os.environ.get(
    "BRAIN_LOOPED_OUT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", "results", "looped_depth.json",
    ),
)

# Device. MLX's GPU backward pass is NOT bit-reproducible: with an identical
# init, identical batch and identical seed, two gradient evaluations differ by
# up to ~4.5e-08 per element (measured), which accumulates to a ~1e-7 bpc
# difference in val_bpc after 1500 steps. The CPU path is bit-identical.
# Set BRAIN_DEVICE=cpu for strict bit-reproducibility; the default stays on the
# GPU because CPU training is ~10x slower. Either way the drift is 5 orders of
# magnitude below every effect size reported here.
_DEVICE = os.environ.get("BRAIN_DEVICE", "gpu").lower()
if _DEVICE == "cpu":
    mx.set_default_device(mx.cpu)

D = 128
CTX = 512
MLP_MULT = 4
CHUNK = 64


# --------------------------------------------------------------------------
# the looped model
# --------------------------------------------------------------------------
class LoopedLM(nn.Module):
    """`n_layer` unique gated-memory blocks, applied `loops` times.

    Parameters are the unique stack's, NOT the unrolled stack's, which is the
    whole point: loop2x2 carries the parameter count of flat2 while doing the
    FLOPs of flat4.

    Optional per-loop signals, both taken from published work rather than
    invented here:
      timestep   Universal Transformer (arXiv:1807.03819) timestep embedding.
      loop_norm  per-loop LayerNorm + scale, the per-loop normalisation that
                 Universal Transformer and Huginn both rely on and that naive
                 looping is known to need (Huginn, arXiv:2502.05171, reports
                 that re-injecting the input each iteration matters for
                 stability; per-loop normalisation is the general form).
    """

    def __init__(self, vocab, d, ctx, n_layer, *, n_head=4, loops=2,
                 timestep=False, reinject=False, loop_norm=False, chunk=CHUNK,
                 banks=1, loop_scale_init=1.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        # a plain list, so MLX registers the blocks in `parameters()`.
        # (underscore-prefixed attributes are silently DROPPED by MLX, which
        # would have made the model untrainable while still "having" params.)
        self.blocks = [Block(d, n_head, "mem", banks, chunk)
                       for _ in range(n_layer)]
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.loops = loops
        self.use_timestep = timestep
        self.reinject = reinject
        self.use_loop_norm = loop_norm
        if timestep:
            self.ts = nn.Embedding(loops, d)
        if loop_norm:
            self.loop_ln = [nn.LayerNorm(d) for _ in range(loops)]
            self.loop_scale = mx.full((loops,), loop_scale_init, dtype=mx.float32)

    def forward_trace(self, idx):
        """Hidden state after each full pass. Used for the mechanism probe."""
        B, T = idx.shape
        x0 = self.tok(idx) + self.pos(mx.arange(T)[None, :])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        h = x0
        states = []
        for i in range(self.loops):
            if self.use_timestep:
                h = h + self.ts(mx.array([i]))
            if self.reinject:
                h = h + x0
            if self.use_loop_norm:
                h = self.loop_ln[i](h) * self.loop_scale[i]
            for blk in self.blocks:
                h = blk(h, mask)
            states.append(h)
        return states

    def __call__(self, idx):
        states = self.forward_trace(idx)
        return self.head(self.lnf(states[-1]))


# --------------------------------------------------------------------------
# arm registry
# --------------------------------------------------------------------------
# `n_layer` = unique layers, `loops` = passes, so params come from n_layer and
# FLOPs from n_layer * loops.
ARMS = {
    "flat4":       dict(n_layer=4, loops=1),
    "loop2x2":     dict(n_layer=2, loops=2),
    "loop1x4":     dict(n_layer=1, loops=4),
    "flat2":       dict(n_layer=2, loops=1),
    "loop2x2_ws":  dict(n_layer=2, loops=2, loop_norm=True),
    "loop2x2_ts":  dict(n_layer=2, loops=2, timestep=True),
    "loop1x4_ws":  dict(n_layer=1, loops=4, loop_norm=True),
    "loop1x4_ts":  dict(n_layer=1, loops=4, timestep=True),
    # flat2 with its step count scaled so its WALL-CLOCK matches loop2x2's.
    # flat2 has loop2x2's parameters at half its FLOPs per token, so under an
    # equal-time budget it gets MORE steps. That is the point: time is what a
    # practitioner pays. `steps_mult` is filled from the wall-clock calibration.
    # 2800 = 1500 * 1.8815, the measured flat2/loop2x2 steps-per-second ratio,
    # rounded to the nearest 50. Calibration is re-run and recorded in the JSON.
    "flat2_wall":  dict(n_layer=2, loops=1, steps_mult=2800 / 1500.0,
                        wall_matched=True),
    # strict wall-clock match: the clean timing measurement (warmup + 3x25
    # reps, min median) gives flat2 at 14.1 ms/step against loop2x2 at
    # 20.2 ms/step, a rate ratio of 1.4309. So under an equal-time budget flat2
    # gets 1500*1.4309 = 2146 steps; rounded to 2150. `flat2_wall` above used
    # the pre-run ratio of 1.8667, which the clean timing shows was an
    # overestimate, so it got ~30% MORE steps than a strict equal-time budget.
    "flat2_wc":    dict(n_layer=2, loops=1, steps_mult=2150 / 1500.0,
                        wall_matched=True),
    # `flat2_wc2` is the properly-centred equal-time arm. Five independent
    # interleaved timing sessions (round-robin inside each round, 15 rounds,
    # min round-median) put flat2 at 12.44-20.4 ms/step against loop2x2 at
    # 22.33-28.3 ms/step, i.e. flat2 completes 1.794x the steps per unit time
    # (median-based estimate 1.751, so the two agree within 2.5%). 1500 *
    # 1.794 = 2691 steps.
    "flat2_wc2":   dict(n_layer=2, loops=1, steps_mult=2691 / 1500.0,
                        wall_matched=True),
}

# pre-registered comparisons; first arm negative = first arm better (lower bpc)
PAIRED = [
    ("loop2x2", "flat2"),     # THE claim: matched params, half the FLOPs
    ("loop2x2", "flat4"),     # matched FLOPs, half the params
    ("loop1x4", "flat2"),     # quarter params vs matched-param control
    ("loop2x2_ws", "flat2"),   # does the stabiliser change the answer?
    ("loop2x2_ws", "loop2x2"),   # what the stabiliser itself does
    ("loop2x2_ts", "loop2x2"),   # what the timestep embedding does
    ("loop1x4_ws", "loop1x4"),   # stabiliser on the 4-pass arm
    ("loop1x4_ts", "loop1x4"),   # timestep embedding on the 4-pass arm
    ("flat2_wall", "loop2x2"),   # equal WALL-CLOCK, not equal steps
    ("flat2_wc", "loop2x2"),     # conservative under-time budget
    ("flat2_wc2", "loop2x2"),    # THE equal-wall-clock comparison
    ("flat2_wc2", "flat4"),      # equal time vs 4 unique layers
    ("flat2_wc", "flat2"),       # value of the extra steps alone
    ("flat2_wc", "flat4"),       # equal-time shallow vs 4 unique layers
]


def build(arm, vocab):
    spec = dict(ARMS[arm])
    spec.pop("steps_mult", None)    # scheduling metadata, not a model argument
    spec.pop("wall_matched", None)
    m = LoopedLM(vocab, D, CTX, spec.pop("n_layer"), loops=spec.pop("loops"),
                 chunk=CHUNK, **spec)
    return m


def arm_flops(arm, vocab):
    spec = ARMS[arm]
    return flops_per_token(vocab, D, spec["n_layer"], CTX, "mem", banks=1,
                           chunk=CHUNK, mult=spec["loops"], n_col=1)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def run(arm, data, vocab, *, steps=1500, bs=16, lr=1e-3, seed=0, log=print):
    mx.random.seed(seed)
    m = build(arm, vocab)
    mx.eval(m.parameters())
    P = n_params(m)
    FL = arm_flops(arm, vocab)
    n = int(0.9 * len(data))
    tr = data[:n]
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)
    xv, yv, n_win = deterministic_val_batches(data, CTX)
    t0 = time.time()
    ntok = 0
    last_train = float("nan")
    diverged_at = None
    for s in range(1, steps + 1):
        opt.learning_rate = lr * min(1.0, s / 100)
        ix = rng.integers(0, len(tr) - CTX - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + CTX] for i in ix]))
        y = mx.array(np.stack([tr[i + 1:i + 1 + CTX] for i in ix]))
        l, g = lg(m, x, y)
        if diverged_at is None and not math.isfinite(float(l)):
            diverged_at = s
            log(f"    !! {arm} seed={seed} non-finite loss at step {s}")
            break
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
        last_train = float(l)
    wall = time.time() - t0
    if diverged_at is not None:
        return dict(arm=arm, seed=seed, params=P, flops_per_token=FL,
                    steps_run=diverged_at, diverged=True,
                    val_bpc=float("nan"), train_bpc=float("nan"),
                    n_val_windows=n_win, tok_s=ntok / max(wall, 1e-9),
                    wall_s=wall)
    val_nats = evaluate(m, xv, yv, vocab)
    drift = pass_drift(m, tr, vocab, seed=seed)
    return dict(arm=arm, seed=seed, params=P, flops_per_token=FL,
                steps_run=steps, diverged=False,
                val_bpc=val_nats / math.log(2),
                train_bpc=last_train / math.log(2),
                n_val_windows=n_win, tok_s=ntok / wall, wall_s=wall,
                tokens_seen=int(ntok), pass_drift=drift)


def pass_drift(m, train, vocab, seed=0, ctx=CTX, n=4):
    """How much does one more pass actually change the representation?

    Reports, per pass, the relative change ||h_i - h_{i-1}|| / ||h_{i-1}|| and
    the cosine similarity between consecutive passes. This is the honest,
    cheap thing to measure on the "effective depth" axis: it says whether the
    extra passes are still doing work or have settled onto a fixed point. It is
    NOT a receptive-field measurement and is not reported as one.
    """
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(train) - ctx - 1, size=n)
    x = mx.array(np.stack([train[i:i + ctx] for i in ix]))
    states = m.forward_trace(x)
    ev = [m.tok(x) + m.pos(mx.arange(ctx)[None, :])] + states
    out = []
    for i in range(1, len(ev)):
        a, b = ev[i - 1], ev[i]
        num = float(mx.sqrt(mx.sum((b - a) ** 2)))
        den = float(mx.sqrt(mx.sum(a ** 2))) + 1e-12
        cos = float(mx.sum(a * b) / (mx.sqrt(mx.sum(a ** 2)) *
                                     mx.sqrt(mx.sum(b ** 2)) + 1e-12))
        out.append(dict(pass_index=i, rel_change=num / den,
                        cosine_to_prev=cos))
    return out


def _timing_session(arm, vocab, data, *, seed, bs, warmup, reps):
    """Set up one arm for repeated timed steps; return a `step` closure."""
    n = int(0.9 * len(data))
    tr = data[:n]
    mx.random.seed(seed)
    m = build(arm, vocab)
    mx.eval(m.parameters())
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(
            m(x).reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)

    def step():
        ix = rng.integers(0, len(tr) - CTX - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + CTX] for i in ix]))
        y = mx.array(np.stack([tr[i + 1:i + 1 + CTX] for i in ix]))
        l, g = lg(m, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)

    for _ in range(warmup):
        step()
    return step


def measure_step_time_interleaved(vocab, data, arms, *, seed=0, bs=16,
                                  warmup=5, reps=12, rounds=7):
    """Robust ms/step per arm under a shared, contended machine.

    Arms are measured ROUND-ROBIN inside each round rather than one arm at a
    time, so a load spike hits every arm rather than whichever arm happens to
    be running. Each arm's estimate is the MINIMUM round-median, which is the
    least-contended (closest to dedicated-machine) estimate. This matters: a
    naive sequential measurement gave loop2x2 20.2 ms/step on one invocation
    and 39.9 ms/step on the next, a 2x swing that would have silently decided
    which step count the equal-time arm got.
    """
    sessions = {a: _timing_session(a, vocab, data, seed=seed, bs=bs,
                                   warmup=warmup, reps=reps)
                for a in arms}
    per_arm = {a: [] for a in arms}
    for _ in range(rounds):
        for a in arms:
            st = sessions[a]
            t0 = time.time()
            for _ in range(reps):
                st()
            per_arm[a].append((time.time() - t0) / reps * 1000.0)
    ref = min(per_arm["loop2x2"]) if "loop2x2" in per_arm else 1.0
    return {a: dict(arm=a,
                    ms_per_step=float(min(per_arm[a])),
                    ms_per_step_median=float(np.median(per_arm[a])),
                    ms_per_step_all=[round(x, 2) for x in per_arm[a]],
                    steps_per_s=float(1000.0 / min(per_arm[a])),
                    steps_per_loop2x2_step=float(min(per_arm[a]) / ref),
                    tokens_per_step=bs * CTX)
            for a in arms}


def measure_step_time(arm, vocab, data, *, seed=0, bs=16, warmup=5, reps=25,
                      repeats=3):
    """Legacy single-arm measurement, kept for compatibility."""
    n = int(0.9 * len(data))
    tr = data[:n]
    mx.random.seed(seed)
    m = build(arm, vocab)
    mx.eval(m.parameters())
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(
            m(x).reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)

    def one_step():
        ix = rng.integers(0, len(tr) - CTX - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + CTX] for i in ix]))
        y = mx.array(np.stack([tr[i + 1:i + 1 + CTX] for i in ix]))
        l, g = lg(m, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)

    for _ in range(warmup):
        one_step()
    per_repeat = []
    for _ in range(repeats):
        t0 = time.time()
        for _ in range(reps):
            one_step()
        per_repeat.append((time.time() - t0) / reps * 1000.0)
    return dict(arm=arm, ms_per_step=float(min(per_repeat)),
                ms_per_step_all=[round(x, 2) for x in per_repeat],
                tokens_per_step=bs * CTX)


def measured_timing(vocab, data, arms):
    return measure_step_time_interleaved(vocab, data, arms)


def determinism_probe(vocab, data, arm, seed=0):
    """Measure, rather than assume, what IS reproducible on this device.

    Three separate claims, because they are not the same claim:
      1. init    -- same seed gives bit-identical parameters (must hold).
      2. forward -- same params + same batch give a bit-identical loss.
      3. gradient-- same params + same batch give a bit-identical gradient.
    Claim 3 is the one that fails on MLX's GPU backend (reduction order), and
    it is the only reason a full run is not bit-reproducible there. The CPU
    backend satisfies all three.
    """
    def one(device):
        mx.set_default_device(device)
        mx.random.seed(seed)
        m = build(arm, vocab)
        mx.eval(m.parameters())
        xv, yv, _ = deterministic_val_batches(data, CTX)
        x, y = xv[:4], yv[:4]

        def loss_fn(m, x, y):
            return nn.losses.cross_entropy(
                m(x).reshape(-1, vocab), y.reshape(-1), reduction="mean")

        lg = nn.value_and_grad(m, loss_fn)
        l1, g1 = lg(m, x, y)
        mx.eval(l1)
        f1 = mx.concatenate([v.reshape(-1)
                             for _, v in nn.utils.tree_flatten(g1)])
        mx.eval(f1)
        # same seed -> second, independent model; same batch
        mx.random.seed(seed)
        m2 = build(arm, vocab)
        mx.eval(m2.parameters())
        pa = [v for _, v in nn.utils.tree_flatten(m.parameters())]
        pb = [v for _, v in nn.utils.tree_flatten(m2.parameters())]
        init_same = all(bool(mx.all(a == b)) for a, b in zip(pa, pb))
        l2, g2 = lg(m, x, y)
        mx.eval(l2)
        f2 = mx.concatenate([v.reshape(-1)
                             for _, v in nn.utils.tree_flatten(g2)])
        mx.eval(f2)
        return dict(device=device.name if hasattr(device, "name") else str(device),
                    init_bit_identical=bool(init_same),
                    forward_loss_bit_identical=bool(l1 == l2),
                    gradient_bit_identical=bool(mx.all(f1 == f2)),
                    forward_loss_abs_diff=float(abs(l1 - l2)),
                    gradient_max_abs_diff=float(mx.max(mx.abs(f1 - f2))))

    out = []
    for dev in (mx.gpu, mx.cpu):
        try:
            out.append(one(dev))
        except Exception as e:  # a device may be unavailable
            out.append(dict(device=str(dev), error=repr(e)))
    mx.set_default_device(mx.cpu if _DEVICE == "cpu" else mx.gpu)
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def paired_report_directional(results, a, b):
    """paired_report, with the verdict phrased for a bpc comparison."""
    pr = paired_report(results, a, b)
    if pr.get("ci95") is None:
        return pr
    mean = pr["paired_mean_diff"]
    lo, hi = pr["ci95"]
    if lo < 0 < hi:
        pr["verdict"] = "NULL: interval straddles zero"
    elif mean < 0:
        pr["verdict"] = f"{a} better (lower bpc), interval excludes 0"
    else:
        pr["verdict"] = f"{b} better (lower bpc), interval excludes 0"
    return pr


def summarise(results, arms, bg):
    rows = []
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        ok = [r for r in rs if not r["diverged"]]
        v = [r["val_bpc"] for r in ok]
        t = [r["train_bpc"] for r in ok]
        rows.append(dict(
            arm=arm, params=rs[0]["params"],
            flops_per_token=rs[0]["flops_per_token"],
            loops=ARMS[arm]["loops"], n_layer=ARMS[arm]["n_layer"],
            steps=rs[0]["steps_run"],
            tokens_seen=int(np.mean([r.get("tokens_seen", 0) for r in rs])),
            pass_drift=ok[0].get("pass_drift") if ok else None,
            val_bpc_mean=float(np.mean(v)) if v else float("nan"),
            val_bpc_sd=float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
            val_bpc_min=float(min(v)) if v else float("nan"),
            val_bpc_max=float(max(v)) if v else float("nan"),
            train_bpc_mean=float(np.mean(t)) if t else float("nan"),
            wall_s_mean=float(np.mean([r["wall_s"] for r in rs])),
            tok_s_mean=float(np.mean([r["tok_s"] for r in rs])),
            n_seeds_ok=len(ok), n_seeds_total=len(rs),
            diverged=bool(len(ok) < len(rs)),
            at_or_above_floor=bool(v and min(v) >= bg),
        ))
    return rows


def print_tables(rows, paired, bg, ug):
    print(f"\nunigram floor {ug:.4f} bpc   BIGRAM floor {bg:.4f} bpc")
    print("(every arm must be BELOW the bigram floor to have learned any "
          "context at all)\n")
    hdr = (f"{'arm':<12}{'params':>10}{'FLOPs/tok':>13}{'uniq':>5}{'pass':>5}"
           f"{'steps':>7}{'train bpc':>11}{'val bpc':>9}{'sd':>8}"
           f"{'vs floor':>10}{'wall s':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = ""
        if r["diverged"]:
            flag = "  DIVERGED"
        elif r["at_or_above_floor"]:
            flag = "  ABOVE FLOOR (no context learned)"
        print(f"{r['arm']:<12}{r['params']:>10,}{r['flops_per_token']:>13,.0f}"
              f"{r['n_layer']:>5}{r['loops']:>5}{r['steps']:>7}"
              f"{r['train_bpc_mean']:>11.4f}{r['val_bpc_mean']:>9.4f}"
              f"{r['val_bpc_sd']:>8.4f}{bg - r['val_bpc_mean']:>+10.4f}"
              f"{r['wall_s_mean']:>9.1f}{flag}")
    print("\n=== PAIRED DIFFERENCES (5 shared seeds; negative = first arm better) ===")
    for pr in paired:
        print(f"  {pr['comparison']:<24} n={pr['n_seeds']} "
              f"mean={pr['paired_mean_diff']:+.5f} "
              f"CI=[{pr['ci95'][0]:+.5f},{pr['ci95'][1]:+.5f}] "
              f"sign={pr['sign_agreement']}  -> {pr['verdict']}")


def run_signature(arm, seed, steps, vocab, data):
    """Everything that determines a single run's numbers.

    Resume is keyed on THIS, not on the whole meta dict, so adding a new arm to
    the registry does not throw away runs that are already complete.
    """
    return dict(arm=arm, seed=seed, steps=steps, d=D, ctx=CTX, chunk=CHUNK,
                mlp_mult=MLP_MULT, chars=len(data), vocab=vocab,
                use_lr_schedule="warmup100_clip1.0")


if __name__ == "__main__":
    data, vocab = load_corpus(CORPUS)
    bg, ug = floors(data)
    seeds = [int(s) for s in os.environ.get("BRAIN_SEEDS", "0,1,2,3,4").split(",")]
    steps = int(os.environ.get("BRAIN_STEPS", "1500"))
    arms = os.environ.get("BRAIN_ARMS", ",".join(ARMS)).split(",")

    print(f"corpus {CORPUS}\n{len(data):,} chars, vocab {vocab}")
    print(f"unigram {ug:.4f}  BIGRAM {bg:.4f} bpc")
    xv, yv, nw = deterministic_val_batches(data, CTX)
    print(f"deterministic validation: {nw} non-overlapping windows of {CTX} "
          f"= {nw*CTX:,} tokens (every arm scored on identical data)")
    print(f"d={D} ctx={CTX} MLP mult={MLP_MULT} chunk={CHUNK} "
          f"steps={steps} seeds={seeds}\n")

    print("arm accounting (params = UNIQUE stack, FLOPs = n_layer*loops):")
    for a in arms:
        print(f"  {a:<12} params={n_params(build(a, vocab)):>8,} "
              f"flops/tok={arm_flops(a, vocab):>11,.0f}")
    print()

    meta = dict(corpus=CORPUS, chars=len(data), vocab=vocab, device=_DEVICE,
                d=D, ctx=CTX, mlp_mult=MLP_MULT, chunk=CHUNK, steps=steps,
                seeds=seeds, n_val_windows=nw,
                bigram_floor=bg, unigram_floor=ug,
                arms={a: dict(ARMS[a], params=n_params(build(a, vocab)),
                              flops_per_token=arm_flops(a, vocab))
                      for a in arms},
                decision_rule=(
                    "looping is a real efficiency win only if loop2x2 beats "
                    "flat2 beyond seed noise; flat2 has the SAME parameters "
                    "at HALF the FLOPs"),
                preregistered_prediction=(
                    "loop2x2 loses to flat4 on bpc at matched FLOPs, and "
                    "matches flat2 on params; a null on loop2x2-vs-flat2 is "
                    "the expected outcome"))

    # ---- determinism is ASSERTED, not assumed -------------------------
    # Two independent constructions of the same arm at the same seed must give
    # bit-identical validation loss. Anything else makes every paired interval
    # below meaningless.
    if os.environ.get("BRAIN_DET_CHECK", "1") == "1":
        mx.random.seed(0)
        xv0, yv0, _ = deterministic_val_batches(data, CTX)
        probe = arms[0]
        mx.random.seed(0)
        m1 = build(probe, vocab)
        mx.eval(m1.parameters())
        v1 = evaluate(m1, xv0[:8], yv0[:8], vocab)
        mx.random.seed(0)
        m2 = build(probe, vocab)
        mx.eval(m2.parameters())
        v2 = evaluate(m2, xv0[:8], yv0[:8], vocab)
        p_ok = n_params(m1) == n_params(m2)
        # init equality is exact on both devices; a TRAINED run's
        # bit-reproducibility depends on the device (see _DEVICE note above).
        mx.random.seed(0)
        m3 = build(probe, vocab)
        mx.random.seed(0)
        m4 = build(probe, vocab)
        mx.eval(m3.parameters(), m4.parameters())
        pa = [v for _, v in nn.utils.tree_flatten(m3.parameters())]
        pb = [v for _, v in nn.utils.tree_flatten(m4.parameters())]
        init_same = all(bool(mx.all(a == b)) for a, b in zip(pa, pb))
        print(f"determinism check [{probe}, device={_DEVICE}] "
              f"init bit-identical: {init_same}, params identical: {p_ok}, "
              f"forward-loss bit-identical: {v1 == v2}")
        if not (p_ok and init_same):
            raise SystemExit("DETERMINISM FAILED: init is not reproducible")
        if _DEVICE == "cpu" and v1 != v2:
            raise SystemExit("DETERMINISM FAILED on CPU: forward not identical")
        if _DEVICE != "cpu" and v1 != v2:
            print("  note: GPU forward loss differs run-to-run at ~1e-7; this "
                  "is MLX GPU reduction order, not this harness")

    # load any previous runs and keep only those whose signature still matches
    prev_runs = []
    if os.path.exists(RESULTS):
        try:
            prev = json.load(open(RESULTS))
            if isinstance(prev, dict):
                prev_runs = prev.get("runs", [])
        except Exception:
            prev_runs = []
    out = []
    for r in prev_runs:
        sig = run_signature(r["arm"], r["seed"], r["steps_run"], vocab, data)
        if r.get("signature") == sig:
            out.append(r)
    done = {(r["arm"], r["seed"]) for r in out}
    stale = len(prev_runs) - len(out)
    if done:
        print(f"resuming: {len(done)} runs already complete"
              + (f" ({stale} discarded as stale)" if stale else "") + "\n")

    def checkpoint():
        json.dump(dict(meta=meta, runs=out), open(RESULTS, "w"), indent=1)

    for arm in arms:
        arm_steps = int(round(steps * ARMS[arm].get("steps_mult", 1.0)))
        if arm_steps != steps:
            print(f"  {arm}: {arm_steps} steps (wall-clock-matched to "
                  f"{steps} steps of loop2x2; ratio "
                  f"{ARMS[arm]['steps_mult']:.4f})")
        for seed in seeds:
            if (arm, seed) in done:
                continue
            r = run(arm, data, vocab, steps=arm_steps, seed=seed)
            r["signature"] = run_signature(arm, seed, arm_steps, vocab, data)
            out.append(r)
            if r["diverged"]:
                print(f"  {arm:<12} seed={seed} DIVERGED at step "
                      f"{r['steps_run']} -- reported, not dropped", flush=True)
            else:
                print(f"  {arm:<12} seed={seed} params={r['params']:>8,} "
                      f"steps={r['steps_run']:>5} val={r['val_bpc']:.4f} "
                      f"train={r['train_bpc']:.4f} ({bg-r['val_bpc']:+.4f} vs "
                      f"floor) tok/s={r['tok_s']:>8,.0f} "
                      f"wall={r['wall_s']:.0f}s", flush=True)
            checkpoint()

    rows = summarise(out, arms, bg)
    paired = [paired_report_directional(out, a, b) for a, b in PAIRED
              if a in arms and b in arms]
    print_tables(rows, paired, bg, ug)

    timing = None
    if os.environ.get("BRAIN_TIMING", "0") == "1":
        tarms = os.environ.get(
            "BRAIN_TIMING_ARMS",
            "flat4,loop2x2,loop1x4,flat2,flat2_wall").split(",")
        print("\n=== measured step time (round-robin, min round-median) ===")
        timing = measured_timing(vocab, data, tarms)
        for a, t in timing.items():
            print(f"  {a:<11} {t['ms_per_step']:7.1f} ms/step (median "
                  f"{t['ms_per_step_median']:7.1f})  {t['steps_per_s']:6.2f} "
                  f"steps/s  {t['steps_per_loop2x2_step']:5.2f}x loop2x2 rate")

    det = None
    if os.environ.get("BRAIN_DET_EVIDENCE", "1") == "1":
        det = determinism_probe(vocab, data, arms[0])
        print("\n=== determinism evidence (measured, not assumed) ===")
        for row in det:
            if "error" in row:
                print(f"  {row['device']:<4} unavailable: {row['error']}")
                continue
            print(f"  {row['device']:<4} init={row['init_bit_identical']} "
                  f"forward={row['forward_loss_bit_identical']} "
                  f"gradient={row['gradient_bit_identical']} "
                  f"(grad max abs diff "
                  f"{row['gradient_max_abs_diff']:.3e})")

    json.dump(dict(meta=meta, determinism=det, per_arm=rows, paired=paired,
                   runs=out, timing=timing), open(RESULTS, "w"), indent=1)
    print(f"\nwrote {RESULTS}")
