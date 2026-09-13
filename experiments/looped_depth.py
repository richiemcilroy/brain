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
    Block, CORPUS, LM, flops_per_token, load_corpus, n_params,
)
from llm_tuned import floors  # noqa: E402
from confirm_headline import (  # noqa: E402
    T95, deterministic_val_batches, evaluate, paired_report,
)

RESULTS = os.environ.get(
    "BRAIN_LOOPED_OUT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", "results", "looped_depth.json",
    ),
)

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


def load_done(meta):
    """Resume support: reuse only runs whose configuration matches exactly."""
    if not os.path.exists(RESULTS):
        return []
    try:
        prev = json.load(open(RESULTS))
    except Exception:
        return []
    if not isinstance(prev, dict) or prev.get("meta") != meta:
        return []
    return prev.get("runs", [])


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

    meta = dict(corpus=CORPUS, chars=len(data), vocab=vocab,
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
        probe = arms[0]
        mx.random.seed(0)
        m1 = build(probe, vocab)
        mx.eval(m1.parameters())
        xv0, yv0, _ = deterministic_val_batches(data, CTX)
        v1 = evaluate(m1, xv0[:8], yv0[:8], vocab)
        mx.random.seed(0)
        m2 = build(probe, vocab)
        mx.eval(m2.parameters())
        v2 = evaluate(m2, xv0[:8], yv0[:8], vocab)
        same = (v1 == v2)
        p1 = n_params(m1) == n_params(m2)
        print(f"determinism check [{probe}] same-init loss identical: {same} "
              f"(params identical: {p1})")
        if not (same and p1):
            raise SystemExit("DETERMINISM FAILED: aborting before the sweep")

    out = load_done(meta)
    done = {(r["arm"], r["seed"]) for r in out}
    if done:
        print(f"resuming: {len(done)} runs already complete\n")

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

    json.dump(dict(meta=meta, per_arm=rows, paired=paired, runs=out),
              open(RESULTS, "w"), indent=1)
    print(f"\nwrote {RESULTS}")
