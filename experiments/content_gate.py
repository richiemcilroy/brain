"""CONTENT-ADDRESSED GATE: does STATE-DEPENDENCE in the forgetting gate help?

THE CAPABILITY THIS TESTS
-------------------------
This repo's carrier is a single-exponential trace whose forget gate is computed
from the input alone:

    h_t = g_t * h_{t-1} + v_t        g_t = sigmoid(W_g x_t)          (arm B)

Two measured results in this repo constrain that design:

  * `docs/MULTISCALE.md` / `experiments/banks_test.py`: multi-bank traces with a
    BETTER fit to attention's recency profile score WORSE perplexity. The repo's
    own words: "fit quality and perplexity are ANTI-CORRELATED here, not
    correlated" (`docs/IMPROVE.md` §2).
  * `docs/PRIOR_ART.md` §7: every published member of this family tunes the gate
    initialisation, and our arm inherits a fixed geometric schedule that has
    never been swept.

The hypothesis under test is that both facts have ONE cause: with an
input-only gate the timescale cannot be chosen per position. A bank at decay
0.9 and a bank at 0.9999 both apply their kernel unconditionally, so a
better-fitting STATIC kernel buys nothing -- what is missing is the ability to
select a timescale from the state. If that is right, adding state-dependence

    g_t = sigmoid(W_g x_t + U_g h_{t-1})                              (arm C)

should (i) improve bpc at matched compute, and (ii) remove the anti-correlation
between kernel-fit quality and bpc, because the static kernel is no longer the
thing that decides the timescale.

PRIOR ART -- READ THIS BEFORE READING ANY NUMBER BELOW
------------------------------------------------------
A gate that reads the previous hidden state makes this a **gated RNN in the
GRU / minimal-gated-unit family**:

  * Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder",
    EMNLP 2014, arXiv:1406.1078 -- the GRU update gate `z_t = sigma(W_z x_t +
    U_z h_{t-1})` is exactly arm C's gate, with our scalar-linear state update.
  * Zhou et al., "Minimal Gated Unit for RNNs", 2016, arXiv:1603.01228 --
    the minGRU simplification.
  * Martin & Cundy, GILR, ICLR 2018, arXiv:1709.04057; Lei et al., SRU, EMNLP
    2018, arXiv:1709.02755 -- input-only gated linear recurrences, i.e. arm B.

The *diagonal linear recurrence* `h_t = g_t*h_{t-1} + v_t` is the S4/Mamba/
RWKV/RG-LRU state-space form (Gu et al. 2021, arXiv:2111.00396; Gu & Dao 2023,
arXiv:2312.00752; Peng et al. RWKV 2023, arXiv:2305.13048; De et al. Griffin
2024, arXiv:2402.19427). Arm C is therefore **the standard GRU update gate
applied to the coordinate-wise state of a linear recurrence**. It is the
smallest possible change to arm B and it is not new.

**NO NOVELTY IS CLAIMED FOR THE GATE FORM.** Arm C is prior art, twice over.
The ONLY question this file answers is narrow and empirical: *given this repo's
specific gated-memory block, at matched parameters and matched per-token FLOPs,
does making the gate read the state improve validation bpc, and does it remove
the measured fit-vs-perplexity anti-correlation?* Anything beyond that
(scientific novelty, brains, dendrites, biology) is unsupported by this file.

ARMS -- ALL CARRY THE SAME PER-TOKEN MAC COUNT
----------------------------------------------
  A_attn      attention, fused causal (mx.fast.scaled_dot_product_attention),
              warmup + gradient clipping. The harness/floor CONTROL.
  B_plain     the repo's exact carrier: g_t = sigmoid(W_g x_t).  (no U_g)
  B_match     B_plain + a redundant input-only projection P (d x d, zero-init)
              so the parameter count and MAC count match arm C exactly.
              It CANNOT leave the input-only function class: P adds to the
              pre-activation, so (W_g + P) is another input-only gate.
  C_statedep  g_t = sigmoid(W_g x_t + U_g h_{t-1}), U_g zero-init.
  D_multiscale  C with a per-channel multi-timescale gate-bias initialisation.

WHY B_match EXISTS (the parameter/FLOP-matching device)
-------------------------------------------------------
C's gate costs one extra d x d matvec per token that B's does not. Comparing
them at equal parameters therefore compares "gate + 16384 MACs/token" against
"gate". Giving B a redundant d x d INPUT projection equalises both counts
exactly, so the single remaining difference between B_match and C is *where
the extra projection reads from*: x_t versus h_{t-1}.

  at d=128, 2 layers, ctx=512, vocab=65:
    A_attn 478,976   B_plain 445,696   B_match 478,464   C_statedep 478,464
  spread A vs B_match vs C = 0.107%   (criterion: within 2%)
  B_match vs C: identical parameter count AND identical per-token MAC count.

B_plain is still reported: it is the repo's real architecture, and it is cheap
to run. B_plain vs B_match should be within seed noise, which is itself the
check that the matching device is not doing work of its own.

HARD FAILURE CONDITION
----------------------
If `A_attn` does not beat the bigram floor (3.5806 bpc on this corpus) the run
raises. An attention arm at the bigram floor has not learned to use context and
nothing else in the file is interpretable.

PRE-REGISTERED FALSIFIERS (stated before the primary run)
---------------------------------------------------------
  F1  C does not beat B_match: the paired 95% interval on the bpc difference
      includes 0. -> The state-dependence hypothesis is REFUTED. Say so plainly.
  F2  The attention control fails to beat the bigram floor -> run is void.
  F3  In the init sweep, "fit quality vs bpc" is anti-correlated in BOTH the
      input-only and the state-dependent family -> the anti-correlation is NOT
      explained by state-dependence, whatever else is true.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import load_corpus, CORPUS  # noqa: E402
from llm_tuned import floors  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CG_OUT", os.path.join(HERE, "results", "content_gate.json"))

CTX = int(os.environ.get("CG_CTX", "512"))
D = int(os.environ.get("CG_D", "128"))
N_LAYER = int(os.environ.get("CG_LAYERS", "2"))
N_HEAD = int(os.environ.get("CG_HEADS", "4"))
MLP = int(os.environ.get("CG_MLP", "4"))
CHUNK = 64
VAL_FRAC = 0.1
# mx.compile on the train step. Arm C's recurrence is 512 sequential
# dispatches per forward pass and is launch-bound, so compilation is worth
# ~4x on it. It is applied to EVERY arm through the same code path, and
# `verify_compile_equivalence` checks that it changes nothing numerically.
COMPILE = os.environ.get("CG_COMPILE", "1") != "0"


# --------------------------------------------------------------------------
# parallel scan for the INPUT-ONLY gate only
# --------------------------------------------------------------------------
def associative_scan(a: mx.array, b: mx.array):
    """Hillis-Steele scan over affine maps (a2,b2)o(a1,b1)=(a2*a1, a2*b1+b2)."""
    T = a.shape[-2]
    step = 1
    while step < T:
        a_shift = mx.concatenate(
            [mx.ones_like(a[..., :step, :]), a[..., : T - step, :]], axis=-2
        )
        b_shift = mx.concatenate(
            [mx.zeros_like(b[..., :step, :]), b[..., : T - step, :]], axis=-2
        )
        b = a * b_shift + b
        a = a * a_shift
        step *= 2
    return a, b


def chunked_gated_scan(v: mx.array, g: mx.array, chunk: int = CHUNK) -> mx.array:
    """h_t = g_t*h_{t-1} + v_t in parallel. VALID ONLY WHEN g DEPENDS ON x ONLY.

    A state-dependent gate destroys the associativity this scan relies on: the
    affine map at step t is not known until h_{t-1} is known. Arm C therefore
    MUST run sequentially, and that is a real cost of the mechanism, not an
    implementation detail. The timing section reports it.
    """
    B, T, Dd = v.shape
    pad = (-T) % chunk
    if pad:
        z = mx.zeros((B, pad, Dd))
        v = mx.concatenate([v, z], axis=1)
        g = mx.concatenate([g, z], axis=1)
    Tp = v.shape[1]
    n_ch = Tp // chunk
    vc = v.reshape(B, n_ch, chunk, Dd)
    gc = g.reshape(B, n_ch, chunk, Dd)
    prod, intra = associative_scan(gc, vc)
    _, carry = associative_scan(prod[:, :, chunk - 1, :], intra[:, :, chunk - 1, :])
    carry_in = mx.concatenate([mx.zeros((B, 1, Dd)), carry[:, :-1, :]], axis=1)
    h = intra + prod * carry_in[:, :, None, :]
    return h.reshape(B, Tp, Dd)[:, :T, :]


def sequential_gated_scan(v: mx.array, g: mx.array, ug, h0=None) -> mx.array:
    """Reference recurrence. Used to VERIFY the chunked scan, and (with ug) as
    the actual arm-C implementation, which cannot use the parallel form."""
    B, T, Dd = v.shape
    h = mx.zeros((B, Dd)) if h0 is None else h0
    acc = []
    for t in range(T):
        gt = g[:, t, :] + ug(h) if ug is not None else g[:, t, :]
        h = mx.sigmoid(gt) * h + v[:, t, :]
        acc.append(h)
    return mx.stack(acc, axis=1)


def verify_scan() -> float:
    """The parallel scan must reproduce the recurrence it replaces.

    CONTRACT, and the first version of this check got it wrong: the two
    functions take DIFFERENT things as `g`.
      * `sequential_gated_scan(v, z, ug)` takes the PRE-SIGMOID logits, because
        with a state-dependent gate the addition `z_t + U_g h_{t-1}` has to
        happen before the nonlinearity.
      * `chunked_gated_scan(v, g, chunk)` takes the POST-SIGMOID decay, because
        the affine map must be known in advance for the scan to be associative.
    Comparing them directly double-applies the sigmoid and reports an error of
    1.75 on a recurrence that is in fact exact. The check below feeds each
    function what its own contract says it takes.
    """
    worst = 0.0
    for T in (1, 2, 7, 33, 64, 65, 129, 512):
        mx.random.seed(0)
        z = mx.random.normal((2, T, 4))
        v = mx.random.normal((2, T, 4))
        ref = sequential_gated_scan(v, z, None)          # applies sigmoid itself
        got = chunked_gated_scan(v, mx.sigmoid(z), CHUNK)  # pre-sigmoided decay
        worst = max(worst, float(mx.max(mx.abs(ref - got))))
    return worst


# --------------------------------------------------------------------------
# gate-bias initialisation
# --------------------------------------------------------------------------
def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


MULTISCALE_DECAYS = (0.90, 0.99, 0.999, 0.9999)


def multiscale_bias(d: int, decays=MULTISCALE_DECAYS) -> mx.array:
    """Per-channel gate bias so that channel j of block 0 starts at a different
    timescale. One bias vector, read by every block -- it is an
    INITIALISATION, not an extra parameter tensor.

    This is arm D's only difference from arm C. It is the same idea the repo
    already ships in `GatedMemory.init_decays`, applied to the gate bias instead
    of to separate trace banks, so arm D costs nothing and dilutes no readout
    (which `docs/IMPROVE.md` §2 identifies as the likely reason multi-bank lost).
    """
    n = len(decays)
    assert d % n == 0, f"d={d} must be divisible by {n} timescales"
    per = d // n
    parts = [mx.full((per,), logit(a), dtype=mx.float32) for a in decays]
    return mx.concatenate(parts, axis=0)


class Block(nn.Module):
    """One pre-norm residual block: mixing primitive + MLP, identical across arms.

    The mixing primitive is chosen by `kind`; everything else is shared code, so
    a difference between arms cannot come from a difference in the block.
    """

    def __init__(self, kind: str, d: int, n_head: int = N_HEAD, mlp: int = MLP,
                 match: bool = False, bias_init=None, state_dep: bool = False):
        super().__init__()
        self.kind = kind
        self.capture_gate = False
        self.last_gate = None
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        if kind == "attn":
            self.q = nn.Linear(d, d, bias=True)
            self.k = nn.Linear(d, d, bias=True)
            self.v = nn.Linear(d, d, bias=True)
            self.o = nn.Linear(d, d, bias=True)
            self.n_head = n_head
            self.scale = (d // n_head) ** -0.5
        else:
            self.wg = nn.Linear(d, d, bias=True)
            self.v = nn.Linear(d, d, bias=False)
            self.o = nn.Linear(d, d, bias=True)
            if bias_init is not None:
                # COPY, not share. Assigning the same array object into every
                # block makes the blocks alias one buffer, so at init all layers
                # have a gate bias that is literally the same tensor. MLX's
                # optimizer replaces arrays rather than mutating them in place,
                # so training breaks the alias after the first step and the run
                # is not actually degenerate -- but the initialisation is
                # shared state that nobody asked for, and it would become a real
                # tied-parameter bug the moment anything updated in place.
                self.wg.bias = mx.array(bias_init)
            # C and D: the gate reads the state.
            self.ug = nn.Linear(d, d, bias=False) if state_dep else None
            if state_dep:
                self.ug.weight = mx.zeros_like(self.ug.weight)
            # B_match: a redundant INPUT-ONLY projection. Zero-init, and it adds
            # to the pre-activation, so the gate stays in the input-only class.
            self.p = nn.Linear(d, d, bias=False) if (match and not state_dep) else None
            if self.p is not None:
                self.p.weight = mx.zeros_like(self.p.weight)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp * d, bias=True),
            nn.GELU(),
            nn.Linear(mlp * d, d, bias=True),
        )

    def __call__(self, x, mask=None):
        h = self.ln1(x)
        if self.kind == "attn":
            B, T, _ = h.shape
            hd = h.shape[-1] // self.n_head
            q = self.q(h).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
            k = self.k(h).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
            v = self.v(h).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
            # mask="causal" is MLX's fused causal path; mask=None would attend
            # to the future, and comparing a causal arm to a non-causal arm
            # measures two different functions.
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal"
            )
            y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
            y = self.o(y)
        else:
            v = self.v(h)
            z = self.wg(h)
            if self.p is not None:
                z = z + self.p(h)
            if self.ug is not None:
                # state-dependent: NOT parallelisable, run the recurrence.
                y = self.o(sequential_gated_scan(v, z, self.ug))
            else:
                y = self.o(chunked_gated_scan(v, mx.sigmoid(z), CHUNK))
            # Record the gate for `measure_effective_decay`. This is a plain
            # scalar assignment on a module attribute, so it costs nothing when
            # unused and does not enter the graph.
            if self.capture_gate:
                if self.ug is not None:
                    # replay the recurrence to collect g_t; the state path is
                    # sequential, so this is the only place g_t exists.
                    hh = mx.zeros((v.shape[0], v.shape[-1]))
                    gs = []
                    for t in range(x.shape[1]):
                        gt = mx.sigmoid(z[:, t, :] + self.ug(hh))
                        hh = gt * hh + v[:, t, :]
                        gs.append(gt)
                    self.last_gate = mx.stack(gs, axis=1)
                else:
                    self.last_gate = mx.sigmoid(z)
        x = x + y
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self, kind: str, vocab: int, d: int = D, ctx: int = CTX,
                 n_layer: int = N_LAYER, match: bool = False,
                 bias_init=None, state_dep: bool = False):
        super().__init__()
        self.kind = kind
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = [
            Block(kind, d, match=match, bias_init=bias_init,
                  state_dep=state_dep)
            for _ in range(n_layer)
        ]
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(mx.arange(T)[None, :])
        mask = None
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.lnf(x))


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
# `bias_init` is the arm's gate-bias initialisation; None means MLX's default
# (uniform) init. Every arm below uses the SAME block code path.
ARMS = {
    # attention control, fused causal, warmup + grad clip in run_arm
    "A_attn": dict(kind="attn", match=False, state_dep=False, bias="zero"),
    # the repo's shipped carrier: input-only gate, no matching device
    "B_plain": dict(kind="mem", match=False, state_dep=False, bias="0.9"),
    # input-only gate + redundant input projection (params & MACs == C)
    "B_match": dict(kind="mem", match=True, state_dep=False, bias="0.9"),
    # THE HYPOTHESIS: gate reads the state
    "C_statedep": dict(kind="mem", match=False, state_dep=True, bias="0.9"),
    # C with a multi-timescale gate-bias init (same parameters as C)
    "D_multiscale": dict(kind="mem", match=False, state_dep=True, bias="multiscale"),
}
# B_match's `match` flag must equal C's extra-projection count for the
# comparison to be honest; assert it rather than trusting the table above.
MATCHED_PAIRS = [("C_statedep", "B_match")]


def make_bias(spec_bias, d: int):
    """A FRESH bias array per call. Returning a cached tensor would be safe only
    because `Block` copies it; returning a fresh one means the copy is
    belt-and-braces rather than load-bearing."""
    if spec_bias is None or spec_bias == "zero":
        return None
    if spec_bias == "multiscale":
        return multiscale_bias(d)
    return mx.full((d,), logit(float(spec_bias)), dtype=mx.float32)


def build(arm: str, vocab: int, d: int = D, ctx: int = CTX,
          n_layer: int = N_LAYER) -> nn.Module:
    spec = ARMS[arm]
    bias = make_bias(spec["bias"], d)
    m = LM(spec["kind"], vocab, d=d, ctx=ctx, n_layer=n_layer,
           match=spec["match"], bias_init=bias, state_dep=spec["state_dep"])
    return m


def n_params(m: nn.Module) -> int:
    return sum(
        int(np.prod(v.shape)) for _, v in nn.utils.tree_flatten(m.parameters())
    )


def flops_per_token(arm: str, vocab: int, d: int = D, ctx: int = CTX,
                    n_layer: int = N_LAYER, mlp: int = MLP) -> float:
    """Analytic forward+backward MACs per token, x3 (fwd + 2x bwd).

    The three projections every arm shares (v, o, mlp) are charged identically.
    What differs is the gate path:

      attn        : 4 projections d->d, plus T^2*d for scores and A@V
      B_plain     : gate proj d->d + scan
      B_match     : gate proj d->d + REDUNDANT input proj d->d + scan
      C / D       : gate proj d->d + STATE proj d->d (U_g h) + scan

    B_match and C therefore carry an identical per-token MAC count by
    construction -- the matching is arithmetic, not hand-waving. The scan is
    charged 3*d*log2(chunk) per token, the same for every memory arm.
    """
    spec = ARMS[arm]
    mlp_cost = 2 * (d * mlp * d + mlp * d * d)
    if spec["kind"] == "attn":
        mix = 2 * 4 * d * d + 2 * 2 * ctx * d
    else:
        gate_projs = 1 + (1 if (spec["match"] or spec["state_dep"]) else 0)
        mix = 2 * gate_projs * d * d + 2 * d * d  # + v and o projections
        mix += 3.0 * d * math.log2(max(CHUNK, 2))
    per = (mix + mlp_cost) * n_layer
    per += 2 * d * vocab          # head
    return 3.0 * per


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def make_splits(data: np.ndarray, ctx: int = CTX, val_frac: float = VAL_FRAC):
    n = int((1.0 - val_frac) * len(data))
    return data[:n], data[n:]


def make_batcher(tr: np.ndarray, va: np.ndarray, ctx: int = CTX):
    def batch(bs: int, rng: np.random.Generator, split: str = "train"):
        d = tr if split == "train" else va
        ix = rng.integers(0, len(d) - ctx - 1, size=bs)
        x = np.stack([d[i:i + ctx] for i in ix])
        y = np.stack([d[i + 1:i + 1 + ctx] for i in ix])
        return mx.array(x), mx.array(y)
    return batch


def val_windows(va: np.ndarray, ctx: int = CTX, max_windows: int = 0):
    """DETERMINISTIC validation: every non-overlapping ctx-token window of the
    split, so all arms are scored on byte-identical data. Sampling the val split
    per arm would let an unlucky draw carry a comparison."""
    n = (len(va) - 1) // ctx
    if max_windows:
        n = min(n, max_windows)
    return [
        (mx.array(va[i * ctx:(i + 1) * ctx][None, :]),
         mx.array(va[i * ctx + 1:(i + 1) * ctx + 1][None, :]))
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def evaluate(m: nn.Module, windows, vocab: int) -> float:
    """Mean cross-entropy in bits/token over deterministic windows."""
    tot, n = 0.0, 0
    for x, y in windows:
        lo = m(x)
        loss = nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )
        mx.eval(loss)
        tot += float(loss) * y.size
        n += y.size
    return (tot / n) / math.log(2)


def run_arm(arm: str, tr, va, vocab: int, *, steps: int, lr: float = 1e-3,
            bs: int = 16, warmup: int = 100, clip: float = 1.0, seed: int = 0,
            wd: float = 0.01, windows=None, log=print, curve_every: int = 0,
            probe: np.ndarray | None = None):
    """Train one arm. Identical recipe for every arm -- warmup + grad clip are
    given to ALL arms, including attention, because the retracted claim in
    `docs/EFFICIENCY.md` failed precisely by sampling a mistuned transformer."""
    mx.random.seed(seed)
    model = build(arm, vocab)
    mx.eval(model.parameters())
    params = n_params(model)
    fl = flops_per_token(arm, vocab)
    batch = make_batcher(tr, va)
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=wd)

    def loss_fn(mod, x, y):
        lo = mod(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(model, loss_fn)
    mx.eval(model.parameters())

    # One compiled step function, used by every arm. `inputs`/`outputs` are the
    # model and optimizer state, so parameters stay DYNAMIC -- compiling a
    # closure over the model instead would freeze the parameters at their
    # initial values and silently train nothing (measured: loss identical to
    # 4 decimal places after 30 steps, which is how that mistake shows up).
    train_step = None
    if COMPILE:
        state = [model.state, opt.state]

        def _body(x, y):
            loss, grads = lg(model, x, y)
            if clip:
                grads, _ = optim.clip_grad_norm(grads, clip)
            opt.update(model, grads)
            return loss

        try:
            train_step = mx.compile(_body, state, state)
        except Exception as exc:
            log(f"    [{arm}] mx.compile unavailable ({exc}); running eager",
                flush=True)
            train_step = None

    curve = []
    t0 = time.time()
    ntok = 0
    for s in range(1, steps + 1):
        opt.learning_rate = lr * min(1.0, s / max(warmup, 1))
        x, y = batch(bs, rng)
        if train_step is not None:
            loss = train_step(x, y)
            mx.eval(loss, model.state, opt.state)
        else:
            loss, grads = lg(model, x, y)
            if clip:
                grads, _ = optim.clip_grad_norm(grads, clip)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
        ntok += x.size
        if curve_every and (s % curve_every == 0 or s == steps):
            curve.append(dict(step=s, train_loss=float(loss),
                              tok_s=ntok / (time.time() - t0)))
    wall = time.time() - t0
    bpc = evaluate(model, windows, vocab)
    # The decay the TRAINED model actually applies. `probe` is the token prefix
    # to measure it on; passing it is opt-in because the measurement is only
    # needed by the recency sweep.
    eff = measure_effective_decay(model, probe, T=min(256, len(probe))) \
        if probe is not None else None
    return dict(arm=arm, seed=seed, params=params, flops_per_token=fl,
                effective_decay=eff,
                bpc=bpc, steps=steps, bs=bs, lr=lr, ctx=CTX, d=D,
                n_layer=N_LAYER, tokens=ntok, wall_s=wall,
                tok_s=ntok / wall, cum_flops=ntok * fl, curve=curve,
                wd=wd, warmup=warmup, grad_clip=clip, compiled=bool(train_step))


# --------------------------------------------------------------------------
# paired statistics (no scipy in this environment; t-interval by hand)
# --------------------------------------------------------------------------
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131}


def paired_interval(diff: np.ndarray) -> dict:
    """95% paired t-interval on the per-seed differences. Seeds are paired, so
    the shared per-seed effect cancels -- the same device used in
    `docs/MATCHED_COMPARISON.md`."""
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = T975.get(n - 1, 2.0)
    lo, hi = mean - t * se, mean + t * se
    return dict(n=n, mean=mean, sd=sd, se=se, lo=lo, hi=hi,
                excludes_zero=bool(lo * hi > 0),
                n_sign_agree=int(max((diff > 0).sum(), (diff < 0).sum())))


def pearson(xs, ys) -> float:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 2 or xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def measure_effective_decay(model: nn.Module, ids: np.ndarray,
                           T: int = 256) -> dict:
    """Measure the gate the model ACTUALLY applies after training.

    THE PROBLEM THIS SOLVES, and it matters for the anti-correlation claim.
    The recency sweep varies the gate-bias INIT, so the cdf_l1 it can compute
    before training is the fit of the *initial* decay. But bpc is produced by
    the *trained* kernel, and training is free to move the gate a long way from
    where it started. Reporting a correlation between "fit of the initial decay"
    and "bpc after training" would be measuring a proxy for a proxy.

    So both are recorded:
      * `cdf_l1_init`    -- fit of the decay the bias was initialised to, which
                            is what the repo's table effectively reports.
      * `cdf_l1_trained` -- fit of the mean gate value the trained model
                            actually applies, per layer.
    The correlation of record is against `cdf_l1_trained`.

    The mean of g is a summary of a per-position quantity, so it is a
    first-order description of a time-varying decay, not the whole kernel. That
    limitation is recorded in the output rather than hidden.
    """
    layers = []
    for blk in model.blocks:
        if blk.kind != "mem":
            return dict(available=False,
                        reason="model has no gated-memory block to measure")
        blk.capture_gate = True
    try:
        x = mx.array(np.asarray(ids[:T], dtype=np.int32)[None, :])
        out = model(x)
        mx.eval(out)
        for blk in model.blocks:
            g = blk.last_gate
            if g is None:
                layers.append(None)
                continue
            mx.eval(g)
            layers.append(dict(
                mean_gate=float(mx.mean(g)),
                sd_gate=float(mx.std(g)),
                min_gate=float(mx.min(g)),
                max_gate=float(mx.max(g)),
                per_layer_cdf_l1=None))
    finally:
        for blk in model.blocks:
            blk.capture_gate = False
            blk.last_gate = None
    return dict(available=True, probe_tokens=T, layers=layers,
                mean_gate=float(np.mean([l["mean_gate"] for l in layers]))
                if layers else None)


def cdf_l1_of_single_exponential(decay: float, profile: np.ndarray) -> float:
    """How well one exponential reproduces a measured recency profile, using the
    repo's own criterion (`fit_decay_from_attention`, llm_hybrid.py:244):
    L1 distance between the cumulative distributions."""
    xs = np.arange(len(profile))
    kern = (1 - decay) * decay ** xs
    kern = kern / kern.sum()
    return float(np.abs(np.cumsum(kern) - np.cumsum(profile)).sum())


def measure_recency_profile(model: nn.Module, ids: np.ndarray,
                            layer: int = 0, T: int = 256) -> np.ndarray:
    """Measure the REAL recency profile of one attention block, from its own
    projections: score q against k under a causal mask and histogram the mean
    attention weight by token distance.

    This mirrors `measure_recency_profile` in `llm_hybrid.py:156`, including the
    reason it takes a slice: a very short probe makes the profile flat by
    construction and any fitted decay is fitted to noise. T=256 is used and the
    profile is normalised before comparison.
    """
    blk = model.blocks[layer]
    x = model.tok(mx.array(ids[:T][None, :])) + model.pos(mx.arange(T)[None, :])
    h = blk.ln1(x)
    B, Tt, _ = h.shape
    hd = h.shape[-1] // blk.n_head
    q = blk.q(h).reshape(B, Tt, blk.n_head, hd).transpose(0, 2, 1, 3)
    k = blk.k(h).reshape(B, Tt, blk.n_head, hd).transpose(0, 2, 1, 3)
    s = (q @ k.transpose(0, 1, 3, 2)) * blk.scale
    # ADDITIVE causal mask. `mx.triu(full(-1e30), k=1)` is -1e30 STRICTLY ABOVE
    # the diagonal; adding it zeroes every future position and leaves the past
    # and the diagonal untouched.
    #
    # The first version of this line was `mx.where(triu(...) > -1e29, -1e30, s)`
    # and it was backwards: that condition is TRUE on the lower triangle and the
    # diagonal, so it masked the PAST and kept the FUTURE. The symptom was a
    # profile with zero mass at distance 0 and argmax 255, i.e. 4.8e-2 of the
    # mass in the first 4 distances against 4.8e-1 for an independent numpy
    # reference of the same math. The attention arm's own bpc was unaffected --
    # its forward pass goes through the fused causal kernel -- but every
    # cdf_l1 in the recency sweep would have been measured against a
    # non-causal profile. Caught by comparing against the reference, not by
    # reading the code.
    s = s + mx.triu(mx.full((Tt, Tt), -1e30), k=1)
    w = mx.softmax(s, axis=-1)
    mx.eval(w)
    w = np.asarray(w, dtype=np.float64).mean(axis=(0, 1))   # (T, T)
    # distance = how far into the PAST: row index minus column index. The
    # inverted convention (column minus row) makes every positive distance a
    # FUTURE token, which the causal mask set to zero -- the profile then has
    # all its mass at distance 0 and a fitted decay degenerates to the bottom
    # of the grid. `llm_hybrid.py:232` records the same bug in the original.
    prof = np.zeros(Tt)
    for dist in range(Tt):
        vals = [w[i, i - dist] for i in range(dist, Tt)]
        prof[dist] = float(np.mean(vals))
    # The FULL profile is returned, not a truncated prefix: the repo's own
    # criterion (`fit_decay_from_attention`) compares the cumulative
    # distribution over every distance in the probe, and truncating to the
    # first 64 distances would change the cdf_l1 being reported to a different
    # statistic than the one the repo quotes.
    return prof / prof.sum()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    seeds = [int(s) for s in os.environ.get("CG_SEEDS", "0,1,2,3,4").split(",")]
    steps = int(os.environ.get("CG_STEPS", "1500"))
    only = os.environ.get("CG_ARMS", "").strip()
    which = only.split(",") if only else list(ARMS)
    max_win = int(os.environ.get("CG_VAL_WINDOWS", "0"))
    do_recency = os.environ.get("CG_RECENCY", "1") != "0"

    print(f"device {mx.default_device()} | d={D} layers={N_LAYER} ctx={CTX} "
          f"mlp={MLP} | steps={steps} seeds={seeds}", flush=True)

    err = verify_scan()
    print(f"scan verification: max abs error vs sequential recurrence = {err:.3e}",
          flush=True)
    if err > 1e-4:
        raise SystemExit("chunked scan does not match the recurrence; aborting")

    data, vocab = load_corpus(CORPUS)
    tr, va = make_splits(data)

    print("causality check (perturbing token p must not move logits < p):",
          flush=True)
    causality = verify_causality(vocab, tr)

    print(f"mx.compile on the train step: {COMPILE}", flush=True)
    if COMPILE and os.environ.get("CG_SKIP_COMPILE_CHECK", "0") != "1":
        print("verifying that compiling the train step changes nothing:",
              flush=True)
        compile_check = verify_compile_equivalence(vocab, tr, va)
    else:
        compile_check = None
    windows = val_windows(va, max_windows=max_win)
    win_tok = sum(int(y.size) for _, y in windows)
    bigram, unigram = floors(data)
    print(f"corpus {len(data):,} chars, vocab {vocab} | train {len(tr):,} "
          f"val {len(va):,} | {len(windows)} deterministic val windows "
          f"({win_tok:,} tokens)", flush=True)
    print(f"unigram floor {unigram:.4f} bpc | BIGRAM FLOOR {bigram:.4f} bpc\n",
          flush=True)

    # ---- parameter / FLOP match table, printed before any training ---------
    table = {}
    for arm in ARMS:
        mx.random.seed(0)
        m = build(arm, vocab)
        mx.eval(m.parameters())
        table[arm] = dict(params=n_params(m),
                          flops_per_token=flops_per_token(arm, vocab),
                          spec={k: v for k, v in ARMS[arm].items()})
    ref = max(table[a]["params"] for a in ARMS)
    for arm in ARMS:
        table[arm]["pct_vs_max"] = 100.0 * (table[arm]["params"] - ref) / ref
    bp, bm, cp = (table["B_plain"]["params"], table["B_match"]["params"],
                  table["C_statedep"]["params"])
    print("arm                  params   vs max    MACs/tok   gate path")
    for arm in ARMS:
        s = ARMS[arm]
        path = ("4 proj + T^2 attn" if s["kind"] == "attn"
                else f"gate{'+redundant in' if s['match'] else ''}"
                     f"{'+state U_g' if s['state_dep'] else ''}")
        print(f"  {arm:<18} {table[arm]['params']:>8,} "
              f"{table[arm]['pct_vs_max']:>7.3f}% {table[arm]['flops_per_token']:>12,.0f}   {path}",
              flush=True)
    spread = max(table[a]["params"] for a in ("A_attn", "B_match", "C_statedep"))
    spread = 100.0 * (spread - min(table[a]["params"]
                                   for a in ("A_attn", "B_match", "C_statedep"))) / spread
    print(f"\nA/B_match/C parameter spread: {spread:.3f}% (criterion: within 2%)",
          flush=True)
    print(f"B_match vs C: params {bm:,} vs {cp:,} -> "
          f"{'IDENTICAL' if bm == cp else 'MISMATCH'}; "
          f"MACs {table['B_match']['flops_per_token']:,.0f} vs "
          f"{table['C_statedep']['flops_per_token']:,.0f} -> "
          f"{'IDENTICAL' if table['B_match']['flops_per_token'] == table['C_statedep']['flops_per_token'] else 'MISMATCH'}",
          flush=True)
    if bm != cp:
        raise SystemExit(
            "the parameter-matching device failed: B_match and C_statedep must "
            f"have identical parameter counts, got {bm:,} vs {cp:,}")
    if abs(table["A_attn"]["params"] - cp) / cp > 0.02:
        raise SystemExit("attention arm is not within 2% of C; fix the widths")

    out = dict(
        meta=dict(host=platform.platform(), machine=platform.machine(),
                  load_avg_at_start=load_avg(),
                  driver="experiments/content_gate.py", mlx=mx.__version__
                  if hasattr(mx, "__version__") else None,
                  device=str(mx.default_device()),
                  corpus=os.path.basename(CORPUS), corpus_tokens=int(len(data)),
                  vocab=vocab, ctx=CTX, d=D, n_layer=N_LAYER, mlp=MLP,
                  n_head=N_HEAD, chunk=CHUNK, steps=steps, seeds=seeds,
                  bs=16, lr=1e-3, warmup=100, grad_clip=1.0, weight_decay=0.01,
                  val_windows=len(windows), val_tokens=win_tok,
                  val_sha1=hashlib.sha1(va.tobytes()).hexdigest(),
                  bigram_bpc=bigram, unigram_bpc=unigram,
                  scan_verify_max_abs_err=err,
                  compile_train_step=COMPILE,
                  causality_max_delta=causality,
                  compile_equivalence=compile_check,
                  started=time.strftime("%Y-%m-%dT%H:%M:%S")),
        param_table=table, arms={}, timing={}, recency={})
    # ---- what is matched, and what cannot be ---------------------------------
    # The STRICTLY matched set is the memory arms, where matching is exact:
    #   B_match == C_statedep == D_multiscale in parameters AND per-token MACs.
    # A_attn is matched in PARAMETERS (0.107%) but not in MACs, and the reason
    # is structural rather than an oversight: attention pays 2*2*ctx*d per token
    # for the score matrix and A@V, which is 67% of its MACs at ctx=512. To pull
    # its MACs down to C's 2,226,432 at 2 layers would need d~88, whose
    # parameter count is 244,816, i.e. 51% of C's -- so at fixed ctx it is
    # impossible to match attention on parameters AND MACs at once. The
    # comparison of record is therefore C vs B_match (exact on both), with
    # A_attn as the floor/harness control at matched parameters.
    a_macs = table["A_attn"]["flops_per_token"]
    c_macs = table["C_statedep"]["flops_per_token"]
    matched = dict(
        exact_set=["B_match", "C_statedep", "D_multiscale"],
        exact_params=bool(bm == cp == table["D_multiscale"]["params"]),
        exact_macs=bool(table["B_match"]["flops_per_token"]
                        == table["C_statedep"]["flops_per_token"]
                        == table["D_multiscale"]["flops_per_token"]),
        attention_params_within_2pct=bool(abs(table["A_attn"]["params"] - cp) / cp <= 0.02),
        attention_macs_ratio_vs_C=float(a_macs / c_macs),
        attention_macs_note=(
            "attention is NOT MAC-matched and structurally cannot be at fixed "
            "ctx: its 2*2*ctx*d score+AV term is 67% of its MACs, and shrinking "
            "d to match MACs would drop its parameters to ~51% of C's. It is "
            "reported as the floor/harness control at matched parameters."),
        B_plain_params_delta_pct_vs_C=float(
            100.0 * (bp - cp) / cp),
        B_plain_note=(
            "B_plain is the architecture as shipped, and is the only memory arm "
            "NOT parameter-matched (it lacks the extra d x d projection). It is "
            "reported for reference; the hypothesis is tested against B_match."),
    )
    out["matched_set"] = matched

    print(f"\nmatched set (exact params AND exact MACs): {matched['exact_set']}", flush=True)
    print(f"  params identical: {matched['exact_params']} | "
          f"MACs identical: {matched['exact_macs']}", flush=True)
    print(f"  A_attn params within 2% of C: {matched['attention_params_within_2pct']} "
          f"| A_attn MACs = {a_macs/c_macs:.2f}x C's (structural, see note)", flush=True)
    print(f"  B_plain is {matched['B_plain_params_delta_pct_vs_C']:+.2f}% vs C "
          f"(as-shipped arm, reported for reference)\n", flush=True)



    print(f"\n=== training {len(which)} arms x {len(seeds)} seeds ===", flush=True)
    t_start = time.time()
    for arm in which:
        out["arms"][arm] = []
        for seed in seeds:
            r = run_arm(arm, tr, va, vocab, steps=steps, seed=seed,
                        windows=windows, curve_every=steps)
            out["arms"][arm].append(r)
            print(f"  {arm:<18} seed {seed} | params {r['params']:>8,} "
                  f"bpc {r['bpc']:.4f} | {r['tok_s']:>10,.0f} tok/s "
                  f"| {r['wall_s']:6.1f}s", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
        # EARLY floor check. The full floor verdict is computed after the last
        # arm, but waiting for it would waste hours when the attention control
        # has failed: nothing downstream is interpretable, so abort now.
        if arm == "A_attn":
            a_bpc = float(np.mean([r["bpc"] for r in out["arms"]["A_attn"]]))
            if bigram - a_bpc <= 0:
                out["verdict"] = dict(attention_beats_bigram=False,
                                      attention_bpc=a_bpc, bigram_bpc=bigram,
                                      run_void=True, aborted_early=True)
                json.dump(out, open(OUT, "w"), indent=1)
                raise SystemExit(
                    f"HARD FAIL (aborted after A_attn, before training the other "
                    f"arms): A_attn scores {a_bpc:.4f} bpc against the bigram "
                    f"floor {bigram:.4f} bpc. An attention arm at the floor has "
                    f"not learned to use context, so no other number would mean "
                    f"anything.")
            print(f"  EARLY CHECK PASS: A_attn {a_bpc:.4f} bpc clears the bigram "
                  f"floor {bigram:.4f} by {bigram - a_bpc:+.4f} bpc\n",
                  flush=True)
    print(f"\ntraining done in {(time.time()-t_start)/60:.1f} min", flush=True)

    # ---- per-arm summary ---------------------------------------------------
    summary = {}
    print("\n=== arm summary ===", flush=True)
    print(f"{'arm':<18}{'params':>10}{'MACs/tok':>14}{'bpc mean':>11}{'sd':>9}"
          f"{'min':>9}{'max':>9}{'tok/s':>11}", flush=True)
    for arm in which:
        runs = out["arms"][arm]
        bpcs = np.array([r["bpc"] for r in runs])
        summary[arm] = dict(
            params=runs[0]["params"], flops_per_token=runs[0]["flops_per_token"],
            bpc_mean=float(bpcs.mean()), bpc_sd=float(bpcs.std(ddof=1)) if len(bpcs) > 1 else 0.0,
            bpc_min=float(bpcs.min()), bpc_max=float(bpcs.max()),
            n_seeds=len(runs),
            tok_s_mean=float(np.mean([r["tok_s"] for r in runs])),
            wall_s_mean=float(np.mean([r["wall_s"] for r in runs])),
            cum_flops_mean=float(np.mean([r["cum_flops"] for r in runs])),
            per_seed_bpc=[float(b) for b in bpcs])
        s = summary[arm]
        print(f"  {arm:<16}{s['params']:>10,}{s['flops_per_token']:>14,.0f}"
              f"{s['bpc_mean']:>11.4f}{s['bpc_sd']:>9.4f}"
              f"{s['bpc_min']:>9.4f}{s['bpc_max']:>9.4f}{s['tok_s_mean']:>11,.0f}",
              flush=True)
    out["summary"] = summary

    # ---- HARD FAILURE CONDITION: attention must clear the bigram floor -----
    a = summary.get("A_attn")
    if a is None:
        print("\n(A_attn not run; the bigram-floor check is skipped -- this is a "
              "partial run and its numbers must not be read as the headline.)",
              flush=True)
        out["verdict"] = dict(partial=True, arms_run=which)
    else:
        margin = bigram - a["bpc_mean"]
        print(f"\nA_attn {a['bpc_mean']:.4f} bpc vs bigram floor {bigram:.4f} "
              f"-> margin {margin:+.4f} bpc", flush=True)
        if margin <= 0:
            out["verdict"] = dict(attention_beats_bigram=False,
                                  attention_bpc=a["bpc_mean"],
                                  bigram_bpc=bigram, run_void=True)
            json.dump(out, open(OUT, "w"), indent=1)
            raise SystemExit(
                f"HARD FAIL: A_attn ({a['bpc_mean']:.4f} bpc) does not beat the "
                f"bigram floor ({bigram:.4f} bpc). An attention arm at the floor "
                f"has not learned to use context, so no other number here means "
                f"anything."
            )
        print("  PASS: attention clears the bigram floor.", flush=True)
        out["verdict"] = dict(attention_beats_bigram=True,
                              attention_margin_vs_bigram=float(margin),
                              bigram_bpc=float(bigram), run_void=False)

    # ---- paired comparisons ------------------------------------------------
    pairs = []
    def add_pair(name, target, base):
        if target not in summary or base not in summary:
            return
        n = min(summary[target]["n_seeds"], summary[base]["n_seeds"])
        d_ = np.array(summary[target]["per_seed_bpc"][:n]) - np.array(
            summary[base]["per_seed_bpc"][:n])
        st = paired_interval(d_)
        st.update(comparison=f"{name}: {target} - {base}",
                  target=target, base=base,
                  param_delta_pct=100.0 * (summary[target]["params"] -
                                           summary[base]["params"]) /
                                  summary[base]["params"],
                  flop_delta_pct=100.0 * (summary[target]["flops_per_token"] -
                                          summary[base]["flops_per_token"]) /
                                  summary[base]["flops_per_token"])
        pairs.append(st)
        print(f"  {st['comparison']:<44} {st['mean']:+.4f} bpc  "
              f"95% CI [{st['lo']:+.4f}, {st['hi']:+.4f}]  "
              f"sign {st['n_sign_agree']}/{st['n']}  "
              f"{'EXCLUDES 0' if st['excludes_zero'] else 'includes 0'}",
              flush=True)
        return st

    print("\n=== paired comparisons (per-seed differences) ===", flush=True)
    primary = add_pair("primary", "C_statedep", "B_match")
    add_pair("secondary", "C_statedep", "B_plain")
    add_pair("matching-device check", "B_match", "B_plain")
    add_pair("multiscale gate init", "D_multiscale", "C_statedep")
    add_pair("attention control", "A_attn", "C_statedep")
    out["paired"] = pairs
    json.dump(out, open(OUT, "w"), indent=1)

    # ---- wall-clock cost of the mechanism (FLOPs are matched; time is not) --
    print("\n=== wall-clock per training step (measured, not modelled) ===",
          flush=True)
    try:
        out["timing"] = timing(tr, va, vocab, steps=int(
            os.environ.get("CG_TIME_STEPS", "10")), bs=16)
        for arm_, t in out["timing"].items():
            print(f"  {arm_:<18} {t['ms_per_step']:8.2f} ms/step  "
                  f"{t['tok_s']:>10,.0f} tok/s", flush=True)
        c, b = out["timing"].get("C_statedep"), out["timing"].get("B_match")
        if c and b:
            tpb = out["param_table"]
            print(f"  C/B_match wall-clock ratio {c['ms_per_step']/b['ms_per_step']:.3f}x "
                  f"on equal per-token MACs ({tpb['C_statedep']['flops_per_token']:,.0f}) "
                  f"-- the sequential recurrence is the cost of state-dependence",
                  flush=True)
    except Exception as exc:      # timing must never void the learning result
        out["timing"] = dict(error=f"{type(exc).__name__}: {exc}")
        print(f"  timing failed: {exc}", flush=True)

    # ---- the recency-fit / bpc anti-correlation ---------------------------
    if do_recency:
        print("\n=== recency fit vs bpc, architecture held fixed ===", flush=True)
        try:
            out["recency"] = recency_sweep(
                tr, va, vocab, windows, steps=int(
                    os.environ.get("CG_REC_STEPS", str(steps))),
                seeds=seeds[:min(2, len(seeds))], log=print)
            rs = out["recency"]
            for arm_ in ("B_plain", "C_statedep"):
                print(f"  {arm_}: corr(cdf_l1_trained, bpc) "
                      f"{rs[f'corr_cdf_l1_trained_vs_bpc_{arm_}']:+.3f} | "
                      f"best-fit decay {rs[f'best_fit_decay_{arm_}']} | "
                      f"best-bpc decay {rs[f'best_bpc_decay_{arm_}']}", flush=True)
            out["recency_verdict"] = recency_verdict(rs)
            print(f"  {out['recency_verdict']['statement']}", flush=True)
        except Exception as exc:
            out["recency"] = dict(error=f"{type(exc).__name__}: {exc}")
            print(f"  recency sweep failed: {exc}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

    finalise(out)


def verify_causality(vocab: int, tr, *, T: int = 33, log=print) -> dict:
    """Perturbing token p must not change any logit at position < p.

    Checked for every arm, because a leak would make every arm's bpc
    meaningless, and because the state-dependent arm is the one with a new way
    to leak: its gate reads h, which already contains the past. That is fine,
    but it would be fatal if the recurrence read ahead.
    """
    ids = np.asarray(tr[:T], dtype=np.int32).copy()
    res = {}
    for arm in ARMS:
        mx.random.seed(0)
        m = build(arm, vocab)
        mx.eval(m.parameters())
        x0 = mx.array(ids[None, :])
        a = m(x0)
        for p in (T // 3, T - 3):
            pert = ids.copy()
            pert[p] = (int(pert[p]) + 7) % vocab
            b = m(mx.array(pert[None, :]))
            mx.eval(a, b)
            da = np.asarray(a, dtype=np.float64)
            db = np.asarray(b, dtype=np.float64)
            delta = float(np.max(np.abs(da[0, :p] - db[0, :p])))
            res.setdefault(arm, {})[f"p{p}"] = delta
        worst = max(res[arm].values())
        log(f"  causality {arm:<14} max |delta logits before p| = {worst:.3e} "
            f"{'OK' if worst < 1e-5 else 'LEAK'}", flush=True)
    bad = [a for a, r in res.items() if max(r.values()) >= 1e-5]
    if bad:
        raise SystemExit(f"causality check FAILED for {bad}: a change at "
                         f"position p moved logits at positions < p")
    return res


def verify_compile_equivalence(vocab: int, tr, va, *, steps: int = 30,
                               log=print) -> dict:
    """mx.compile MUST NOT change what training does.

    The failure it guards against is specific and silent: compiling a closure
    that captures the model freezes the parameters at their initial values, and
    the loss then does not move at all. A run like that costs an hour and
    reports nothing. This trains the same arm twice -- compiled and eager -- for
    `steps` steps from the same seed and requires the final losses to agree.

    Both paths consume the same batches, drawn identically, so a mismatch means
    compilation changed the computation.
    """
    res = {}
    for arm in ("C_statedep", "B_match", "A_attn"):
        outs = {}
        for mode in ("eager", "compiled"):
            mx.random.seed(0)
            model = build(arm, vocab)
            mx.eval(model.parameters())
            batch = make_batcher(tr, va)
            rng = np.random.default_rng(0)
            opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

            def loss_fn(mod, x, y):
                lo = mod(x)
                return nn.losses.cross_entropy(
                    lo.reshape(-1, vocab), y.reshape(-1), reduction="mean")

            lg = nn.value_and_grad(model, loss_fn)
            state = [model.state, opt.state]

            def _body(x, y):
                loss, grads = lg(model, x, y)
                grads, _ = optim.clip_grad_norm(grads, 1.0)
                opt.update(model, grads)
                return loss

            step = mx.compile(_body, state, state) if mode == "compiled" else None
            first, last = None, None
            for s in range(1, steps + 1):
                opt.learning_rate = 1e-3 * min(1.0, s / 10)
                x, y = batch(8, rng)
                if step is not None:
                    loss = step(x, y)
                    mx.eval(loss, model.state, opt.state)
                else:
                    loss, grads = lg(model, x, y)
                    grads, _ = optim.clip_grad_norm(grads, 1.0)
                    opt.update(model, grads)
                    mx.eval(model.parameters(), opt.state, loss)
                if s == 1:
                    first = float(loss)
                last = float(loss)
            outs[mode] = dict(first=first, last=last)
        moved = abs(outs["eager"]["last"] - outs["eager"]["first"]) > 1e-4
        agree = abs(outs["eager"]["last"] - outs["compiled"]["last"]) < 1e-4
        res[arm] = dict(**outs, loss_moves=moved, agrees=agree,
                        delta=abs(outs["eager"]["last"] - outs["compiled"]["last"]))
        log(f"  compile check {arm:<12} eager {outs['eager']['first']:.4f}->"
            f"{outs['eager']['last']:.4f} | compiled "
            f"{outs['compiled']['first']:.4f}->{outs['compiled']['last']:.4f} "
            f"| delta {res[arm]['delta']:.2e} | "
            f"{'OK' if (moved and agree) else 'MISMATCH'}", flush=True)
    bad = [a for a, r in res.items() if not (r["loss_moves"] and r["agrees"])]
    if bad:
        raise SystemExit(
            f"compile equivalence check FAILED for {bad}: compiled and eager "
            f"training disagree, or the loss did not move at all (frozen "
            f"parameters). Refusing to run."
        )
    return res


REC_DECAYS = tuple(float(x) for x in os.environ.get(
    "CG_REC_DECAYS", "0.5,0.7,0.8,0.9,0.95,0.99,0.999").split(","))


def recency_sweep(tr, va, vocab: int, windows, *, steps: int, seeds,
                  decays=REC_DECAYS,
                  log=print) -> dict:
    """Does STATE-DEPENDENCE explain the repo's 'better recency fit -> worse
    perplexity' anti-correlation?

    THE CONFOUND IN THE REPO'S OWN COMPARISON, AND HOW THIS AVOIDS IT
    ----------------------------------------------------------------
    `docs/IMPROVE.md` §2 reads the anti-correlation off a table in which the
    arms with the best kernel fits are also the arms with FOUR timescale banks.
    `single_0.8` -- the winner -- has no cdf_l1 recorded at all, and the four-bank
    arms are compared against it. So "better fit loses" and "more banks lose"
    are the same rows, and the measured anti-correlation is therefore
    confounded with bank count.

    This sweep holds the architecture FIXED at one bank and varies ONLY the gate
    bias init, which moves the static decay and hence the kernel fit. Fit error
    is then measured against the recency profile of a TRAINED attention block in
    this same harness, using the repo's own L1-of-CDF criterion
    (`fit_decay_from_attention`, llm_hybrid.py:244). Whatever correlation comes
    out is not a bank-count artifact.

    The same sweep is run in BOTH families. The repo's claimed anti-correlation
    means better fit but WORSE bpc, so fit error and bpc must be negatively
    correlated in the input-only family and not in the state-dependent family.

    SIGN CONVENTION, stated because it is easy to get backwards:
      r = corr(cdf_l1, bpc).  cdf_l1 is a FIT ERROR (lower = better fit).
      r > 0  -> better fit goes with LOWER bpc.  Fit helps.
      r < 0  -> better fit goes with HIGHER bpc.  Fit HURTS.  Anti-correlated,
                which is the repo's reported direction.
      r ~ 0  -> the static kernel's fit quality does not predict bpc at all;
                the repo's anti-correlation was not about fit.
    """
    # 1. Train attention with the standard recipe and measure ITS recency
    #    profile. The profile is a property of the trained block, so measuring it
    #    on an untrained projection would measure the wrong geometry.
    mx.random.seed(seeds[0])
    model = build("A_attn", vocab)
    mx.eval(model.parameters())
    batch = make_batcher(tr, va)
    rng = np.random.default_rng(seeds[0])
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

    def loss_fn(mod, x, y):
        lo = mod(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(model, loss_fn)
    for s in range(1, steps + 1):
        opt.learning_rate = 1e-3 * min(1.0, s / 100)
        x, y = batch(16, rng)
        loss, grads = lg(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
    prof = measure_recency_profile(model, tr[:512], layer=0, T=256)
    # normalise over the full profile we measured, not just the first 8 bins
    prof = prof / prof.sum()
    attn_bpc = evaluate(model, windows, vocab)
    log(f"  attention profile measured (trained arm, {attn_bpc:.4f} bpc); "
        f"first-4 mass {prof[:4].sum():.4f}, dominant distance "
        f"{int(np.argmax(prof))}", flush=True)

    rows = []
    for decay in decays:
        fit = cdf_l1_of_single_exponential(decay, prof)
        row = dict(decay=float(decay), cdf_l1=fit, arms={})
        for arm in ("B_plain", "C_statedep"):
            spec_bias = ARMS[arm]["bias"]
            ARMS[arm]["bias"] = str(decay)
            bpcs, gates = [], []
            for seed in seeds:
                r = run_arm(arm, tr, va, vocab, steps=steps, seed=seed,
                            windows=windows, probe=tr[:512])
                bpcs.append(r["bpc"])
                ed = r.get("effective_decay") or {}
                if ed.get("available"):
                    gates.append(ed["mean_gate"])
            ARMS[arm]["bias"] = spec_bias
            mean_gate = float(np.mean(gates)) if gates else decay
            row["arms"][arm] = dict(
                bpc_mean=float(np.mean(bpcs)),
                bpc_sd=float(np.std(bpcs, ddof=1)) if len(bpcs) > 1 else 0.0,
                per_seed=bpcs,
                mean_gate_trained=mean_gate,
                cdf_l1_init=fit,
                cdf_l1_trained=float(cdf_l1_of_single_exponential(mean_gate, prof)))
        rows.append(row)
        log(f"    decay {decay:<7} cdf_l1 {fit:7.3f} | B {row['arms']['B_plain']['bpc_mean']:.4f} "
            f"| C {row['arms']['C_statedep']['bpc_mean']:.4f}", flush=True)

    fits_init = [r["cdf_l1"] for r in rows]
    out = dict(profile_first16=[float(v) for v in prof[:16]],
               profile_mass_4=float(prof[:4].sum()),
               profile_mass_16=float(prof[:16].sum()),
               profile_dominant_distance=int(np.argmax(prof)),
               attn_bpc=float(attn_bpc),
               decays=[float(d_) for d_ in decays],
               steps=steps, seeds=list(seeds), rows=rows)
    for arm in ("B_plain", "C_statedep"):
        bpcs = [r["arms"][arm]["bpc_mean"] for r in rows]
        fits_t = [r["arms"][arm]["cdf_l1_trained"] for r in rows]
        out[f"corr_cdf_l1_trained_vs_bpc_{arm}"] = pearson(fits_t, bpcs)
        out[f"corr_cdf_l1_init_vs_bpc_{arm}"] = pearson(fits_init, bpcs)
        out[f"cdf_l1_trained_values_{arm}"] = [float(v) for v in fits_t]
        out[f"mean_gate_trained_{arm}"] = [r["arms"][arm]["mean_gate_trained"] for r in rows]
    # `recency_verdict` reads the of-record correlation; point it at the TRAINED
    # fit, because that is the kernel that produced the bpc.
    out["cdf_l1"] = fits_init
    return out


def load_avg():
    """Wall-clock numbers from a loaded machine are not trustworthy. This
    repo already records load for this reason (`experiments/fair_bench.py`);
    the same applies here, and more so: the FLOPs comparison in this file is
    computed analytically and is unaffected by load, while the timing row is
    not."""
    try:
        return round(float(os.getloadavg()[0]), 2)
    except Exception:
        return None


def timing(tr, va, vocab: int, *, steps: int = 12, bs: int = 16, seeds=(0,)):
    """Measured wall-clock per training step at matched batch. Arm C cannot use
    the parallel scan (its gate depends on h), so this quantifies the real cost
    of the mechanism rather than hiding it. FLOPs are matched by construction;
    wall clock is NOT, and is reported separately for exactly that reason.

    Load average is recorded per row and included in the output, because on a
    shared machine a 10x wall-clock ratio can be mostly contention. The ratio
    here is reported as an upper bound on the mechanism's cost unless load was
    low."""
    rows = {}
    for arm in ARMS:
        m = build(arm, vocab)
        mx.eval(m.parameters())
        batch = make_batcher(tr, va)
        rng = np.random.default_rng(0)
        opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

        def loss_fn(mod, x, y):
            lo = mod(x)
            return nn.losses.cross_entropy(
                lo.reshape(-1, vocab), y.reshape(-1), reduction="mean")

        lg = nn.value_and_grad(m, loss_fn)
        state = [m.state, opt.state]

        def _body(x, y):
            loss, grads = lg(m, x, y)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(m, grads)
            return loss

        step = mx.compile(_body, state, state) if COMPILE else None

        def one(x, y):
            if step is not None:
                loss = step(x, y)
                mx.eval(loss, m.state, opt.state)
            else:
                loss, grads = lg(m, x, y)
                grads, _ = optim.clip_grad_norm(grads, 1.0)
                opt.update(m, grads)
                mx.eval(m.parameters(), opt.state, loss)
            return loss

        x, y = batch(bs, rng)
        one(x, y)
        one(x, y)                    # second call, so compile time is amortised
        t0 = time.time()
        for _ in range(steps):
            x, y = batch(bs, rng)
            one(x, y)
        dt = (time.time() - t0) / steps
        rows[arm] = dict(ms_per_step=dt * 1e3, tok_s=bs * CTX / dt,
                         params=n_params(m), compiled=bool(step),
                         load_avg=load_avg())
    return rows


def recency_verdict(rs: dict) -> dict:
    """Turn the two correlations into a plain statement about the hypothesis.

    The repo's reported finding is "better recency fit -> worse perplexity"
    (`docs/IMPROVE.md` §2). The hypothesis under test here is that this is
    explained by the gate being input-only. That predicts:
      * the anti-correlation (r < 0) is PRESENT in B (input-only), and
      * ABSENT in C (state-dependent), because the static kernel no longer
        decides the timescale.
    """
    rb = rs.get("corr_cdf_l1_trained_vs_bpc_B_plain",
               rs.get("corr_cdf_l1_vs_bpc_B_plain"))
    rc = rs.get("corr_cdf_l1_trained_vs_bpc_C_statedep",
                rs.get("corr_cdf_l1_vs_bpc_C_statedep"))
    out = dict(corr_input_only=rb, corr_state_dep=rc)
    if rb is None or rc is None:
        out["statement"] = "recency verdict unavailable (sweep incomplete)"
        return out
    floor = rs.get("meta", {}).get("bigram_bpc")
    if (rs.get("steps", 0) < 100 or len(rs.get("seeds", [])) < 2
            or len(rs.get("decays", [])) < 3 or floor is None
            or rs.get("attn_bpc", float("inf")) >= floor):
        out["valid_run"] = False
        out["explains_anti_correlation"] = False
        out["statement"] = (
            "INVALID recency verdict: require at least 100 steps, two seeds, "
            "three decays, and a trained attention arm below the bigram floor")
        return out
    out["valid_run"] = True
    anti_b = rb < -0.3
    anti_c = rc < -0.3
    out["anti_correlation_present_input_only"] = bool(anti_b)
    out["anti_correlation_present_state_dep"] = bool(anti_c)
    if anti_b and not anti_c:
        out["explains_anti_correlation"] = True
        out["statement"] = (
            f"SUPPORTED: anti-correlation present with the input-only gate "
            f"(r={rb:+.3f}) and absent with the state-dependent gate "
            f"(r={rc:+.3f}); the anti-correlation tracks the gate, not the kernel.")
    elif anti_b and anti_c:
        out["explains_anti_correlation"] = False
        out["statement"] = (
            f"REFUTED (F3): anti-correlation present in BOTH families "
            f"(r={rb:+.3f} input-only, r={rc:+.3f} state-dependent), so "
            f"state-dependence does NOT explain it.")
    elif rb > 0.3:
        out["explains_anti_correlation"] = False
        out["statement"] = (
            f"REVERSED: input-only r={rb:+.3f} means better kernel fit goes "
            f"with better bpc, the opposite of the repo's anti-correlation.")
    elif not anti_b:
        out["explains_anti_correlation"] = False
        out["statement"] = (
            f"INCONCLUSIVE: input-only r={rb:+.3f} does not reproduce the "
            f"claimed anti-correlation in this one-bank harness.")
    else:
        out["explains_anti_correlation"] = False
        out["statement"] = f"mixed: r={rb:+.3f} (input-only), r={rc:+.3f} (state-dep)"
    return out


def finalise(out, out_path=OUT):
    """Write the verdict block, print it, and persist."""
    bigram = out["meta"]["bigram_bpc"]
    summary = out.get("summary", {})
    primary = next((p for p in out.get("paired", [])
                    if p["comparison"].startswith("primary")), None)
    line = {}
    if primary:
        line["primary_comparison"] = primary["comparison"]
        line["primary_mean_diff_bpc"] = primary["mean"]
        line["primary_ci95"] = [primary["lo"], primary["hi"]]
        line["primary_excludes_zero"] = primary["excludes_zero"]
        line["primary_n_sign_agree"] = f"{primary['n_sign_agree']}/{primary['n']}"
        # the hypothesis is "C is BETTER", i.e. LOWER bpc -> negative difference
        if primary["excludes_zero"] and primary["mean"] < 0:
            line["C_beats_B_match"] = True
            line["hypothesis"] = ("SUPPORTED: state-dependent gate beats the "
                                  "input-only gate at matched params and matched "
                                  "per-token FLOPs; interval excludes 0")
        elif primary["excludes_zero"] and primary["mean"] > 0:
            line["C_beats_B_match"] = False
            line["hypothesis"] = ("REFUTED, IN THE OPPOSITE DIRECTION: the "
                                  "state-dependent gate is reliably WORSE")
        else:
            line["C_beats_B_match"] = False
            line["hypothesis"] = ("REFUTED: the paired 95% interval on "
                                  "C_statedep - B_match includes 0, so adding "
                                  "state-dependence is not measurably better")
    if "A_attn" in summary:
        line["attention_bpc"] = summary["A_attn"]["bpc_mean"]
        line["attention_margin_vs_bigram"] = bigram - summary["A_attn"]["bpc_mean"]
    rs = out.get("recency") or {}
    for arm in ("B_plain", "C_statedep"):
        k = f"corr_cdf_l1_vs_bpc_{arm}"
        if k in rs:
            line[k] = rs[k]
    out["verdict"].update(line)
    json.dump(out, open(out_path, "w"), indent=1)

    print("\n=== VERDICT ===", flush=True)
    if primary:
        print(f"primary: {primary['comparison']}", flush=True)
        print(f"  mean {primary['mean']:+.4f} bpc, 95% CI "
              f"[{primary['lo']:+.4f}, {primary['hi']:+.4f}], "
              f"sign {primary['n_sign_agree']}/{primary['n']}", flush=True)
        print(f"  C BEATS B_match: {line.get('C_beats_B_match')}", flush=True)
        print(f"  {line.get('hypothesis','')}", flush=True)
    if rs:
        print(f"  corr(cdf_l1_trained, bpc) | input-only (B): "
              f"{rs.get('corr_cdf_l1_trained_vs_bpc_B_plain', float('nan')):+.3f}  "
              f"| state-dep (C): {rs.get('corr_cdf_l1_trained_vs_bpc_C_statedep', float('nan')):+.3f}",
              flush=True)
    print(f"wrote {out_path}", flush=True)
    return out


def recency_main():
    """Second question, same harness: is the repo's recency-fit/bpc
    anti-correlation explained by the gate being input-only?

    Run as `CG_MODE=recency python3 experiments/content_gate.py`. Bank count is
    fixed at 1 for every row, so the measured correlation cannot be the
    bank-count confound described in `recency_sweep`'s docstring.
    """
    steps = int(os.environ.get("CG_REC_STEPS", "800"))
    seeds = [int(s) for s in os.environ.get("CG_REC_SEEDS", "0,1").split(",")]
    decays = REC_DECAYS
    out_path = os.environ.get("CG_REC_OUT", os.path.join(
        HERE, "results", "content_gate_recency.json"))

    print(f"device {mx.default_device()} | ONE bank, architecture fixed | "
          f"steps={steps} seeds={seeds}", flush=True)
    print(f"decays swept (gate-bias init): {decays}", flush=True)
    print("sign convention: cdf_l1 is a FIT ERROR (lower = better fit). "
          "r>0 fit helps; r<0 fit hurts (the repo's reported direction)\n",
          flush=True)

    err = verify_scan()
    print(f"scan verification: {err:.3e}", flush=True)
    if err > 1e-4:
        raise SystemExit("scan verification failed; aborting")

    data, vocab = load_corpus(CORPUS)
    tr, va = make_splits(data)
    causality = verify_causality(vocab, tr)
    windows = val_windows(va)
    bigram, unigram = floors(data)
    print(f"corpus {len(data):,} vocab {vocab} | {len(windows)} deterministic "
          f"val windows ({sum(int(y.size) for _, y in windows):,} tokens)",
          flush=True)
    print(f"unigram {unigram:.4f} bpc | BIGRAM FLOOR {bigram:.4f} bpc\n",
          flush=True)

    t0 = time.time()
    res = recency_sweep(tr, va, vocab, windows, steps=steps, seeds=seeds,
                        decays=decays, log=print)
    res["meta"] = dict(
        host=platform.platform(), driver="experiments/content_gate.py "
        "(CG_MODE=recency)", steps=steps, seeds=seeds, bank=1, ctx=CTX, d=D,
        n_layer=N_LAYER, val_windows=len(windows), bigram_bpc=bigram,
        unigram_bpc=unigram, scan_verify_max_abs_err=err,
        causality_max_delta=causality,
        criterion="cdf_l1 of a single exponential vs the TRAINED attention "
                  "recency profile (same criterion as llm_hybrid.py:244), "
                  "one bank, architecture and recipe fixed across rows",
        confound_removed="the repo's table mixes bank count with decay "
                         "placement; here bank=1 for every row",
        elapsed_s=time.time() - t0,
        preregistered_falsifier=(
            "r<0 in BOTH families -> state-dependence does NOT explain the "
            "anti-correlation (F3); r~0 in both -> the anti-correlation was a "
            "bank-count artifact, not a kernel property"))
    res["verdict"] = recency_verdict(res)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print("\n=== verdict ===", flush=True)
    print(res["verdict"]["statement"], flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    if os.environ.get("CG_MODE", "").strip() == "recency":
        recency_main()
    else:
        main()
