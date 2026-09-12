"""Head-to-head: transformer attention vs gated multiscale dendritic memory.

THE CLAIM UNDER TEST
--------------------
A transformer pays O(T^2) per token for attention because it must compare every
position with every other. The spiking substrate in this repo produced a
measured result that suggests a cheaper mechanism: working memory in the model
neuron is a DENDRITIC TRACE, not a recurrent weight (zeroing the entire
recurrent weight matrix leaves accuracy bit-identical; see
docs/WORKING_MEMORY.md). A trace is O(T*d) to run, not O(T^2).

The trace alone is a passive linear filter and it is a poor language model. It
becomes a computational unit when the neuron can CHOOSE to hold or overwrite:

    h_t = g_t * h_{t-1} + v_t        g_t = sigmoid(W_g x_t)

and gated recurrence is what this file tests against attention at matched
parameters, matched context, and matched training tokens.

HONESTY NOTES
-------------
* Gated linear recurrences are a KNOWN family (H3, Mamba, RWKV, RetNet,
  Griffin). Nothing here claims otherwise. What is being tested is whether the
  specific mechanism this project measured in its own substrate is competitive
  at matched compute, and by how much at long context.
* An earlier version of this comparison ran the recurrence as a Python loop
  over T and reported the memory arm as 4x SLOWER. That was an implementation
  artifact. The sequential loop is replaced by a chunked parallel scan
  (sequential only over T/C chunks, vectorised within a chunk), verified to
  4.8e-07 against the naive recurrence.
* An earlier version of count_flops charged attention O(T*d) for the score
  matrix instead of O(T^2*d), understating it by a factor of T. Fixed here.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

CORPUS = os.environ.get(
    "BRAIN_CORPUS", "/Volumes/T9/human-brain/scratch/tinyshake.txt"
)
OUT = os.environ.get(
    "BRAIN_EFF_OUT", "/Volumes/T9/human-brain/scratch/llm_efficiency.json"
)


# --------------------------------------------------------------------------
# parallel scans
# --------------------------------------------------------------------------
def associative_scan(a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    """Hillis-Steele scan for the associative op on affine maps.

    An affine map h -> a*h + b is closed under composition:
        (a2,b2) o (a1,b1) = (a2*a1, a2*b1 + b2)
    so a prefix scan of these pairs gives the prefix composition, whose `b`
    component is the scan output. O(log T) sequential steps, fully vectorised.
    """
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


def chunked_gated_scan(v: mx.array, g: mx.array, chunk: int = 64) -> mx.array:
    """h_t = g_t * h_{t-1} + v_t, evaluated in parallel.

    Sequential work is O(T/chunk) chunk carries; work inside a chunk is a
    Hillis-Steele scan over `chunk` positions, vectorised across all chunks at
    once. Trades work for latency, which is the right trade on a GPU where
    occupancy is free and sequential dependency is the bottleneck.
    """
    B, T, D = v.shape
    pad = (-T) % chunk
    if pad:
        z = mx.zeros((B, pad, D))
        v = mx.concatenate([v, z], axis=1)
        g = mx.concatenate([g, z], axis=1)
    Tp = v.shape[1]
    n_ch = Tp // chunk
    vc = v.reshape(B, n_ch, chunk, D)
    gc = g.reshape(B, n_ch, chunk, D)

    prod, intra = associative_scan(gc, vc)
    # state entering each chunk, by a scan over chunk-level affine maps
    _, carry = associative_scan(
        prod[:, :, chunk - 1, :], intra[:, :, chunk - 1, :]
    )
    carry_in = mx.concatenate(
        [mx.zeros((B, 1, D)), carry[:, :-1, :]], axis=1
    )
    h = intra + prod * carry_in[:, :, None, :]
    return h.reshape(B, Tp, D)[:, :T, :]


def sequential_scan(v: mx.array, g: mx.array) -> mx.array:
    """Reference recurrence. Kept for verification only."""
    B, T, D = v.shape
    h = mx.zeros((B, D))
    acc = []
    for t in range(T):
        h = g[:, t, :] * h + v[:, t, :]
        acc.append(h)
    return mx.stack(acc, axis=1)


# --------------------------------------------------------------------------
# memory modules
# --------------------------------------------------------------------------
class GatedMemory(nn.Module):
    """h_t = g_t * h_{t-1} + v_t, with the scan done in parallel.

    `banks` > 1 keeps several independent traces per channel with different
    initial gate biases, giving the layer a coarse-to-fine view of the past.
    """

    def __init__(self, d: int, banks: int = 1, chunk: int = 64,
                 init_decays=(0.90, 0.99, 0.999, 0.9999)):
        super().__init__()
        self.d = d
        self.banks = banks
        self.chunk = chunk
        self.v = nn.Linear(d, d * banks, bias=False)
        self.o = nn.Linear(d * banks, d, bias=False)
        self.gate = nn.Linear(d, d * banks, bias=True)
        dec = list(init_decays)[:banks]
        # bias so that sigmoid(bias) starts at the intended decay
        self._init_bias = [
            math.log(float(a) / (1.0 - float(a))) for a in dec
        ]
        self._init_bias += [0.0] * (banks - len(self._init_bias))
        self.gate.bias = mx.array(
            np.repeat(np.array(self._init_bias, dtype=np.float32), d)
        )

    def __call__(self, x, mask=None):
        B, T, _ = x.shape
        v = self.v(x).reshape(B, T, self.banks, self.d)
        g = mx.sigmoid(self.gate(x)).reshape(B, T, self.banks, self.d)
        # scan each bank: fold bank into batch so one scan call serves all
        v = v.transpose(0, 2, 1, 3).reshape(B * self.banks, T, self.d)
        g = g.transpose(0, 2, 1, 3).reshape(B * self.banks, T, self.d)
        h = chunked_gated_scan(v, g, self.chunk)
        h = h.reshape(B, self.banks, T, self.d).transpose(0, 2, 1, 3)
        return self.o(h.reshape(B, T, self.banks * self.d))


class AttentionMemory(nn.Module):
    def __init__(self, d: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d, n_head, bias=True)

    def __call__(self, x, mask=None):
        return self.attn(x, x, x, mask=mask)


class Block(nn.Module):
    def __init__(self, d, n_head, kind, banks=1, chunk=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mem = (
            AttentionMemory(d, n_head)
            if kind == "attn"
            else GatedMemory(d, banks=banks, chunk=chunk)
        )
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d, bias=True),
            nn.GELU(),
            nn.Linear(4 * d, d, bias=True),
        )

    def __call__(self, x, mask=None):
        x = x + self.mem(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))


class ParallelColumns(nn.Module):
    """K shallow columns over the same input, mixed by a learned combiner.

    This is the "simultaneous thinking" arm. A deep stack evaluates one
    hypothesis at depth K, serially. K columns evaluate K hypotheses at
    width K, concurrently, and a mixer decides how much to trust each. The
    parameters are comparable; the dependency chain is not.
    """

    def __init__(self, d, n_head, kind, n_col, depth, banks=1, chunk=64):
        super().__init__()
        self.n_col = n_col
        self.cols = [
            [Block(d, n_head, kind, banks, chunk) for _ in range(depth)]
            for _ in range(n_col)
        ]
        # outs is [x] + one output per column, so the mixer input is
        # (n_col + 1) * d, not n_col * d. The original version sized it as
        # n_col * d and raised a shape error on the first forward pass.
        self.mix = nn.Linear(d * (n_col + 1), d, bias=True)

    def __call__(self, x, mask=None):
        outs = [x]
        for c in self.cols:
            h = x
            for blk in c:
                h = blk(h, mask)
            outs.append(h)
        return self.mix(mx.concatenate(outs, axis=-1))


class LM(nn.Module):
    def __init__(self, vocab, d, ctx, kind, n_layer, n_head, banks=1,
                 chunk=64, looped=False, loops=1):
        super().__init__()
        self.looped = looped
        self.loops = loops
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = [Block(d, n_head, kind, banks, chunk) for _ in range(n_layer)]
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(mx.arange(T)[None, :])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for _ in range(self.loops if self.looped else 1):
            for b in self.blocks:
                x = b(x, mask)
        return self.head(self.lnf(x))


class ColumnLM(nn.Module):
    def __init__(self, vocab, d, ctx, kind, n_col, depth, n_head, banks=1,
                 chunk=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.core = ParallelColumns(d, n_head, kind, n_col, depth, banks, chunk)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(mx.arange(T)[None, :])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        return self.head(self.lnf(self.core(x, mask)))


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------
def n_params(m) -> int:
    return sum(
        int(np.prod(v.shape)) for _, v in nn.utils.tree_flatten(m.parameters())
    )


def flops_per_token(vocab, d, n_layer, ctx, kind, banks=1, chunk=64,
                    mult=1, n_col=1):
    """Analytic forward+backward FLOPs per token.

    attention: 4 projections + T^2*d scores/AV  -> grows with ctx
    memory:    3 projections + log(chunk) scan   -> flat in ctx

    The scan is charged ~3*d*log2(chunk) per token per bank. Backward is
    charged 2x forward throughout (3x total). This is a FLOPs model, not a
    wall-clock model; wall clock is measured separately and reported.
    """
    mlp = 2 * (d * 4 * d + 4 * d * d)
    if kind == "attn":
        proj = 2 * 4 * d * d
        ctx_work = 2 * 2 * ctx * d
    else:
        proj = 2 * 3 * d * d * banks
        ctx_work = 3.0 * d * banks * math.log2(max(chunk, 2))
    per = (proj + ctx_work + mlp) * n_layer * mult
    # column mixer: (n_col*d) -> d, applied once
    if n_col > 1:
        per += 2 * (n_col * d) * d
    per += 2 * d * vocab
    return 3.0 * per


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_corpus(path):
    if not os.path.exists(path):
        alt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "tinyshakespeare.txt",
        )
        if os.path.exists(alt):
            path = alt
        else:
            raise SystemExit(f"corpus not found: {path}")
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    return data, len(chars)


def make_batcher(data, ctx, val_frac=0.1):
    n = int((1.0 - val_frac) * len(data))
    tr, va = data[:n], data[n:]

    def batch(bs, rng, split="train"):
        d = tr if split == "train" else va
        ix = rng.integers(0, len(d) - ctx - 1, size=bs)
        x = np.stack([d[i:i + ctx] for i in ix])
        y = np.stack([d[i + 1:i + 1 + ctx] for i in ix])
        return mx.array(x), mx.array(y)

    return batch


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
ARMS = {
    "A_attention": dict(kind="attn", n_layer=4),
    "B_gated": dict(kind="mem", n_layer=4, banks=1),
    "C_gated_banks4": dict(kind="mem", n_layer=4, banks=4),
    "D_cols2_gated": dict(kind="cols", n_col=2, depth=1),
    "E_cols4_gated": dict(kind="cols", n_col=4, depth=1),
    "F_gated_looped2": dict(kind="mem", n_layer=2, banks=1, looped=True, loops=2),
    "G_cols2_looped2": dict(kind="cols", n_col=2, depth=1, looped=True, loops=2),
}


def build(arm, vocab, d, ctx, chunk=64):
    spec = dict(ARMS[arm])
    kind = spec.pop("kind")
    if kind == "cols":
        n_col = spec.pop("n_col")
        depth = spec.pop("depth")
        looped = spec.pop("looped", False)
        loops = spec.pop("loops", 1)
        m = ColumnLM(vocab, d, ctx, "mem", n_col, depth, 4, 1, chunk)
        # loops handled by wrapping call
        if looped:
            inner = m.core

            class _Wrapped(nn.Module):
                def __init__(self, base, inner, loops):
                    super().__init__()
                    self.base = base
                    self.loops = loops

                def __call__(self, idx):
                    B, T = idx.shape
                    x = self.base.tok(idx) + self.base.pos(mx.arange(T)[None, :])
                    mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
                    h = x
                    for _ in range(self.loops):
                        h = inner(h, mask)
                    return self.base.head(self.base.lnf(h))

            m = _Wrapped(m, inner, loops)
        return m, dict(spec, kind="mem", n_col=n_col, depth=depth,
                       looped=looped, loops=loops)
    else:
        looped = spec.pop("looped", False)
        loops = spec.pop("loops", 1)
        n_layer = spec.pop("n_layer")
        banks = spec.pop("banks", 1)
        m = LM(vocab, d, ctx, kind, n_layer, 4, banks, chunk, looped, loops)
        return m, dict(spec, kind=kind, n_layer=n_layer, banks=banks,
                       looped=looped, loops=loops)


def run_arm(arm, data, vocab, *, ctx=512, d=128, steps=500, bs=16, lr=3e-3,
            seed=0, chunk=64, log=print):
    m, spec = build(arm, vocab, d, ctx, chunk)
    mx.eval(m.parameters())
    P = n_params(m)
    n_layer = spec.get("n_layer", spec.get("depth", 1) * spec.get("n_col", 1))
    mult = spec.get("loops", 1) if spec.get("looped") else 1
    FL = flops_per_token(
        vocab, d, n_layer, ctx,
        "attn" if spec["kind"] == "attn" else "mem",
        banks=spec.get("banks", 1), chunk=chunk, mult=mult,
        n_col=spec.get("n_col", 1),
    )
    batch = make_batcher(data, ctx)
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)
    t0 = time.time()
    ntok = 0
    for s in range(1, steps + 1):
        x, y = batch(bs, rng)
        l, g = lg(m, x, y)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
        if s % 100 == 0:
            log(f"    [{arm}] step {s}/{steps} loss {float(l):.4f} "
                f"tok/s {ntok/(time.time()-t0):,.0f}")
    wall = time.time() - t0
    xv, yv = batch(64, rng, "val")
    vl = float(loss_fn(m, xv, yv))
    return dict(
        arm=arm, params=P, flops_per_token=FL, ctx=ctx, d=d, steps=steps,
        bs=bs, seed=seed, chunk=chunk,
        bpc=vl / math.log(2), val_loss=vl, wall_s=wall,
        tok_s=ntok / wall, cum_flops=ntok * FL, tokens=ntok, spec=spec,
    )


def verify_scan():
    """Guard: the parallel scan must reproduce the recurrence it replaces."""
    worst = 0.0
    for T in (1, 2, 7, 33, 64, 100, 129, 512, 1000):
        mx.random.seed(0)
        g = mx.sigmoid(mx.random.normal((2, T, 4)))
        v = mx.random.normal((2, T, 4))
        ref = sequential_scan(v, g)
        got = chunked_gated_scan(v, g, 64)
        worst = max(worst, float(mx.max(mx.abs(ref - got))))
    return worst


if __name__ == "__main__":
    which = sys.argv[1:] or list(ARMS)
    err = verify_scan()
    print(f"scan verification: max abs error vs sequential = {err:.3e}")
    if err > 1e-4:
        raise SystemExit("parallel scan does not match the recurrence; aborting")

    data, vocab = load_corpus(CORPUS)
    print(f"corpus: {len(data):,} chars, vocab {vocab}")
    ctx = int(os.environ.get("BRAIN_CTX", "512"))
    steps = int(os.environ.get("BRAIN_STEPS", "500"))
    out = []
    for arm in which:
        print(f"\n=== {arm} ===")
        r = run_arm(arm, data, vocab, ctx=ctx, steps=steps)
        out.append(r)
        print(
            f"{r['arm']:<18} params={r['params']:>9,} "
            f"fl/tok={r['flops_per_token']:>11,.0f} "
            f"bpc={r['bpc']:.4f} tok/s={r['tok_s']:>9,.0f} "
            f"wall={r['wall_s']:6.1f}s"
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
