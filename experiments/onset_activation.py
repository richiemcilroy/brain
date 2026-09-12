"""A brain-derived activation: band-pass detection instead of monotonic gating.

WHERE THIS COMES FROM
---------------------
This is the one architectural idea in this repo that is derived from a
measurement on our own substrate rather than borrowed from the sequence-model
literature. `docs/NEURON_OPERATING_POINT.md` established, with real spikes, that
the dCaAP neuron here is a **transient onset detector**: its transfer function

    phi(z) = z * exp(1 - z)

peaks at z = 1 and DECAYS ON BOTH SIDES. It fires once, at ms 12-14 of a 15 ms
window, and then stops responding even if the drive persists.

That is a band-pass / scale-selective nonlinearity. Every activation in
mainstream sequence models is MONOTONIC: ReLU, GELU, SiLU, and the SiLU-based
gates in Mamba/H3/RWKV all increase with input magnitude forever. A monotonic
activation cannot represent "this input is the right SIZE"; it can only
represent "this input is big enough". A band-pass unit responds to a preferred
magnitude and is suppressed by both too-weak and too-strong input.

WHAT IS TESTED
--------------
Swap the MLP's GELU for phi, with NO other change, in the gated-memory arm and
in the attention arm. Arms:

  gelu        h = gelu(W1 x),  W2 h                    (baseline, monotonic)
  onset       h = phi(W1 x),   W2 h                    (band-pass)
  onset_bank  h = phi_b(W1 x) for B different scales b,
              concatenated then W2                   (multiscale band-pass:
                                                     a bank of detectors tuned
                                                     to different magnitudes --
                                                     this is the multiscale idea
                                                     applied where it has not
                                                     been refuted, i.e. to the
                                                     activation rather than to
                                                     the dendritic time constant)
  mix         half the hidden units gelu, half phi     (is any gain from the
                                                       nonlinearity, or is it
                                                       just more capacity?)

`onset_bank` is scale-matched in parameter count to `gelu` by making each of the
B banks narrower, so the comparison is capacity-neutral.

PRE-REGISTERED PREDICTION, stated before running: I expect `onset` alone to be
NEUTRAL or slightly WORSE than gelu, because a monotonic activation is a strict
superset in terms of the functions it can represent cheaply and the network has
to learn the right input scale. I expect `onset_bank` to be the only arm that
could win, and only if the task actually rewards scale-selectivity. **If no arm
beats gelu, that is a clean negative result and the correct conclusion is that
the substrate's most distinctive nonlinearity does not transfer to this task.**
That would be worth knowing and I will report it as such.
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
from llm_efficiency import load_corpus, CORPUS  # noqa: E402
from llm_tuned import floors  # noqa: E402

OUT = "/Volumes/T9/human-brain/scratch/onset_activation.json"


def phi(z, scale=1.0):
    """The measured dCaAP transfer function: peaks at z*scale == 1.

    phi(z) = z * exp(1 - z) has its maximum value of exactly 1.0 at z = 1, and
    falls to 0 at z = 0 and to ~0 as z grows. `scale` shifts the preferred
    input magnitude, giving a bank of detectors tuned to different sizes.
    """
    zz = z * scale
    return zz * mx.exp(1.0 - zz)


class OnsetMLP(nn.Module):
    """MLP whose hidden nonlinearity is the onset detector, optionally banked.

    `banks=1, onset=False` reproduces the standard GELU MLP.
    """

    def __init__(self, d, hidden_mult=4, banks=1, onset=False, init_scales=None):
        super().__init__()
        self.banks = banks
        self.onset = onset
        total_hidden = hidden_mult * d
        # equal parameter count regardless of bank count
        self.hidden = max(1, total_hidden // banks)
        self.w1 = nn.Linear(d, self.hidden * banks, bias=True)
        self.w2 = nn.Linear(self.hidden * banks, d, bias=True)
        if init_scales is None:
            init_scales = [2.0 ** (k - (banks - 1) / 2.0) for k in range(banks)]
        self._scales = [float(s) for s in init_scales]

    def __call__(self, x):
        h = self.w1(x)                                  # (..., hidden*banks)
        if not self.onset:
            return self.w2(nn.gelu(h))
        B = self.banks
        shape = h.shape[:-1] + (B, self.hidden)
        h = h.reshape(shape)
        outs = [phi(h[..., k, :], self._scales[k]) for k in range(B)]
        return self.w2(mx.concatenate(outs, axis=-1).reshape(h.shape[:-2] + (-1,)))


class Block(nn.Module):
    def __init__(self, d, kind, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.kind = kind
        if kind == "attn":
            self.mem = nn.MultiHeadAttention(d, 4, bias=True)
        else:
            self.mem = GatedMem(d)
        self.mlp = mlp

    def __call__(self, x, mask=None):
        if self.kind == "attn":
            x = x + self.mem(self.ln1(x), self.ln1(x), self.ln1(x), mask=mask)
        else:
            x = x + self.mem(self.ln1(x))
        return x + self.mlp(self.ln2(x))


def associative_scan(a, b):
    T = a.shape[-2]
    step = 1
    while step < T:
        a_s = mx.concatenate([mx.ones_like(a[..., :step, :]), a[..., :T - step, :]], axis=-2)
        b_s = mx.concatenate([mx.zeros_like(b[..., :step, :]), b[..., :T - step, :]], axis=-2)
        b = a * b_s + b
        a = a * a_s
        step *= 2
    return a, b


def chunked_scan(v, g, chunk=64):
    Bt, T, D = v.shape
    pad = (-T) % chunk
    if pad:
        z = mx.zeros((Bt, pad, D))
        v = mx.concatenate([v, z], axis=1)
        g = mx.concatenate([g, z], axis=1)
    Tp = v.shape[1]
    nch = Tp // chunk
    vc = v.reshape(Bt, nch, chunk, D)
    gc = g.reshape(Bt, nch, chunk, D)
    prod, intra = associative_scan(gc, vc)
    _, carry = associative_scan(prod[:, :, chunk - 1, :], intra[:, :, chunk - 1, :])
    carry_in = mx.concatenate([mx.zeros((Bt, 1, D)), carry[:, :-1, :]], axis=1)
    h = intra + prod * carry_in[:, :, None, :]
    return h.reshape(Bt, Tp, D)[:, :T, :]


class GatedMem(nn.Module):
    def __init__(self, d, chunk=64, init_decay=0.99):
        super().__init__()
        self.chunk = chunk
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=True)
        self.gate.bias = mx.full((d,), math.log(init_decay / (1 - init_decay)))

    def __call__(self, x, mask=None):
        v = self.v(x)
        g = mx.sigmoid(self.gate(x))
        return self.o(chunked_scan(v, g, self.chunk))


class LM(nn.Module):
    def __init__(self, vocab, d, ctx, kind, n_layer, mlp_factory):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = [Block(d, kind, mlp_factory(d)) for _ in range(n_layer)]
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def __call__(self, idx):
        Bt, T = idx.shape
        x = self.tok(idx) + self.pos(mx.arange(T)[None, :])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.lnf(x))


def make_mlp(arm):
    def f(d):
        if arm == "gelu":
            return OnsetMLP(d, 4, 1, False)
        if arm == "onset":
            return OnsetMLP(d, 4, 1, True)
        if arm == "onset_bank":
            return OnsetMLP(d, 4, 4, True)
        if arm == "onset_bank8":
            return OnsetMLP(d, 4, 8, True)
        if arm == "mix":
            return OnsetMLP(d, 4, 1, True)   # placeholder, replaced below
        raise ValueError(arm)
    return f


def nparams(m):
    return sum(int(np.prod(v.shape)) for _, v in nn.utils.tree_flatten(m.parameters()))


def det_val(data, ctx, frac=0.1):
    n = int((1.0 - frac) * len(data))
    va = data[n:]
    nw = (len(va) - 1) // ctx
    x = np.stack([va[i * ctx:i * ctx + ctx] for i in range(nw)])
    y = np.stack([va[i * ctx + 1:i * ctx + 1 + ctx] for i in range(nw)])
    return mx.array(x), mx.array(y)


def evaluate(m, xv, yv, vocab, bs=32):
    tot, n = 0.0, 0
    for i in range(0, xv.shape[0], bs):
        lo = m(xv[i:i + bs])
        l = nn.losses.cross_entropy(lo.reshape(-1, vocab),
                                    yv[i:i + bs].reshape(-1), reduction="sum")
        mx.eval(l)
        tot += float(l)
        n += int(yv[i:i + bs].size)
    return tot / n


def run(arm, kind, data, vocab, ctx=512, steps=1500, bs=16, lr=1e-3, seed=0):
    mx.random.seed(seed)
    m = LM(vocab, 128, ctx, kind, 2, make_mlp(arm))
    mx.eval(m.parameters())
    P = nparams(m)
    n = int(0.9 * len(data))
    tr = data[:n]
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def lf(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(lo.reshape(-1, vocab), y.reshape(-1),
                                       reduction="mean")

    lg = nn.value_and_grad(m, lf)
    xv, yv = det_val(data, ctx)
    t0 = time.time()
    ntok = 0
    for s in range(1, steps + 1):
        opt.learning_rate = lr * min(1.0, s / 100)
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + ctx] for i in ix]))
        y = mx.array(np.stack([tr[i + 1:i + 1 + ctx] for i in ix]))
        l, g = lg(m, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
    wall = time.time() - t0
    return dict(arm=arm, kind=kind, seed=seed, params=P,
                val_bpc=evaluate(m, xv, yv, vocab) / math.log(2),
                tok_s=ntok / wall)


if __name__ == "__main__":
    data, vocab = load_corpus(CORPUS)
    bg, ug = floors(data)
    print(f"unigram {ug:.4f}  BIGRAM {bg:.4f} bpc")
    print("phi(z) = z*exp(1-z) peaks at z=1; gelu is monotonic\n")
    arms = os.environ.get("BRAIN_ARMS", "gelu,onset,onset_bank,onset_bank8").split(",")
    seeds = [int(s) for s in os.environ.get("BRAIN_SEEDS", "0,1").split(",")]
    kinds = os.environ.get("BRAIN_KINDS", "mem,attn").split(",")
    out = []
    for kind in kinds:
        for arm in arms:
            for seed in seeds:
                r = run(arm, kind, data, vocab, seed=seed)
                out.append(r)
                print(f"  {kind:<5} {arm:<13} seed={seed} params={r['params']:>8,} "
                      f"val={r['val_bpc']:.4f} ({bg-r['val_bpc']:+.4f} vs floor) "
                      f"tok/s={r['tok_s']:>8,.0f}", flush=True)
                json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== SUMMARY (mean over seeds) ===")
    print(f"{'kind':<6}{'arm':<14}{'params':>9}{'val bpc':>10}{'vs gelu':>10}")
    for kind in kinds:
        base = {}
        for arm in arms:
            rs = [r for r in out if r["arm"] == arm and r["kind"] == kind]
            if not rs:
                continue
            m = np.mean([r["val_bpc"] for r in rs])
            base[arm] = m
        for arm in arms:
            if arm not in base:
                continue
            rs = [r for r in out if r["arm"] == arm and r["kind"] == kind]
            d = base["gelu"] - base[arm]
            print(f"{kind:<6}{arm:<14}{rs[0]['params']:>9,}{base[arm]:>10.4f}"
                  f"{d:>+10.4f}")
    print(f"\nbigram floor = {bg:.4f} bpc")
    print(f"wrote {OUT}")
