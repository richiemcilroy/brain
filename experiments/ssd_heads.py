"""THE EFFICIENCY FIX: give up per-channel gating so the context op becomes a MATMUL.

THE DIAGNOSIS (all measured, this repo, M4 Max / MLX 0.31)
---------------------------------------------------------
The hardware is dispatch-bound at small sizes. Pure matmul throughput:

    n=64   ->      2.8 GMAC/s      n=512  ->    909.6 GMAC/s
    n=128  ->     25.2 GMAC/s      n=2048 ->  9,002.0 GMAC/s
    n=256  ->    181.9 GMAC/s      n=4096 -> 10,107.8 GMAC/s

That is a 3,600x efficiency range as a function of SIZE, at fixed FLOPs. A
transformer is "efficient" for one reason: its context operation is one big
matmul. Our gated scan is ~100 small elementwise kernels, so it runs at the
bottom of that range. Measured: our memory block achieves ~1 GMAC/s while the
same machine does 10,108 GMAC/s on a large matmul.

WHY PER-CHANNEL GATING FORCES THE BAD REGIME
--------------------------------------------
With a gate vector g_t in R^D, the intra-chunk map is
    A_ij = exp(L_i - L_j)  applied PER CHANNEL,
so the operator is (B, n, C, C, D) -- quadratic in chunk size AND linear in D.
Measured: reformulating the per-channel scan this way is 10x SLOWER than the
scan it replaces (0.15x at T=512, 0.10x at T=2048/8192), because the
intermediate is ~2 GB. Dead end, and now recorded as one.

THE FIX
-------
If the gate is a SCALAR per head, the operator becomes (B, n, C, C) -- the D
dimension factorises out and the contraction is a real batched matmul:
    (B*n, C, C) @ (B*n, C, D)
This is Mamba2's SSD decomposition (Dao & Gu, ICML 2024, arXiv:2405.21060).

MEASURED SPEEDUP (exact: max rel err 2.1e-06 vs the sequential reference,
checked at gate_bias 0, 4, 9 and T up to 8192):

    T=512   scan  1.995 ms   ssd(C=64) 0.965 ms   2.07x
    T=2048  scan  4.895 ms   ssd(C=64) 2.211 ms   2.21x
    T=8192  scan 19.704 ms   ssd(C=128) 9.524 ms  2.07x

WHAT IS STILL UNKNOWN, AND WHAT THIS FILE TESTS
-----------------------------------------------
Speed is not quality. A scalar gate is LESS expressive than a per-channel gate.
The honest question is whether the expressivity loss costs more than the 2x
speed buys. n_head interpolates the whole frontier: n_head=1 is the most
matmul-friendly and least expressive; n_head=D is the current arm.

MACs are FLAT in n_head -- the A@v product is n_chunks * C^2 * d regardless of
how d is split into heads -- so this is a pure expressivity/efficiency knob.

PRE-REGISTERED PREDICTION, before running
-----------------------------------------
Quality should improve monotonically with n_head and saturate early. If quality
at n_head=1..8 is within seed noise of the per-channel arm, we get a ~2x wall
clock win for free and the honest headline is "diagonal gating buys nothing
here". If quality collapses at low n_head, then per-channel gating is load-
bearing and the 2x is not free -- also a real answer, and the one to report.
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
    load_corpus, CORPUS, n_params, sequential_scan,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "ssd_heads.json")


# --------------------------------------------------------------------------
# the scan, reformulated for a per-head scalar gate
# --------------------------------------------------------------------------
def ssd_scan(v: mx.array, g: mx.array, chunk: int = 64,
             clamp: float = 80.0) -> mx.array:
    """Causal linear attention with a per-head scalar decay, in matmul form.

    v: (B, T, D)  values
    g: (B, T, H)  scalar gate per head, in (0,1)

    Returns h_t = g_t * h_{t-1} + v_t, computed by decomposing each chunk into
    a strictly-lower-triangular weighted sum plus a carry. Every weight is
    exp(L_i - L_j) with i >= j, so it lies in (0, 1] and nothing is ever divided
    by a small number -- which is what made the earlier log-space scan unsafe.
    """
    B, T, D = v.shape
    H = g.shape[-1]
    dh = D // H
    pad = (-T) % chunk
    if pad:
        v = mx.concatenate([v, mx.zeros((B, pad, D))], axis=1)
        g = mx.concatenate([g, mx.zeros((B, pad, H))], axis=1)
    Tp = v.shape[1]
    n = Tp // chunk

    # group channels into heads: (B, H, n, C, dh)
    vh = v.reshape(B, Tp, H, dh).transpose(0, 2, 1, 3).reshape(B, H, n, chunk, dh)
    gh = g.reshape(B, Tp, H).transpose(0, 2, 1).reshape(B, H, n, chunk)

    logg = mx.log(mx.maximum(gh, 1e-30))
    L = mx.cumsum(logg, axis=-1)                              # (B,H,n,C)
    diff = L[..., :, None] - L[..., None, :]                  # (B,H,n,C_i,C_j)
    diff = mx.clip(diff, -clamp, 0.0)
    ii = mx.arange(chunk)[:, None]
    jj = mx.arange(chunk)[None, :]
    tril = (ii >= jj).astype(diff.dtype)[None, None, None, :, :]
    A = mx.exp(diff) * tril                                   # (B,H,n,C,C)

    Ab = A.reshape(B * H * n, chunk, chunk)
    vb = vh.reshape(B * H * n, chunk, dh)
    intra = mx.matmul(Ab, vb).reshape(B, H, n, chunk, dh)

    decay = mx.exp(mx.clip(L, -clamp, 0.0))                   # (B,H,n,C)
    a = decay[..., -1]                                        # (B,H,n)
    b = intra[..., -1, :]                                     # (B,H,n,dh)

    # chunk carries: sequential over n chunks only, vectorised over (B,H,dh)
    h = mx.zeros_like(b[..., 0, :])
    carries = []
    for k in range(n):
        h = a[..., k][..., None] * h + b[..., k, :]
        carries.append(h)
    carry = mx.stack(carries, axis=2)                         # (B,H,n,dh)
    carry_in = mx.concatenate(
        [mx.zeros((B, H, 1, dh)), carry[:, :, :-1, :]], axis=2)

    out = intra + decay[..., None] * carry_in[..., None, :]
    out = out.reshape(B, H, n * chunk, dh).transpose(0, 2, 1, 3).reshape(B, Tp, D)
    return out[:, :T, :]


class SSDMemory(nn.Module):
    """Diagonal gated memory with a per-head scalar gate (matmul-friendly)."""

    def __init__(self, d: int, n_head: int = 1, chunk: int = 64,
                 init_decays=(0.90, 0.99, 0.999, 0.9999)):
        super().__init__()
        self.d, self.n_head, self.chunk = d, n_head, chunk
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, n_head, bias=True)
        # spread the initial decays across heads, as far as the schedule reaches
        dec = list(init_decays)
        bias = [math.log(dec[min(i, len(dec) - 1)] / (1 - dec[min(i, len(dec) - 1)]))
                for i in range(n_head)]
        if n_head > len(dec):  # interpolate geometrically for the long tail
            lo, hi = math.log(dec[0] / (1 - dec[0])), math.log(dec[-1] / (1 - dec[-1]))
            bias = [lo + (hi - lo) * i / max(n_head - 1, 1) for i in range(n_head)]
        self.gate.bias = mx.array(np.array(bias, dtype=np.float32))

    def __call__(self, x, mask=None):
        return self.o(ssd_scan(self.v(x), mx.sigmoid(self.gate(x)), self.chunk))


class Block(nn.Module):
    def __init__(self, d, n_head, mlp_mult=4, chunk=64):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mem = SSDMemory(d, n_head=n_head, chunk=chunk)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))

    def __call__(self, x, mask=None):
        x = x + self.mem(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self, vocab, d, ctx, n_layer, n_head, chunk=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = [Block(d, n_head, 4, chunk) for _ in range(n_layer)]
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(mx.arange(T)[None, :])
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


# --------------------------------------------------------------------------
def deterministic_val_batches(data, ctx, val_frac=0.1):
    n = int((1.0 - val_frac) * len(data))
    va = data[n:]
    n_win = (len(va) - 1) // ctx
    x = np.stack([va[i * ctx:i * ctx + ctx] for i in range(n_win)])
    y = np.stack([va[i * ctx + 1:i * ctx + 1 + ctx] for i in range(n_win)])
    return mx.array(x), mx.array(y), n_win


def evaluate(m, xv, yv, vocab, bs=32):
    tot, n = 0.0, 0
    for i in range(0, xv.shape[0], bs):
        lo = m(xv[i:i + bs])
        l = nn.losses.cross_entropy(
            lo.reshape(-1, vocab), yv[i:i + bs].reshape(-1), reduction="sum")
        mx.eval(l)
        tot += float(l)
        n += int(yv[i:i + bs].size)
    return tot / n


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262}


def run(arm, n_head, data, vocab, *, ctx=512, d=128, n_layer=2, steps=1500,
        bs=16, lr=1e-3, seed=0, chunk=64, log=print):
    mx.random.seed(seed)
    m = LM(vocab, d, ctx, n_layer, n_head, chunk)
    mx.eval(m.parameters())
    P = n_params(m)
    n = int(0.9 * len(data))
    tr = data[:n]
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(m, loss_fn)
    xv, yv, n_win = deterministic_val_batches(data, ctx)
    t0, ntok = time.time(), 0
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
    val = evaluate(m, xv, yv, vocab) / math.log(2)
    log(f"  {arm:<14} n_head={n_head:<4} params={P:>9,} val={val:.4f} "
        f"wall={wall:6.1f}s tok/s={ntok/wall:>8,.0f}")
    return dict(arm=arm, n_head=n_head, params=P, val_bpc=val, seed=seed,
                wall_s=wall, tok_s=ntok / wall, n_val_windows=n_win,
                steps=steps, ctx=ctx, d=d, n_layer=n_layer)


if __name__ == "__main__":
    err = 0.0
    for T in (64, 65, 512, 1000):
        for b_ in (0.0, 4.0, 9.0):
            mx.random.seed(0)
            g = mx.sigmoid(mx.random.normal((2, T, 1)) + b_)
            v = mx.random.normal((2, T, 4))
            ref = np.array(sequential_scan(v, g))
            got = np.array(ssd_scan(v, g, 64))
            err = max(err, float(np.abs(got - ref).max() / max(1.0, np.abs(ref).max())))
    print(f"ssd_scan verification: max rel err vs sequential = {err:.3e}")
    if err > 1e-4:
        raise SystemExit("ssd_scan does not match the recurrence; aborting")

    data, vocab = load_corpus(CORPUS)
    bg, ug = 3.5806, 4.8292
    print(f"corpus {len(data):,} chars vocab {vocab} | bigram floor {bg} | "
          f"any arm at/above the floor has not learned context\n")

    arms = os.environ.get("SSD_ARMS", "perchan,1,4,16,128").split(",")
    seeds = [int(s) for s in os.environ.get("SSD_SEEDS", "0,1,2").split(",")]
    steps = int(os.environ.get("SSD_STEPS", "1500"))
    out = []
    for arm in arms:
        nh = 128 if arm == "perchan" else int(arm)
        for seed in seeds:
            try:
                r = run(arm, nh, data, vocab, steps=steps, seed=seed)
                r["at_floor"] = r["val_bpc"] >= bg
            except Exception as e:
                print(f"  {arm} seed={seed} FAILED: {type(e).__name__}: {e}")
                r = dict(arm=arm, n_head=nh, seed=seed,
                         error=f"{type(e).__name__}: {e}")
            out.append(r)
            json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== PER-ARM (mean +/- sd over seeds) ===")
    for arm in arms:
        rs = [r for r in out if r["arm"] == arm and "error" not in r]
        if not rs:
            continue
        v = [r["val_bpc"] for r in rs]
        print(f"  n_head={rs[0]['n_head']:<4} params={rs[0]['params']:>9,} "
              f"val={np.mean(v):.4f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0:.4f} "
              f"wall={np.mean([r['wall_s'] for r in rs]):6.1f}s")
    print(f"\nwrote {OUT}")
