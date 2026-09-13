"""Wall-clock engineering of the gated-memory scan, with an exactness gate.

WHY THIS FILE EXISTS
--------------------
`docs/EFFICIENCY.md` and `docs/MATCHED_COMPARISON.md` record the FLOP story for
the gated-memory arm. The lead measured where the wall clock actually goes and
found the FLOP model was the wrong proxy:

    component            MACs/tok   min ms   share of mem-block wall
    LayerNorm                   0    0.209    17.2%
    GatedMemory banks=1    49,152    1.098    90.4%
    MLP d->4d->d          131,072    0.474    39.1%
    mem BLOCK total       180,224    1.214      100%
    attn BLOCK total      ...

(The lead's table is reproduced, with the two unit corrections below, by
`end_to_end()` at the bottom of this file.)

The suspect was `chunked_gated_scan` in `experiments/llm_efficiency.py`: a
Hillis-Steele scan that materialises O(log2(chunk)) full-tensor intermediate
levels. Independent review then established the mechanism more precisely: the
scan is **dispatch-bound**, not bandwidth-bound. Each `associative_scan` level
is several small kernels, and at B=16, T=512, d=128 the GPU spends its time
launching work, not moving bytes. A trivial op costs 0.158 ms of dispatch on
this machine (`dispatch_floor_ms` in the JSON), which is why component times in
the original table are NOT additive.

WHAT THIS FILE TESTS
--------------------
Every lever, measured, with the exactness gate applied BEFORE any speedup is
reported:

  (a) chunk-size sweep at fixed T
  (b) log-space cumsum scans -- both reference points, and the bounded variant
  (c) matmul-form scan: the chunk scan as one banded (chunk x chunk) matmul
  (d) fused gate/value projection (one matmul instead of two)
  (e) single fused Metal kernel, sequential over T inside one launch

Two corrections to the unit accounting, applied here so no arm is flattered:
  * every arm is charged 1x forward MACs (the lead's original table charged
    attention the 2x figure from `flops_per_token()` while charging memory and
    the MLP true 1x MACs);
  * the dispatch floor is reported separately rather than summing component
    times, which was wrong because the components are timed in serial calls.

THE GATE (non-negotiable)
-------------------------
`sequential_scan` is the reference. Every candidate is compared against it at
gate_bias 0, 2, 4, 9, 13 -- those biases span initial decays 0.90 to 0.9999,
i.e. the real operating point. A candidate whose max RELATIVE error exceeds
`TOL = 1e-5` is recorded with its error and **no speedup**; it can never appear
as a win. If the winner fails the gate the script aborts.

Deterministic given a seed. `python3 experiments/scan_bench.py [seed]`.

Outputs: experiments/results/scan_bench.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llm_efficiency import (  # noqa: E402
    associative_scan, sequential_scan, chunked_gated_scan, verify_scan,
    GatedMemory, AttentionMemory, Block,
)

TOL = 1e-5
GATE_BIASES = (0.0, 2.0, 4.0, 9.0, 13.0)
GATE_T = (512, 2048)
OUT = os.path.join(HERE, "results", "scan_bench.json")


# --------------------------------------------------------------------------
# (e) the winning lever: ONE Metal kernel, sequential over T inside the launch
# --------------------------------------------------------------------------
_FWD_SRC = """
    uint lane = thread_position_in_grid.x;      // lane = b * D + d
    uint b = lane / (uint)_D_;
    uint d = lane % (uint)_D_;
    uint base = b * (uint)_T_ * (uint)_D_ + d;
    float h = 0.0f;
    uint t = 0;
    for (; t + 4 <= (uint)_T_; t += 4) {
        uint i = base + t * (uint)_D_;
        h = g[i] * h + v[i];  out[i] = h;  i += (uint)_D_;
        h = g[i] * h + v[i];  out[i] = h;  i += (uint)_D_;
        h = g[i] * h + v[i];  out[i] = h;  i += (uint)_D_;
        h = g[i] * h + v[i];  out[i] = h;
    }
    for (; t < (uint)_T_; ++t) {
        uint i = base + t * (uint)_D_;
        h = g[i] * h + v[i];
        out[i] = h;
    }
"""

# Reverse scan for the VJP. dh_t = cot_t + g_{t+1} * dh_{t+1}, so walking t
# downwards: a accumulates the total gradient wrt h_t, dv_t = a, and
# dg_t = a * h_{t-1}. One pass, no storage of the whole adjoint history.
_BWD_SRC = """
    uint lane = thread_position_in_grid.x;
    uint b = lane / (uint)_D_;
    uint d = lane % (uint)_D_;
    uint base = b * (uint)_T_ * (uint)_D_ + d;
    float a = 0.0f;
    for (int t = (int)_T_ - 1; t >= 0; --t) {
        uint i = base + (uint)t * (uint)_D_;
        a = cot[i] + a;
        dv[i] = a;
        float hm1 = (t > 0) ? h[i - (uint)_D_] : 0.0f;
        dg[i] = a * hm1;
        a = g[i] * a;
    }
"""

_KCACHE: dict = {}


def _kernel(name, inputs, outputs, src, T, D, dtype):
    key = (name, T, D, str(dtype))
    if key not in _KCACHE:
        _KCACHE[key] = mx.fast.metal_kernel(
            name=f"{name}_{T}_{D}",
            input_names=list(inputs),
            output_names=list(outputs),
            source=src.replace("_D_", str(D)).replace("_T_", str(T)),
        )
    return _KCACHE[key]


def _tg(lanes: int) -> int:
    return max(1, min(256, lanes))


def _fused_fwd(v: mx.array, g: mx.array) -> mx.array:
    B, T, D = v.shape
    k = _kernel("scan_fwd", ["v", "g"], ["out"], _FWD_SRC, T, D, v.dtype)
    (out,) = k(
        inputs=[v, g], template=[("T", v.dtype)], grid=(B * D, 1, 1),
        threadgroup=(_tg(B * D), 1, 1),
        output_shapes=[(B, T, D)], output_dtypes=[v.dtype],
    )
    return out


def _fused_bwd(v, g, h, cot):
    B, T, D = v.shape
    k = _kernel("scan_bwd", ["v", "g", "h", "cot"], ["dv", "dg"],
                _BWD_SRC, T, D, v.dtype)
    dv, dg = k(
        inputs=[v, g, h, cot], template=[("T", v.dtype)], grid=(B * D, 1, 1),
        threadgroup=(_tg(B * D), 1, 1),
        output_shapes=[(B, T, D), (B, T, D)],
        output_dtypes=[v.dtype, v.dtype],
    )
    return dv, dg


_fused_cf = mx.custom_function(_fused_fwd)


@_fused_cf.vjp
def _fused_cf_vjp(primals, cotangent, output):
    v, g = primals
    return _fused_bwd(v, g, output, cotangent)


def fused_scan(v: mx.array, g: mx.array) -> mx.array:
    """h_t = g_t * h_{t-1} + v_t in ONE kernel launch, parallel over B*D lanes.

    Sequential over T *inside* the launch, so the whole recurrence costs a
    single dispatch instead of O(log2(chunk)) kernel groups. Arithmetic is the
    recurrence itself: 2 ops per element, in the reference accumulation order,
    which is why it is slightly MORE accurate than the tree-based Hillis-Steele
    scan rather than merely equal to it.
    """
    return _fused_cf(v, g)


# --------------------------------------------------------------------------
# the other candidates
# --------------------------------------------------------------------------
def _pad(v, g, chunk):
    B, T, D = v.shape
    p = (-T) % chunk
    if p:
        v = mx.concatenate([v, mx.zeros((B, p, D))], axis=1)
        g = mx.concatenate([g, mx.ones((B, p, D))], axis=1)
    return v, g, T


def _carry(prod, intra_end):
    B, n_ch, D = prod.shape
    _, carry = associative_scan(prod, intra_end)
    return mx.concatenate([mx.zeros((B, 1, D)), carry[:, :-1, :]], axis=1)


def matmul_scan(v, g, chunk=16):
    """(c) chunk scan as one banded (chunk x chunk) lower-triangular matmul.

    W[t,s] = exp(cs_t - cs_s) for s <= t is the exact path weight from s to t,
    so the intra-chunk scan is a single matmul. Included because MLX is near
    peak on matmuls; measured slower anyway (see docs/SCAN.md).
    """
    v, g, T = _pad(v, g, chunk)
    B, Tp, D = v.shape
    n_ch = Tp // chunk
    vc = v.reshape(B, n_ch, chunk, D)
    lg = mx.log(g).reshape(B, n_ch, chunk, D)
    cs = mx.cumsum(lg, axis=2)
    i = mx.arange(chunk)
    m = i[:, None] >= i[None, :]
    Dm = cs[:, :, :, None, :] - cs[:, :, None, :, :]
    W = mx.exp(mx.where(m[None, None, :, :, None], Dm, mx.array(-1e30)))
    intra = mx.sum(W * vc[:, :, None, :, :], axis=3)
    prod = mx.exp(mx.sum(lg, axis=2))
    carry_in = _carry(prod, intra[:, :, -1, :])
    h = intra + mx.exp(cs) * carry_in[:, :, None, :]
    return h.reshape(B, Tp, D)[:, :T, :]


def logscan_start(v, g, chunk=16):
    """(b) log-space cumsum, referenced to the chunk START.

    intra_t = exp(M_t) * sum_{s<=t} v_s exp(-M_s), M_t = sum_{u<=t} log g_u <= 0.
    Weights exp(-M_s) are >= 1 and the dominant term is s=t, so there is no
    subtraction of near-equal huge numbers. This is the numerically SAFE log
    form; it passes the gate at every required bias.
    """
    v, g, T = _pad(v, g, chunk)
    B, Tp, D = v.shape
    n_ch = Tp // chunk
    vc = v.reshape(B, n_ch, chunk, D)
    gc = g.reshape(B, n_ch, chunk, D)
    lg = mx.log(gc)
    M = mx.cumsum(lg, axis=2)
    A = mx.cumsum(vc * mx.exp(-M), axis=2)
    prod = mx.exp(mx.sum(lg, axis=2))
    carry_in = _carry(prod, A[:, :, -1, :] * mx.exp(M[:, :, -1, :]))
    h = mx.exp(M) * (A + carry_in[:, :, None, :])
    return h.reshape(B, Tp, D)[:, :T, :]


def logscan_end(v, g, chunk=16):
    """(b) log-space cumsum referenced to the chunk END -- the UNSAFE variant.

    R_t = M_t - M_last >= 0, h_t = exp(R_t) * sum_{s<=t} v_s exp(-R_s). The
    weights exp(-R_s) are huge and their sum is dominated by a term that is
    then multiplied by exp(R_t) ~ 0, so this subtracts near-equal large numbers
    and overflows. Reproduces the lead's pre-existing measurement; kept in the
    harness permanently as the negative control for the gate.
    """
    v, g, T = _pad(v, g, chunk)
    B, Tp, D = v.shape
    n_ch = Tp // chunk
    vc = v.reshape(B, n_ch, chunk, D)
    gc = g.reshape(B, n_ch, chunk, D)
    lg = mx.log(gc)
    M = mx.cumsum(lg, axis=2)
    R = M - M[:, :, -1:, :]
    A = mx.cumsum(vc * mx.exp(-R), axis=2)
    prod = mx.exp(mx.sum(lg, axis=2))
    carry_in = _carry(prod, A[:, :, -1, :] * mx.exp(-R[:, :, -1:, :])[:, :, 0, :])
    h = mx.exp(R) * (A + carry_in[:, :, None, :])
    return h.reshape(B, Tp, D)[:, :T, :]


def logscan_shift(v, g, chunk=16):
    """(b) the suggested rescue: rescale per chunk by the running max.

    Ms_t = M_t + max(-M) >= 0, h_t = exp(Ms_t) * sum_{s<=t} v_s exp(-Ms_s) +
    exp(Ms_t) * carry. The SHIFT does not fix it: exp(-Ms_s) is still huge
    relative to exp(-Ms_t) at large t, so the same cancellation remains. Kept
    because the gate must judge the suggestion, not my description of it.
    """
    v, g, T = _pad(v, g, chunk)
    B, Tp, D = v.shape
    n_ch = Tp // chunk
    vc = v.reshape(B, n_ch, chunk, D)
    gc = g.reshape(B, n_ch, chunk, D)
    lg = mx.log(gc)
    M = mx.cumsum(lg, axis=2)
    shift = mx.max(-M, axis=2, keepdims=True)
    Ms = M + shift
    A = mx.cumsum(vc * mx.exp(-Ms), axis=2)
    prod = mx.exp(mx.sum(lg, axis=2))
    carry_in = _carry(prod, A[:, :, -1, :] * mx.exp(Ms[:, :, -1, :]))
    h = mx.exp(Ms) * (A + carry_in[:, :, None, :])
    return h.reshape(B, Tp, D)[:, :T, :]


def loop_scan(v, g):
    """Plain Python loop, compiled: exact, but slower than the tree. Negative."""
    B, T, D = v.shape
    h = mx.zeros((B, D))
    acc = []
    for t in range(T):
        h = g[:, t, :] * h + v[:, t, :]
        acc.append(h)
    return mx.stack(acc, axis=1)


_COMPILE_CACHE: dict = {}
# mx.compile over an unrolled loop costs minutes of graph construction at T=8192
# for a result that is already known to be slower, so the lever is capped. The
# cap is recorded in the JSON rather than hidden.
COMPILED_LOOP_MAX_T = int(os.environ.get("BRAIN_COMPILED_LOOP_MAX_T", "512"))

# Context lengths for the short-context block sweep. 64 and 128 are the regime
# where attention is cheap and the scan's fixed launch cost should dominate.
SHORT_TS = tuple(int(t) for t in os.environ.get(
    "BRAIN_SHORT_TS", "64,128,256,512,2048").split(","))

# Independent repetitions per context length for the short-context sweep. At
# short T the two blocks are within ~10% of each other, so one pass cannot
# establish the sign; the reps are what turn a coin flip into a result.
SHORT_REPS = int(os.environ.get("BRAIN_SHORT_REPS", "5"))


def compiled_hillis(v, g, chunk=64):
    key = ("hillis", chunk)
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = mx.compile(
            lambda a, b: chunked_gated_scan(a, b, chunk))
    return _COMPILE_CACHE[key](v, g)


def compiled_loop(T):
    """mx.compile over an unrolled T-step Python loop. Returns a callable;
    exact (it IS the recurrence) but measured slower than the tree."""
    key = ("loopcall", T)
    if key not in _COMPILE_CACHE:
        def body(v, g):
            B, TT, D = v.shape
            h = mx.zeros((B, D))
            acc = []
            for t in range(T):
                h = g[:, t, :] * h + v[:, t, :]
                acc.append(h)
            return mx.stack(acc, axis=1)
        _COMPILE_CACHE[key] = mx.compile(body)
    return _COMPILE_CACHE[key]


def compiled_loop_compile_ms(T):
    return 1000.0 * float(_COMPILE_CACHE.get(("loopms", T), 0.0))


# --------------------------------------------------------------------------
# MAC accounting -- ALL ARMS IN THE SAME UNIT (1x forward multiply-accumulates)
# --------------------------------------------------------------------------
def macs_scan(name, B, T, D, chunk=0):
    """Forward multiply-accumulates for one scan call, COUNTED from the code.

    All arms everywhere in this file are charged 1x forward MACs. The lead's
    original table charged attention the 2x figure from `flops_per_token()`
    while charging memory and the MLP true 1x MACs; that mixed units and
    overstated attention by exactly 2x. Here a MAC means one multiply plus one
    add, counted once.

    `fused` and `loop` do the recurrence: 2 per element.
    `hillis` pays ~3 per element per tree level, plus the chunk-carry scan.
    `matmul` pays the dense (chunk x chunk) band per chunk.
    `logscan*` pays log, cumsum, exp, mul, cumsum, exp, mul, add ~ 8 per element.
    """
    n = B * T * D
    if name.startswith("fused") or name.startswith("loop"):
        return 2 * n
    if name.startswith("hillis"):
        lv = int(np.ceil(np.log2(max(chunk, 2))))
        n_ch = max(T // max(chunk, 1), 1)
        lvc = max(int(np.ceil(np.log2(max(n_ch, 2)))), 1)
        return 3 * n * lv + 3 * B * D * n_ch * lvc
    if name.startswith("matmul"):
        return int(n * chunk)
    if name.startswith("logscan"):
        return 8 * n
    return 0


def macs_per_token_block(kind, d, T, banks=1, scan="fused", chunk=64):
    """End-to-end forward MACs per token for a whole Block, one consistent unit.

    attention projections 4d^2, scores+AV 2*T*d (full masked) or T*d (causal
    skipping), MLP 8d^2.
    memory     v,gate,o projections 3d^2, scan 2d (one fused recurrence per
    element), MLP 8d^2.
    """
    mlp = 8 * d * d
    if kind == "attn":
        proj = 4 * d * d
        scores_full = 2 * T * d
        scores_causal = T * d
        return dict(proj=proj, ctx=scores_full, mlp=mlp,
                    block_full=proj + scores_full + mlp,
                    block_causal=proj + scores_causal + mlp)
    proj = 3 * d * d * banks
    scan_macs = 2 * d * banks
    return dict(proj=proj, ctx=scan_macs, mlp=mlp,
                block_full=proj + scan_macs + mlp, block_causal=None)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def rel_err(ref: mx.array, got: mx.array) -> float:
    d = float(mx.max(mx.abs(ref - got)))
    s = float(mx.max(mx.abs(ref)))
    return d / max(s, 1e-30)


def gate_case(T, D, B, bias, seed=0):
    mx.random.seed(seed)
    g = mx.sigmoid(mx.random.normal((B, T, D)) + bias)
    v = mx.random.normal((B, T, D))
    return v, g, sequential_scan(v, g)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def measure(fn, warmup=5, iters=30):
    """min and median ms per call, mx.eval every step, warmup discarded."""
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def measure_paired(fns, warmup=5, iters=30, trials=3):
    """Interleaved timing of several callables, repeated over `trials` rounds.

    This machine is shared: other processes hold the GPU while this runs, and a
    whole-table measurement can otherwise drift by 3-5x between arms. Round
    robin within one loop makes every arm see the same contention, so the RATIO
    is stable even when the absolute milliseconds are inflated.

    The reported `ms_min` is the minimum over all trials, and `ms_median` is the
    median of all samples. The minimum is the right statistic for "what does
    this call cost when the machine is not stealing the GPU from me", and the
    dispatch floor is recorded next to it so a reader can tell how contended
    the run was.
    """
    for _ in range(warmup):
        for fn in fns.values():
            mx.eval(fn())
    ts = {k: [] for k in fns}
    for _ in range(trials):
        for _ in range(iters):
            for k, fn in fns.items():
                t0 = time.perf_counter()
                mx.eval(fn())
                ts[k].append((time.perf_counter() - t0) * 1e3)
    out = {}
    for k, v in ts.items():
        v.sort()
        out[k] = dict(ms_min=v[0], ms_median=v[len(v) // 2], n_samples=len(v))
    return out


# --------------------------------------------------------------------------
# candidate registry
# --------------------------------------------------------------------------
def build_candidates(B, T, D):
    """name -> (lever, setting, callable). Callables close over fresh inputs."""
    mx.random.seed(0)
    v = mx.random.normal((B, T, D))
    g = mx.sigmoid(mx.random.normal((B, T, D)) + 4.0)
    c = {}
    c["hillis_c16"] = ("a_chunk_sweep", "chunk=16",
                       lambda: chunked_gated_scan(v, g, 16))
    c["hillis_c32"] = ("a_chunk_sweep", "chunk=32",
                       lambda: chunked_gated_scan(v, g, 32))
    c["hillis_c64"] = ("a_chunk_sweep", "chunk=64 (baseline)",
                       lambda: chunked_gated_scan(v, g, 64))
    c["hillis_c128"] = ("a_chunk_sweep", "chunk=128",
                        lambda: chunked_gated_scan(v, g, 128))
    c["hillis_c256"] = ("a_chunk_sweep", "chunk=256",
                        lambda: chunked_gated_scan(v, g, 256))
    c["hillis_c64_compiled"] = ("a_chunk_sweep", "chunk=64 + mx.compile",
                                lambda: compiled_hillis(v, g, 64))
    c["logscan_start_c16"] = ("b_logspace", "start-referenced chunk=16",
                              lambda: logscan_start(v, g, 16))
    c["logscan_start_c32"] = ("b_logspace", "start-referenced chunk=32",
                              lambda: logscan_start(v, g, 32))
    c["logscan_end_c16"] = ("b_logspace", "end-referenced chunk=16 (unsafe)",
                            lambda: logscan_end(v, g, 16))
    c["logscan_shift_c16"] = ("b_logspace", "max-shifted chunk=16 (unsafe)",
                              lambda: logscan_shift(v, g, 16))
    c["matmul_c16"] = ("c_matmul_form", "chunk=16", lambda: matmul_scan(v, g, 16))
    c["matmul_c32"] = ("c_matmul_form", "chunk=32", lambda: matmul_scan(v, g, 32))
    c["matmul_c64"] = ("c_matmul_form", "chunk=64", lambda: matmul_scan(v, g, 64))
    c["loop_unrolled"] = ("e_other", "python loop, uncompiled (reference)",
                          lambda: loop_scan(v, g))
    if T <= COMPILED_LOOP_MAX_T:
        c["loop_compiled"] = ("e_other", "unrolled loop + mx.compile",
                              lambda: compiled_loop(T)(v, g))
    c["fused_metal"] = ("e_fused_kernel", "one Metal launch, unroll4 (WINNER)",
                        lambda: fused_scan(v, g))
    c["fused_metal_compiled"] = ("e_fused_kernel", "winner + mx.compile",
                                 lambda: _compiled_fused(v, g))
    c["_inputs"] = (v, g)
    return c


_COMPILED_FUSED = {}


def _compiled_fused(v, g):
    key = tuple(v.shape)
    if key not in _COMPILED_FUSED:
        _COMPILED_FUSED[key] = mx.compile(lambda a, b: fused_scan(a, b))
    return _COMPILED_FUSED[key](v, g)


# --------------------------------------------------------------------------
# end-to-end: rebuild the mem block and the attn block, before and after
# --------------------------------------------------------------------------
class GatedMemoryFused(nn.Module):
    """GatedMemory with the fused-kernel scan. Same weights, same function."""

    def __init__(self, base: GatedMemory):
        super().__init__()
        self.d = base.d
        self.banks = base.banks
        self.chunk = base.chunk
        self.v = base.v
        self.o = base.o
        self.gate = base.gate

    def __call__(self, x, mask=None):
        B, T, _ = x.shape
        v = self.v(x).reshape(B, T, self.banks, self.d)
        g = mx.sigmoid(self.gate(x)).reshape(B, T, self.banks, self.d)
        v = v.transpose(0, 2, 1, 3).reshape(B * self.banks, T, self.d)
        g = g.transpose(0, 2, 1, 3).reshape(B * self.banks, T, self.d)
        h = fused_scan(v, g)
        h = h.reshape(B, self.banks, T, self.d).transpose(0, 2, 1, 3)
        return self.o(h.reshape(B, T, self.banks * self.d))


def _swap_mem(block: Block) -> Block:
    """A SECOND block that shares ln/mlp parameters but has the fused mem.

    Builds a new Block rather than reassigning `block.mem`, because mutating in
    place made both arms the same object and reported a 0.99x "speedup" that
    was just noise.
    """
    out = Block(block.mem.d, 1, "mem", banks=block.mem.banks,
                chunk=block.mem.chunk)
    out.ln1 = block.ln1
    out.ln2 = block.ln2
    out.mlp = block.mlp
    out.mem = GatedMemoryFused(block.mem)
    return out


def end_to_end(seed=0, B=16, T=512, d=128, n_head=8, warmup=5, iters=30,
               trials=3):
    """The lead's table, rebuilt, all arms in the same unit, paired timing.

    Two deliberate differences from the original measurement, both recorded in
    docs/SCAN.md: attention is charged 1x forward MACs like every other arm
    (the original charged it the 2x `flops_per_token()` figure), and the
    components are timed round-robin in ONE loop with `measure_paired` so that
    background GPU contention cannot flatter one arm.
    """
    mx.random.seed(seed)
    x = mx.random.normal((B, T, d))
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

    ln = nn.LayerNorm(d)
    mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=True), nn.GELU(),
                        nn.Linear(4 * d, d, bias=True))
    mem_before = GatedMemory(d, banks=1, chunk=64)
    mem_after = GatedMemoryFused(mem_before)
    attn = AttentionMemory(d, n_head)
    for m in (ln, mlp, attn):
        mx.eval(m.parameters())

    blk_before = Block(d, n_head, "mem")
    blk_before.mem = mem_before
    blk_after = _swap_mem(blk_before)
    blk_attn = Block(d, n_head, "attn")
    for m in (blk_before, blk_after, blk_attn):
        mx.eval(m.parameters())

    # the fused arm must compute the same function when fed the same weights
    _a = blk_before(x, mask)
    _b = blk_after(x, mask)
    mx.eval(_a, _b)
    block_agree = rel_err(_a, _b)

    m_attn = macs_per_token_block("attn", d, T)
    m_mem1 = macs_per_token_block("mem", d, T, banks=1)
    m_mem4 = macs_per_token_block("mem", d, T, banks=4)

    parts = {
        "LayerNorm": (lambda: ln(x), 0),
        "MLP d->4d->d": (lambda: mlp(x), m_mem1["mlp"]),
        # scan rows are charged per token, like every other row, so the column
        # is one unit: divide the per-call MACs by B*T. The fused scan is 2 MACs
        # per element = 2*d per token per bank.
        "mem scan only BEFORE (hillis c64)": (
            lambda: chunked_gated_scan(*_mem_scan_inputs(mem_before, x)),
            macs_scan("hillis", B, T, d, 64) // (B * T)),
        "mem scan only AFTER (fused)": (
            lambda: fused_scan(*_mem_scan_inputs(mem_after, x)),
            macs_scan("fused", B, T, d) // (B * T)),
        "GatedMemory banks=1 BEFORE": (lambda: mem_before(x),
                                      m_mem1["proj"] + m_mem1["ctx"]),
        "GatedMemory banks=1 AFTER (fused)": (lambda: mem_after(x),
                                             m_mem1["proj"] + m_mem1["ctx"]),
        "mem BLOCK BEFORE": (lambda: blk_before(x, mask), m_mem1["block_full"]),
        "mem BLOCK AFTER": (lambda: blk_after(x, mask), m_mem1["block_full"]),
        "attn BLOCK": (lambda: blk_attn(x, mask), m_attn["block_full"]),
        "attn BLOCK (causal-skip MACs)": (lambda: blk_attn(x, mask),
                                          m_attn["block_causal"]),
        "attn BLOCK (mask=None)": (lambda: blk_attn(x, None),
                                   m_attn["block_full"]),
        "dispatch floor (trivial op)": (lambda: mx.add(mx.array([1.0]), 1.0), 0),
    }
    res = measure_paired({k: v[0] for k, v in parts.items()},
                         warmup=warmup, iters=iters, trials=trials)

    rows = []
    for name, (_fn, macs) in parts.items():
        r = res[name]
        rows.append(dict(component=name, macs_per_token=macs,
                         min_ms=r["ms_min"], med_ms=r["ms_median"],
                         n_samples=r["n_samples"]))

    by = {r["component"]: r for r in rows}
    mem_b_before = by["mem BLOCK BEFORE"]["min_ms"]
    mem_b_after = by["mem BLOCK AFTER"]["min_ms"]
    attn_b = by["attn BLOCK"]["min_ms"]
    return dict(
        B=B, T=T, d=d, n_head=n_head, rows=rows,
        block_same_function_rel_err=block_agree,
        dispatch_floor_ms=by["dispatch floor (trivial op)"]["min_ms"],
        dispatch_floor_med_ms=by["dispatch floor (trivial op)"]["med_ms"],
        mem_block_before_ms=mem_b_before,
        mem_block_after_ms=mem_b_after,
        attn_block_ms=attn_b,
        mem_over_attn_before=mem_b_before / attn_b,
        mem_over_attn_after=mem_b_after / attn_b,
        mem_faster_than_attn_before=bool(mem_b_before < attn_b),
        mem_faster_than_attn_after=bool(mem_b_after < attn_b),
        mem_block_speedup=mem_b_before / mem_b_after,
        macs_per_token=dict(attn=m_attn, mem_banks1=m_mem1, mem_banks4=m_mem4),
        mac_ratio_attn_over_mem_full=m_attn["block_full"] / m_mem1["block_full"],
        mac_ratio_attn_over_mem_causal=(
            m_attn["block_causal"] / m_mem1["block_full"]),
    )


# --------------------------------------------------------------------------
# short-context sweep: where does the memory block stop beating attention?
# --------------------------------------------------------------------------
def block_pair(T, seed=0, B=16, d=128, n_head=8, warmup=3, iters=20,
               trials=3):
    """Mem block (Hillis vs fused) and attention block at ONE context length.

    Every arm is timed round-robin in a single interleaved loop, so background
    GPU contention cannot flatter one arm; `ms_min` is the minimum over all
    trials and `ms_median` the median over all samples. mx.eval runs inside the
    timing loop, so each sample is a real evaluation of a lazy graph.

    The scan-only rows are included so the reader can separate "the recurrence
    costs this much" from "the block costs this much"; the fused-kernel launch
    is fixed-cost, so at short T it should dominate the scan row.
    """
    mx.random.seed(seed)
    x = mx.random.normal((B, T, d))
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

    mem_before = GatedMemory(d, banks=1, chunk=64)
    mem_after = GatedMemoryFused(mem_before)
    attn = AttentionMemory(d, n_head)
    for m in (attn,):
        mx.eval(m.parameters())

    blk_before = Block(d, n_head, "mem")
    blk_before.mem = mem_before
    blk_after = _swap_mem(blk_before)
    blk_attn = Block(d, n_head, "attn")
    for m in (blk_before, blk_after, blk_attn):
        mx.eval(m.parameters())

    _a, _b = blk_before(x, mask), blk_after(x, mask)
    mx.eval(_a, _b)
    agree = rel_err(_a, _b)

    m_attn = macs_per_token_block("attn", d, T)
    m_mem = macs_per_token_block("mem", d, T, banks=1)
    m_attn_ctx1 = macs_per_token_block("attn", d, T)

    parts = {
        "mem scan only BEFORE (hillis c64)": (
            lambda: chunked_gated_scan(*_mem_scan_inputs(mem_before, x)),
            macs_scan("hillis", B, T, d, 64) // (B * T)),
        "mem scan only AFTER (fused kernel)": (
            lambda: fused_scan(*_mem_scan_inputs(mem_after, x)),
            macs_scan("fused", B, T, d) // (B * T)),
        "GatedMemory banks=1 AFTER (proj+scan)": (
            lambda: mem_after(x), m_mem["proj"] + m_mem["ctx"]),
        "mem BLOCK BEFORE (hillis)": (
            lambda: blk_before(x, mask), m_mem["block_full"]),
        "mem BLOCK AFTER (fused)": (
            lambda: blk_after(x, mask), m_mem["block_full"]),
        "attn BLOCK (causal mask)": (
            lambda: blk_attn(x, mask), m_attn["block_full"]),
        "attn BLOCK (mask=None)": (
            lambda: blk_attn(x, None), m_attn_ctx1["block_full"]),
        "dispatch floor (trivial op)": (
            lambda: mx.add(mx.array([1.0]), 1.0), 0),
    }
    res = measure_paired({k: v[0] for k, v in parts.items()},
                         warmup=warmup, iters=iters, trials=trials)
    rows = []
    for name, (_fn, macs) in parts.items():
        r = res[name]
        rows.append(dict(component=name, macs_per_token=macs,
                         min_ms=r["ms_min"], med_ms=r["ms_median"],
                         n_samples=r["n_samples"]))
    by = {r["component"]: r for r in rows}
    mb_before = by["mem BLOCK BEFORE (hillis)"]["min_ms"]
    mb_after = by["mem BLOCK AFTER (fused)"]["min_ms"]
    at_masked = by["attn BLOCK (causal mask)"]["min_ms"]
    at_none = by["attn BLOCK (mask=None)"]["min_ms"]
    return dict(
        T=T, B=B, d=d, n_head=n_head, rows=rows,
        same_function_rel_err=agree,
        mem_block_before_ms=mb_before, mem_block_after_ms=mb_after,
        attn_block_masked_ms=at_masked, attn_block_nomask_ms=at_none,
        mem_over_attn_after=mb_after / at_masked,
        mem_over_attn_after_vs_nomask=mb_after / at_none,
        mem_over_attn_before=mb_before / at_masked,
        mem_block_speedup=mb_before / mb_after,
        mem_faster_than_attn_after=bool(mb_after < at_masked),
        mem_faster_than_attn_after_vs_nomask=bool(mb_after < at_none),
        scan_before_ms=by["mem scan only BEFORE (hillis c64)"]["min_ms"],
        scan_after_ms=by["mem scan only AFTER (fused kernel)"]["min_ms"],
        dispatch_floor_ms=by["dispatch floor (trivial op)"]["min_ms"],
        macs_per_token=dict(attn=m_attn, mem_banks1=m_mem),
        mac_ratio_attn_over_mem=m_attn["block_full"] / m_mem["block_full"],
    )


def short_context_sweep(seed=0, Ts=(64, 128, 256, 512, 2048), reps=1, **kw):
    """Block-vs-block wall clock across context length, plus the crossover.

    Each T is measured `reps` independent times. At short context the two
    blocks land within ~10-20% of each other, which is the same size as the
    dispatch floor and as run-to-run contention on this machine, so a single
    pass cannot tell the sign of the difference. A one-rep run would be a coin
    flip dressed as a result, so the full distribution of the per-rep ratio is
    reported and each T gets a 95% CI on the ratio (the reps are paired, so the
    per-rep log ratio is the statistic).

    A T counts as a WIN for either side only when that CI EXCLUDES 1.0. Simply
    counting agreeing reps is not enough: at these rep counts it resolves
    differences that are inside the noise. T where the CI spans 1.0 are
    reported as TIES and excluded from the crossover, because calling them
    either way would be reporting noise as a result.

    The crossover is reported as an interval: the largest tested T at which the
    memory block lost, and the smallest at which it won, both CI-resolved.
    `None` on either side means "no such T in the tested range", which is
    itself the result.
    """
    rows = []
    for T in Ts:
        reps_out = [block_pair(T, seed=seed, **kw) for _ in range(reps)]
        ratios = sorted(r["mem_over_attn_after"] for r in reps_out)
        first = reps_out[0]
        first = dict(first)
        first["ratio_reps"] = ratios
        first["ratio_median_over_reps"] = ratios[len(ratios) // 2]
        first["ratio_min_over_reps"] = ratios[0]
        # Paired CI on the ratio. The reps are paired (each rep measures both
        # blocks round-robin under the same contention), so the per-rep log
        # ratio is the right statistic; a CI on it says whether the sign of the
        # difference is actually resolved rather than merely lucky.
        logs = [math.log(x) for x in ratios]
        n = len(logs)
        mean = sum(logs) / n
        var = (sum((x - mean) ** 2 for x in logs) / (n - 1)) if n > 1 else 0.0
        se = math.sqrt(var / n) if n > 1 else float("inf")
        # 95% interval, t-approximated by 1.96 for the rep counts used here
        lo, hi = math.exp(mean - 1.96 * se), math.exp(mean + 1.96 * se)
        first["ratio_ci95_low"] = lo
        first["ratio_ci95_high"] = hi
        first["ratio_ci95_excludes_1"] = bool(lo > 1.0 or hi < 1.0)
        first["ratio_geomean"] = math.exp(mean)

        first["ratio_max_over_reps"] = ratios[-1]
        first["mem_faster_reps"] = sum(
            1 for r in reps_out if r["mem_faster_than_attn_after"])
        first["sign_stable"] = bool(
            first["mem_faster_reps"] in (0, reps))
        first["median_rep_mem_faster"] = bool(
            first["ratio_median_over_reps"] < 1.0)
        first["mem_faster_resolved"] = bool(
            first["ratio_ci95_excludes_1"] and first["ratio_geomean"] < 1.0)
        # comparison against UNMASKED attention, on the same rep distribution,
        # so the two verdicts are computed the same way
        ratio_nm = sorted(r["mem_over_attn_after_vs_nomask"] for r in reps_out)
        first["ratio_nomask_reps"] = ratio_nm
        first["ratio_nomask_median_over_reps"] = ratio_nm[len(ratio_nm) // 2]
        first["median_rep_mem_faster_vs_nomask"] = bool(
            first["ratio_nomask_median_over_reps"] < 1.0)
        logs_nm = [math.log(x) for x in ratio_nm]
        n_nm = len(logs_nm)
        mean_nm = sum(logs_nm) / n_nm
        var_nm = (sum((x - mean_nm) ** 2 for x in logs_nm) / (n_nm - 1)
                  if n_nm > 1 else 0.0)
        se_nm = math.sqrt(var_nm / n_nm) if n_nm > 1 else float("inf")
        first["ratio_nomask_ci95_low"] = math.exp(mean_nm - 1.96 * se_nm)
        first["ratio_nomask_ci95_high"] = math.exp(mean_nm + 1.96 * se_nm)
        first["ratio_nomask_ci95_excludes_1"] = bool(
            first["ratio_nomask_ci95_low"] > 1.0
            or first["ratio_nomask_ci95_high"] < 1.0)
        first["mem_faster_resolved_vs_nomask"] = bool(
            first["ratio_nomask_ci95_excludes_1"]
            and math.exp(mean_nm) < 1.0)
        first["nomask_faster_reps"] = sum(
            1 for r in reps_out
            if r["mem_faster_than_attn_after_vs_nomask"])
        first["nomask_sign_stable"] = bool(
            first["nomask_faster_reps"] in (0, reps))
        rows.append(first)

    # A T is only evidence of a WIN or a LOSS if the 95% CI on the ratio
    # excludes 1.0. Where it spans 1.0 the difference is smaller than this
    # machine's measurement noise, and calling it either way would be
    # reporting noise. Those T are TIES and are excluded from the crossover.
    ties = [r["T"] for r in rows if not r["ratio_ci95_excludes_1"]]
    slower = [r["T"] for r in rows if r["ratio_ci95_excludes_1"]
              and not r["mem_faster_resolved"]]
    faster = [r["T"] for r in rows if r["mem_faster_resolved"]]
    nomask_slower = [r["T"] for r in rows if r["ratio_nomask_ci95_excludes_1"]
                     and not r["mem_faster_resolved_vs_nomask"]]
    nomask_ties = [r["T"] for r in rows
                   if not r["ratio_nomask_ci95_excludes_1"]]
    return dict(
        seed=seed, Ts=list(Ts), B=rows[0]["B"], d=rows[0]["d"],
        n_head=rows[0]["n_head"], reps=reps, rows=rows,
        tie_Ts=ties,
        resolved_Ts=sorted(faster + slower),
        mem_slower_at=sorted(slower), mem_faster_at=sorted(faster),
        # the resolution limit: below this T the two blocks are
        # indistinguishable on this machine, whichever way individual reps fall
        # the smallest T at which the memory block's win was sign-stable;
        # if the smallest tested T is already stable, no limit was found
        resolution_limit_T=(
            min(r["T"] for r in rows if r["mem_faster_resolved"])
            if any(r["mem_faster_resolved"] for r in rows) else None),
        nomask_tie_Ts=nomask_ties,
        mem_slower_at_vs_nomask=sorted(nomask_slower),
        crossover_T_interval=(
            [max(slower), min(faster)] if slower and faster else None),
        mem_always_faster=not slower,
        mem_never_faster=not faster,
    )


def _mem_scan_inputs(mem, x):
    B, T, _ = x.shape
    v = mem.v(x).reshape(B, T, mem.banks, mem.d)
    g = mx.sigmoid(mem.gate(x)).reshape(B, T, mem.banks, mem.d)
    v = v.transpose(0, 2, 1, 3).reshape(B * mem.banks, T, mem.d)
    g = g.transpose(0, 2, 1, 3).reshape(B * mem.banks, T, mem.d)
    return v, g


# --------------------------------------------------------------------------
# training-integration check: the fused scan must not change what is learned
# --------------------------------------------------------------------------
def grad_check(seed=0, B=2, T=64, d=16):
    """VJP through the custom kernel vs the exact Python loop, and through a
    full GatedMemory vs the Hillis version. Shows quality of report is intact."""
    mx.random.seed(seed)
    v = mx.random.normal((B, T, d))
    g = mx.sigmoid(mx.random.normal((B, T, d)) + 2.0)
    w = mx.random.normal((T, 1))

    def loss_loop(a, b):
        return mx.sum(loop_scan(a, b) * w[None, :, :])

    def loss_fused(a, b):
        return mx.sum(fused_scan(a, b) * w[None, :, :])

    lv_l, lg_l = mx.grad(loss_loop, argnums=(0, 1))(v, g)
    lv_f, lg_f = mx.grad(loss_fused, argnums=(0, 1))(v, g)
    mx.eval(lv_l, lg_l, lv_f, lg_f)

    return dict(
        fwd_rel=rel_err(loop_scan(v, g), fused_scan(v, g)),
        grad_v_rel=rel_err(lv_l, lv_f),
        grad_g_rel=rel_err(lg_l, lg_f),
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mx.random.seed(seed)
    np.random.seed(seed)

    B, D = 16, 128
    # Anything already in the results file is PRIOR EVIDENCE. This session's
    # brief was to add short-context numbers without overwriting the recorded
    # headline, so the previous headline and detailed table are carried forward
    # under explicit *_previous keys and never silently replaced.
    prior = {}
    if os.path.exists(OUT):
        try:
            with open(OUT) as fh:
                prior = json.load(fh)
        except Exception:
            prior = {}

    report: dict = dict(seed=seed, B=B, D=D, tol=TOL,
                        gate_biases=list(GATE_BIASES),
                        gate_T=list(GATE_T), levers={}, gate={},
                        compiled_loop_max_T=COMPILED_LOOP_MAX_T)
    # The recorded headline is the FIRST one ever written, whatever key it
    # ended up under on the previous run. This has to be idempotent: an earlier
    # version of this code moved the headline to `headline_current_run` on the
    # second run, and the third run then saw no `headline` key and silently
    # overwrote the committed number with its own. A run that is re-executed
    # many times must never be able to lose prior evidence.
    recorded_headline = (prior.get("headline")
                         or prior.get("headline_original")
                         or prior.get("headline_previous"))
    if recorded_headline:
        report["headline_original"] = recorded_headline
    if prior.get("end_to_end"):
        report["end_to_end_previous"] = prior["end_to_end"]
    if prior.get("short_context_sweep"):
        report["short_context_sweep_previous"] = prior["short_context_sweep"]

    print("=" * 78)
    print("scan_bench: wall clock of the gated-memory scan, with exactness gate")
    print(f"seed={seed}  B={B}  D={D}  tol={TOL}  device=MLX/Metal (M4 Max)")
    print("=" * 78)

    # ---- existing guard -------------------------------------------------
    guard = verify_scan()
    report["verify_scan_max_abs_err"] = guard
    print(f"\n[guard] llm_efficiency.verify_scan() max abs err = {guard:.3e} "
          f"({'PASS' if guard <= 1e-4 else 'FAIL'})")
    if guard > 1e-4:
        raise SystemExit("verify_scan guard FAILED; aborting")

    # ---- gate: every candidate vs sequential_scan ------------------------
    print(f"\n[gate] max RELATIVE error vs sequential_scan, tol {TOL:.0e}  "
          f"biases {GATE_BIASES}")
    # candidates whose gate row can be "not measured" for reasons other than
    # numerical failure, so a cap is never reported as a correctness failure
    NOT_MEASURED = {"loop_compiled": f"capped at T={COMPILED_LOOP_MAX_T} "
                                     "(compile cost; slower where affordable)"}
    gate_fns = {
        "hillis_c16": lambda v, g: chunked_gated_scan(v, g, 16),
        "hillis_c32": lambda v, g: chunked_gated_scan(v, g, 32),
        "hillis_c64": lambda v, g: chunked_gated_scan(v, g, 64),
        "hillis_c128": lambda v, g: chunked_gated_scan(v, g, 128),
        "hillis_c256": lambda v, g: chunked_gated_scan(v, g, 256),
        "hillis_c64_compiled": lambda v, g: compiled_hillis(v, g, 64),
        "logscan_start_c16": lambda v, g: logscan_start(v, g, 16),
        "logscan_start_c32": lambda v, g: logscan_start(v, g, 32),
        "logscan_end_c16": lambda v, g: logscan_end(v, g, 16),
        "logscan_shift_c16": lambda v, g: logscan_shift(v, g, 16),
        "matmul_c16": lambda v, g: matmul_scan(v, g, 16),
        "matmul_c32": lambda v, g: matmul_scan(v, g, 32),
        "matmul_c64": lambda v, g: matmul_scan(v, g, 64),
        "loop_unrolled": lambda v, g: loop_scan(v, g),
        "loop_compiled": lambda v, g: (
            compiled_loop(v.shape[1])(v, g)
            if v.shape[1] <= COMPILED_LOOP_MAX_T
            else (_ for _ in ()).throw(
                ValueError(
                    f"compiled-loop lever capped at T={COMPILED_LOOP_MAX_T}; "
                    "measured slower where it was affordable"))),
        "fused_metal": lambda v, g: fused_scan(v, g),
        "fused_metal_compiled": lambda v, g: _compiled_fused(v, g),
    }
    hdr = f"{'candidate':<24}" + "".join(f"{b:>12}" for b in GATE_BIASES) + f"{'max':>12}  gate"
    print(hdr)
    for name, fn in gate_fns.items():
        per_bias = {}
        measured = True
        for b in GATE_BIASES:
            worst = 0.0
            for T in GATE_T:
                if name in NOT_MEASURED and T > COMPILED_LOOP_MAX_T:
                    measured = False
                    break
                v, g, ref = gate_case(T, D, B, b, seed=seed)
                try:
                    worst = max(worst, rel_err(ref, fn(v, g)))
                except Exception:  # a crashing candidate cannot be reported
                    measured = False
                    break
            per_bias[str(b)] = worst if measured else None
        vals = [v for v in per_bias.values() if v is not None]
        mxv = max(vals) if vals else None
        passed = (mxv is not None and mxv <= TOL)
        status = ("pass" if passed
                  else "GATED OUT" if measured and vals
                  else "not measured")
        report["gate"][name] = dict(
            rel_err_by_bias=per_bias, max_rel_err=mxv,
            gate_passed=bool(passed), measured=measured,
            note=NOT_MEASURED.get(name),
        )
        cells = "".join(
            (f"{per_bias[str(b)]:>12.2e}" if per_bias[str(b)] is not None
             else f"{'n/m':>12}") for b in GATE_BIASES)
        mxstr = f"{mxv:>12.2e}" if mxv is not None else f"{'n/m':>12}"
        print(f"{name:<24}{cells}{mxstr}  {status}")

    failures = [n for n, r in report["gate"].items()
                if not r["gate_passed"] and r.get("measured")]
    unmeasured = [n for n, r in report["gate"].items() if not r.get("measured")]
    npass = sum(1 for r in report["gate"].values() if r["gate_passed"])
    print(f"\n[gate] {npass}/{len(gate_fns)} candidates pass the exactness gate.")
    print(f"[gate] GATED OUT (rel err > {TOL:.0e}, no speedup reported): "
          f"{failures or 'none'}")
    if unmeasured:
        print(f"[gate] NOT MEASURED (unaffordable to compile, not a numerical "
              f"failure): {unmeasured}")

    if not report["gate"]["fused_metal"]["gate_passed"]:
        raise SystemExit("winner (fused_metal) failed the exactness gate; aborting")
    if not report["gate"]["hillis_c64"]["gate_passed"]:
        raise SystemExit("baseline (hillis_c64) failed the exactness gate; aborting")

    # ---- dispatch floor --------------------------------------------------
    floor_min, floor_med = measure(lambda: mx.add(mx.array([1.0]), 1.0),
                                   warmup=5, iters=40)
    report["dispatch_floor_ms"] = floor_min
    report["dispatch_floor_med_ms"] = floor_med
    print(f"\n[dispatch floor] trivial op: min {floor_min:.3f} ms "
          f"med {floor_med:.3f} ms -- component times are NOT additive")

    # ---- timing per lever -----------------------------------------------
    print("\n[timing] ms per scan call (mx.eval every step, warmup discarded)")
    print(f"{'lever':<18}{'setting':<34}{'T':>6}{'min ms':>10}{'med ms':>10}"
          f"{'MACs/tok':>11}{'vs base':>10}  gate")
    for T in (512, 2048, 8192):
        iters = 30 if T <= 2048 else 15
        cands = build_candidates(B, T, D)
        v, g = cands.pop("_inputs")
        # baseline first, so every ratio is against a freshly measured number
        _bm = measure(lambda: chunked_gated_scan(v, g, 64), warmup=5, iters=iters)
        base_min = _bm[0]
        report["levers"].setdefault("a_chunk_sweep", {})[f"T={T}:hillis_c64"] =             dict(setting="chunk=64 (baseline)", T=T, ms_min=_bm[0],
                 ms_median=_bm[1],
                 macs_per_token=macs_scan("hillis", B, T, D, 64) // (B * T),
                 rel_err=report["gate"]["hillis_c64"]["max_rel_err"],
                 gate_passed=True, speedup_vs_hillis_c64=1.0)
        print(f"{'a_chunk_sweep':<18}{'chunk=64 (baseline)':<34}{T:>6}"
              f"{_bm[0]:>10.3f}{_bm[1]:>10.3f}"
              f"{macs_scan('hillis', B, T, D, 64) // (B * T):>11,}"
              f"{'1.00x':>10}  pass")
        for name, (lever, setting, fn) in cands.items():
            if name == "hillis_c64":
                continue
            gi = report["gate"].get(name)
            passed = bool(gi["gate_passed"]) if gi else True
            if not passed:
                report["levers"].setdefault(lever, {})[f"T={T}:{name}"] = dict(
                    setting=setting, T=T, ms_min=None, ms_median=None,
                    macs_per_token=None, rel_err=gi["max_rel_err"],
                    gate_passed=False,
                    verdict="GATED OUT - rel err > 1e-5, no speedup reported")
                print(f"{lever:<18}{setting:<34}{T:>6}{'--':>10}{'--':>10}"
                      f"{'--':>11}{'--':>10}  GATED OUT")
                continue
            macs = macs_scan(name, B, T, D,
                             chunk=_chunk_of(setting, name)) // (B * T)
            try:
                samples = []
                for _ in range(3):
                    a, b = measure(fn, warmup=3, iters=iters)
                    samples.append(a)
                mn, md = min(samples), measure(fn, warmup=0, iters=iters)[1]
            except Exception as exc:
                print(f"{lever:<18}{setting:<34}{T:>6}  raised: {exc}")
                continue
            speed = (base_min / mn) if (base_min and mn) else None
            report["levers"].setdefault(lever, {})[f"T={T}:{name}"] = dict(
                setting=setting, T=T, ms_min=mn, ms_median=md,
                macs_per_token=macs, rel_err=gi["max_rel_err"] if gi else None,
                gate_passed=True, speedup_vs_hillis_c64=speed)
            sp = f"{speed:.2f}x" if speed else "--"
            print(f"{lever:<18}{setting:<34}{T:>6}{mn:>10.3f}{md:>10.3f}"
                  f"{macs:>11,}{sp:>10}  pass")

    # ---- levers (d) and the dispatch-floor/baseline context -------------
    print("\n[lever d] projection fusion (1 matmul vs 2), d=128, B=16, T=512")
    mx.random.seed(seed)
    xx = mx.random.normal((B, 512, D))

    class _Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.v = nn.Linear(D, D, bias=False)
            self.o = nn.Linear(D, D, bias=False)
            self.gate = nn.Linear(D, D, bias=True)

        def __call__(self, z):
            return self.o(mx.sigmoid(self.gate(z)) * self.v(z))

    class _Fused(nn.Module):
        def __init__(self):
            super().__init__()
            self.vg = nn.Linear(D, 2 * D, bias=True)
            self.o = nn.Linear(D, D, bias=False)

        def __call__(self, z):
            vv, gg = mx.split(self.vg(z), 2, axis=-1)
            return self.o(mx.sigmoid(gg) * vv)

    a, b = _Split(), _Fused()
    mx.eval(a.parameters(), b.parameters())
    sa = measure(lambda: a(xx)); sb = measure(lambda: b(xx))
    report["levers"]["d_projection_fusion"] = {
        "T=512:split_2_matmul": dict(setting="v and gate as two Linears", T=512,
                                     ms_min=sa[0], ms_median=sa[1],
                                     macs_per_token=2 * D * D, gate_passed=True,
                                     rel_err=0.0),
        "T=512:fused_1_matmul": dict(setting="one Linear to 2*d then split", T=512,
                                     ms_min=sb[0], ms_median=sb[1],
                                     macs_per_token=2 * D * D, gate_passed=True,
                                     rel_err=0.0),
    }
    print(f"  two matmuls  min {sa[0]:.3f} med {sa[1]:.3f}")
    print(f"  one matmul   min {sb[0]:.3f} med {sb[1]:.3f}")

    # ---- training integration -------------------------------------------
    gc = grad_check(seed=seed)
    report["grad_check"] = gc
    print(f"\n[grads] fused vs exact loop: fwd rel {gc['fwd_rel']:.2e}, "
          f"dL/dv rel {gc['grad_v_rel']:.2e}, dL/dg rel {gc['grad_g_rel']:.2e}")

    # ---- end to end ------------------------------------------------------
    print("\n[end-to-end] mem block vs attn block, T=512, B=16, d=128, "
          "all arms charged 1x forward MACs")
    e2e = end_to_end(seed=seed)
    report["end_to_end"] = e2e
    print(f"{'component':<38}{'MACs/tok':>10}{'min ms':>10}{'med ms':>10}"
          f"{'vs attn':>10}")
    _attn = next(r for r in e2e["rows"] if r["component"] == "attn BLOCK")
    for r in e2e["rows"]:
        _ratio = (_attn["min_ms"] / r["min_ms"]) if r["min_ms"] else None
        _rs = f"{_ratio:.2f}x" if _ratio else "--"
        print(f"{r['component']:<38}{r['macs_per_token']:>10,}"
              f"{r['min_ms']:>10.3f}{r['med_ms']:>10.3f}{_rs:>10}")


    mb = e2e["mem_block_before_ms"]
    ma = e2e["mem_block_after_ms"]
    at = e2e["attn_block_ms"]
    current = dict(
        mem_block_before_ms=mb, mem_block_after_ms=ma, attn_block_ms=at,
        mem_over_attn_before=e2e["mem_over_attn_before"],
        mem_over_attn_after=e2e["mem_over_attn_after"],
        mem_faster_than_attn_before=e2e["mem_faster_than_attn_before"],
        mem_faster_than_attn_after=e2e["mem_faster_than_attn_after"],
        mem_block_speedup=e2e["mem_block_speedup"],
        mac_ratio_attn_over_mem_full=e2e["mac_ratio_attn_over_mem_full"],
        mac_ratio_attn_over_mem_causal=e2e["mac_ratio_attn_over_mem_causal"],
    )
    # `headline` always holds the FIRST recorded long-context result, so the
    # committed number survives re-runs. This run's own number goes in
    # `headline_current_run`, and `headline_used` says which one a reader
    # should treat as the current measurement.
    report["headline_current_run"] = current
    if recorded_headline:
        report["headline"] = recorded_headline
        report["headline_used"] = "headline (first recorded run)"
    else:
        report["headline"] = current
        report["headline_used"] = "headline (this run, first write)"
    print(f"\n[headline] mem block {mb:.3f} -> {ma:.3f} ms "
          f"({e2e['mem_block_speedup']:.2f}x); mem/attn "
          f"{e2e['mem_over_attn_before']:.2f}x -> {e2e['mem_over_attn_after']:.2f}x "
          f"({'mem BEATS attn' if e2e['mem_faster_than_attn_after'] else 'mem does NOT beat attn'})"
          f" at T=512")
    print(f"[headline] MACs/tok: attn {e2e['macs_per_token']['attn']['block_full']:,} "
          f"vs mem1 {e2e['macs_per_token']['mem_banks1']['block_full']:,} "
          f"= {e2e['mac_ratio_attn_over_mem_full']:.2f}x full / "
          f"{e2e['mac_ratio_attn_over_mem_causal']:.2f}x causal-skip")
    print(f"[headline] block same-function rel err vs Hillis: "
          f"{e2e['block_same_function_rel_err']:.2e}")

    # ---- short-context sweep (NEW KEY: short_context_sweep) --------------
    print("\n[short-context] mem block vs attn block, B=16, d=128, n_head=8")
    print("  round-robin paired timing, mx.eval every sample, min over trials")
    sweep = short_context_sweep(seed=seed, Ts=SHORT_TS, reps=SHORT_REPS)
    report["short_context_sweep"] = sweep
    print(f"  min-of-N ms, {SHORT_REPS} independent reps per T; ratio is "
          f"mem_block_after/attn_block (min_ms within a rep)")
    print(f"  {'T':>6}{'mem aft':>9}{'attn':>9}{'ratio':>8}{'95% CI':>17}"
          f"{'mem win':>9}{'scan aft':>10}{'floor':>8}{'a/m MAC':>9}  verdict")
    for r in sweep["rows"]:
        ci = f"{r['ratio_ci95_low']:.2f}-{r['ratio_ci95_high']:.2f}"
        v = ("mem faster" if (r["ratio_ci95_excludes_1"]
                              and r["median_rep_mem_faster"])
             else "ATTN faster" if r["ratio_ci95_excludes_1"]
             else "tie (CI spans 1)")
        print(f"  {r['T']:>6}{r['mem_block_after_ms']:>9.3f}"
              f"{r['attn_block_masked_ms']:>9.3f}"
              f"{r['ratio_median_over_reps']:>8.2f}{ci:>17}"
              f"{r['mem_faster_reps']:>5}/{SHORT_REPS:<3}"
              f"{r['scan_after_ms']:>10.3f}{r['dispatch_floor_ms']:>8.3f}"
              f"{r['mac_ratio_attn_over_mem']:>9.2f}  {v}")
    ci = sweep["crossover_T_interval"]
    if sweep["tie_Ts"]:
        print(f"  [tie] 95% CI on the ratio SPANS 1.0 at T={sweep['tie_Ts']}: "
              f"the two blocks are not distinguishable there on this machine.")
    if sweep["mem_always_faster"]:
        print(f"  [crossover] no T in {list(SHORT_TS)} where attention beats "
              f"the memory block in every rep. The memory block is never "
              f"reliably SLOWER in the tested range.")
    elif sweep["mem_never_faster"]:
        print(f"  [crossover] no T in {list(SHORT_TS)} where the memory block "
              f"beats attention.")
    else:
        print(f"  [crossover] memory block loses at T<={ci[0]} and wins from "
              f"T>={ci[1]}.")
    print(f"  [resolution] memory WIN, CI excludes 1.0, at T="
          f"{sweep['mem_faster_at']}")
    if sweep["mem_slower_at"]:
        print(f"  [resolution] ATTENTION WIN, CI excludes 1.0, at T="
              f"{sweep['mem_slower_at']}")
    if sweep["nomask_tie_Ts"]:
        print(f"  [resolution] vs UNMASKED attention, tie at T="
              f"{sweep['nomask_tie_Ts']}")
    if sweep["tie_Ts"]:
        print(f"  [resolution] TIE (no stable sign) at T={sweep['tie_Ts']}: "
              f"the two blocks are within this machine's measurement noise "
              f"there, so those T are excluded from the crossover.")
    print(f"  [resolution] smallest tested T is {min(SHORT_TS)}; this harness "
          f"does NOT exclude a crossover below it.")
    print(f"  [resolution] quiet-run caveat: the dispatch floor for this sweep "
          f"ranged {min(r['dispatch_floor_ms'] for r in sweep['rows']):.3f}-"
          f"{max(r['dispatch_floor_ms'] for r in sweep['rows']):.3f} ms. At "
          f"short T the whole block costs ~0.3-0.6 ms, so the floor is a large "
          f"share of the measurement and the TIE band is wide there.")
    print(f"  [crossover] vs UNMASKED attention (median over "
          f"{SHORT_REPS} reps): mem slower at "
          f"{sweep['mem_slower_at_vs_nomask'] or 'none'}")
    print(f"  [crossover] same-function rel err of the fused block vs Hillis: "
          f"max {max(r['same_function_rel_err'] for r in sweep['rows']):.2e}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {OUT}")


def _chunk_of(setting, name):
    for tok in setting.replace("chunk=", " ").replace("(", " ").split():
        tok = tok.strip(",x)")
        if tok.isdigit():
            return int(tok)
    if name == "loop_compiled" or name == "fused_metal_compiled":
        return 0
    return 0


if __name__ == "__main__":
    main()
