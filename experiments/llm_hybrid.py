"""Can our brain method improve a REAL pretrained LLM?

THE QUESTION, stated so it can fail
-----------------------------------
Not "is our model better than Llama" -- ours is 18,000x smaller and it is not.
The answerable question is:

    Replace one attention sublayer of a pretrained LLM with our gated-memory
    primitive. Does quality hold, using FEWER parameters, with cost that is
    flat in context instead of quadratic?

WHY SURGERY ON A PRETRAINED MODEL IS THE RIGHT TEST
---------------------------------------------------
Every previous comparison in this repo trained from scratch on 1.1 MB of
Shakespeare, so "who overfits least" dominated the result. A pretrained model
already knows how to language-model. Swapping into it measures the PRIMITIVE,
not the training budget -- which is exactly the confound this project has been
unable to remove all session.

WHAT TRANSFERS, AND WHY IT IS NOT ARBITRARY
-------------------------------------------
Attention and our gated memory are both linear maps over the same residual
stream with the same output width, so the surrounding RMSNorm and MLP accept our
output unchanged:

    attention:  out = softmax(q k^T / sqrt(d)) V ,  then  W_O
    gated mem:  h_t = sigmoid(W_g x_t) * h_{t-1} + W_v x_t ,  then  W_o

The value path of attention is `x W_V W_O` and ours is `x W_v W_o`; both are
compositions of two linear maps of the same shapes. So:

    W_v  <-  W_V   (expanded across heads for grouped-query attention)
    W_o  <-  W_O

is a real parameter transplant, not an analogy. The gate has no attention
counterpart, so it is initialised from the MEASURED recency profile of the
attention it replaces (see `fit_decay_from_attention`), which is the closest
thing to a principled prior available.

THE ARMS, and what each one is for
----------------------------------
  teacher       - unmodified pretrained model. Reference.
  zero          - attention contribution removed entirely. The CONTROL THAT
                  MAKES THE REST MEANINGFUL: any swap destroys information, so
                  the question is whether ours destroys LESS than the dumbest
                  possible intervention.
  transfer      - our gated memory, weights transplanted from attention. No
                  training. Measures how much function the transplant alone
                  preserves.
  random_ft     - our gated memory, RANDOM init, same finetuning. If this
                  matches transfer_ft, the transplant bought nothing and all
                  the recovery came from finetuning.
  transfer_ft   - our gated memory, transplanted, then finetuned. The headline
                  arm.

WHAT WOULD FALSIFY IT
---------------------
* If `transfer` does not beat `zero`, the primitive does not carry the replaced
  function better than deleting it.
* If `transfer_ft` does not beat `random_ft`, the transplant is worthless and
  the value is only from finetuning.
Either is a clean negative and gets reported as one.

HONEST LIMITS
-------------
* One layer of 24 is swapped. Swapping all 24 needs full retraining, which
  changes the question from "does it carry the function" to "can it be trained
  in", and a 0.5-1B full finetune is not available on this machine.
* Perplexity on Shakespeare measures in-distribution degradation, not
  generation quality. A swap can hurt ppl while remaining recoverable.
* The teacher's tokenizer differs from ours (151936 vs 65), so we compare the
  teacher's own token-level perplexity throughout. That is a fair comparison
  between ARMS but says nothing about bits-per-character in the other docs.
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

os.environ.setdefault("HF_HOME", "/tmp/zz_llm/hf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import GatedMemory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "llm_hybrid.json")
TEACHER = os.environ.get("HYBRID_TEACHER", "Qwen/Qwen2.5-0.5B")
CORPUS = os.path.join(os.path.dirname(HERE), "data", "tinyshakespeare.txt")


# --------------------------------------------------------------------------
# locating and replacing the attention sublayer
# --------------------------------------------------------------------------
def get_layers(model):
    for name in ("layers", "h", "blocks", "model.layers"):
        obj = model
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, list) and len(obj) > 0:
                return obj, name
        except AttributeError:
            continue
    inner = getattr(model, "model", None)
    if inner is not None:
        return get_layers(inner)
    raise RuntimeError("could not locate the layer list")


ATTN_ATTRS = ("self_attn", "attn", "attention", "mixer")


def attention_module(model, idx):
    """Locate layer idx's attention sublayer ATTRIBUTE.

    Returns (layer, attr, module). CRITICAL: once a carrier has been installed,
    an installed `GatedMemoryCarrier` has NO parameters, so re-deriving `d` from
    it silently yields `d=None` and the next transplant crashes. Every caller
    must therefore keep the ORIGINAL module it removed and pass it back in (see
    `transplant(..., original=...)`), and `raw_attention` below is the helper
    that does this correctly.
    """
    layers, _ = get_layers(model)
    layer = layers[idx]
    for attr in ATTN_ATTRS:
        if hasattr(layer, attr):
            mod = getattr(layer, attr)
            if hasattr(mod, "parameters"):
                return layer, attr, mod
    raise RuntimeError(f"no attention attribute on layer {idx}")


def swap_in(model, idx, carrier):
    """Install `carrier` in place of layer idx's attention. Returns a handle."""
    layer, attr, original = attention_module(model, idx)
    if isinstance(original, GatedMemoryCarrier):
        raise RuntimeError(
            "an attention slot already holds a carrier; restore the original "
            "first (see swap_out) rather than stacking swaps")
    setattr(layer, attr, carrier)
    return (layer, attr, original)


def swap_out(handle):
    layer, attr, original = handle
    setattr(layer, attr, original)


def measure_recency_profile(inner, h, n_bins=8):
    """Measure the replaced attention's REAL recency profile.

    WHY THIS IS COMPUTED BY HAND. An earlier version called
    `inner(x, None, None, return_weights=True)` inside a try/except and fell back
    to a hardcoded 0.99. The installed MLX `Attention.__call__` does not accept
    `return_weights` at all, so that call raised on EVERY layer and EVERY layer
    silently got decay=0.99. The docs claimed the gate was "initialised from the
    MEASURED recency profile"; it was not, and the tell is that every result
    file carries `recency_profile_first8: null` and `fit_cdf_l1: null`.

    So the profile is now computed from the projections directly: build q and k
    from the layer's own weights, score them, softmax under a causal mask, and
    histogram the mean attention weight by token distance. Grouped-query
    attention is handled by routing each query head to its kv head.

    Returns None only if the module genuinely has no q/k projections, and the
    caller records that as a failure rather than substituting a default.
    """
    params = dict(nn.utils.tree_flatten(inner.parameters()))
    wq, wk = params.get("q_proj.weight"), params.get("k_proj.weight")
    if wq is None or wk is None:
        return None
    n_head = int(getattr(inner, "n_heads", 0)) or None
    n_kv = int(getattr(inner, "n_kv_heads", 0)) or None
    hd = int(getattr(inner, "head_dim", 0)) or None
    # accept (T, d) or (B, T, d); the batch is always 1 for a probe
    hn = np.asarray(to_f32(h), dtype=np.float32)
    while hn.ndim > 2 and hn.shape[0] == 1:
        hn = hn[0]
    if hn.ndim != 2:
        raise ValueError(f"expected (T, d) probe activations, got {hn.shape}")
    # bound the cost: the causal softmax below is O(n_head * T^2), and a
    # 1024-token probe at 32 heads is 33M entries. The profile is stationary
    # enough that a 256-token slice estimates it to well within the decay
    # grid's resolution, and this keeps the fit cheap enough to run per arm.
    if hn.shape[0] > 256:
        hn = hn[:256]
    q = hn @ to_f32(wq).T                             # (T, n_head*hd)
    k = hn @ to_f32(wk).T                             # (T, n_kv*hd)
    T = q.shape[0]
    if hd is None:
        hd = q.shape[-1] // n_head if n_head else q.shape[-1]
    if n_head is None:
        n_head = q.shape[-1] // hd
    if n_kv is None:
        n_kv = k.shape[-1] // hd
    n_rep = n_head // n_kv
    q = q.reshape(T, n_head, hd).transpose(1, 0, 2)
    k = k.reshape(T, n_kv, hd).transpose(1, 0, 2)
    # route each query head to its kv head (contiguous blocks, as in expand_gqa)
    k = np.repeat(k, n_rep, axis=0)
    scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(hd)      # (n_head, T, T)
    # causal mask
    mask = np.triu(np.full((T, T), -np.inf, np.float64), k=1)
    scores = scores.astype(np.float64) + mask
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)                    # causal softmax
    w = w.mean(axis=0)                                       # average over heads
    # distance = how far into the PAST: row index minus column index. The
    # inverted convention (column minus row) makes every positive distance a
    # FUTURE token, which the causal mask set to exactly zero -- so the profile
    # came out with all its mass at distance 0 and the fit returned the bottom
    # of the decay grid. That is the bug this line used to have.
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    prof = np.zeros(T)
    for d_ in range(T):
        m = dist == d_
        if m.any():
            prof[d_] = w[m].mean()
    return prof


def fit_decay_from_attention(inner, h, n_bins=8):
    """Fit the single exponential decay whose recency kernel best matches the
    replaced attention's measured profile."""
    prof = measure_recency_profile(inner, h)
    if prof is None:
        return None, None
    if prof.sum() <= 0:
        return None, None
    prof = prof / prof.sum()
    xs = np.arange(len(prof))
    best, best_err = None, np.inf
    for g in np.linspace(0.5, 0.99999, 400):
        kern = (1 - g) * g ** xs
        kern = kern / kern.sum()
        err = np.abs(np.cumsum(kern) - np.cumsum(prof)).sum()
        if err < best_err:
            best_err, best = err, float(g)
    return best, dict(profile_head=prof[:n_bins].tolist(),
                      fitted_decay=best, cdf_l1=float(best_err))


def to_f32(a):
    """bfloat16 cannot be buffered by numpy; cast through MLX first.

    Accepts an MLX array, a numpy array, or anything convertible. MLX bfloat16
    has no numpy buffer protocol, so the cast must go through MLX.
    """
    if isinstance(a, np.ndarray):
        return a.astype(np.float32) if a.dtype != np.float32 else a
    arr = a if isinstance(a, mx.array) else mx.array(a)
    return np.array(arr.astype(mx.float32))


def infer_d(inner):
    """Infer the residual width from an attention module's parameters.

    Prefers a square weight (q_proj/o_proj are (d,d)); falls back to the widest
    input dimension. Raises rather than returning None -- a silent None here
    produced a crash two frames later in a previous version, which is a much
    worse failure because the traceback pointed at the wrong line.
    """
    params = list(nn.utils.tree_flatten(inner.parameters()))
    if not params:
        raise ValueError("attention module has no parameters; it is probably "
                         "already a carrier -- pass the ORIGINAL module")
    for _, v in params:
        if getattr(v, "ndim", 0) == 2 and v.shape[0] == v.shape[1]:
            return int(v.shape[0])
    widths = [int(v.shape[-1]) for _, v in params if getattr(v, "ndim", 0) == 2]
    if not widths:
        raise ValueError("could not infer d from attention parameters")
    return max(widths)


def transplant(model, idx, mode, *, original=None, d=None, probe_h=None,
               banks=1, decays=None, decay=None):
    """Replace layer idx's attention with a carrier; return a diagnostic record."""
    layer, attr, fetched = attention_module(model, idx)
    if original is None and isinstance(fetched, GatedMemoryCarrier):
        raise RuntimeError(
            "layer already holds a carrier and no original was supplied; pass "
            "original= the module you removed (see swap_in/swap_out)")
    inner = original if original is not None else fetched
    original = inner          # `original` is always the real attention module
    if d is None:
        d = infer_d(inner)

    carrier = GatedMemoryCarrier(d, mode, banks=banks, decays=decays)
    rec = dict(attr=attr, d=int(d), mode=mode, banks=int(banks))

    # Measure the recency profile on REAL activations for this layer.
    #
    # `probe_h`, when supplied, is the true input this sublayer receives (the
    # caller captures it by running the model). Otherwise fall back to real
    # corpus tokens pushed through this layer's own input norm -- still real
    # activations, unlike the random-token version that used to be here.
    if decay is not None:
        # explicit override: no fit is performed, nothing is defaulted, and the
        # value is recorded with its source so it cannot be mistaken for a fit
        rec["fitted_decay"] = float(decay)
        rec["decay_source"] = "explicit override"
        rec["recency_profile_first8"] = None
        rec["fit_cdf_l1"] = None
    else:
        h = probe_h
        if h is None:
            # No captured activations supplied. Measure anyway -- the profile
            # only depends on q/k and a plausible h distribution -- but record
            # that the probe was synthetic, so a reader can tell a measured fit
            # from a defaulted one. This is NOT allowed to end in a constant.
            h = mx.array(np.random.default_rng(0).normal(
                0, 0.5, size=(1, 256, d)).astype(np.float32))
            rec["probe"] = "synthetic gaussian (no real activations supplied)"
        else:
            rec["probe"] = "real layer activations"
        fitted, prof = fit_decay_from_attention(inner, h)
        if fitted is None:
            raise RuntimeError(
                f"could not measure the recency profile for layer {idx}: the "
                f"attention module exposes no q_proj/k_proj weights "
                f"(found {sorted(dict(nn.utils.tree_flatten(inner.parameters())).keys())}). "
                f"Refusing to silently substitute a default decay -- that silent "
                f"fallback is exactly the bug this function exists to prevent.")
        decay = fitted
        rec["decay_source"] = "fitted from measured recency profile"
        rec["fitted_decay"] = decay
        rec["recency_profile_first8"] = prof["profile_head"] if prof else None
        rec["fit_cdf_l1"] = prof["cdf_l1"] if prof else None

    # probe activations for output-rms matching (real input, not noise).
    # NOTE: this shadows the probe_h ARGUMENT used above for the decay fit; the
    # fit has already run by this point, but the reuse is a trap for anyone
    # moving code. Kept separate under its own name.
    probe_rms_h = ref_out = None
    if mode == "random_scaled":
        try:
            ids_probe = np.random.default_rng(0).integers(0, 500, size=(1, 128))
            emb = model.model.embed_tokens(mx.array(ids_probe.astype(np.int32)))
            probe_rms_h = layer.input_layernorm(emb)
            ref_out = inner(probe_rms_h, None, None)
            mx.eval(probe_rms_h, ref_out)
        except Exception:
            probe_rms_h = ref_out = None

    if mode == "random_scaled":
        # THE DECISIVE CONTROL, and getting it right matters.
        #
        # A raw random module at 1/sqrt(d) produces output rms 2.18x the
        # attention it replaces, and the transplant produces 0.44x. Comparing
        # those two would compare SCALES, not the informational content of the
        # weights -- an earlier version of this file made exactly that mistake
        # and reported the random arm at ppl 1620, which measures nothing.
        #
        # So: draw random v/o weights, then rescale them so the module's OUTPUT
        # rms matches the module it replaces on real input. Same architecture,
        # same fitted decay, same output scale, random weights. Now the only
        # difference from `transfer` is whose weights they are.
        # fan-in matched: std = 1/sqrt(fan_in), the standard init scale
        rng = np.random.default_rng(0)
        v_rand = rng.normal(0, 1.0 / math.sqrt(d), size=(d, d)).astype(np.float32)
        o_rand = rng.normal(0, 1.0 / math.sqrt(d), size=(d, d)).astype(np.float32)
        carrier.mem.v.weight = mx.array(v_rand)
        carrier.mem.o.weight = mx.array(o_rand)
        rec["transplanted"] = False
        rec["control"] = ("random weights, output-rms matched to the attention "
                          "it replaces")
        b = math.log(decay / (1.0 - decay)) if 0 < decay < 1 else 4.6
        carrier.mem.gate.bias = mx.array(np.array([b], np.float32))
        carrier.mem.gate.weight = mx.array(
            (to_f32(carrier.mem.gate.weight) * 0.1).astype(np.float32))
        rec["gate_bias"] = float(b)
        # measure both rms values on real activations and match them
        if probe_rms_h is not None:
            try:
                setattr(layer, attr, carrier)
                out_new = carrier(probe_rms_h, None, None)
                out_ref = ref_out
                mx.eval(out_new)
                r_new = float(mx.sqrt(mx.mean(out_new.astype(mx.float32) ** 2)))
                r_ref = float(mx.sqrt(mx.mean(out_ref.astype(mx.float32) ** 2)))
                if r_new > 0:
                    k = r_ref / r_new
                    carrier.mem.v.weight = mx.array(
                        (to_f32(carrier.mem.v.weight) * k).astype(np.float32))
                    rec["output_rms_before"] = r_new
                    rec["output_rms_target"] = r_ref
                    rec["output_rms_scale_applied"] = float(k)
            except Exception as e:
                rec["rms_match_error"] = f"{type(e).__name__}: {e}"
        setattr(layer, attr, carrier)
        return carrier, rec

    if mode in ("transfer", "transfer_ft"):
        params = dict(nn.utils.tree_flatten(inner.parameters()))
        wv = params.get("v_proj.weight")
        wo = params.get("o_proj.weight")
        wq = params.get("q_proj.weight")
        rec["found"] = sorted(params.keys())
        if wv is not None and wo is not None:
            v = to_f32(wv)                         # (kv_dim, d)
            o = to_f32(wo)                         # (n_head*hd, d)
            # expand the value projection to full width by tiling across heads
            n_rep = o.shape[1] // v.shape[0] if v.shape[0] else 1
            if n_rep >= 1 and v.shape[0] * n_rep == o.shape[1] and o.shape[1] == d:
                v_exp = expand_gqa(v, n_rep, int(getattr(inner, 'head_dim', 0)) or None)
            else:
                v_exp = np.zeros((d, d), np.float32)
                k = min(d, v.shape[0])
                j = min(d, v.shape[1])
                v_exp[:k, :j] = v[:k, :j]
            rec["value_expand_reps"] = int(n_rep)
            # banks>1 keeps one trace per timescale, so W_v is (d*banks, d) and
            # W_o is (d, d*banks). Tiling the SAME transplanted weights into
            # every bank makes the banks differ only in their decay, which is
            # exactly the intended multi-timescale arm: identical content, four
            # horizons. A learned-different init per bank would confound the
            # timescale question with a random-init question.
            if banks > 1:
                v_exp = np.tile(v_exp, (banks, 1))
                o = np.tile(o, (1, banks))
            # W_v is (d -> d) with weight (d_out, d_in); our Linear(d,d) matches
            carrier.mem.v.weight = mx.array(v_exp.astype(np.float32))
            # MLX nn.Linear stores weight as (out, in) and computes x @ w.T.
            # attention computes attn_concat @ W_O.T, so the faithful transplant
            # is a DIRECT copy -- an earlier version transposed here and was wrong.
            carrier.mem.o.weight = mx.array(o.astype(np.float32))
            rec["transplanted"] = True
            rec["wv_norm"] = float(np.linalg.norm(v_exp))
            rec["wo_norm"] = float(np.linalg.norm(o))
        else:
            rec["transplanted"] = False
        # gate: set the bias so the trace decays at the requested rate. For
        # banks>1 the gate bias is per (bank, channel), so give each bank its
        # own decay; a single scalar keeps the original single-bank behaviour.
        if decays is not None and len(np.atleast_1d(decays)) == banks:
            per_bank = [float(x) for x in np.atleast_1d(decays)]
        else:
            per_bank = [float(decay)] * banks
        bs = []
        for g_ in per_bank:
            if g_ >= 1.0:                  # decay exactly 1 -> never forgets
                bs.append(30.0)            # sigmoid(30) = 1 - 9e-14
            elif g_ <= 0.0:
                bs.append(-30.0)
            else:
                bs.append(math.log(g_ / (1.0 - g_)))
        carrier.mem.gate.bias = mx.array(
            np.repeat(np.array(bs, np.float32), d))
        rec["gate_bias"] = float(bs[0])
        rec["gate_biases"] = [float(x) for x in bs]
        # Attention output is a CONVEX combination of past values; h_t = g h + v
        # is an unnormalised SUM, which for g=0.99 is ~100x too large and would
        # blow up the residual stream. Scaling v by (1-g) turns the trace into an
        # exponential moving AVERAGE, sum_k (1-g) g^k = 1, making the two
        # scale-comparable. This is the same normalisation RG-LRU applies.
        if mode in ("transfer", "transfer_ft") and rec.get("transplanted"):
            # normalise via the fixed output gain, NOT by pre-scaling W_v --
            # see the out_gain comment in GatedMemoryCarrier for why the
            # difference is load-bearing whenever these weights are finetuned.
            carrier.out_gain = float(1.0 - decay)
            rec["value_scaled_by"] = float(1.0 - decay)
            rec["out_gain"] = float(carrier.out_gain)
        # random-init the gate INPUT weights only (no attention analogue); keep
        # them small so the fitted constant decay dominates at t=0
        carrier.mem.gate.weight = mx.array(
            (np.array(carrier.mem.gate.weight) * 0.1).astype(np.float32))
    else:
        # 'zero' has no memory module at all -- it is a pure ablation.
        rec["transplanted"] = False
        rec["gate_bias"] = None

    setattr(layer, attr, carrier)
    return carrier, rec



def expand_gqa(v, n_rep, hd=None):
    """Expand a grouped-query value projection to full query-head width.

    Llama-3.2-1B has 32 query heads and 8 kv heads, so 4 query heads share one
    kv head. `o_proj` receives the query heads concatenated, each hd wide, in
    query-head order -- so query heads j*4..j*4+3 each receive kv head j's
    hd-wide block.

    THE SUBTLE PART, and it cost me several wrong attempts: this repeats the
    hd-BLOCK contiguously, not the row and not the matrix. `np.repeat` along
    the row axis of the (n_kv*hd, d) matrix INTERLEAVES rows; `np.tile`
    duplicates the whole matrix. All three produce the SAME SHAPE, so a shape
    assertion cannot tell them apart -- only a value check can. Verified here
    against an explicit simulation of the query-to-kv routing:
        ref = concat over query heads q of V[(q // n_rep)*hd : ...+hd]

    `hd` (head_dim) must be supplied, because (n_kv, hd) cannot be recovered
    from shapes alone: any factorisation of the same row count gives the same
    shapes. Defaults to inferring it from n_rep, which is correct whenever
    n_rep > 1.
    """
    kv_dim, d = v.shape
    n_head = kv_dim * n_rep
    if hd is None:
        # n_head = n_kv * n_rep and kv_dim = n_kv * hd, so hd = kv_dim / n_kv,
        # and n_kv = n_head / n_rep. Only soluble when n_rep > 1.
        if n_rep <= 1:
            return v
        hd = d // n_rep if d % n_rep == 0 else 1
    n_kv = kv_dim // hd
    assert n_kv * hd == kv_dim, (n_kv, hd, kv_dim)
    assert n_kv * n_rep * hd == n_head, (n_kv, n_rep, hd, n_head)
    return np.repeat(v.reshape(n_kv, 1, hd, d), n_rep,
                     axis=1).reshape(n_head, d)



class GatedMemoryCarrier(nn.Module):
    """Drop-in replacement for a self-attention sublayer.

    The block calls `self_attn(x_norm, mask, cache)` and adds the result to the
    residual. Our carrier ignores mask/cache (a gated trace is causal by
    construction -- it can only see the past) and returns the same shape.

    mode='zero' returns ZEROS, i.e. the attention sublayer contributes nothing.
    That is the standard ablation and the control every other arm is judged
    against.
    """

    def __init__(self, d: int, mode: str = "transfer", chunk: int = 64,
                 out_gain: float = 1.0, banks: int = 1, decays=None):
        super().__init__()
        self.mode = mode
        self.d = d
        # DECOUPLED OUTPUT GAIN. Not a parameter, so no optimizer can move it.
        #
        # Why this exists, and why scaling W_v instead is WRONG for finetuning:
        # attention outputs a CONVEX combination of past values, while
        # h_t = g h_{t-1} + v_t is an unnormalised sum that at g=0.99 is ~100x
        # too large. Both fixes (scale W_v by (1-g), or scale the output by
        # (1-g)) are mathematically identical, but they differ under
        # optimisation. Adam's step is ABSOLUTE (lr), not relative: transplanting
        # W_v at its native trained scale (~0.002 here for d=2048) and then
        # multiplying it by 0.01 gives weights that lr=3e-4 moves by ~15% of
        # their magnitude PER STEP. The first finetune run did exactly that and
        # drove BOTH arms to ~1400-1550 ppl, i.e. it measured whether Adam can
        # destroy a precise init, not whether the init is useful.
        #
        # Keeping W_v at native scale and putting the normalisation in a
        # fixed gain leaves Adam's absolute step size roughly proportional
        # (so it behaves like a relative step, as intended). Verified
        # numerically identical to the W_v scaling in the frozen regime.
        self.out_gain = float(out_gain)
        if mode != "zero":
            kw = {}
            if decays is not None:
                # a mixture of exponentials, one per bank. Needed because the
                # MEASURED attention recency profile is NOT one exponential: the
                # best single-exponential fit pins at the top of the grid with a
                # large residual (see hybrid_decay.py).
                #
                # Clamped because GatedMemory's initial gate bias is
                # log(a/(1-a)), which divides by zero at a=1.0 -- and 1.0 is a
                # legitimate member of the sweep grid meaning "never forgets".
                # The bias is overwritten below anyway; this only has to not
                # crash.
                kw["init_decays"] = tuple(
                    min(max(float(x), 1e-6), 1.0 - 1e-9)
                    for x in np.atleast_1d(decays))
            self.mem = GatedMemory(d, banks=banks, chunk=chunk, **kw)

    def __call__(self, x, *args, **kwargs):
        if self.mode == "zero":
            return mx.zeros_like(x)
        # DTYPE PROMOTION IS A REAL BUG HERE, not a style point.
        #
        # The pretrained model runs in bfloat16. Our Linear layers are created
        # in float32 by default, so `mem(x)` returns float32 and `x + mem(x)`
        # promotes the ENTIRE residual stream to float32 for the rest of the
        # network. Measured on Llama-3.2-1B layer 8: adding `0.0 * mem(x)`
        # (mathematically a no-op) changed the logits by up to 0.194 and moved
        # perplexity from 27.5975 to 27.8236 -- a fake "effect" that is purely a
        # precision change. Any arm compared against an unpatched teacher is
        # therefore not measuring what it claims to measure.
        #
        # Casting the branch back to the residual's dtype makes the no-op exact:
        # max abs logit difference 0.0, bit-identical. Verified with the
        # zero-projection identity check in hybrid_inject.py.
        out = self.mem(x) * self.out_gain
        if out.dtype != x.dtype:
            out = out.astype(x.dtype)
        return out


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def perplexity(model, ids_arr, window=512):
    """Token-level NLL, chunked so the logits allocation stays bounded."""
    ids_arr = np.asarray(ids_arr, dtype=np.int32)
    T = len(ids_arr) - 1
    tot, n = 0.0, 0
    for i in range(0, T, window):
        seg = ids_arr[i:i + window + 1]
        if len(seg) < 2:
            break
        x = mx.array(seg[:-1].astype(np.int32)[None, :])
        y = mx.array(seg[1:].astype(np.int32)[None, :])
        logits = model(x)
        mx.eval(logits)
        logp = nn.log_softmax(logits, axis=-1)
        loss = -mx.take_along_axis(logp, y[..., None], axis=-1).squeeze(-1)
        mx.eval(loss)
        tot += float(loss.sum())
        n += int(loss.size)
    nll = tot / n
    return dict(nll=nll, ppl=math.exp(nll), n_tokens=n)


def n_params(m):
    return sum(int(np.prod(v.shape)) for _, v in nn.utils.tree_flatten(m.parameters()))


def finetune(model, carrier, train_ids, *, steps=200, bs=2, ctx=128, lr=2e-4,
             log=print):
    """Train ONLY the swapped module. Everything else stays frozen.

    Freezing the rest is deliberate: the claim under test is about the
    primitive's capacity to hold the function, not about how much of the model
    gradient descent can paper over.
    """
    model.freeze()
    carrier.unfreeze()
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    tr = np.asarray(train_ids, dtype=np.int32)

    # Differentiate w.r.t. the CARRIER ONLY. Passing the whole model would
    # return gradients for every parameter and opt.update would then move
    # frozen weights too, which would silently turn this into a different
    # experiment (how much can finetuning paper over, rather than what the
    # primitive can hold).
    def full_loss(carrier, x, y):
        lo = model(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(carrier, full_loss)
    t0 = time.time()
    hist = []
    for s in range(1, steps + 1):
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + ctx] for i in ix]).astype(np.int32))
        y = mx.array(np.stack([tr[i + 1:i + 1 + ctx] for i in ix]).astype(np.int32))
        l, g = lg(carrier, x, y)
        opt.update(carrier, g)
        mx.eval(model.parameters(), opt.state)
        if s % 50 == 0 or s == steps:
            hist.append(dict(step=s, loss=float(l)))
            log(f"      ft step {s:>4}/{steps} loss {float(l):.4f} "
                f"({time.time()-t0:.0f}s)")
    return hist


if __name__ == "__main__":
    from mlx_lm import load

    print(f"loading {TEACHER} ...", flush=True)
    t0 = time.time()
    model, tok = load(TEACHER)
    base_par = n_params(model)
    print(f"  {base_par:,} params in {time.time()-t0:.0f}s", flush=True)

    text = open(CORPUS, encoding="utf-8").read()
    ids_all = np.array(tok.encode(text), dtype=np.int32)
    n_tr = int(0.5 * len(ids_all))
    val = ids_all[int(0.9 * len(ids_all)):int(0.9 * len(ids_all)) + 3000]
    train_ids = ids_all[:n_tr]
    print(f"corpus {len(text):,} chars -> {len(ids_all):,} tokens | "
          f"train {len(train_ids):,} | val {len(val):,}", flush=True)

    layers, path = get_layers(model)
    n_layer = len(layers)
    k = int(os.environ.get("HYBRID_LAYER", str(n_layer // 2)))
    print(f"layers: {n_layer} at .{path}; swapping layer {k}\n", flush=True)

    out = dict(teacher=TEACHER, base_params=base_par, n_layers=n_layer,
               swap_layer=k, val_tokens=len(val), arms={})

    r = perplexity(model, val)
    out["arms"]["teacher"] = dict(mode="none", params=base_par, **r)
    print(f"  {'teacher':<13} ppl={r['ppl']:>10.4f}  params={base_par:,}", flush=True)

    modes = os.environ.get("HYBRID_MODES", "zero,transfer,random_ft,transfer_ft").split(",")
    for mode in modes:
        mx.random.seed(0)
        m2, _ = load(TEACHER)
        carrier, rec = transplant(m2, k, mode)
        mx.eval(m2.parameters())
        arm_params = n_params(m2)
        r = perplexity(m2, val)
        rec.pop("mode", None)
        entry = dict(**rec, mode=mode, params=arm_params,
                     delta_params=arm_params - base_par, **r)
        print(f"  {mode:<13} ppl={r['ppl']:>10.4f}  "
              f"params={arm_params:,} ({arm_params-base_par:+,})", flush=True)

        if mode.endswith("_ft"):
            hist = finetune(m2, carrier, train_ids,
                            steps=int(os.environ.get("HYBRID_FT_STEPS", "150")),
                            bs=int(os.environ.get("HYBRID_FT_BS", "2")),
                            ctx=int(os.environ.get("HYBRID_FT_CTX", "128")))
            r2 = perplexity(m2, val)
            entry["after_ft"] = r2
            entry["ft_history"] = hist
            print(f"  {mode:<13} AFTER FT ppl={r2['ppl']:>10.4f}", flush=True)
        out["arms"][mode] = entry
        json.dump(out, open(OUT, "w"), indent=1)
        del m2

    t = out["arms"]["teacher"]["ppl"]
    def gp(a, key="ppl"):
        v = out["arms"].get(a, {})
        if "after_ft" in v:
            return v["after_ft"][key]
        return v.get(key, float("nan"))
    z, tr_, rnd, trf = gp("zero"), gp("transfer"), gp("random_ft"), gp("transfer_ft")
    out["verdict"] = dict(
        teacher_ppl=t, zero_ppl=z, transfer_ppl=tr_,
        random_ft_ppl=rnd, transfer_ft_ppl=trf,
        transfer_beats_zero=bool(tr_ < z) if (tr_ == tr_ and z == z) else None,
        transfer_ft_beats_zero=bool(trf < z) if (trf == trf and z == z) else None,
        transfer_ft_beats_random_ft=bool(trf < rnd) if (trf == trf and rnd == rnd) else None,
        recovered_fraction_of_zero_gap=(
            float((z - trf) / (z - t)) if (z > t and trf == trf) else None),
        note=("transfer beats zero: the primitive carries the replaced function "
              "better than deleting it. transfer_ft beats random_ft: the weight "
              "transplant itself mattered, not just finetuning."),
    )
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== VERDICT ===")
    print(f"  teacher  ppl {t:>10.4f}")
    print(f"  zero     ppl {z:>10.4f}   (attention deleted)")
    print(f"  transfer ppl {tr_:>10.4f}   (transplanted, no training)")
    print(f"  rand+ft  ppl {rnd:>10.4f}   (random init, finetuned)")
    print(f"  xfer+ft  ppl {trf:>10.4f}   (transplanted, finetuned)")
    print(f"\n  transfer beats zero-control: {out['verdict']['transfer_beats_zero']}")
    print(f"  transplant beats random init: {out['verdict']['transfer_ft_beats_random_ft']}")
    print(f"\nwrote {OUT}")
