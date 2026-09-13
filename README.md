# Making a pretrained LLM better by giving it a brain's memory

A pretrained transformer processes the past by recomputing it. Every token, it
rebuilds its context from scratch through the attention matrix. A brain does
something different: it carries a **running trace** of the past that decays at
multiple timescales, updated in one operation per token.

This repository tests whether swapping attention for that mechanism — and
against a pretrained model — actually helps, with an **adversarial control at
every step**.

**The short version: it works at 1B and fails to replicate at 8B.** Both results
are reported here with their controls, because the interesting part of a result
is what survives when you try to kill it, and the failure to replicate is the
more informative of the two.

Everything runs locally on one Apple M4 Max, 128 GB, MLX/Metal. No cluster.

> **Scope, up front.** This is not a simulated human brain and does not claim to
> be one. The 86-billion-neuron gap cannot be closed on a laptop, and the Human
> Brain Project spent ~EUR 607M over ten years establishing that a full replica
> is neither achievable nor of clear practical use at any near-term scale. What
> *is* portable across scale is a **mechanism**. This repo isolates one
> mechanism, implements it at full precision, and measures it against the
> strongest control available. An earlier phase of this project built a spiking
> substrate to test a different hypothesis (three-factor local plasticity for
> continual learning) and **falsified it**; that work is archived at
> [`docs/README_ARCHIVE_substrate.md`](docs/README_ARCHIVE_substrate.md) and its
> negative results are part of the record.

---

## The one-paragraph version

A gated multi-timescale memory trace, inserted beside attention as a
**zero-initialised low-rank branch**, improves a frozen pretrained Llama-3.2-1B
from **20.3756 to 19.8922 perplexity** on held-out text. When attention is
*deleted* and the memory must take over, a carrier whose weights are
**transplanted from the attention it replaced** recovers **38.6%** of the
lost performance, while an otherwise identical carrier with **random weights**
makes things *worse than deleting attention entirely* (−53.1%).

That is the 1B result, and it is real. **At 8B it does not replicate**: under the
same held-out selection protocol the transplanted carrier *loses* to the random
control at every control seed ([§4](#4-does-it-survive-at-8b-no--and-that-is-the-most-important-result-here)).
This repository reports both, because the failure to replicate is the more
informative finding — it says the earlier win was about decay selection rather
than about the transplanted weights.

### Headline numbers (all measured, all reproducible)

| experiment | teacher | our arm | control | verdict |
|---|---|---|---|---|
| **Injection** (attention kept, 1B, layer 8) | 20.3756 | **19.8922** | 19.9987 (random) | any trained low-rank branch helps; only 22% of the gain is from our weights |
| **Finetune** (attention deleted, 1B, layer 8) | 20.3756 | **22.6638** | 26.0826 (random) | transplant recovers 38.6%; random is worse than deletion |
| **Decay sweep** (same, decay chosen per arm) | 20.3756 | 23.0357 @ g=0.6 | 25.2227 @ g=1.0 | interior optimum for transplant; control pinned to the grid edge |
| **8B replication** (attention deleted, Llama-3.1-8B-4bit, layer 16) | 6.4999 | 8.7064 | **8.6875** (random) | **the effect does NOT replicate at 8B** — see below |

The 8B row is why this repo is honest rather than promotional. Details in
[§4](#4-does-it-survive-at-8b-no--and-that-is-the-most-important-result-here).

---

## 1. What the mechanism actually is

```
x  ->  x + attn(ln1(x))        [unchanged, pretrained]
x  ->  x + up(down(mem(ln1(x))))   [our branch]
```

`mem` is a **gated multi-timescale trace**: `h_t = g * h_{t-1} + v_t`, with the
decay `g` set per-channel, and `v_t = W_v x_t`, read out through `W_o`.

Three design choices matter, and each was forced by a measured failure:

1. **Zero-initialised output projection.** `up` starts at exactly zero, so the
   branch contributes nothing and the model **is** the teacher at step 0. This is
   asserted in code (`abs(before_val - base_val) < 1e-12`) and the run aborts if
   it fails. Without this property, "our model beats the teacher" is
   unfalsifiable — you cannot tell improvement from a lucky initialisation.
2. **Low rank (r=64).** A full-rank projection is 4.19M parameters on a few
   hundred training batches. The first attempt measured exactly that failure:
   training loss fell 4.31 → 3.37 while **validation perplexity exploded 20.4 →
   69.1**. That run was not testing whether the module helps; it was measuring
   how fast 4M parameters memorise 800 samples. At rank 64 the branch is 262,144
   parameters — **0.02%** of the 1.24B model.
3. **Normalisation to attention's output scale.** Memory output rms is ~64x
   attention's (measured 3.23 vs 0.0505). Adam's step size is *absolute*, not
   relative, so an unscaled branch is badly conditioned and blows up within 100
   steps. The normalisation lives in a fixed non-parameter `out_gain`, not in
   `W_v`, precisely so no optimizer can move it.

---

## 2. The result that can be falsified: injection

`experiments/hybrid_inject.py` → `experiments/results/hybrid_inject.json`

Attention is **untouched**. Only the low-rank injection projection trains.
3000 held-out validation tokens, deterministic (no sampling).

| arm | val perplexity | vs teacher |
|---|---|---|
| teacher (unmodified) | **20.3756** | — |
| **transfer** (our weights) | **19.8922** | **−0.4834** |
| summary (learned EMA, no transplant) | 19.9454 | −0.4302 |
| random (matched random weights) | 19.9987 | −0.3769 |

**The honest decomposition.** A trained low-rank branch helps almost regardless
of its content: `random` already gains 0.3769. Only **0.1065 of the 0.4834**
improvement — about **22%** — is attributable to the transplanted weights rather
than to "having a trainable branch at all". A reader who quotes −0.4834 as
"our method's gain" is quoting a number that mostly measures the branch, not the
brain-derived content.

---

## 3. The decisive experiment: delete attention and see who survives

`experiments/hybrid_finetune.py` → `results/hybrid_finetune.json`

If the mechanism is real, it should be able to *replace* attention rather than
sit beside it. Attention is removed. Both arms get the identical finetuning
protocol; the only difference is whose weights start in the carrier.

| arm | val perplexity | recovery of the deletion gap |
|---|---|---|
| teacher | 20.3756 | — |
| **transfer** | **22.6638** | **+38.6%** |
| random (matched) | 26.0826 | **−53.1%** |

Deleting attention costs ~3.73 perplexity. The transplanted carrier recovers
about 38.6% of that. A matched random carrier is **worse than deleting attention
entirely** — it does not merely fail to help, it actively harms. This is a much
stronger separation than the injection result, because the control has every
advantage except the weights.

### The decay was never fitted, and that changed the conclusion

`experiments/hybrid_decay.py` → `results/hybrid_decay.json`

The gate decay `g` had been silently hardcoded to 0.99 through a `try/except`
around an API call that always raised — the tell being `recency_profile_first8:
null` in every result file. Fitting it properly:

| arm | best decay | perplexity at best | perplexity at the old 0.99 |
|---|---|---|---|
| transfer | **0.6** (interior optimum) | **23.0357** | 24.2981 |
| random_matched | 1.0 (grid edge) | 25.2227 | — |

**The committed constant 0.99 was near the transfer arm's *worst* point.** The
control is worse than outright deletion at every decay. An earlier version of
this project's documentation quoted a layer-dependence sweep that had been run at
this wrong decay; that sweep is not quoted here and carries a retraction banner.

### Multi-seed adversarial verification

`experiments/verify_transplant.py` → `results/verify_transplant.json`

An independent implementation, re-deriving the protocol rather than importing
it, sweeping **8 decay values × 6–8 control seeds**:

- transfer best **22.7305** @ g=0.8, control best **25.2227** @ g=1.0
- margin **+2.49 ppl**; across the control-seed sweep, mean 25.0007 ± 0.2841
- **`n_seeds_beating_transfer: 0`** — no control seed beat transfer
- minimum paired margin **+1.83 ppl**, all paired margins positive

---

## 4. Does it survive at 8B? No — and that is the most important result here

`experiments/hybrid_8b.py` → `results/hybrid_8b.json`, `results/hybrid_8b_b.json`

Llama-3.1-8B-Instruct (4-bit) is 6.5x larger, uses **grouped-query attention**
(8 KV heads, 32 query heads — the value projection is not `d×d` and must be
expanded), and is **quantized**, a code path the bf16 1B model never exercised.
Getting it running required fixing two silent-corruption bugs, both documented in
[§6](#6-three-bugs-that-produced-plausible-wrong-numbers).

**At 8B the effect does not replicate under the protocol that worked at 1B.**

Teacher 6.4999, attention deleted (zero ppl 8.1610), layer 16 of 32. Raw sweep —
as in §3, every decay reported so nothing is hidden:

| decay | transfer | random (matched) | gap |
|---|---|---|---|
| 0.5 | **9.0001** | 9.9353 | −0.9353 (transfer wins) |
| 0.9 | **9.3239** | 10.0252 | −0.7013 (transfer wins) |
| 0.99 | 9.7499 | 9.7743 | −0.0244 (tie) |
| 1.0 | 8.7064 | **8.6875** | +0.0188 (random wins) |

If you pick the decay by looking at validation, transfer wins at 0.5. **That is
not the protocol.** The 1B result selected the decay on a held-out `SELECT`
split and reported once on `VAL`, specifically to avoid this. Applying the same
rule at 8B:

| arm | decay chosen on SELECT | SELECT ppl | **VAL ppl** |
|---|---|---|---|
| transfer | 1.0 | 13.6154 | 8.7064 |
| random_matched | 1.0 | 13.6291 | **8.6875** |

SELECT picks 1.0 for both arms, and at 1.0 **transfer loses**. The control-seed
sweep at that decay is unambiguous: random gives **8.6875 / 8.5353 / 8.5496**
against transfer's **8.7064** — **3 of 3 control seeds beat it**, mean 8.5908.
The 1B result had `n_seeds_beating_transfer: 0` with a minimum margin of +1.83
ppl. At 8B the sign reverses.

**Why this matters more than the 1B win.** A result that appears at two decay
values chosen post hoc, and disappears at the decay chosen by held-out
selection, is a result about *decay choice*, not about transplanted weights.
Note also that deleting attention costs only **1.66 ppl** at 8B (6.4999 → 8.1610)
against **3.73 ppl** at 1B — a 32-layer model with 15 other attention layers is
far less dependent on any single one, so the 8B test is a harder place to see an
effect, and it may simply be underpowered. But the honest summary is the plain
one: **the transplant advantage is a 1B-scale finding and does not currently
reproduce at 8B.** The two candidate explanations — an underpowered single-layer
test, or a genuine scale limit — are not distinguished by this run, and
distinguishing them is the obvious next experiment.

## 5. Efficiency: what is measured and what is still unknown

**No end-to-end speedup is claimed.** A gated trace has fewer multiply-accumulates
than attention at short context, and the scan was the wall-clock bottleneck —
measured, then fixed.

`experiments/scan_bench.py` → `results/scan_bench.json`, `docs/WALLCLOCK.md`

- The memory block went from **1.41x slower** than attention to **1.56x faster**
  (mem block 9.867 ms → 4.484 ms; attention block 6.990 ms), a **2.20x**
  speedup on the memory block itself.
- MAC ratio, attention over memory: **1.82x** (full masked T×T) or **1.45x**
  (kernel skips masked work). An earlier version of this repo claimed 2.9x;
  that was a **unit error** — attention was charged FLOPs while memory was
  charged MACs, double-counting attention by exactly 2x. Caught by review.
- Chunk-size sweep: `chunk=16` gives a further **1.16x** over the `chunk=64`
  baseline with fewer MACs; `chunk=256` is *slower* than the baseline.
- Short-context sweep (T=4…2048): the ratio is **noisy below T≈256 and often
  favours attention** — measured ratios of memory-block/attention-block time are
  1.84 (T=4), 1.32 (T=32), 1.10 (T=48), i.e. attention faster at those lengths,
  and 0.51 (T=8), 0.76 (T=16), 0.73 (T=64), 0.98 (T=128). From **T=256 upward
  the memory advantage is consistent and grows monotonically**: ratios 0.64,
  0.36, 0.26, 0.17 at T=256/512/1024/2048. An independent 5-rep reproduction
  agrees on every shared T except T=32, where it found a small memory win
  against the primary sweep's tie; the conservative reading is kept. **There is no clean short-context
  crossover claim here** — the sub-256 region is dominated by dispatch overhead
  and noise.
- An independent reproduction of this result was commissioned and its verdict is
  in [§7](#7-what-is-not-claimed).

**The honest position:** these are kernel-level and block-level measurements.
Whether the advantage survives in a full training run at realistic context
lengths is **not established here**, and the block-level win is small enough that
fusion and dispatch overheads could erase it.

---

## 6. Three bugs that produced plausible wrong numbers

Each of these returned a confident, well-formatted, incorrect result. They are
documented because the failure mode — not the bug — is the reusable lesson.

1. **Dtype promotion silently changed the model.** The model is bfloat16; our
   `nn.Linear` returned float32; so `x + mem(x)` promoted the entire residual
   stream to float32. A **multiply-by-zero** no-op moved perplexity from 27.5975
   to 27.8236 — a "no-op" that changed the answer. Fix: cast the branch back to
   `x.dtype`. Verified bit-identical (max diff 0.0). See `docs/DTYPE_BUG.md`.
2. **`module.unfreeze()` is recursive.** Freezing the model and unfreezing our
   wrapper also unfroze the *pretrained attention* inside it. The run trained the
   whole layer, drove perplexity to **1889**, and left the model corrupted for
   every subsequent arm. Fix: unfreeze only `down`/`up`, then **assert** the
   trainable set is exactly those two tensors.
3. **Quantized weights are packed, not dense.** A 4-bit `QuantizedLinear` stores
   a logical 4096×4096 projection as uint32 **(4096, 512)** plus scales/biases.
   Reading `.weight` as dense is silently wrong — and the width heuristic then
   inferred `d=512` instead of 4096, which routed the transplant into a
   zero-padded fallback that copied a 512×512 corner and **reported success**.
   Fixed by dequantizing properly (`mx.dequantize`, verified against the
   module's own forward pass to 8.4e-4) and by replacing the silent fallback
   with a hard error. The bf16 1B model never hit this; the 8B model hit it
   immediately.

---

## 7. What is *not* claimed

- **Not "better than a transformer".** No end-to-end training-run speedup is
  demonstrated. The wins are: a 0.48 ppl injection gain (78% of which is not
  ours), a 38.6% recovery when attention is deleted, and block-level speedups.
- **Not a brain simulation.** This is one mechanism inspired by cortical
  memory, isolated and measured. The full spiking substrate that preceded it
  produced *negative* results and is archived, not deleted.
- **Not novel as an architecture.** Gated multi-timescale memory is the
  mechanism of Feng et al. (arXiv:2410.01201) and the gate form predates this
  work (SRU/QRNN/GILR); looped depth was settled by ALBERT and Saunshi et al.
  See `docs/PRIOR_ART.md`. The contribution here is the **transplant protocol,
  the adversarial controls, and an honest decomposition of the gain.**
- **Three-factor plasticity was falsified**, not confirmed, by this project's own
  thesis experiment (`docs/THREEFACTOR.md`): `criterion.verdict = "FALSIFIED"`.
  Eligibility-trace rank stays at 1 regardless of hidden width.
- **Single seed, single layer, one corpus** for the 1B results. Multi-layer
  injection (1 of 16 layers) is the largest untested dimension.

---

## 8. Quickstart

```bash
cd human-brain

# validation suite (118 tests: neuron dynamics, plasticity direction,
# connectivity, determinism, exact spike compaction)
python3 -m pytest tests/ -q

# the falsification experiments over a language model (~4 GB model download)
export HF_HOME=~/zbrain/hf
python experiments/hybrid_inject.py      # injection: attention kept
python experiments/hybrid_finetune.py    # attention deleted; transplant vs random
python experiments/hybrid_decay.py       # the decay sweep
python experiments/verify_transplant.py  # adversarial multi-seed verification
python experiments/scan_bench.py         # block-level wall-clock benchmarks

# 8B replication (needs ~4.2 GB, runs from anywhere HF_HOME points)
python experiments/hybrid_8b.py

# the archived spiking substrate
python3 experiments/xor_neuron.py        # single-neuron XOR from dendrites
python3 bench/bench_scale.py             # scale ceiling on this machine
```

Requires `numpy`, `mlx`, and `mlx-lm`. With no MLX the substrate half falls back
to NumPy (`BRAIN_BACKEND=numpy`).

---

## 9. Repository layout

- `experiments/hybrid_*.py` — the language-model experiments and their controls
- `experiments/results/*.json` — every number quoted above, machine-readable
- `experiments/scan_bench.py`, `bench/` — wall-clock and scale measurements
- `brain/` — the spiking substrate (archived line of work, 118 tests passing)
- `docs/INJECTION.md`, `FINETUNE.md`, `HYBRID_DECAY.md` — per-experiment detail
- `docs/DTYPE_BUG.md` — the dtype-promotion bug
- `docs/WALLCLOCK.md` — the MAC/FLOP unit error and its correction
- `docs/PRIOR_ART.md` — what is genuinely new vs. already published
- `docs/THREEFACTOR.md` — the falsified thesis experiment
- `docs/README_ARCHIVE_substrate.md` — the earlier brain-simulation README

## 10. References

Prior art, provenance and the falsification criteria are in
[`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) and
[`docs/PARADIGM.md`](docs/PARADIGM.md).
