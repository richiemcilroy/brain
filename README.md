# Testing recurrent memory against pretrained transformer attention

A pretrained transformer uses attention over the past. During cached
autoregressive generation it **stores prior keys and values**, rather than
recomputing all prior projections, but each new query still reads a growing
cache. Prefill and training also pay for causal attention over the sequence.
A gated recurrent trace instead carries a bounded state forward.

This repository tests whether swapping attention for that mechanism — and
against a pretrained model — actually helps, with an **adversarial control at
every step**.

**The current result:** a bounded, query-addressable state beside an exact
64-token local window recovers some quality lost by converting one pretrained
1B Llama layer. A small teacher-output transfer improves held-out Shakespeare
and WikiText quality; a matched rank-8 next-token adjustment then overfits
and worsens fresh quality. The converted model still trails the unchanged
teacher and has not shown a material complete-model speed gain. Earlier
recurrent-transplant results also failed an 8B replication and matched static
adapter control. The results and controls are retained below so each claim can
be checked.

Everything runs locally on one Apple M4 Max, 128 GB, MLX/Metal. No cluster.
The first [complete-model cached-inference baseline](docs/OSS_BASELINE.md)
finds **no material throughput gain** from an untrained one-layer replacement:
the cache shrinks, but held-out perplexity worsens and peak Metal memory rises.
The [matched-adapter control audit](docs/CONTROL_STATUS.md) further limits the
claim that the injected trace itself improves next-token quality.
The [local-window conversion](docs/LOCAL_HYBRID.md) keeps a bounded cache.
A signed recurrent gain beats its local-only control on four fresh cached
windows, but the unchanged 1B teacher still has better quality and no
complete-model speed gain was verified. A separate [five-seed small-model
training test](docs/TRAINING_PARETO.md) reaches the same loss in less time
and estimated compute. That training signal has not transferred to the
pretrained 1B model.
The newer [query-addressable local/global conversion](docs/QUERY_GLOBAL.md)
uses an exact 64-token window and a constant-size global feature state.
Attention-output training improves five fixed WikiText-2 test windows without
WikiText training data, but teacher perplexity remains better and the
one-layer serving benchmark does not yet show a speed win.
The [matched rank-8 LoRA follow-up](docs/QUERY_GLOBAL_LORA_RESULT.md)
overfit on the tiny Shakespeare adjustment set: query state still beat
local-only LoRA on nine fresh windows, but every LoRA arm lost to the
unchanged teacher and query LoRA worsened its frozen stage-1 map.
The [broader two-text LoRA run](docs/QUERY_GLOBAL_LORA_BROAD_RESULT.md)
improved fresh WikiText quality but degraded Shakespeare, including its
full-attention control, and query-state training was about 9% slower.
A [balanced feature-map transfer](docs/QUERY_GLOBAL_TRANSFER_BROAD_RESULT.md)
then reduced attention-output error on every fresh prefix but slightly
worsened next-token quality on both texts; its predeclared gate rejected
the new map.
A [direct next-token map adjustment](docs/QUERY_GLOBAL_MAP_NLL_RESULT.md)
selected after one update and remained behind the teacher on Shakespeare,
WikiText and an unadjusted literary text, so its three-text gate also failed.
The [layer and complete-model timing profile](docs/QUERY_GLOBAL_PERFORMANCE.md)
finds a layer-only prefill crossover near 32K context, but that one
attention layer is only about 3.5% of complete-model prefill time there.
An [exact one-token decode path](docs/QUERY_GLOBAL_FAST_DECODE.md) sped up
the converted layer by about 17–18% but gave no reliable whole-model gain.

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
from **20.3756 to 19.8922 perplexity** on held-out text. The newer matched
rank-64 static adapter improves it further, so this injection result is not
evidence that the added recurrent memory is necessary. When attention is
*deleted* and the memory must take over, a carrier whose weights are
**transplanted from the attention it replaced** recovers **38.6%** of the
lost performance, while an otherwise identical carrier with **random weights**
makes things *worse than deleting attention entirely* (−53.1%).

That is the narrow 1B transplant result. **At 8B it does not replicate**: under the
same held-out selection protocol the transplanted carrier *loses* to the random
control at every control seed ([§4](#4-does-it-survive-at-8b-no--and-that-is-the-most-important-result-here)).
This repository reports both, because the failure to replicate is the more
informative finding — it says the earlier win was about decay selection rather
than about the transplanted weights. The obvious defence that the 8B test is
merely underpowered is **refuted by measurement**: in nats the 8B deletion gap is
*1.35x larger* than the 1B one, and yet transfer recovers −28% of it instead of
+35%. So are three other candidate explanations — output alignment, grouped-query
expansion, and scale matching — each tested and each coming back clean. The
remaining suspect is the single-layer design of the 8B test.

### Headline numbers (all measured, all reproducible)

| experiment | teacher | our arm | control | verdict |
|---|---|---|---|---|
| **Injection** (attention kept, 1B, layer 8) | 20.3756 | **19.8922** | 19.9987 (random); newer matched static adapter beats transfer | an added trainable branch helps; recurrent memory has not shown a unique gain |
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
than to "having a trainable branch at all" *against that random-branch control*.
A newer five-seed rank-matched **static attention-output adapter** does better
than the transferred memory branch in every seed
([control audit](docs/CONTROL_STATUS.md)). Thus the random comparison is not
enough to establish a unique gain from the memory content.

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

### The obvious explanation is wrong

The natural defence is "the 8B test is underpowered". **The data refutes it.**
Perplexities are not comparable across models, but *nats* are, and in nats the
8B test has **more** headroom, not less:

| | 1B | 8B |
|---|---|---|
| deletion cost (nats) | 0.1681 | **0.2276** (1.35x larger) |
| transfer's recovery of that gap | **+34.9%** | **−28.4%** |

If 8B recovered the same fraction of its (larger) gap, transfer would land at
**7.54**. It landed at 8.71 — worse than deleting attention outright. A larger
signal with a worse outcome is not a power problem.

Three further candidate explanations were tested and **also refuted**:

| hypothesis | test | result |
|---|---|---|
| Transplanted weights don't align with attention at 8B | cosine similarity of carrier output vs the attention it replaces, on real activations | **+0.49 at 1B, +0.44 at 8B** — essentially the same; random control ≈0.00 at both |
| Grouped-query expansion picks the wrong head width | compare module `head_dim` to the true `d / n_heads`, and the inferred `n_kv` to the model config | **correct at both**: 1B 64/32/8, 8B 128/32/8 |
| The arms are being compared on scale, not content | rms normalisation landing error | **0.00%** at 1B and 8B — every arm hits its target exactly |

So the effect is not absent for want of signal, alignment, expansion correctness,
or scaling. **The one structural difference not yet excluded is that only one
layer was replaced.** At 1B, deleting layer 8 of 16 removes a large share of the
model's total attention; at 8B, deleting layer 16 of 32 leaves 31 attention
layers intact around it, and the surviving layers may simply route around the
damage. That is a hypothesis under test, not a conclusion — a multi-layer 8B
sweep is running, and an independent adversarial review has been commissioned to
try to break the null on six specific harness mechanisms and to confirm the
grouped-query expansion against an explicit routing simulation rather than
shapes. **Until those land, the honest summary is: the transplant advantage is a
1B-scale finding that does not currently reproduce at 8B, and the most likely
remaining explanation is the single-layer design of the 8B test, not scale.**

## 4b. What does *not* work: the obvious fixes are refuted

`docs/IMPROVE.md` records two hypotheses I formed for making the carrier better,
both of which **measurement killed**. They are kept because a result that only
lists survivors is not a result.

1. **"The decay fit uses the wrong objective."** Refuted: least squares and the
   repo's CDF-L1 criterion select the *same* decay (g = 0.99999). The objective
   was never the problem.
2. **"The carrier needs more timescales."** Refuted, and this one is important
   because it was the main hypothesis. Attention's real recency profile is
   heavy-tailed and is fit **15x better** by a set of banks clustered near 1
   (cdf_l1 7.50) than by the shipped default (cdf_l1 111.27). That fit does not
   translate: at matched bank count, `experiments/banks_test.py` measures

   | arm | decays | val ppl | fit cdf_l1 |
   |---|---|---|---|
   | **single bank** | (0.8,) | **22.7305** | — |
   | near-1 banks ×4 | (0.98 … 0.99999) | 23.9123 | 25.87 |
   | default geometric ×4 | (0.90 … 0.9999) | 23.9761 | 70.94 |
   | optimal placement ×4 | (0.99245 … 0.99999) | 23.9761 | **30.56** |

   **A single bank beats every multi-bank arm, including the one whose fit is 3x
   better.** Fit quality and perplexity are *anti*-correlated here. Multi-bank is
   strictly worse, which also explains why `hybrid_decay.json` contains no
   `banks_multi` rows — they should stay unrun.

The likely reason, and it is a prediction rather than a result: with `banks=4`
the trace is 4d wide, so the same rank-64 readout and the same rms budget are
spread across four timescales. The *learned* readout, not the fixed kernel, is
what recovers the local structure. That suggests testing a near-field
convolution (which adds the fast component without widening the trace) and a
larger readout, rather than more timescales.

---

## 5. Efficiency: measured against the fused kernel

**The earlier efficiency claims in this repo were retracted**, because they were
measured against hand-rolled attention rather than MLX's fused
`mx.fast.scaled_dot_product_attention`. The correction is in
`docs/RETRACTION_FUSED_BASELINE.md`; the clean re-measurement is in
`docs/FAIR_BENCH.md`.

`experiments/fair_bench.py` re-measures with the fair baseline, and -- because
this is the third time a handicapped baseline has fooled this project -- it
**verifies before it times**. Both attention arms share their projections and
differ only in the attention op, so they must compute the same function; measured
discrepancy **1.6e-7 relative**. The chunked scan is checked against the
sequential recurrence to **4.8e-7**. The run aborts on disagreement.

That check earned its keep immediately: it caught that
`mx.fast.scaled_dot_product_attention(mask=None)` is **unmasked** (it attends to
the future), so the first attempt was timing two different functions.

### The result (B=4, min-of-15, load 5.6-7.1, 15/15 rows clean)

| d | T | fused attn | memory | ratio | winner |
|---|---|---|---|---|---|
| 512 | 512 | 0.701 ms | 0.984 | 1.40 | attention |
| 512 | 2048 | 3.204 | 3.553 | 1.11 | attention |
| 512 | 4096 | 10.158 | 8.459 | **0.83** | **memory** |
| 2048 | 512 | 6.088 | 6.703 | 1.10 | attention |
| 2048 | 2048 | 29.990 | 27.839 | **0.93** | **memory** |
| 2048 | 4096 | 84.721 | 56.581 | **0.67** | **memory** |
| 4096 | 1024 | 65.970 | 47.438 | **0.72** | **memory** |
| 4096 | 4096 | 347.762 | 249.055 | **0.72** | **memory** |

Full 15-point table in `docs/FAIR_BENCH.md`.

- The memory block is faster than fused attention at **7 of 15** points, and
  **3 of 3 at T=4096** (1.20x, 1.50x, 1.40x at d=512/2048/4096).
- It is **slower at 8 of 15**, including at *every* width at T=512.
- The fused kernel is a median **1.56x** faster than the hand-rolled arm the
  earlier results were compared against -- exactly the margin that flattered them.

### The corrected claim

> Replacing a causal-attention sublayer with a gated multi-timescale memory
> sublayer is **slower at short context and faster at long context**. Against
> MLX's fused attention at B=4, the memory block wins at every width tested at
> T=4096 and loses at every width at T=512.

That is narrower than "our architecture is more efficient", and it is what the
data supports. A two-parameter fit of the form
`(a*d^2 + b*T*d) / (4*d^2 + 2*T*d)` reproduces the measured ratios to an RMSE of
0.085-0.135 across ratios spanning 0.67-1.40, and predicts crossovers in the
right region for d=512 and d=2048.

**Still not claimed:** any end-to-end training or inference speedup on a real
model. This measures isolated sublayers at B=4. A real transformer layer also has
LayerNorms, residual adds and an MLP, and 16-32 such layers -- none of which is in
this measurement.

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
- `docs/RETRACTION_FUSED_BASELINE.md` — the efficiency claims were measured
  against hand-rolled attention, not the fused kernel
- `docs/FAIR_BENCH.md` — the clean re-measurement against the fused baseline
- `docs/IMPROVE.md` — how to improve the carrier, including two refuted hypotheses
- `docs/PRIOR_ART.md` — what is genuinely new vs. already published
- `docs/THREEFACTOR.md` — the falsified thesis experiment
- `docs/README_ARCHIVE_substrate.md` — the earlier brain-simulation README

## 10. References

Prior art, provenance and the falsification criteria are in
[`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) and
[`docs/PARADIGM.md`](docs/PARADIGM.md).
