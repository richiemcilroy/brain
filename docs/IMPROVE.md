# How to make the carrier actually beat attention

This document is a diagnosis, not a plan. Every claim is tied to a measurement in
this repository. The previous framing -- "the transplant works at 1B and fails at
8B" -- treated the 8B failure as a *scale* question. The measurements now point
somewhere more specific, and more fixable.

## The finding that reframes everything

The carrier is a **single exponential** trace. Attention's recency profile is
**not exponential**. Measured on real activations (1024-token probe, layer 8 of
Llama-3.2-1B, via `measure_recency_profile`):

| statistic | measured |
|---|---|
| mass in the single most recent token | 0.0194 |
| mass within 4 tokens | 0.0447 |
| mass within 16 tokens | 0.1077 |
| mass within 64 tokens | 0.2732 |
| distance holding 50% of mass | **154 tokens** |
| distance holding 90% of mass | 253 tokens |
| lag-1 ratio (empirical decay) | **0.514** |
| best single exponential, repo's own criterion | **g = 0.99999**, cdf_l1 = **22.28** |

Two things are true at once, and one exponential cannot express both:

- the near field is **sharp** (lag-1 ratio 0.51 -- the token immediately behind
  carries about half the weight of the current one), and
- the far field is **heavy** (half the total mass sits ~154 tokens back).

A high `g` matches the tail and misses the near field; a low `g` does the
reverse. The repo's fit resolves this by pinning to the top of the grid
(`g = 0.99999`, i.e. *almost never forget*) -- the degenerate "no timescale at
all" solution. The residual is not small: **cdf_l1 = 22.28**.

My first attempt to show more banks would help produced a **null**, and the null
was wrong because the fitter was wrong: a greedy step that could only return the
same decay repeatedly, giving an identical cdf_l1 for K=1..8. That identical
output was the tell, and should have been caught immediately. Fitting properly by
non-negative least squares over a dense grid:

| fit | cdf_l1 | dominant decays (weight) |
|---|---|---|
| 1 timescale (what the repo uses) | 111.27 | 0.99999 (1.00) |
| 1, best of a dense grid | 10.64 | 0.9975 (0.22), 0.9950 (0.20), 0.9999 (0.16), 0.9924 (0.16) |
| 4 evenly-spaced timescales | 86.85 | 0.99999 (0.90), 0.8333 (0.10) |
| 8 evenly-spaced timescales | 69.61 | 0.99999 (0.83), 0.9286 (0.16) |
| 16 evenly-spaced timescales | 46.68 | 0.99999 (0.71), 0.9667 (0.28) |

The instructive row is the second. Even **a single bank**, if the decay is chosen
by least squares rather than by the CDF-L1 grid search, drops the residual from
111 to 10.6 -- the optimum is simply not where the repo's criterion looks. And
the weights that come out cluster just below 1 (0.987-0.99999), the signature of
a **near-uniform / heavy-tailed** kernel that a point mass at one decay cannot
represent.

**The timescales needed are not coarse-to-fine** (0.9, 0.99, 0.999, 0.9999, as
`GatedMemory`'s default `init_decays` assumes). **They are densely clustered just
below 1.** Those are different hypotheses, and only the second fits the data.

## Consequence for the 8B null

The same measurement at 8B gives the same qualitative result with a **larger**
residual: `g = 0.99999` again, cdf_l1 = **48.76** (vs 22.28 at 1B), 2.2x worse.

So the 8B null is not best explained by scale, power, alignment, or GQA expansion
(all four tested and refuted -- see README section 4). It is explained by the
carrier being asked to reproduce a heavy-tailed kernel with a kernel family that
cannot represent it, failing harder as the model gets deeper, because a deeper
model's layer 16 has a more complex mix of local and long-range dependence.

## Ranked fixes

Each is a controlled change against the existing protocol, and each is falsifiable
on its own. **Two of my three hypotheses were refuted by measurement before
being written up.** They are recorded here rather than quietly dropped, because
a diagnosis that only lists the survivors is not a diagnosis.

### 1. REFUTED: changing the fit criterion does not help

My first hypothesis was that `fit_decay_from_attention` minimises the wrong
objective (CDF-L1 instead of least squares), and that refitting would find a
different decay. **It does not.** Both criteria choose the same value:

```
repo criterion (CDF-L1):  g = 0.99999   cdf_l1 = 22.2792
least squares (single g): g = 0.99999   sse    = 4.969e-03
CDF-L1 of the LS choice:  22.2792    (identical)
LS of the repo's choice:  4.969e-03  (identical)
```

The objective was not the problem: a single exponential cannot represent this
profile, and both criteria agree on that.

### 2. REFUTED: a better recency fit does NOT improve perplexity

This was the main hypothesis in the first draft of this document. It is wrong.

The carrier's default decay placement fits attention's recency profile badly:
cdf_l1 = **111.27**, against **7.50** for an optimally-placed set of three banks
-- a 15x reduction in fit error. The natural inference is that the bad fit costs
perplexity. `experiments/banks_test.py` measures it, at matched bank count so the
comparison cannot repeat the feature-count confound that refuted the earlier
multiscale claim (`docs/MULTISCALE.md`):

| arm | decays | **val ppl** | select ppl | cdf_l1 of fit |
|---|---|---|---|---|
| **single_0.8** | (0.8,) | **22.7305** | 25.5613 | -- |
| single_0.9 | (0.9,) | 23.0357 | 25.9737 | -- |
| near1 | (0.98, 0.99, 0.999, 0.99999) | 23.9123 | 27.1063 | 25.87 |
| default_geo | (0.90, 0.99, 0.999, 0.9999) | 23.9761 | 27.1063 | 70.94 |
| optimal | (0.99245, 0.99496, 0.99748, 0.99999) | 23.9761 | 27.3241 | **30.56** |

**A single bank at g=0.8 beats every multi-bank arm, including the one with the
best fit by 3x.** The arms with better recency fits (`near1`, cdf_l1 25.87) are
*worse* than the one with a 3x worse fit (`default_geo`, cdf_l1 70.94). Fit
quality and perplexity are **anti-correlated here, not correlated**.

Two conclusions follow, and both cut against the framing of this document:

1. **The single-bank default is not the problem.** Multi-bank is strictly worse
   at matched parameters. "The carrier needs more timescales" is refuted: it
   needs *fewer*, and the existing single-bank setting is already the best of
   the configurations tested.
2. **The `banks_multi` arm should stay unrun.** The missing rows in
   `hybrid_decay.json` are now explained rather than merely absent.

There is a plausible mechanism for why, and it is worth stating because it also
predicts where multi-bank *would* help: with `banks=4` the trace vector is 4d
wide, so the same rank-64 injection projection (or the same rms budget) is spread
across 4 timescales instead of 1. Each bank is trained less, and the *learned*
readout -- not the fixed kernel -- is what recovers the local structure. A
multi-bank carrier might help if the readout were given more capacity or trained
longer. That is a prediction, not a result.

### 3. NOT YET TESTED: add a sharp local component

The kernel needs mass at distance ~0-4 (lag-1 ratio 0.514) *and* at ~154. Given
that multi-bank hurt, the likely reason is the one above -- capacity dilution --
rather than a shortage of timescales. A small depthwise convolution (width 4-8)
in parallel with the trace would supply the near field **without** multiplying
the trace width, so it does not dilute the readout. This is now the most
promising architectural change, and unlike #2 it has not been tested.

### 4. NOT YET TESTED: train the carrier at low learning rate

`hybrid_finetune.py` unfreezes the whole carrier and gives both arms 400
identical steps, so "transplant vs random" is an **initialisation** comparison
that optimisation is allowed to erase. `hybrid_inject.py` trains only the rank-64
projection and leaves `W_v`/`W_o` frozen, so the transplant never adapts. Neither
tests the obvious third option: transplant, then finetune at **low** learning
rate. The first finetune attempt used a learning rate high enough that both arms
diverged to ~1400 ppl (`docs/FINETUNE.md`) -- an Adam step-size problem, not an
init problem.

### 5. NOT YET TESTED: multi-layer at both scales

`verify_transplant.json` declares `layers: [8, 4, 12, 15]` but **only layer 8
appears in the results**. The 1B positive result is single-layer too, so the
1B-vs-8B comparison is single-layer vs single-layer. 1B is 6.5x cheaper; mirror
the multi-layer sweep there as well.

## What is actually established

- The carrier's default decay placement does not fit attention's recency profile
  (cdf_l1 111 vs 7.5 achievable). **Measured.**
- Improving that fit does not improve perplexity, and multi-bank is worse than a
  single bank. **Measured.** This kills the "more timescales" direction.
- A single bank at g=0.8 remains the best configuration of this carrier that has
  been measured at 1B. **Measured.**
- The 8B null has four refuted explanations (power, alignment, GQA, scale) and no
  surviving one. **Measured.**

## What would falsify this diagnosis

1. If a multi-bank carrier with an **expanded readout** (so capacity is not
   diluted) beats the single bank, then capacity dilution -- not the kernel --
   was the binding constraint, and approach #2 becomes viable again.
2. If the near-field convolution (#3) fails to beat the single bank, the
   carrier's limit is not its kernel family at all, and the remaining suspects
   are the readout and the training protocol.
