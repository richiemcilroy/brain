> ## !!! CORRECTION, READ THIS FIRST !!!
>
> The numbers below were measured with the gate decay **hardcoded to 0.99** while
> the code claimed it was "fitted from the measured recency profile". The fit
> never ran: an MLX API mismatch raised inside a `try/except` and fell through to
> the constant on every layer and every run. The tell is that every committed
> result carries `recency_profile_first8: null` and `fit_cdf_l1: null`.
>
> 0.99 turns out to be **near the worst point on the module's own curve**. When
> the decay is swept honestly (chosen on a held-out SELECT split), the transplant
> recovers **33%** of the deletion gap instead of the 10% reported below, and the
> matched random control is worse than deleting attention at **every** decay
> tested rather than only at one setting.
>
> The layer sweep in "Result 2" below was likewise run at the wrong decay and
> should not be quoted. See **`docs/HYBRID_DECAY.md`** for the corrected
> measurements and the full account. The text below is retained as the record of
> what was claimed at the time, not as current results.

# Does our brain method improve a real pretrained LLM?

**Question.** Not "is our model better than Llama" — ours is 18,000x smaller and it
is not. The answerable question: *can our gated-memory primitive carry part of a
pretrained attention layer's function, using fewer parameters?*

**Short answer: yes, but weakly and not uniformly.** Transplanting attention's
weights into our module beats deleting attention entirely, and beats a
matched random control by a wide margin — but it recovers only ~10-30% of the
damage at the layers where it works, and it makes things *worse* at other layers.

Teacher: `unsloth/Llama-3.2-1B` (1,235,814,400 params, 16 layers, GQA 32 query / 8 kv
heads). Evaluation: token-level perplexity on 3000 held-out TinyShakespeare tokens,
deterministic (no sampling), so these are exact numbers. Scripts:
`experiments/llm_hybrid.py`, `experiments/hybrid_sweep.py`.

## Why surgery on a pretrained model is the right test

Every previous comparison in this repo trained from scratch on 1.1 MB of
Shakespeare, so "who overfits least" dominated the result — and an independent
review showed the headline comparison *replicates* Feng et al. arXiv:2410.01201.
A pretrained model already knows how to language-model, so swapping into it
isolates **the primitive**, not the training budget. That is the confound this
project has been unable to remove all session.

## What transfers, and precisely how

Attention and our gated memory are both linear maps over the same residual
stream with the same output width, so the surrounding RMSNorm and MLP accept our
output unchanged:

    attention:  out = softmax(q k^T / sqrt(d)) V · W_O
    gated mem:  h_t = sigmoid(W_g x_t) · h_{t-1} + W_v x_t , then W_o

Both value paths are compositions of two linear maps of the same shape, so
`W_v <- W_V` and `W_o <- W_O` is a real transplant, not an analogy. Three details
are load-bearing, and each cost a wrong attempt:

1. **GQA expansion.** With 32 query heads and 8 kv heads, query heads `j*4..j*4+3`
   each receive kv head `j`'s `head_dim`-wide block. That means repeating the
   **head block contiguously** — not `np.tile` (duplicates the whole matrix) and
   not `np.repeat` along rows (interleaves). **All three produce identical
   shapes**, so a shape assertion cannot tell them apart; only a value check can.
   `expand_gqa` carries the ground-truth construction it was verified against,
   and is checked on 6 layouts including `n_rep=1`.

2. **The trace is an unnormalised sum, attention is a convex combination.** With
   decay 0.99 the trace is ~100x too large and blows up the residual stream.
   Scaling the value path by `(1-g)` makes it an exponential moving average
   (`sum_k (1-g) g^k = 1`), which is the same normalisation RG-LRU applies.

3. **The random control must be output-matched.** A raw random module at
   `1/sqrt(d)` produces output rms **2.18x** the attention it replaces, and the
   transplant produces **0.44x**. Comparing those compares *scales*, not
   information. An earlier version of this file did exactly that and reported the
   random arm at ppl 1620 — which measures nothing. The control now rescales to
   match the replaced module's measured output rms (achieved ratio **1.019**).

## Result 1: the matched control (layer 8 of 16, 3 seeds)

| arm | ppl | vs teacher | vs zero |
|---|---|---|---|
| teacher (unmodified) | **20.3756** | — | — |
| `zero` (attention deleted) | 24.1044 | +3.7288 | — |
| `random_output_matched` | **74.7101** | +54.3345 | 3.2x **worse** than deleting |
| **`transfer`** | **23.7169** | +3.3413 | **beats zero** |

The standard deviation across seeds is under 0.005 for `transfer` (23.7228 /
23.7137 / 23.7142), and `zero` is deterministic (24.1044 three times), because
evaluation uses no sampling.

**This is the decisive result.** A random module with the *same architecture,
same fitted decay, and same output scale* is 3.2x worse than simply deleting
attention — while the transplant beats deletion. So the benefit is genuinely
carried by **attention's weights**, not by "any recurrence with a sensible time
constant". A correctly-matched control turns a meaningless comparison into a
decisive one.

## Result 2: it is layer-dependent, and not uniformly positive

5 layers, 1 seed each. Positive gain = transfer better than deleting attention.

| layer | zero | transfer | gain | recovers |
|---|---|---|---|---|
| 0 | 513.9267 | 354.0462 | **+159.88** | 32.4% |
| 4 | 33.3760 | 51.5984 | −18.22 | −140% |
| 8 | 24.1044 | 23.7228 | +0.38 | 10.2% |
| 12 | 23.0972 | 25.3469 | −2.25 | −83% |
| 15 | 23.2208 | 22.3732 | **+0.85** | **29.8%** |

It beats deletion at layers **0, 8, 15** and loses at **4, 12**. The layer-0
number is the largest in absolute terms but the weakest evidence, because the
zero-ablation there is catastrophic (ppl 514) — when deletion destroys the model,
beating it is a low bar. The meaningful wins are layer 8 (+0.38) and layer 15
(+0.85, recovering 29.8% of the deletion damage).

## What this does and does not establish

**Established:** our gated-memory module, initialised from a pretrained
attention layer's own weights, carries measurable function — it beats both
deleting that layer and a rigorously matched random control, and the effect
reproduces across three seeds.

**Not established, and these are the honest limits:**

- **This is not "better than a transformer".** Recovery is 10-30% of the damage
  at the good layers, and negative at others. The teacher is still strictly
  better everywhere.
- **One layer of 16, and no finetuning in this result.** The transplant is
  frozen. Whether finetuning recovers the rest — and whether it recovers *more*
  from a transplanted init than from a random one — is the next experiment and
  is not answered here.
- **Perplexity on Shakespeare is in-distribution degradation**, not generation
  quality.
- **No end-to-end speedup is claimed.** The module is cheaper per token in
  theory (flat in context vs quadratic), but at 512 tokens and `d=2048` no
  wall-clock claim has been measured for this configuration, and
  `docs/WALLCLOCK.md` shows this class of comparison is kernel-count-bound at
  short context.
- **Llama-3.2-1B is 1.2B parameters.** Nothing here speaks to behaviour at
  frontier scale.
