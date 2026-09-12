# Prior art the project was missing, and what it means for our claims

Written after an adversarial fact-check (independent model, given the measured
numbers and asked to attack them). Three of its findings are corrections to
things this repo asserts. One is a lead worth running. All are recorded here
because a claim that has not been checked against the literature is not a claim.

## 1. The confirmed result is a replication, not a discovery

`docs/MATCHED_COMPARISON.md` reports a 2-layer diagonal gated linear recurrence
(`h_t = sigmoid(W_g x_t)·h_{t-1} + W_v x_t`) at 2.21 bpc against a 4-layer
causal-attention transformer at 2.28 bpc, matched recipe, matched seeds.

**This exact experiment is already published, on this exact corpus.**

- **Feng, Tung, Ahmed, Bengio, Hajimirsadeghi, "Were RNNs All We Needed?",**
  arXiv:2410.01201, §4 — trains minGRU, minLSTM, Mamba and a Transformer on
  character-level Shakespeare in the nanoGPT harness, and reports the
  Transformer needing on the order of 2.5x the steps of minGRU to reach
  comparable loss, with all four arms plateauing within a few hundredths of a
  nat of each other (~1.55-1.59 nats). Our 2.21 / 2.28 bpc sit inside that range.

The gate form itself predates the family we had been citing by roughly seven
years:

- **Lei et al., SRU**, EMNLP 2018, arXiv:1709.02755.
- **Martin & Cundy, GILR**, ICLR 2018, arXiv:1709.04057 — this layer with a
  convex gate.
- **Bradbury et al., QRNN**, ICLR 2017, arXiv:1611.01576.

**Consequence.** Our minGRU/HGRN/RG-LRU attribution was accurate but late, and
the comparison itself is a *replication*. What it is worth is calibration: it
reproduces a published number on our own harness, which is real but is not a
contribution. The strongest defensible statement, and the one to use:

> On character-level TinyShakespeare at context 512 without dropout, a two-layer
> diagonal gated linear recurrence of the SRU/minGRU family reaches 2.21 bpc
> against 2.28 for a four-layer causal-attention transformer with twice its
> parameters under the same optimizer recipe, with the gap holding at 1500 steps
> on five paired seeds.

**Do not claim:** wall-clock efficiency (see `docs/WALLCLOCK.md` — at `d=128`
this is a kernel-count comparison); "half the parameters" as if parameter count
were the binding constraint, since the transformer overfits on 1 MB of data;
anything about memory, brains or dendritic traces; anything beyond this corpus.

## 2. The leading candidate for the surviving artifact: no dropout

This is the most useful finding in the check, because it is testable cheaply and
it predicts our own data.

Our own 5000-step single-seed curves (recorded in `docs/MATCHED_COMPARISON.md`):

| arm | train bpc | best val | final val |
|---|---|---|---|
| `A_attention` | 1.47 | 2.23 | 2.40 |
| `depth2_mem` | 1.67 | 2.17 | 2.20 |

The transformer is **memorizing**, not capacity-limited. On a 1 MB corpus passed
~12 times, the game is who overfits least, and the arm with the stronger
locality prior and fewer parameters wins that game. nanoGPT's `shakespeare_char`
config uses **dropout 0.2** for exactly this reason, and Feng et al. inherited
nanoGPT's settings. A baseline missing the regulariser its standard config uses
is mistuned in the same sense as the bigram-floor baseline we already retracted.

**The experiment:** dropout in {0, 0.1, 0.2, 0.3} on attention and residual
paths, both arms, 5 paired seeds, evaluate every 100 steps, compare best-of-curve.
Prediction, stated before running: attention's best validation improves by
0.05-0.1 bpc and the gap against the 2-layer memory arm goes null or reverses.
If the gap survives per-arm tuned dropout, the claim is materially stronger.

**A second hint already in our data:** the 4-layer memory arm scored *worse* than
the 2-layer one. Adding capacity hurt. Consistent with the overfitting account.

## 3. The rank-1 eligibility argument is wrong and must be withdrawn

`docs/THREEFACTOR.md` explains the plasticity falsification by observing that the
eligibility matrix is rank 1 (top singular value carries 99.3% of energy), so a
third factor can only rescale a fixed input direction.

**That argument does not hold.** The per-sample weight gradient of *any* dense
layer under backprop is also rank 1 — it is the outer product of the error vector
and the input vector. So "static input through a fixed projection gives rank-1
eligibility" is equally true of exact gradient descent and cannot explain why our
rule lost to its own frozen ablation.

**The falsification measurement stands.** Only the mechanism paragraph is
withdrawn. A better-supported candidate mechanism is one already in that document:
the LTD term shifts every weight of neuron `j` by the same amount, changing each
neuron's total drive, and we measured that drive changes push these onset
detectors off their operating point.

## 4. The lead: decompose validation loss by associative-recall tokens

Not yet run. This is the cheapest experiment that yields a *mechanism* rather
than another number.

**Arora et al., "Zoology", ICLR 2024, arXiv:2312.04927** showed that most of the
perplexity gap between attention and gated-recurrent or convolutional models is
concentrated on tokens whose bigram already appeared earlier in the context.
Character-level Shakespeare is full of these (speaker names, repeated words).

**Procedure:** score both arms per token, split by whether the current bigram
occurred earlier in the window, report both means over seeds. Index by training
step and the induction-head event (Olsson et al. 2022, arXiv:2209.11895) should
land on the recall slice.

**Why it is worth running:** it is diagnostic, not confirmatory. If attention
wins on recall hits and loses elsewhere, the headline becomes "the recurrence's
advantage is diffuse and regularisation-driven while attention's is real and
localized", and both the dropout experiment and a hybrid model are predicted
before they are run. If the memory arm *wins* on recall hits too, that
contradicts the Zoology account at this scale and is the first thing in this
project worth calling new.

## 5. Looped depth: the direction is already settled

For the recursive-depth experiment, at matched FLOPs unique layers win on
perplexity; at matched parameters looping wins. The relevant prior:

- **ALBERT** (arXiv:1909.11942) Table 3 — all-layer sharing costs ~2 points of
  downstream average at the same FLOPs. The direct matched-FLOP answer.
- **Universal Transformer** (arXiv:1807.03819) — beats base Transformer at equal
  parameters with more compute; no matched-FLOP unique-depth control.
- **Saunshi et al., "Reasoning with Latent Thoughts"** (arXiv:2502.17416) — a
  k-layer block looped L times nearly matches a kL-layer model on reasoning at
  iso-FLOPs, and lags on perplexity.
- **Geiping et al., Huginn** (arXiv:2502.05171) — 3.5B recurrent-depth model; the
  authors state they did not train an iso-FLOP non-recurrent baseline.
- **MobileLLM** (arXiv:2402.14905) — immediate block-wise repetition gives ~0.5
  points at fixed parameters.

**Implementation note that would otherwise handicap the arm:** a looped stack has
no per-iteration signal unless one is added. Universal Transformer adds a
timestep embedding; Huginn re-injects the token embedding at every iteration and
reports that it matters for stability. Without one of these the loop arm cannot
tell which pass it is on.

## 6. The scan: the fix we were missing

`docs/WALLCLOCK.md` records that the scan is dispatch-bound. Two points from the
check:

- **Mamba's actual approach is no parallel scan at all** (Gu & Dao,
  arXiv:2312.00752 §3.3): the selective scan is *sequential over time inside
  on-chip memory* and parallel over batch and channel. At `B×D = 2048` lanes and
  `T=512` that is one launch of 2048 threads each doing 512 fused multiply-adds.
  MLX 0.31 has `mx.fast.metal_kernel` for custom Metal from Python and
  `mx.custom_function` for the VJP, so this is reachable here. **This is the
  highest-value scan change and it is not the chunk sweep.**
- **The banded-matmul trick (Mamba2 SSD, Dao & Gu ICML 2024, arXiv:2405.21060)
  does not apply to a per-channel gate.** It needs a scalar decay per head so the
  chunk decay matrix is shared across channels and becomes a GEMM; with `d` gates
  per token it is an elementwise-weighted causal sum. The numerically safe
  version of our failed log-space scan is that per-channel form with weights
  `exp(cum_t − cum_s)`, all ≤ 1. GLA (arXiv:2312.06635 §4) documents the
  half-precision instability of cumulative gates and uses fp32 secondary
  chunking for the same reason.
- **Our recurrence has no input normalisation.** RG-LRU scales the input by
  `sqrt(1 − a²)` precisely so a bank at decay 0.9999 does not accumulate a state
  orders of magnitude larger than the input scale. Our unnormalised slow banks
  become a running mean of `W_v x`, which carries little at character level —
  a plausible part of why four banks never helped, and a one-line change worth
  checking before the multiscale refutation is treated as final. Related:
  "Stuffed Mamba" (arXiv:2410.07145) on state overflow in this family.

## 7. Gate initialisation is a known lever we have not swept

Every published member of this family tunes the gate init, and our arm currently
inherits a fixed geometric schedule:

- HGRN uses a depth-dependent lower bound (arXiv:2311.04823).
- RG-LRU uses `a^(c·r_t)` with `c=8` (Griffin, arXiv:2402.19427).
- LRU uses ring initialisation (arXiv:2303.06349).

Gate init plausibly explains the slow start noted in the 160-step run; it is an
initialisation property, not a compute property, so it does not by itself
undermine an efficiency framing. But it is un-swept and cheap to sweep.
