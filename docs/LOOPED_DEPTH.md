# Recursive cortical depth: a clean NULL on quality, with a measured parameter saving

**Verdict: looping is NOT a parameter-efficiency win on held-out bits per
character. The deciding number is `loop2x2 - flat2 = -0.00085 bpc`, 95% paired
t-interval `[-0.01753, +0.01583]`, sign agreement 4/5. That interval straddles
zero, so it is a NULL, not a win, and the pre-registered decision rule says a
non-zero-beating interval is a failure of the claim.**

Three things are true at once and all three are reported below:

1. **Quality is flat.** At matched parameters, `loop2x2` (2 passes over 2 unique
   layers) scores 2.2083 bpc against 2.2092 for `flat2` (1 pass over the same 2
   layers). The difference is 0.00085 bpc — 1/20th of the smallest real effect
   this design can detect. Extra passes over shared weights bought nothing.
2. **Parameters are genuinely saved — 44.9%.** Against the 4-unique-layer
   baseline `flat4`, the looped arm matches quality *at equal FLOPs* while
   carrying 55.1% of the parameters. It does not match `flat4`'s quality; it
   lands 0.0258 bpc behind it (interval excludes zero), so the honest statement
   is "0.026 bpc worse with 45% fewer parameters at equal FLOPs", not "as good".
3. **Under a matched wall-clock budget, the unique-layer control does not lose.**
   `flat2` is 1.86x faster per step than `loop2x2`, so a strict equal-time
   budget gives it 2,691 steps against 1,500, and it then scores 2.1857 bpc.
   The loop arm is 0.0226 bpc *behind*, with all 5 seeds pointing the same way
   (5/5 sign) — but the interval `[-0.056, +0.011]` still straddles zero, so on
   the pre-registered standard **this is also a NULL, not a win for either
   arm.** The sign consistency is suggestive and is reported as such; it is not
   an interval excluding zero and must not be read as one.

## Pre-registered decision rule (stated before the run, in `looped_depth.py`)

> Looping is a real efficiency win ONLY if `loop2x2` beats `flat2` by more than
> seed noise, since `flat2` has the SAME parameter count (445,440) at HALF the
> FLOPs (2,226,432 vs 4,402,944 per token).

**The claim fails.** `loop2x2` does not beat `flat2`; it is 0.00085 bpc better
in the mean with an interval straddling zero and only 4/5 seeds agreeing in
sign. Under the rule as written this is a NULL and must be reported as one.

## Pre-registered prediction (stated before the run)

> `loop2x2` LOSES to `flat4` on bpc at matched FLOPs, and MATCHES `flat2` on
> params.

**Both halves are confirmed.** `loop2x2` loses to `flat4` by +0.0258 bpc
(5/5 seeds, interval excludes zero) and matches `flat2` on parameters exactly
(445,440 = 445,440) and on quality within noise. The prior art said this would
happen (ALBERT at matched FLOPs, Saunshi et al. on perplexity vs reasoning) and
it happened.

## The matched-budget table

Corpus: the vendored `data/tinyshakespeare.txt` (1,115,394 chars, vocab 65),
which is `llm_efficiency.py`'s default `CORPUS` path. The floors were recomputed
from this in-repo file rather than any external copy and reproduce exactly
(unigram 4.8292, bigram 3.5806), so no external corpus is needed — the earlier
`/Volumes/T9/.../enwik8` reference is not used and was not required.

All arms: char-level TinyShakespeare (`data/tinyshakespeare.txt`, 1,115,394
chars, vocab 65), `d=128`, `ctx=512`, MLP mult 4, gated-memory mixing, chunk 64,
batch 16, AdamW `lr=1e-3` with 100-step warmup and gradient clip 1.0.
Validation is the deterministic full 217-window pass (111,104 tokens, every arm
scored on identical data). **5 seeds.** Unigram floor 4.8292, **bigram floor
3.5806** — every arm is far below the floor, so all of them learned to use
context.

| arm | params | FLOPs/tok | unique layers | passes | steps | train bpc | **val bpc** | sd | vs bigram floor |
|---|---|---|---|---|---|---|---|---|---|
| `flat4` | 808,448 | 4,402,944 | 4 | 1 | 1500 | 1.8072 | **2.1826** | 0.0113 | +1.3980 |
| `loop2x2` | **445,440** | 4,402,944 | 2 | 2 | 1500 | 1.8928 | **2.2083** | 0.0133 | +1.3722 |
| `loop1x4` | **263,936** | 4,402,944 | 1 | 4 | 1500 | 1.9842 | **2.2701** | 0.0148 | +1.3105 |
| `flat2` | **445,440** | 2,226,432 | 2 | 1 | 1500 | 1.8979 | **2.2092** | 0.0055 | +1.3714 |
| `loop2x2_ws` | 445,954 | 4,402,944 | 2 | 2 | 1500 | 1.8900 | 2.2075 | 0.0090 | +1.3731 |
| `loop2x2_ts` | 445,696 | 4,402,944 | 2 | 2 | 1500 | 1.8971 | 2.2080 | 0.0060 | +1.3726 |
| `loop1x4_ws` | 264,964 | 4,402,944 | 1 | 4 | 1500 | 2.0093 | 2.2847 | 0.0184 | +1.2959 |
| `loop1x4_ts` | 264,448 | 4,402,944 | 1 | 4 | 1500 | 2.0120 | 2.2892 | 0.0073 | +1.2914 |
| `flat2_wall` | 445,440 | 2,226,432 | 2 | 1 | 2800 | 1.7871 | **2.1803** | 0.0228 | +1.4003 |
| `flat2_wc` | 445,440 | 2,226,432 | 2 | 1 | 2150 | 1.8268 | 2.1889 | 0.0132 | +1.3917 |
| `flat2_wc2` | 445,440 | 2,226,432 | 2 | 1 | 2691 | 1.7943 | **2.1857** | 0.0164 | +1.3948 |

No arm diverged, no arm is at or above the floor, nothing was dropped. In
perplexity: `flat4` 4.540, `loop2x2` 4.621, `flat2` 4.624, `loop1x4` 4.824.

### Which axis is matched, and which is not

| comparison | params matched | FLOPs/token matched | steps matched | wall-clock matched |
|---|---|---|---|---|
| `loop2x2` vs `flat4` | no (55.1%) | **yes** (equal) | yes | no |
| `loop2x2` vs `flat2` | **yes** (equal) | no (2.0x) | yes | no |
| `flat2_wc2` vs `loop2x2` | **yes** | no (0.51x) | **no** (2691 vs 1500) | **yes** |
| `loop1x4` vs `flat4` | no (32.6%) | **yes** | yes | no |

Stated plainly: **no single arm matches all three of parameters, FLOPs and
wall-clock, because matching any two pins the third.** That is why there are
three separate comparisons rather than one.

## The question as asked: at matched parameters AND matched wall-clock, what does the extra effective depth buy?

**Answer: nothing measurable, and the point estimate is negative.**

Hold both budgets fixed and the two arms are `flat2_wc2` (2 unique layers, 1
pass, 2,691 steps) and `loop2x2` (2 unique layers, 2 passes, 1,500 steps). Both
carry exactly 445,440 parameters and both are given the same wall-clock. The
difference is one extra pass over the same weights per step, bought by taking
1.79x fewer steps:

| | `flat2_wc2` | `loop2x2` |
|---|---|---|
| parameters | 445,440 | 445,440 |
| passes per step | 1 | 2 |
| steps in the budget | 2,691 | 1,500 |
| FLOPs/token | 2,226,432 | 4,402,944 |
| val bpc | **2.1857** | **2.2083** |

**The extra effective depth buys −0.0226 bpc** (i.e. the looped arm is worse),
95% CI `[−0.05646, +0.01123]`, 5/5 seeds agreeing in sign. By the pre-registered
standard that interval straddles zero, so **this too is a NULL.** At matched
parameters and matched time, deeper-but-repeated computation is not better than
shallower-but-more-iterated computation; the only thing the loop arm's
parameters buy is a higher FLOP cost per step, which the flat arm spends on more
steps instead.

## Paired differences (5 paired seeds, 95% paired t-intervals)

Negative = first arm has lower bpc (better).

| comparison | mean diff | 95% CI | sign | verdict |
|---|---|---|---|---|
| `loop2x2` − `flat2` | −0.00085 | [−0.01753, +0.01583] | 4/5 | **NULL** |
| `loop2x2` − `flat4` | +0.02579 | [+0.00728, +0.04429] | 5/5 | `flat4` better |
| `loop1x4` − `flat2` | +0.06090 | [+0.04307, +0.07874] | 5/5 | `flat2` better |
| `flat2_wc2` − `loop2x2` | −0.02261 | [−0.05646, +0.01123] | 5/5 | **NULL** (equal time) |
| `flat2_wall` − `loop2x2` | −0.02806 | [−0.07133, +0.01522] | 5/5 | **NULL** (equal time) |
| `flat2_wc2` − `flat4` | +0.00317 | [−0.01645, +0.02280] | 4/5 | **NULL** |
| `flat2_wc` − `flat2` | −0.02033 | [−0.03554, −0.00511] | 5/5 | more steps help |
| `flat2_wc` − `loop2x2` | −0.01948 | [−0.04861, +0.00964] | 4/5 | **NULL** (equal time) |
| `flat2_wc` − `flat4` | +0.00631 | [−0.00822, +0.02083] | 4/5 | **NULL** |
| `loop2x2_ws` − `flat2` | −0.00171 | [−0.01615, +0.01273] | 2/5 | **NULL** |
| `loop2x2_ws` − `loop2x2` | −0.00086 | [−0.01207, +0.01034] | 2/5 | **NULL** |
| `loop2x2_ts` − `loop2x2` | −0.00036 | [−0.01812, +0.01741] | 2/5 | **NULL** |
| `loop1x4_ts` − `loop1x4` | +0.01906 | [+0.00079, +0.03734] | 4/5 | `loop1x4` better |
| `loop1x4_ws` − `loop1x4` | +0.01457 | [−0.01986, +0.04899] | 4/5 | **NULL** |

At n=5 the paired design's standard error on the headline comparison is
0.00601 bpc, so the smallest effect it can resolve is about **0.0167 bpc**. The
observed `loop2x2` − `flat2` difference of 0.00085 is about **20x smaller** than
that. This is a null by design limits as well as by interval: the effect, if it
exists at all here, is below 0.017 bpc.

## Does the per-loop stabiliser change the answer? No.

A looped stack has no per-iteration signal — the same blocks and the same
LayerNorms cannot tell which pass they are on. Both published fixes were tested:

| fix | source | effect on `loop2x2` | effect on `loop1x4` |
|---|---|---|---|
| per-loop LayerNorm + scale | Universal Transformer / Huginn | −0.00086 bpc (NULL, 2/5) | +0.0146 bpc (NULL, 4/5) |
| per-loop timestep embedding | Universal Transformer | −0.00036 bpc (NULL, 2/5) | +0.0191 bpc (worse, 4/5) |

**Neither stabiliser changes the verdict on the claim.** Against `flat2`,
`loop2x2_ws` is −0.00171 bpc and still a NULL. So the null is not an artifact of
handicapping the loop arm: giving the loop arm the per-loop normalisation that
its published cousins use does not rescue it. If anything the timestep embedding
*helps* `flat2`'s case at 4 passes (+0.019 bpc).

Worth recording honestly: `loop2x2_ws` has a marginally better mean than
`loop2x2` and 2/5 sign agreement, i.e. the stabiliser is noise-level here. It
did **not** prevent divergence, because no arm diverged in the first place —
the "naive looping diverges without per-loop normalisation" failure mode did not
appear at 4 passes with this architecture.

## What looping actually saved

| arm | params vs `flat4` | saved | FLOPs vs `flat4` | measured ms/step | quality vs `flat4` |
|---|---|---|---|---|---|
| `loop2x2` | 55.1% | 363,008 (44.9%) | 100% | 19.2 vs 19.6 | +0.0258 bpc worse |
| `loop1x4` | 32.6% | 544,512 (67.4%) | 100% | 18.5 vs 19.6 | +0.0875 bpc worse |
| `flat2` | 55.1% | 363,008 (44.9%) | 50.5% | 10.3 vs 19.6 | +0.0266 bpc worse |

**Wall-clock: looping saved nothing, and in the compute-bound limit it should
save nothing.** Measured with round-robin interleaving (arms measured
alternately inside each round so a load spike hits all arms, warmup + 12 reps ×
10 rounds, minimum round-median):

| arm | ms/step | steps/s | rate vs `loop2x2` |
|---|---|---|---|
| `loop2x2` | 19.21 | 52.1 | 1.00x |
| `loop1x4` | 18.47 | 54.2 | 1.04x |
| `flat4` | 19.64 | 50.9 | 0.98x |
| `flat2` | 10.33 | 96.8 | 1.86x |

`loop2x2`, `loop1x4` and `flat4` are the same speed to within 4% — they have
identical analytic FLOPs, and at `d=128`, `ctx=512`, batch 16 this machine is
launch/memory-bound rather than FLOP-bound, so ~1.9x the FLOPs costs ~0%
wall-clock. **Looping's parameter saving is real (44.9%); its FLOP saving is
exactly zero by construction; its wall-clock saving is zero-to-negative.**
`flat2` is the arm that actually converts its parameter saving into a 1.86x
step-rate advantage, so under an equal wall-clock budget it is the loop arm that
must justify itself — and it does not: the equal-time comparison is a NULL with
a negative point estimate (above).

## Mechanism: why flat (and why the second pass is nearly idle)

The honest measurement on the "effective depth" axis is what one more pass
changes. Relative change `||h_i − h_{i−1}|| / ||h_{i−1}||` and cosine to the
previous pass, averaged over 5 seeds (train split):

| arm | pass 1 | pass 2 | pass 3 | pass 4 | cos(pass1→2) | cos(pass2→3) | cos(pass3→4) |
|---|---|---|---|---|---|---|---|
| `loop2x2` | 12.45 | 0.88 | — | — | 0.037 | 0.840 | — |
| `loop1x4` | 9.20 | 0.68 | 0.52 | 0.40 | 0.013 | 0.814 | 0.938 / 0.974 |
| `loop1x4_ts` | 9.58 | 0.63 | 0.52 | 0.40 | 0.035 | 0.859 | 0.949 / 0.978 |

Pass 1 changes the representation by 12.4x its own norm at `loop2x2` and 9.2x
at `loop1x4`; **pass 2 changes it by 0.88x, and passes 3–4 by 0.5x/0.4x while
correlating at 0.94–0.97 with the previous pass.** The extra passes are not diverging or exploding — they are
converging toward something close to a fixed point of the shared block, and
successive passes carry rapidly diminishing new information. This is the
mechanism behind the null: **the repeated block reaches a near-fixed-point
regime after roughly one extra pass, so passes 3, 4, ..., L mostly re-encode
what pass 2 already produced.** That is also why `loop1x4` pays a real cost
(+0.061 bpc vs `flat2`) while `loop2x2` does not: the third and fourth passes
are the ones that stop buying anything but still shift the representation.

Note this is a *representation-change* measurement, not a receptive-field or
information-theoretic depth measurement. It says the extra passes do
progressively less; it does not by itself prove they do literally nothing.

## Prior art: what is new here and what is not

**Not new (and it predicted this result).** Loop/recurrence-at-depth is
established and is settled in the direction *against* a naive perplexity win:

- **ALBERT** (arXiv:1909.11942), Table 3: all-layer parameter sharing costs
  ~2 points of downstream average at the same FLOPs. This is the direct
  matched-FLOP answer, and it goes against looping.
- **Saunshi et al., "Reasoning with Latent Thoughts"** (arXiv:2502.17416): a
  k-layer block looped L times nearly matches a kL-layer model on *reasoning* at
  iso-FLOPs but **lags on perplexity**. bpc is a perplexity-family metric, so
  their result predicts exactly the null found here.
- **Universal Transformer** (arXiv:1807.03819): beats the base Transformer at
  equal *parameters* with more compute; no matched-FLOP unique-depth control.
- **Huginn / Geiping et al.** (arXiv:2502.05171): the authors state they did
  **not** train an iso-FLOP non-recurrent baseline — which is precisely the
  gap this run fills.
- **MobileLLM** (arXiv:2402.14905): immediate block-wise repetition gives ~0.5
  points at fixed parameters.

**Genuinely new here, and the reason to keep the result.** A single controlled
sweep in which a *k-pass looped* stack is compared against BOTH the matched-FLOP
unique-depth stack (2 vs 4 layers) AND the matched-parameter unique-depth stack
(2 vs 2 layers) at equal training steps, with 5 paired seeds, deterministic full
validation, on the gated-memory primitive this project uses — plus a third
equal-wall-clock arm. That triple is what makes the null attributable: it
separates "more FLOPs help" from "re-use of weights helps". ALBERT's matched-FLOP
number does not isolate the shared-weights effect at matched parameters; the
Huginn paper does not contain the iso-FLOP control at all. **This run supplies
the missing control, and it comes out flat.**

**What is NOT claimed.** Nothing here generalises to attention-based retrieval,
to reasoning tasks (where Saunshi et al. find looping competitive), to
`ctx` beyond 512, to larger `d`, or to corpora beyond 1.1M characters of
Shakespeare. bpc at a 12.3M-token budget is a perplexity-family measurement and
says nothing about the inductive-bias benefits of recurrence that the
reasoning-literature reports.

## Determinism: asserted, and one real caveat

Determinism was **measured, not assumed** (`determinism.py` logic inside
`looped_depth.py`, recorded in the JSON):

| device | init bit-identical | forward loss bit-identical | gradient bit-identical | grad max abs diff |
|---|---|---|---|---|
| GPU (Metal, default) | yes | yes | **no** | 3.7e-08 (up to 4.5e-08 across sessions) |
| CPU | yes | yes | **yes** | 0.0 |

**MLX's GPU backward pass is not bit-reproducible**, even with an identical
init, identical batch and identical seed: repeated gradient evaluations differ
by ~4e-08 per element because parallel reductions sum in a non-deterministic
order. Over 1,500 steps this accumulates to a ~1e-7 bpc difference in `val_bpc`
(measured: 2.207151171475598 vs 2.2071512475600885). On CPU two full 1,500-step
runs are bit-identical (2.207151602621045 both times, across separate
processes). Set `BRAIN_DEVICE=cpu` for strict bit-reproducibility; CPU is ~10x
slower. **This does not affect any claim here**: 1e-7 bpc is five orders of
magnitude below the 0.017 bpc resolution of the design, and seeds are paired.

A separate and larger source of variance is the wall-clock axis: the macro was
shared during the sweep, load average moved between 16 and 40, and the same arm
took 25s on one seed and 124s on another. That is why the timing table uses
round-robin interleaved measurement with a minimum round-median rather than the
per-run `wall_s`, and why the equal-time step counts were fixed from that
measurement.

## Harness validity: an inherited result was reproduced exactly

`flat2` in this script is a re-implementation (`LoopedLM` with `loops=1`), not a
call into the project's validated arm. It was verified equivalent: identical
parameter keys, `max abs param diff = 0.0` and `max abs forward diff = 0.0`
against `LM(vocab, 128, 512, "mem", 2, 4, 1, 64)`. It then reproduced
`experiments/confirm_headline.py`'s archived `depth2_mem` per-seed values
**exactly** — 2.207151374367573 (seed 0), 2.2016, 2.2168, 2.2100, 2.2104 — and
the archived 5-seed mean 2.2092. The floors also reproduce exactly (unigram
4.8292, bigram 3.5806) from the in-repo corpus.

## How to falsify this

1. **The null is falsifiable by more seeds or a lower-variance design.** The
   design resolves 0.0167 bpc at n=5. If `loop2x2` genuinely beats `flat2` by
   less than that, this run cannot see it. Re-run with n≥20 paired seeds, or
   average several batch sizes per seed to shrink the paired SE, and report the
   interval again. A finding of "better by 0.01-0.017 bpc" would be a real
   effect this run was underpowered to confirm.
2. **Falsified if the fixed-point explanation is wrong.** The mechanism claim is
   that extra passes nearly stop changing the representation. Test it directly:
   force pass 3+ to be non-redundant (per-loop input injection scaled up,
   per-loop block parameter deltas, or a per-loop readout) and check whether
   `loop1x4` recovers the 0.061 bpc it loses to `flat2`. If it does not, the
   fixed-point story is not the whole explanation.
3. **Falsified if the primitive is the problem.** All arms use gated memory at
   `d=128`. If looping pays off with attention or with a larger `d`
   (e.g. 512/768) where the block is further from fixed-point saturation, the
   null is a property of this configuration, not of looping.
4. **Pre-registered for the next run:** a scale sweep of `d ∈ {128, 256, 512}`
   and passes `L ∈ {2, 4}` against matched-FLOP unique stacks, with the
   prediction that the looped arm improves relative to the flat arm as `d`
   grows (larger blocks have further to travel before reaching a fixed point).
   If the gap instead stays flat or widens with `d`, the "cortex is a repeated
   circuit" intuition is not buying anything at these scales.

## Reproduce

```
python3 experiments/looped_depth.py                 # ~100 min, 55 runs, GPU
BRAIN_DEVICE=cpu python3 experiments/looped_depth.py  # bit-reproducible, ~10x slower
BRAIN_TIMING=1 BRAIN_TIMING_ROUNDS=10 python3 experiments/looped_depth.py  # + timing
```

Deterministic given a seed; results resume from the JSON so completed runs are
never recomputed. Raw output: `experiments/results/looped_depth.json`. No
arm diverged and no arm was dropped.
