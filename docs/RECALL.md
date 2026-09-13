# Associative-recall decomposition of the matched-depth result

**Status: PRE-REGISTERED, NOT YET RUN.**

Written `2026-09-12T22:45Z`, at repo commit `e4e9f1d`, *before* the experiment in
`experiments/recall_decomp.py` was executed for the first time. The prediction
below is frozen as of this commit. Any later change to it is recorded as an
amendment with a timestamp, never as a silent edit.

## Why this experiment exists

`docs/MATCHED_COMPARISON.md` reports the project's one surviving positive
result: at matched depth (2 layers), matched width (`d=128`), matched seeds and
matched budget, a gated-memory stack scores **2.2092 bpc** against **2.3329**
for an attention stack, paired difference **-0.1237 bpc, 95% CI
[-0.1382, -0.1092], 5/5 seeds agreeing**.

`docs/PRIOR_ART.md` §1 records the problem with that number: it is a
**replication**. Feng et al. (arXiv:2410.01201) ran essentially this comparison
on essentially this corpus, and the gated linear recurrence family
(SRU/QRNN/GILR → minGRU/HGRN/RG-LRU) is theirs, not ours. Reproducing a published
number on our own harness is calibration, not a contribution.

What is *not* established is the **mechanism**. §4 of `docs/PRIOR_ART.md`
proposes the cheapest experiment that replaces a number with a mechanism:

> **Arora et al., "Zoology", ICLR 2024, arXiv:2312.04927** showed that most of
> the perplexity gap between attention and gated-recurrent or convolutional
> language models is concentrated on tokens whose **bigram already appeared
> earlier in the context** — "associative-recall" tokens.

Character-level Shakespeare is dense with these: speaker names, repeated words,
common letter pairs. So the val loss of both arms is decomposed token-by-token
by whether the token's bigram (and, as a harder control, its trigram) occurred
earlier in the same 512-token window.

## The pre-registered prediction

**I predict the Zoology account carries to this scale, and the interaction will
be POSITIVE on the recall slice.**

Concretely, with `mem = depth2_mem` and `attn = depth2_attn`, and the
interaction defined as

```
interaction = (mean_mem_loss - mean_attn_loss)_recall_hits
            - (mean_mem_loss - mean_attn_loss)_non_hits      [nats/token]
```

I predict:

| quantity | predicted sign | reasoning |
|---|---|---|
| `mem - attn` on **bigram-recall hits** | **positive** (memory *worse*) | recall hits are where an induction/copy circuit pays off; attention has one, our gated recurrence is an LTI filter with no content-based lookup |
| `mem - attn` on **non-hits** | **negative** (memory *better*) | the memory arm's advantage is diffuse locality/regularisation, matching the overfitting account in `docs/PRIOR_ART.md` §2 |
| **interaction (bigram)** | **positive**, CI excludes 0 | consequence of the two rows above |
| **interaction (trigram)** | **positive** | harder control; should be attenuated relative to bigram and may fall to null |

**Why I expect this, stated plainly:** the memory arm's advantage is already
known to be in the *overfitting* regime — `docs/PRIOR_ART.md` §2 shows the
4-layer attention arm *memorises* (train 1.47 bpc, val degrading to 2.40) while
the memory arm stays stable, and the 4-layer memory arm scores *worse* than the
2-layer one. A capacity/regularisation effect should show up as a diffuse
advantage spread over ordinary tokens, not as a win on the slice where a
content-addressable lookup is required. Meanwhile attention's one genuinely
architectural superpower at short context is exactly content-based retrieval of
an earlier occurrence — the induction head (Olsson et al. arXiv:2209.11895). If
the head forms in 1500 steps at `d=128` and there is enough data with repeated
bigrams, attention should be *relatively* better there, even though it is worse
overall.

**The falsification case, stated before the run.** If the memory arm is *also*
better on recall hits — interaction ≤ 0 with a CI excluding zero — then the
Zoology account does not carry to this scale, and that is the first genuinely
new thing this project would have produced. It would mean a purely
gate-modulated linear recurrence is beating an in-context content lookup at
in-context recall, which is a substantive mechanistic claim about this
architecture class and needs its own adversarial treatment before being
believed. **I do not expect this outcome.** I expect to report a positive
interaction, i.e. a *mechanism* for a result that was previously only a number —
and, in the honesty accounting of this repo, a mechanism that is itself a
replication of Zoology rather than a contribution.

**Secondary prediction, also frozen:** the recall-hit fraction on validation
will be substantial — order 30–45% for bigrams, and I expect higher for
trigrams than bigrams (longer n-grams are more likely to have recurred than
shorter ones, because char-level English repeats long sequences in words,
names and verse meter, and trigram hits count bigram hits as a subset by
construction). If the bigram hit fraction comes in under 10%, the stated
consequence is that this experiment is **underpowered** and must be reported as
a null regardless of the point estimate — that is written here in advance so it
cannot be renegotiated after seeing the number.

## Method (frozen)

- Harness: `experiments/confirm_headline.py` imported directly. `build()`,
  `deterministic_val_batches()`, `evaluate()`, `paired_report()`, `T95`, and
  `run()`'s exact recipe — AdamW, `lr 1e-3`, 100-step linear warmup, grad clip
  1.0, weight decay 0.01, `bs 16`, `ctx 512`, `d 128`, 1500 steps — are reused,
  not reimplemented. Nothing in `llm_efficiency.py` or `confirm_headline.py` is
  modified.
- Arms: `depth2_mem` (445,440 params) and `depth2_attn` (478,976 params).
- Seeds: 5 paired seeds (`0,1,2,3,4`).
- Evaluation: all 217 non-overlapping 512-token windows of the validation
  split, 111,104 scored tokens, identical for every arm and seed. Per-token
  cross-entropy in **nats**, reduced with `reduction="none"`; BPC is nats/ln 2
  and is reported alongside for comparability with the committed result.
- Masks are **strictly causal**: position `p` may only consult positions
  `< p` (`< p-1` for bigrams, `< p-2` for trigrams). The script asserts this
  rather than asserting it in prose.
- Slice means are **weighted by token count**, so the recall and non-recall
  means are not distorted by differing slice sizes.
- Intervals: 95% **paired** t-intervals over 5 seeds on every slice difference,
  on the between-arm difference within each slice, and on the interaction.
  A CI straddling zero is reported as a NULL, not as a win.

## Reproduction gate

The script first re-derives each arm's validation loss with the harness's own
`evaluate()` and compares its nats-per-token against the per-token mean. If the
two disagree by more than 1e-6 nats/token, or if either arm lands absurdly far
from the committed per-arm means, the script **stops and reports a broken
harness** instead of reporting a slice result. A slice decomposition computed on
a model that does not reproduce the committed number is not evidence about
anything.

## Results

*Not yet run.* Filled in below after the run, in this same file, without
altering anything above this line.

---

## AMENDMENT 1 — `2026-09-12T22:52Z`, still before any training run

I checked my own secondary prediction above and it is **provably wrong**, so I am
correcting it under the amendment rule rather than letting it stand.

**The error.** I wrote that I expect the trigram hit fraction to be *higher* than
the bigram hit fraction. It cannot be. A trigram hit is a **subset** of a bigram
hit under the declared predicate:

> If the trigram ending at `p` has a duplicate ending at `j ≤ p-1`, then the
> bigram `(x_{p-1}, x_p)` — the last two positions of that duplicate — also
> occurs ending at `j ≤ p-1`. So trigram-hit ⟹ bigram-hit, always.

Therefore `fraction(trigram hits) ≤ fraction(bigram hits)`, strictly, by
construction, for any corpus. My stated reason ("longer n-grams are more likely
to have recurred") is false on char-level text: long sequences recur less often,
and every trigram hit is already counted as a bigram hit. The correct prediction
is the reverse, and the subset relation becomes an assertion in the code
(`check_trigram_subset`) rather than a prediction.

**Corrected secondary prediction.** `fraction(trigram hits) < fraction(bigram
hits)`, with the trigram slice strictly smaller. The operational consequence:
the trigram slice is the *harder* control but also the *noisier* one, so its
interval should be wider and it is more likely to produce a null — a null there
is informative about power, not about mechanism.

**A power concern I should have stated the first time, and now do.** My original
worry was that a hit fraction *below* 10% would leave the experiment
underpowered. Having thought about it, the more likely failure mode on
character-level English inside a 512-token window is the **opposite**: the bigram
hit fraction may be so high (the alphabet is only 65 symbols and a window holds
511 bigram occurrences) that the *non-hit* slice is the small one. Both slices
matter for the interaction, so I record now, before seeing the data, that:

- if the bigram hit fraction is > 90%, the non-hit slice is small and the
  non-hit estimate is the noisier half of the interaction — the bigram
  interaction should then be treated as dominated by the hit slice;
- if either slice holds < 5% of tokens, I will report the experiment as
  underpowered for that predicate regardless of the point estimate.

This is an amendment to *my own* secondary prediction only. **The primary
prediction, the interaction sign, the falsification condition, the recipe, the
seeds, the mask definitions and the reporting rules above are unchanged.**

**One further decision fixed here, before the run.** The primary predicate
follows the task specification literally — the duplicate must end at position
`≤ p-1`, i.e. strictly before `p`, so the mask never consults any position
`≥ p`. A second, stricter variant requires the duplicate and the current n-gram
to share **no positions at all** (duplicate ends at `≤ p-n`). The stricter
variant is reported as a robustness control, not as the primary result, because
the looser one is what was specified. If the two disagree in sign, that is
reported as a finding about the predicate, not resolved by picking one.
