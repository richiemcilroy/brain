# Finetuning: the transplant beats a matched random init once the decay is right

**One-line:** after identical finetuning of only the gated-memory module, the
transplanted init recovers **38.6%** of the damage from deleting attention, while
a scale-matched random init ends up **worse than deleting attention** (−53.1%).

Teacher `unsloth/Llama-3.2-1B`, layer 8 of 16, val = last 10% of the corpus
(3000 tokens), select = 60-80%, train = first 50%. Disjoint by construction and
asserted. Decay and lr are both chosen on **select**; val is only ever reported.

| arm | before ft | after ft | recovers |
|---|---|---|---|
| teacher (unmodified) | 20.3756 | — | — |
| `zero` (attention deleted) | 24.1044 | — | — |
| **`transfer`** | 22.8985 | **22.6638** | **38.6%** |
| `random_matched` | 25.6698 | 26.0826 | −53.1% |

The margin is **3.4188 ppl** in transfer's favour. Per-arm decay was chosen on
select: transfer 0.7, random 0.999.

## Why the earlier run of this experiment said the opposite

An earlier version reported that the random control *beat* the transplant
(22.9197 vs 23.6220) and this file's conclusion was briefly headed the other
way. Three separate defects were found, each of which produced plausible numbers:

1. **Adam's step is absolute, not relative.** The trace was normalised by
   pre-scaling `W_v` by `(1-g)=0.01`. At `d=2048` that makes each element ~0.002,
   so `lr=3e-4` moved every weight by ~15% of its own magnitude per step. The
   *first* finetune run drove **both** arms to ~1400-1550 ppl — it measured how
   fast Adam destroys a precise init. Fixed by moving the normalisation into a
   fixed non-parameter output gain.
2. **The decay was never fitted.** It was hardcoded to 0.99 via a `try/except`
   around an MLX API that does not exist (see `docs/HYBRID_DECAY.md`). 0.99 is
   near the worst point on the module's own curve. With the decay chosen per arm
   on select, the ranking reverses.
3. **Both arms were not scale-matched.** The random arm was output-rms matched
   but the transplant was not, so the arms differed in score scale as well as in
   weight provenance.

## Honest limits

- **One seed.** The 3.4188 ppl margin is large relative to the 0.1065 ppl margin
  in the parallel-injection experiment, but it has no error bars yet.
- **Machine load was high** (~39) during this run, including a `rustc` compile.
  Wall-clock figures from it are meaningless; the ppl numbers are deterministic
  given the data order, so they are unaffected, but this should be re-run on an
  idle machine before being quoted.
- **Still not better than the teacher.** 38.6% recovery of self-inflicted damage
  is not an improvement on the unmodified model. For that, see
  `docs/INJECTION.md`.
- One layer of 16.

## Reproduce

```sh
cd /Users/richie/Documents/github/human-brain
HF_HOME=~/zbrain/hf FT_STEPS=400 FT_LRS=3e-5 FT_ARMS=transfer,random_matched \
  ~/zbrain/venv/bin/python experiments/hybrid_finetune.py
```
