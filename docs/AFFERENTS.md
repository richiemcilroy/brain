# A signed projection fixes the input path — but not the way the hypothesis said

**Result: replacing `brain/cortex.py`'s 1:1 pixel routing with a frozen random
projection raises a linear probe on the substrate's own spike-count code from
36.17% to 71.67% on 10-way MNIST (`verdict.q1_projection_effect_real.delta = 0.355`,
3 seeds). The stated cause was afferent *convergence* — neurons combining many
pixels. That cause is not supported: a projection with one pixel per neuron and
no convergence at all scores 64.00% once its receptive fields are signed
(`verdict.q2_is_it_convergence`). Polarity is worth +27.83 points, routing +1.83
points, centering −2.83 points. And the better projection does not rescue
continual learning: the end-to-end substrate remains 45.27 points behind the
parameter-matched dense MLP on Permuted-MNIST (`verdict.q3_consequence`).**

This document reports `experiments/results/afferents.json`, produced by
`experiments/afferents.py`, using the frozen projection interface in
`brain/afferents.py`. Every number below is a value from a named key in that
JSON.

---

## 1. Method

**What was varied.** The input projection only, via three parameters recorded on
every `sweep[]` row: `kind` (`identity`, `dense`, `sparse`, `conv`, plus the
experiment-local `ctrl_*` controls), `fan_in` (1-784), and `gain` (1.0-12.0).
The projection is **frozen**: it is sampled once from the seed and never trained.
Only the downstream probe is fitted.

**What was held fixed.** Everything else: `config.n_neurons = 900`,
`config.n_input = 784`, `config.t_ms = 15`, `config.n_train = 500`,
`config.n_test = 200`, `config.backend = "numpy"`, `config.seeds = 3`. The codes
are spike counts over the 15 ms window from the real `CortexClassifier` path with
the drive supplied by the projection. `config.quick = False` and
`config.modern_protocol = False`, so this is the full grid under the published
protocol's compatibility flags.

**The interface under test** (`brain/afferents.py`): `AfferentConfig` →
`Afferents(...).project(x)` returns a float32 `(n_neurons,)` drive. `KINDS` is
`("identity", "dense", "sparse", "conv")`; `stats()` reports `n_weights`,
`n_connections`, `fan_in_effective`, `divergence` and `macs_per_sample`. All
kinds scale by `gain` so the per-neuron current is comparable across kinds, which
is what makes the fan-in sweep a test of convergence rather than of gain. The
module's own `python3 brain/afferents.py` self-check passes on this machine
(identity bit-exact against `cortex.py`, dense/sparse RMS ratio 1.121,
same-seed determinism, invalid configs rejected).

**The probe and the metric.** A closed-form least-squares solve on `[code, 1]`
for the linear probe and `[code, code**2, 1]` for the quadratic one, fitted on the
500 train codes and scored on the 200 held-out codes. `sweep[].linear` is
top-1 accuracy (`lstsq` with `rcond=1e-3`, the earlier measurement's convention).
`sweep[].linear_robust` averages four other solve conventions (`lstsq_full`,
`ridge_1`, `ridge_10`, `ridge_100`) because that convention is numerically
unstable at these shapes; both are reported throughout.

**Chance level.** The JSON contains no chance key. Arithmetic from the task
definition: a 10-way argmax over balanced classes, so uniform guessing is
**10%**. It is not a key in the file.

**Sample size.** `sweep` has 109 rows, each an average over `n_seeds = 3`
independent seeds; `sweep_raw_runs = 327` is the number of underlying runs. Each
seed re-draws **both** the frozen projection and the neuron noise, which the
script's protocol section calls the honest unit of replication. Every figure is
mean ± 95% CI (`1.96 * SEM`), and each cell's CI is a seed-to-seed CI over 3
seeds.

**The published baseline is verified, not recalled.** Before any comparison,
`verify_published` re-reads `experiments/results/continual_permuted.json` and
`continual_split.json` and asserts the MLP numbers against hard-coded values. All
assertions pass (`published_reference.<bench>.asserted = true`), with
`published_reference.permuted.n_seeds.mlp = 2` and
`published_reference.split.n_seeds.mlp = 3`.

---

## 2. The effect is real and large — Q1

`verdict.q1_projection_effect_real`, 3 seeds:

| arm | `sweep` row | accuracy | ±95% CI |
|---|---|---:|---:|
| `identity` g=2.2 (current `cortex.py` behaviour) | `identity, fan_in=0, gain=2.2` | 36.17 | 2.14 |
| best mixed (`sparse` f=784 g=4.0, `winner`) | `sparse, fan_in=784, gain=4.0` | 71.67 | 5.70 |

`delta = 0.355`, `answer = true`. The `identity` arm is bit-exact with the
existing `drive[:n_input] = pix * gain` line, so this is a like-for-like
replacement of the input stage and nothing else.

The effect survives the more conservative solve convention: under
`sweep[].linear_robust` the `identity` arm is 61.58 ± 2.59 and the winner is
76.67 ± 3.83, so the 15.08-point gap on that convention still exceeds the sum of
the two CIs (6.42). The headline's +35.50 points on the `rcond=1e-3` convention
and the +15.08 on the robust convention are not the same size, and the robust one
is the number to quote if only one is quoted.

Two other things in `sweep` are worth recording because they were the reason the
rigorous re-measurement existed:

- **The gain response is non-monotonic.** `identity`: g=1.0 → 32.33, g=2.2 →
  36.17, g=3.0 → 35.67, g=6.0 → 14.67, g=12.0 → 14.00. Activity falls with gain
  over the same range (9.14% → 13.12% → 13.74% → 3.63% → 2.23%), so "more gain"
  is not "more signal".
- **Activity cannot be raised by identity routing.**
  `matched_activity.identity_max_active_frac = 0.13737` (13.74%) is the highest
  active fraction any `identity` gain reaches, and `identity_at_max` records that
  it occurs at `gain = 3.0` with `linear = 0.35667`. The winner sits at
  `winner.active_frac = 0.25948` (25.95%). A pure 1:1 map cannot reach the
  winning arm's activity, so the matched-activity control (§3) has to use the
  zero-mixing controls rather than extrapolating `identity`'s gain.

---

## 3. The stated cause is not supported — Q2

The hypothesis was *convergence*: neurons that mix several pixels acquire
multi-pixel selectivity. The decomposition holds one pixel per neuron and
manipulates only routing and polarity (`verdict.q2_is_it_convergence.polarity_decomposition`,
all at gain 2.2, 3 seeds):

| arm | what it changes | pixels per neuron | accuracy | ±95% CI | active |
|---|---|---:|---:|---:|---:|
| `identity` | nothing | 1 | 36.17 | 2.14 | 13.12% |
| `ctrl_perm` | random pixel→neuron routing | 1 | 38.00 | 4.93 | 13.04% |
| `ctrl_centered` | image mean subtracted | 1 | 33.33 | 5.58 | 12.00% |
| `ctrl_sign` | random ±1 per-neuron polarity | 1 | **64.00** | 1.96 | 6.57% |
| `ctrl_both` | permutation **and** polarity | 1 | **63.50** | 0.98 | 6.42% |

`polarity_effect = +0.2783`, `routing_effect = +0.0183`,
`centering_effect = −0.0283`. The script's verdict string is
`"PARTIALLY RETRACTED, CAUSE CORRECTED ... the operative variable is NOT afferent
convergence. It is receptive-field POLARITY"`, and its stated grounding is that
the identity path is rectified (pixels lie in [0,1], so every receptive field is
all-positive) whereas cortical receptive fields are signed.

The residual case for convergence is not nothing, but it is inside its own noise.
`verdict.q2_is_it_convergence` compares the best mixed arm against the
best zero-mixing arm within 2 activity points: best mixed 71.67 at 25.95% active
vs `best_zero_mixing_kind = "sparse"`, `best_zero_mixing_gain = 3.0` at
`best_zero_mixing_linear = 0.6417` and `best_zero_mixing_active_frac = 0.2412`.
The `gap` is 0.075 against `combined_ci95 = 0.0998`, and
`answer_is_convergence = false`.
Reading `matched_activity.pairs` directly: of 77 mixed↔zero-mixing pairs, 43 have a
positive accuracy gap and 25 of those have disjoint CIs, but 33 have a *negative*
gap, and `matched_activity.pairs[].activity_gap` is not zero — its median is
0.0064 and 24 of 77 pairs differ by more than 0.01 activity. The mixed arms are
not systematically better than their activity-matched counterparts.

---

## 4. The consequence test: no rescue — Q3

The sweep's winner was re-run end-to-end through the published continual-learning
runner by monkeypatching only `_code` (stated plainly in the script; `cortex.py`
is not modified), with the published protocol's flags
(`reset_between_samples=False`, binary readout) and 800 neurons
(`CONS_NEURONS`), 5 tasks, single pass, no replay, 3 seeds.
`consequence.permuted.arms`:

| arm | final retained | ±95% CI | just-trained | forgetting | params | SynOps/sample |
|---|---:|---:|---:|---:|---:|---:|
| `identity_control` | 21.73 | 1.25 | 25.53 | 4.80 | 8,000 | 13,397 |
| `winner2_sparse_f256_g3.0` | **22.87** | 0.13 | 28.67 | 5.80 | 8,000 | 18,176 |
| `winner1_sparse_f784_g4.0` | 21.67 | 1.14 | 26.33 | 4.87 | 8,000 | 19,349 |
| `zero_mixing_sparse_g2.2` | 21.07 | 3.80 | 26.00 | 5.87 | 8,000 | 15,573 |
| `mlp` | **68.13** | 3.75 | 74.40 | 6.33 | 101,770 | -- |
| `mlp_replay` | 72.93 | 3.06 | 72.73 | 1.60 | 101,770 | -- |

`gap_to_mlp_pp = 45.2667`, `rescues_continual_learning = false`. The winner
improves on the identity control by **+1.13 points** — a projection that is worth
+35.50 points to a linear probe on a single task is worth just over a point to
the full continual-learning protocol. On split, the same comparison is
`best_projection_split_final = 0.3960` vs `identity_control_split_final = 0.3680`
(+2.80 points), with `mlp_split_final = 0.1867` and
`mlp_replay_split_final = 0.3093`.

---

## 5. The quadratic-readout anomaly does not reproduce

An earlier observation was that a quadratic readout rescued a thin linear one.
`quadratic_check` re-measured it: `mean_delta_identity = 0.0050` and
`mean_delta_mixed = 0.000364`, i.e. the quadratic bonus is 0.50 percentage
points on the unmixed identity arms and 0.04 percentage points on the mixed arms.
`verdict.quadratic_anomaly.explained_by_missing_mixing = true`. In `sweep`, 82 of
109 rows have `quadratic_minus_linear` exactly 0.0, and the winner's own
`quadratic_minus_linear` is 0.0 with `quadratic_minus_linear_robust = −0.0100`.

---

## 6. What this does NOT show

1. **It does not show a replacement for the substrate's learning.** Every probe
   here is a closed-form least-squares solve. The projection is frozen and random
   and is never trained; the result is about what the *feature space* supports,
   not about a mechanism a neuron could implement. Nothing here is a local rule.
2. **It does not show convergence is irrelevant.** The claim that fails is "the
   gain is due to convergence". The residual mixed-vs-zero-mixing gap is +7.50
   points but its combined CI is 9.98, so it is inconclusive rather than refuted —
   and 25 individual matched pairs do show positive, CI-disjoint gaps.
3. **It does not validate the argument from biology.** That cortical afferents
   are convergent and signed is the motivation, not a measurement. The experiment
   measures a rectified 1:1 path against signed random ones; it does not measure
   cortex.
4. **It does not generalise past this task family.** `config.n_train = 500`,
   `config.n_test = 200`, one dataset (MNIST), one population size
   (`config.n_neurons = 900` for the sweep, 800 for the consequence test), one
   time window (`config.t_ms = 15`).
5. **It does not report energy or throughput.** The JSON's only timing key is
   `wall_seconds` (1125.6 s for the whole committed run); there is no per-arm
   wall-clock, no joule, and no throughput key at all. `macs_per_sample` is the
   projection's own operation count and `synops_per_sample` counts substrate
   synaptic operations; neither is an energy measurement, and no arm on either
   side of the comparison was run under a power meter.
6. **It does not establish that the best zero-mixing arm is the best possible one.**
   The zero-mixing arms in `matched_activity.zero_mixing_arms` are 27 rows from
   four experiment-local control kinds plus `sparse` f=1; `identity` is excluded
   from them by construction, so "no convergence achieves 64%" is a statement
   about this control set, not about all zero-convergence projections.

## 7. Confounds visible in the data and the script

1. **The winner is selected on the same 200 samples it is scored on.** The sweep
   reports the mean over seeds, but `select_winner` takes the argmax over 109
   configurations using `sweep[].linear`, so the reported 71.67 is an in-sample
   choice over configs. The script says this outright and re-runs the top two
   end-to-end — which is how we know the choice does not transfer: on the
   continual-learning benchmark, second-place `sparse` f=784 g=4.0 scores
   21.67, *below* the identity control's 21.73. Treat the sweep's top rows as a
   ranking over a fixed test set, not as three independent confirmations.
2. **The end-to-end arms use a different population size from the sweep.** 900
   neurons in `sweep` (`config.n_neurons`), 800 in `consequence`
   (`CONS_NEURONS` in the script) — the published benchmark's own count, chosen so
   the MLP baseline is comparable. The +35.50 point probe gain and the +1.13 point
   end-to-end gain are therefore measured at different network sizes, and the
   projection that wins the sweep is not necessarily the one that would win at 800.
3. **Backend changes the number.** `backend_parity` runs one seed per row on
   NumPy and MLX with identical configs:

   | kind | NumPy linear | MLX linear | difference |
   |---|---:|---:|---:|
   | `identity` f=0 g=2.2 | 0.340 | 0.395 | 0.055 |
   | `sparse` f=64 g=6.0 | 0.670 | 0.645 | 0.025 |

   The sweep is NumPy-only (`config.backend = "numpy"`), for a stated performance
   reason, so every sweep number carries an unquantified backend offset of up to
   ~5.5 points at one seed. Parity rows have no CI (one seed each). Note the
   NumPy identity value, 0.340, is one of the three per-seed values that average
   to the headline 36.17; it is not a discrepancy, but it is why the parity table
   and the sweep table should not be read against each other directly.
4. **Two solve conventions give two different stories for the same arm.** For
   `ctrl_sign` g=2.2: `linear = 64.00` but `linear_robust = 72.46`. The
   polarity effect is present under both (identity `linear_robust` 61.58), but it
   shrinks from +27.83 to +10.88 points when the unstable convention is replaced.
   `_probe_all`'s own docstring records the reason: the code matrix is ~800 wide
   with many near-zero-variance columns, so accuracy is non-monotonic in training
   size under `rcond=1e-3`.
5. **The control kinds are not shipped code.** `ctrl_perm`, `ctrl_sign`,
   `ctrl_both` and `ctrl_centered` are defined in `experiments/afferents.py` as
   `_ControlProjection`, deliberately kept out of `brain/afferents.py` so the
   shipped interface stays as specified. The library equivalent of `ctrl_both` is
   `sparse` with `fan_in=1` (`n_weights` 1800, `n_connections` 900), and the two
   agree to within a point (63.50 vs 63.17 at g=2.2), which is the cross-check the
   script intended.
6. **The script rewrites its own committed artifact and has no `--out` flag.**
   `main` writes `RESULTS_DIR / "afferents.json"` unconditionally. Re-running the
   experiment to check it therefore overwrites the file this document cites; the
   327-run sweep plus parity plus consequence took `wall_seconds = 1125.6` s for
   the committed run. Determinism was instead checked by re-running individual
   `sweep_job` cells, which reproduced the committed values: `identity` g=2.2 mean
   0.3617 (per-seed 0.34/0.375/0.37) and `ctrl_sign` g=2.2 mean 0.6400
   (per-seed 0.63/0.63/0.66), matching `sweep[]` exactly for both.
7. **`seeds_finite` is true everywhere, and that is a real check.** Every sweep row
   reports `seeds_finite = true` and the four consequence *projection* arms report
   `weights_finite = true` (the `mlp`/`mlp_replay` baselines are built by a
   different function and carry no `weights_finite` key), so no accuracy here is a
   NaN that a mean silently dropped.

## 8. Reproducing

```bash
cd /Users/richie/Documents/github/human-brain
python3 brain/afferents.py            # interface self-check (fast, no data)
python3 experiments/afferents.py      # full experiment, ~19 min, overwrites results/afferents.json
```

What would change the conclusion:

- **A strict no-convergence control that beats the winner.** If a zero-convergence
  projection reached 71.67% at ~26% active, the polarity story would collapse into
  "any de-correlating random projection of the right polarity suffices" — and
  §3's +7.50 point residual, already inside its CI, would go to zero.
- **A signed 1:1 projection with a *learned* readout.** The polarity result currently
  rests on closed-form probes. If a local rule cannot exploit the signed projection,
  the fix is a property of the feature space only.
- **A second dataset.** The whole result is MNIST with 500 train / 200 test samples.
  A polarity gain that does not appear on another task family is a property of this
  dataset's pixel statistics (which are non-negative and sparse), not of cortex.
- **An activity-matched mixed arm with a CI-disjoint, positive gap at a comparable
  activity level.** That is the direct test of convergence, and `matched_activity.pairs`
  already contains 25 candidate pairs that pass it while 33 fail it; the question is
  whether they survive as a group rather than as selected pairs.
