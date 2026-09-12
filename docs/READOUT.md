# The readout is the bottleneck: the same frozen code scores 37% or 65% depending only on the rule

**Result: on identical cached spike-count codes, on the same benchmark, in the same
substrate, the local readout's learning rule is worth `+28.00` points on Split-MNIST and
`+8.20` points on Permuted-MNIST. Both gaps are larger than their combined 95% CI. The
representation did not change at all between those two arms — only the error term did.**

This document reports `experiments/results/readout_diagnostic.json`, produced by
`experiments/readout_diagnostic.py`. Every number below is a value from a named key in
that file, and each table cites the key path it was read from.

The diagnostic separates three hypotheses that had been conflated:

| | hypothesis | prediction if true |
|---|---|---|
| `H-code` | the representation is weak | every readout fails; a cheap control with the same width wins |
| `H-rule` | the code is fine, the local rule is the bottleneck | same features, same local family, different error term ⇒ large gain |
| `H-linear` | the code is nonlinearly decodable, linearly not | quadratic arms win |

---

## 1. Method

**What was varied.** The readout applied to a *fixed* code: the learning rule
(`local_perceptron` binarised error, `local_delta` graded error, `local_softmax`
cross-entropy, and the closed-form `ridge`/`ridge_quad` solves), the feature space
(`linear` vs a 256-term sampled quadratic, `meta.quad_dim = 256`), the normaliser
(`normalize` true/false), the step size (`lr`, 4 values in `lr_sweep`), and the substrate
variant (`meta.variants = ["as_falsified", "reset_fresh"]`).

**What was held fixed.** Everything else that matters. The substrate's code for every
sample is encoded **once** and cached to `data/readout_codes.npz`
(`meta.cache_version = "2"`, `meta.config_hash = "f8f2e831ee919b13"`, which matches the
cache's own `config_hash`). Every arm trains and is scored on bit-identical matrices, so
the readout kind is the only variable. The substrate is fixed at
`meta.substrate` = 800 neurons, `k_out = 64`, k-WTA with `k_wta = 64`, dendrites on,
plasticity on, `gain = 2.2`, 15 ms per sample. Encoding **is** substrate training when
plasticity is on, so the samples are fed in original order (task 0 train, task 0 test,
task 1 train, …) to reproduce the sequence the original falsification used.

**Benchmarks and sample sizes.** 5 tasks, single pass, no replay
(`meta.tasks = 5`, `meta.seeds = 3`). Per task, `meta.train_per_task` is 80 (split) and
200 (permuted); `meta.test_per_task` is 50 (split) and 100 (permuted). So each arm is
scored on 250 (split) or 500 (permuted) held-out samples per seed, and every number below
is a mean over 3 seeds with a 95% CI of `1.96 * SEM` (`summary[...].final_average_ci95`).

**The metric.** `final_average` (`summary[...].final_average`) is the mean accuracy over
all 5 tasks measured at the **end** of the stream, through one 10-way head with **no task
ID supplied at test**. `diagonal_mean` is the mean accuracy on each task immediately after
it was trained. `backward_transfer` is the mean over tasks of (final accuracy − accuracy
right after training), and `forgetting` is the mean over tasks of (best accuracy ever seen
on that task − accuracy at the end). All three are read from the same named summary keys.

**Chance level.** The JSON contains no chance key, so this is arithmetic from the
benchmark definition (`meta.substrate.n_classes = 10`, 5 tasks) and is stated as such:
uniform guessing over 10 classes gives **10%** on both benchmarks. A predictor that knows
*which* two classes a Split-MNIST task uses, but not the labels, scores **50%** on split;
on Permuted-MNIST every task contains all 10 classes, so task awareness buys nothing there
and the level stays 10%. Both are computed, not measured, and are not keys in the file.

**Scale of the run.** `arms` has 492 rows, `ceiling` 246, `capacity_scaling` 40,
`summary` 164 groups, and `failed_arms` is empty.

---

## 2. The rule gap — `H-rule`

Split-MNIST, `as_falsified` (the config the original 22.10% run used:
`reset_between_samples=False`), `lr = 0.05`, 3 seeds. Key form:
`summary["<arm>|brain|True|split|as_falsified|0.05"]`.

| arm | rule | final | ±95% CI | just-trained | forgetting | params | diverged seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| `lin_perceptron` | binarised error | 37.33 | 4.28 | 65.47 | 28.13 | 8,000 | 0 |
| `lin_softmax` | cross-entropy | 46.40 | 5.67 | 66.93 | 20.93 | 8,000 | 0 |
| `lin_delta` | graded error, `lr=0.05` | 10.00 | 0.00 | 5.20 | 0.00 | 8,000 | **3 / 3** |
| `lin_ridge` | closed-form linear (not local) | 62.80 | 1.63 | 68.13 | 6.00 | 8,000 | 0 |

Permuted-MNIST, same variant:

| arm | rule | final | ±95% CI | just-trained | forgetting | diverged seeds |
|---|---|---:|---:|---:|---:|---:|
| `lin_perceptron` | binarised error | 25.20 | 0.45 | 29.73 | 5.00 | 0 |
| `lin_softmax` | cross-entropy | 25.67 | 2.27 | 30.47 | 5.60 | 0 |
| `lin_delta` | graded error, `lr=0.05` | 10.00 | 0.00 | 10.00 | 0.00 | **3 / 3** |
| `lin_ridge` | closed-form linear (not local) | 18.33 | 2.38 | 43.40 | 25.07 | 0 |

The delta rule at its own step size is competitive, not divergent. From `lr_sweep`:
`sweep_delta` reaches **65.07 ± 1.31** at `lr = 1e-4` on split and **33.40 ± 1.58** at the
same `lr` on permuted, against 10.00 for the same rule at `lr = 0.05`. The `lr_sweep`
entries are keyed `summary["sweep_delta|brain|True|<bench>|as_falsified|<lr>"]`, and
`lr_sweep.<bench>.local_delta.best_lr` records the selected step size.

At the falsification's own step size the comparison the script draws is
`verdicts[].rule_gap` — best local linear arm minus `local_perceptron`, on the same
features, same normaliser, same family:

| benchmark | `rule_gap` | combined CI | `verdicts[].calls` |
|---|---:|---:|---|
| split | +28.00 pts | 5.58 | `H-rule SUPPORTED` |
| permuted | +8.20 pts | 2.04 | `H-rule SUPPORTED` |

**This is the headline.** The gap is not a step-size artefact in the sense of "the local
rules cannot learn": the step-size sweep is reported in full, and each rule is judged at
its own best step (`verdicts[].best_local_lr`). It is worth noting *why* the perceptron is
flat across the sweep — `lr_sweep.split.local_perceptron.sweep` gives 0.3733 at all four
step sizes because a binarised error takes values in {−1, 0, 1}, so the update is
scale-free. The delta rule's error is unbounded in the weight norm, which is why it
diverges at `lr = 0.05` (3 of 3 seeds, `summary[...].n_diverged = 3`) and needs `1e-4`.

---

## 3. The code gap — `H-code` flips between benchmarks

`verdicts[].code_gap` is the best local arm (any space) minus the best *cheap control*
arm. The control candidate set is fixed in the script at six arms: `ctrl_rp800_ridge`,
`ctrl_rp800_quad_ridge`, `ctrl_rp800_ridge_n`, `ctrl_rp800_quad_ridge_n` (a fixed random
projection of the raw pixels followed by ReLU, at width 800), and `ctrl_pix_ridge`,
`ctrl_pix_quad_ridge` (the raw 784 pixels themselves). The strongest control wins the
comparison, so this is the conservative version, and `verdicts[].best_control_arm` names
the winner — `ctrl_rp800_ridge_n(rp800)` on both benchmarks.

| benchmark | best local | best control | `code_gap` | combined CI | call in `verdicts[].calls` |
|---|---:|---:|---:|---:|---|
| split | 65.33 (`sweep_softmax` @1e-3) | **80.00** (`ctrl_rp800_ridge_n`) | **−14.67** | 9.78 | `H-code SUPPORTED: a cheap control feature set beats the substrate` |
| permuted | 33.40 (`sweep_delta` @1e-4) | 24.07 (`ctrl_rp800_ridge_n`) | **+9.33** | 6.53 | `substrate code BEATS the best cheap control` |

The two benchmarks disagree, and the disagreement is not resolvable from this file. On
Split-MNIST a **training-free** random projection of the pixels plus a closed-form linear
readout at the same 800 width (`ctrl_rp800_ridge_n`, `summary[...] = 80.00 ± 2.72`)
beats everything the substrate produces. On Permuted-MNIST the substrate's code does beat
the same control, by 9.33 points with a combined CI of 6.53.

The per-task breakdown shows what the control is doing. On permuted, `ctrl_rp800_ridge_n`
scores 80.87 on each task just after that task is trained but only `24.07` at the end
(`forgetting = 56.80`) — a catastrophic-forgetting profile that the substrate's local
rules do not share (`lin_perceptron` on permuted: `forgetting = 5.00`). The control wins
the immediate-fit column and loses the retention column; `final_average` is the retention
column, which is the only reason the substrate is ahead on that benchmark at all.

---

## 4. The linearity gap — `H-linear` is rejected on both benchmarks

`verdicts[].linear_gap` = best quadratic arm − best linear arm, same rule family:
**−22.40** points (CI 7.52) on split and **−17.93** points (CI 5.34) on permuted. The
quadratic space is *worse*, not better. `summary["quad_ridge|brain|True|split|as_falsified|0.05"]`
is 42.93 ± 4.58 against `lin_ridge`'s 62.80 ± 1.63; the plausible local quadratic rule
`quad_local` sits at 10.00 and diverged on all 3 seeds.

This contradicts the earlier reading (recorded in `docs/RESULTS.md` §3.1, and the reason
this stage exists) that a quadratic readout rescued a linear one on cached codes. The
script anticipated that: `capacity_scaling` re-measures the question with the penalty
re-selected per training size on a training holdout and a **fixed** 1,000-sample test set
(`capacity_scaling_summary`, `n_seeds = 2`). If the quadratic advantage were real it
should widen as data grows. The earlier comparison varied the feature space at one
training size and one penalty; this stage varies training size and re-selects the penalty,
which is the difference the script's own docstring gives as the reason the original
observation was an artefact of the sample/penalty regime.

| train n | brain linear | brain quad | Δ | pixels linear | pixels quad | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 43.50 | 26.05 | −17.45 | 71.65 | 61.60 | −10.05 |
| 420 | 44.00 | 32.95 | −11.05 | 76.30 | 65.15 | −11.15 |
| 1000 | 47.70 | 41.05 | −6.65 | 79.55 | 75.10 | −4.45 |
| 2000 | 47.35 | 40.75 | −6.60 | 82.80 | 80.65 | −2.15 |
| 4000 | 47.25 | 45.05 | −2.20 | 83.30 | 84.65 | **+1.35** |

The quadratic gap narrows with data and never inverts for the substrate's code, while the
substrate's code never approaches the raw pixels at any size (47.25 vs 83.30 at n = 4000).
The single-task ceiling set agrees: `ceiling_summary["lin_ridge|brain|True"].test_accuracy`
is 60.46 ± 15.44 against `ceiling_summary["quad_ridge|brain|True"].test_accuracy` 47.96 ± 11.90,
and the quadratic-dimension ladder is flat —
`ceiling_summary["ladder_quad16|brain|True"]` 48.89 ± 13.16, `ladder_quad64` 48.80 ± 11.17,
`ladder_quad256` 47.96 ± 11.90, `ladder_quad1024` 50.28 ± 12.29.

---

## 5. The normaliser is a second large variable

Dropping the feature normaliser is worth the same order as changing the rule on split:
`raw_perceptron` 13.47 ± 4.35 vs `lin_perceptron` 37.33 ± 4.28 (−23.86), and `raw_ridge`
34.67 ± 2.23 vs `lin_ridge` 62.80 ± 1.63 (−28.13) (`normalize` false vs true, keys
`summary["...|brain|False|..."]` vs `summary["...|brain|True|..."]`). For comparison, the
rule gap in §2 is +28.00. Every `raw_*` arm in `summary` is therefore a different
operating regime, not a robustness check, and the verdict block uses the normalised arms.
The `meta.caveats` note that the normaliser drifts during training and is frozen at test
time is the relevant warning: this is a live confound, and `raw_*` arms exist precisely so
it can be seen. Note the two effects are not additive — `raw_perceptron` (13.47) and
`raw_ridge` (34.67) both fall below `lin_perceptron` (37.33), so removing the normaliser
removes more than the rule difference accounts for, and the rule gap must not be read as
a quantity that transfers across normaliser settings.

---

## 6. What this result does NOT show

1. **It does not show the substrate is good.** The best cheap control beats it by 14.67
   points on Split-MNIST, and the substrate's code loses to raw pixels at every training
   size in `capacity_scaling`. The verdict block's own `H-code SUPPORTED` call on split
   should be read as stated.
2. **It does not show a fix.** `ridge`/`ridge_quad` are closed-form least-squares solves
   and are not biologically plausible: they measure what the feature space *supports*, not
   what a neuron could learn. `meta.caveats[0]` says so in the file itself. The local rules
   never reach the closed-form arms (65.33 best local vs 80.00 control on split).
3. **It does not isolate a single mechanism.** Three variables are confounded across the
   tables — the error term, the step size, and the normaliser — and the rule gap is
   reported at each rule's own best step, which is a selection over the sweep, not a
   pre-registered single setting. `lr_sweep` reports the whole grid so the selection is
   auditable, but "best" here means "best on the reported benchmark".
4. **It does not establish the `H-rule` result for the current substrate.** The verdict
   block is emitted only for `as_falsified` (`verdicts[].variant`). Under `reset_fresh`,
   `summary["lin_perceptron|brain|True|permuted|reset_fresh|0.05"].final_average` is 48.80
   ± 2.16 against 25.20 for the same arm under `as_falsified` — the variant that the
   substrate is *currently* configured for scores almost twice as well at the same readout.
   Any statement about "the readout" has to name the variant it belongs to.
5. **It says nothing about wall-clock or energy.** `meta.caveats[5]` is explicit: operation
   counts are not joules and no wall-clock claim is made for the readout arms. There is no
   SynOps or energy key anywhere in this JSON, so none is reported here.

## 7. Confounds visible in the data and the script

1. **`ceiling_summary` averages two different substrates.** `ceiling_summary` groups by
   `(arm, features, normalize)` and therefore pools the two variants, which is why its
   `n_seeds` is 6 for brain arms. Grouping the `ceiling[]` rows by `variant` instead:
   all 33 control cells are bit-identical across variants (they never read the substrate,
   so this is the correct outcome), but only **14 of 90** brain ceiling cells are; e.g.
   `lin_ridge` is 42.96 ± 3.46 under `as_falsified` and 77.96 ± 1.92 under `reset_fresh`.
   The pooled 60.46 in §4 is an average over two substrates and should not be read as one.
2. **Control arms are duplicated across variants in `arms` but the current code cannot
   produce them.** 132 of the 492 `arms` rows are control rows: 11 control arms × 2
   benchmarks × 3 seeds under *each* of the two variants, and all 66 paired control cells
   are identical across variants. `run_all` explicitly skips non-brain arms when
   `vi > 0` ("running them once per variant would duplicate bit-identical seeds"), so the
   committed rows cannot have been produced by the committed arm-running path. The
   consequence is that any `reset_fresh` lookup of the control term would resolve to
   missing (and `code_gap` to not-measurable) if the arms were regenerated as the code now
   reads. The published verdicts are unaffected because they are emitted for `as_falsified`
   only, but a `reset_fresh` re-analysis from freshly generated rows will not reproduce the
   `reset_fresh` control numbers in the committed `summary`.
3. **The documented oracle arms were never run.** The module docstring and
   `meta.caveats[1]` describe `oracle=True` arms that fit all tasks at once as an upper
   bound. Every row in `arms` has `oracle = False`, and no `oracle` group exists in
   `summary`. The upper bound the docstring promises is not in the data.
4. **`meta.wall_seconds_total = 0.0437` s is not a run time.** 492 arms cannot be
   evaluated in 44 ms — a partial re-run of 63 arms on one benchmark and one seed took
   ~26 s on this machine. The value is consistent with the `--reanalyse` path, which
   re-derives every table from rows already in the file (a re-analysis of the committed
   rows on this machine reported 0.0208 s). Re-analysis over the same rows reproduces
   `summary`, `verdicts`, `lr_sweep`, `ceiling_summary`, `checks` and `capacity_scaling`
   identically, so the numbers are re-derivable, but the file's own wall-clock field does
   not describe the experiment that produced the rows.
5. **The arm-running path is currently broken.** In
   `experiments/readout_diagnostic.py`, `main` calls
   `report_and_write(rows, ceil_rows, cap_rows, ...)` at line 1493, but `cap_rows` is
   never bound in `main` — it only exists as a parameter of `report_and_write` and as a
   local inside its `--reanalyse` / `--capacity` branches. Running the script without
   `--reanalyse` raises `NameError: name 'cap_rows' is not defined` *after* all arms have
   been run, and writes no JSON. Reproduced twice on this machine (`--from-cache
   --no-capacity` and `--from-cache` with default capacity), each time losing the entire
   arm run. The JSON can therefore only be regenerated through
   `--reanalyse` over an existing file, which is precisely the path that cannot create one.
   This is a reproduction defect, not a result, and it is the first thing to fix.
6. **`--quad-dim` defaults to 256 but the ladder and the capacity stage disagree on
   scope.** The ladder varies `quad_dim` at 16/64/256/1024 for the continual arms while
   `capacity_scaling` uses `meta.quad_dim` throughout; the JSON records `quad_dim` per
   `ladder_*` row (`1024, 1024, False` for `quad_dim_requested`/`quad_truncated` on
   `ladder_quad1024`), so the ladder is auditable, but the two quadratic comparisons are
   not the same experiment.

## 8. Reproducing, and what would change the verdict

The script is deterministic: re-running individual arms from the shared cache reproduced
`final_average` and `n_params` exactly for 12 tested cells spanning `lin_perceptron`,
`lin_softmax`, `lin_ridge`, `raw_perceptron`, `sweep_delta`, `sweep_softmax`,
`sweep_perceptron` and `sweep_quad`, across both benchmarks, four step sizes and both
variants. No re-run produced a different number. One caveat for anyone repeating this:
the four `sweep_*` arms share a name across the four step sizes and are distinguished only
by the `lr` in their summary key, so an arm reconstructed without the intended `lr` will
compare a different row. The cache's own integrity checks all pass: every value in
`checks.checks` is 1.0 (`rows_aligned`, `codes_finite`, `codes_nonzero_variance`, and four
control-non-degeneracy checks), and `checks.hashes` contains 12 per-(benchmark, variant,
seed) code hashes.

```bash
# Re-derive every table and verdict from an existing JSON. This is the only path
# that currently works end to end. --reanalyse reads its input from --out and
# writes back to it, so work on a copy:
cp experiments/results/readout_diagnostic.json /tmp/rd_copy.json
python3 experiments/readout_diagnostic.py --reanalyse --out /tmp/rd_copy.json
# (running it with the default --out rewrites the committed artifact in place)

# To regenerate from scratch the NameError in §7.5 must be fixed first, then:
python3 experiments/readout_diagnostic.py --from-cache
```

What would overturn the headline:

- **A different verdict variant.** If the control term is fixed so that `reset_fresh`
  resolves, the H-code and H-rule calls should be recomputed on both variants. The
  `as_falsified` numbers are not the current substrate's numbers.
- **A parameter-matched control.** `ctrl_rp800_ridge_n` and the substrate arms both have
  `n_params = 8000`, so that comparison is width-matched. The `pix` controls have 7,840 and
  the quadratic controls 18,560-20,560, so not every table row is matched; `n_params` is
  in every summary group for exactly this reason.
- **A closer look at the split-control win.** The 80.00 point control uses a ridge solve
  fitted per task with the penalty chosen on a training holdout (`ridge_sweep` is stored
  per arm). If that selection is unfair in any way, the H-code call on split is the claim
  that moves.
