# Independent adversarial review

**Reviewer:** independent adversarial reviewer (did not write any of this code)
**Date:** 2026-09-12
**Method:** read `README.md`, `docs/RESULTS.md`, `docs/PARADIGM.md`, `docs/DECISIONS.md`,
`docs/SCALING.md`, then re-ran every command; recomputed every statistic from the raw
`runs` arrays; reproduced or falsified each mechanism claim with isolated probes in
`/tmp`; re-introduced two historical bugs to test whether the suite catches them.

**Revision reviewed** (md5, captured 00:03:50 — see "Concurrency" below, this moved repeatedly):

| file | md5 |
|---|---|
| `README.md` | `6f62752f0c131bed90f8c06ceb763ffa` |
| `docs/RESULTS.md` | `8e3e0d4c81d7a96ea60fb063293295d4` |
| `docs/DECISIONS.md` | `1bca5c4cacbbac3d823fdc7824a3a4e9` |
| `docs/SCALING.md` | `5ce6467b61b17cc9af53a56e21f60e18` |
| `brain/cortex.py` | `2c9cfcf34434cbe5c691a8925edf99d6` |
| `brain/simulator.py` | `7c7d28c7c016dbc387942149854f5c31` |
| `experiments/results/continual_split.json` | `d4f82149a9a1a5a1adb1d6af2b8fd635` |
| `bench/results_scale.json` | `1c9aba103fcac88ab0138788b91c0cc8` |

**Environment:** Apple M4 Max / macOS 27.0 arm64 / CPython 3.14.3 / NumPy 2.4.1 /
MLX 0.31.0 (`Device(gpu, 0)`) / 137.44 GB RAM.

---

## Verdict summary

| claim | verdict |
|---|---|
| C1 — single-neuron XOR from `dcaap`, impossible for `linear` | **SUPPORTED** |
| C2 — `pytest tests/ -q` passes with 114/2/2 | **SUPPORTED** (with a serious caveat about what the xfail marks actually guard) |
| C3 — JSON consistent with RESULTS §2 and CIs arithmetically correct | **SUPPORTED as arithmetic; NOT SUPPORTED as reproducible provenance** |
| C4 — all three null controls land at chance | **NOT SUPPORTED** |
| C5 — dendrite ablation is better, a genuine negative result | **NOT SUPPORTED as written** (wiring is real; the numbers, the "disjoint CIs" claim, and the unconfoundedness claim all fail) |
| C6 — the nine D5/D5b defects are fixed | **PARTIALLY:** 6 genuinely fixed and reproduced; 1 partially (k-WTA); 1 stale/contradictory across docs; 1 in doubt under a deliberate stress |

**One sentence:** the mechanism-level work is real and reproduces; the
continual-learning comparison is not reproducible from the current revision's
defaults, and the "null controls prove it is the substrate's plasticity" argument
is falsified by a control the authors did not run.

---

## Concurrency warning — the repo changed under this review

This is a material threat to every number in the repo, so it comes first.

The results file was **overwritten twice during the review window**:

```
22:11:48  experiments/results/continual_split.json   (arms: brain_nodend, mlp, mlp_replay, frozen, shuffled; brain_noplast "not run")
22:51:06  experiments/results/continual_split.json   (re-generated again)
00:03:50  md5 d4f82149a9a1a5a1adb1d6af2b8fd635
```

I observed the file change identity mid-session: a `json.load` that returned
`brain_noplast -> 9.33 +/- 1.38` early in the session returned
`brain_noplast -> {"status": "failed", "error": "not run"}` minutes later, and later
`0.1000 +/- 0.0000`. A concurrent `python3 experiments/continual.py --permute ...`
process (PID 69419) was running in the repo for part of the review.

The consequence is that **documentation and artifacts are not version-pinned to each
other**. `README.md` (22:40), `docs/RESULTS.md` (22:51) and
`experiments/results/continual_split.json` (22:51) were each written by a different
process at a different time. Every number below is therefore reported with the file
state I actually measured.

---

## C1 — single-neuron XOR

### Command

```
python3 experiments/xor_neuron.py
```

### Observed output (exit 0)

```
dend_mode = dcaap
  magnitudes      : [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
  dendritic drive : [0.0, 0.8244, 1.0, 0.9098, 0.7358, 0.406]
  monotonic       : False
  separable threshold for XOR: 0.8679
  observed spikes : [0, 1, 1, 0]
  XOR truth table : [0, 1, 1, 0]
  XOR computed    : True

dend_mode = linear
  dendritic drive : [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
  monotonic       : True
  separable threshold for XOR: NONE (drive never falls below its one-input value) -> XOR impossible

dcaap computes XOR          : True
linear ablation computes XOR: False
VERDICT: dendritic nonlinearity is causal: True
C1 EXIT: 0
```

### Independent adversarial check of the "provably impossible" half

The script's `linear` verdict is *analytic* — it never actually tries to find a
threshold, it just declines to compute one. I swept every threshold the neuron
could have:

```
# threshold sweep 0.05 .. 3.00 in 0.05 steps, real spiking neuron, 80 ms
thresholds yielding exactly [0,1,1,0]: NONE

Achievable patterns with the linear dendrite:
  [0, 0, 0, 0] at thresholds [2.0, 2.5]
  [0, 0, 0, 1] at thresholds [1.0, 1.05, 1.5]
  [0, 1, 1, 1] at thresholds [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
```

Because a linear drive is monotone in total input, `drive(1,1) >= drive(1,0)`
always, so any threshold that fires on `(1,0)` also fires on `(1,1)`. The claim is
correct, and I could not construct a counterexample. **C1 is SUPPORTED.**

### Two caveats worth recording

1. **Margin is thin.** `drive(one) = 1.0000`, `drive(two) = 0.7358`,
   `theta = 0.8679`. The separation is 0.13 on a peak of 1.0. The XOR result is
   not robust to parameter drift: at `tau_soma = 3.0` the pattern becomes
   `[0,1,1,1]` (wrong), and at `n_ms <= 20` the neuron has not integrated far
   enough and emits `[0,0,0,0]`. Defaults (`tau_soma=12`, `n_ms=80`) work.
2. **The threshold is chosen using the same table it is tested on.** That is
   legitimate for demonstrating the existence of a separating threshold, but it is
   not a capacity claim. It should not be read as "this neuron learns XOR".

---

## C2 — `python3 -m pytest tests/ -q`

### Command and observed output

```
python3 -m pytest tests/ -q
........................................................................ [ 61%]
..........................................XXxx                           [100%]
114 passed, 2 xfailed, 2 xpassed in 3.00s
```

Counts match the claim exactly, on both backends. The two XPASS are indeed the
padding tests:

```
XPASS tests/test_simulator.py::test_compact_padding_is_not_consumed_as_a_spike_index[numpy]
XPASS tests/test_simulator.py::test_compact_padding_is_not_consumed_as_a_spike_index[mlx]
```

**C2 is SUPPORTED as literally stated.** But the green suite is much weaker than it
looks, for two reasons I verified by experiment.

### C2-defect-1: the suite stays green if the phantom-drive bug regresses

I reverted the fix in an isolated copy (`/tmp/rev/hb/brain/simulator.py`, changing
`spike_idx = spike_buf[:n_spk]` back to `spike_idx = spike_buf`) — the exact
correctness-critical bug from D5b#5 — and re-ran the suite:

```
REGRESSION INTRODUCED: spike_idx now uses the whole padded buffer
114 passed, 4 xfailed in 3.16s
PYTEST EXIT CODE: 0
```

Exit code 0. The suite cannot fail on the return of the bug that "contaminated
every result in this repo". It silently converts an XPASS into an XFAIL. There is
no `xfail_strict = true` in any pytest configuration (`pytest.ini`, `setup.cfg`,
`pyproject.toml`, `tox.ini` do not exist), so this will never be noticed by a
green/red check. Given that this bug is the one defect the team judged
correctness-critical, the regression guard for it should be an ordinary assertion,
not an xfail.

### C2-defect-2: the two XFAILs are for a defect the changelog says was fixed

`tests/test_simulator.py:245` marks `test_kwta_enforces_at_most_k_winners` as
xfail with the reason:

> "known bug ... `_kwta_mask` reads `v_soma` AFTER `NeuronState.update` has reset
> spikers to `v_reset=0` ... k_wta=8 allowed 11 spikes in one step"

But `docs/DECISIONS.md` D5b#6 claims exactly this was fixed:

> "**k-WTA ranked the wrong neurons.** ... **Fixed** by exposing `v_pre_reset`."

The fix *is* present — `brain/simulator.py:145` scores by `self.neurons.v_pre_reset`
and `brain/neurons.py:140` sets it before the reset. Yet the test still fails:

```
python3 -m pytest tests/ -q --runxfail
FAILED tests/test_simulator.py::test_kwta_enforces_at_most_k_winners[numpy]
FAILED tests/test_simulator.py::test_kwta_enforces_at_most_k_winners[mlx]
2 failed, 116 passed in 3.86s
```

The test does not fail for the reason documented. It fails because the test sets
`brain.neurons.v_soma = be.zeros((64,))` (and `v_pre_reset` stays at its default
`e_leak = 0.0`), so *all* candidate scores tie. I isolated it:

```
numpy: distinct v_pre_reset -> kept=8  (k=8)   <- correct
numpy: TIED v_pre_reset     -> kept=20         <- bound violated
mlx:   distinct v_pre_reset -> kept=8  (k=8)   <- correct
mlx:   TIED v_pre_reset     -> kept=20         <- bound violated
```

So D5b#6 is *half* fixed: the ranking-source defect is gone, but `_kwta_mask`
(`brain/simulator.py:138-149`) still cannot enforce `k_wta` under ties, because
`score >= cut` admits every neuron equal to the k-th value. The stale xfail reason
actively conceals this: a reader of `DECISIONS.md` believes k-WTA is fixed, and a
reader of the test summary believes the failure is the old, already-fixed one.

**This is not academic.** In the real experiment configuration I instrumented
`_kwta_mask` over 900 steps and found raw candidates exceeded `k_wta` on
**804 of 900 steps** (raw mean 77.8, p95 96.0, max 113 against `k_wta = 64`). The
bound is load-bearing for the sparsity claim on most steps, and it is enforced only
because the potentials happen to be distinct.

Also missing: `pytest` reports `114 passed` but `README.md:208` states "**114
passing tests**", and the new modules `brain/readout.py`, `brain/hierarchy.py`,
`brain/afferents.py` have **zero** test-file references (`grep -rl` over `tests/`
returns 0 for each). Those are the modules that carry the current explanation for
the failure, so the parts of the system under most active change are untested.

---

## C3 — JSON/RESULTS consistency and CI arithmetic

### Method

Recomputed mean and CI directly from the `runs` array of each JSON, with no
reference to the `summary` block:

```python
vals = [r["final_average"] for r in runs if r["arm"] == arm]
mean = sum(vals)/n
sem  = sqrt(sum((v-mean)**2 for v in vals)/(n-1) / n)
ci   = 1.96 * sem
```

### Result — arithmetic matches exactly

```
### continual_split.json   (seeds=3, tasks=5, permute=False)
arm            per-seed final_average           recomputed mean  CI        json mean  json CI  match
brain          [0.380, 0.416, 0.296]                  36.40%     6.97%     36.40%   6.97%   True
brain_nodend   [0.392, 0.412, 0.452]                  41.87%     3.46%     41.87%   3.46%   True
brain_noplast  [0.100, 0.100, 0.100]                  10.00%     0.00%     10.00%   0.00%   True
mlp            [0.188, 0.188, 0.184]                  18.67%     0.26%     18.67%   0.26%   True
mlp_replay     [0.292, 0.312, 0.324]                  30.93%     1.83%     30.93%   1.83%   True
frozen         [0.064, 0.104, 0.088]                   8.53%     2.28%      8.53%   2.28%   True
shuffled       [0.140, 0.132, 0.116]                  12.93%     1.38%     12.93%   1.38%   True
```

Every mean and every CI reproduces to the last digit. I also cross-checked
`diagonal_mean`, `backward_transfer` and `forgetting` against the stored
`acc_matrix` arrays (BWT = `mean_j(acc[n-1,j] - acc[j,j])`, forgetting =
`mean_j(max_i>=j acc[i,j] - acc[n-1,j])`); all matched, e.g. `brain`
bwt `-0.2573` vs stored `-0.2573`, `shuffled` forgetting `0.0573` vs `0.0573`.
**The CI arithmetic is correct.** CIs are `1.96 * SEM`, as documented in
`docs/RESULTS.md:100`.

### C3-defect-1: the stored numbers are not reproducible from current defaults

`docs/RESULTS.md:84` documents the command that produced the table:

```
python3 experiments/continual.py --tasks 5 --seeds 1 --epochs 1 --neurons 800 \
  --train-per-task 80 --test-per-task 50 --lr 1e-2
```

(note: that command literally says `--seeds 1` while the table says 3 seeds, but
that is cosmetic — I ran the 3-seed version.)

I ran exactly that on the current revision, in two isolated read-only copies:

```
arm                   final acc  task-diag       BWT    forget     params   SynOps/sample
brain            10.00 +/- 0.00      5.73%     4.27%     0.00%      8,000           5,611
brain_nodend     10.00 +/- 0.00      5.60%     4.40%     0.00%      8,000           5,995
brain_noplast    10.00 +/- 0.00      6.00%     4.00%     0.00%      8,000           5,461
mlp              18.67 +/- 0.26     96.00%   -77.33%    77.33%    101,770               0
mlp_replay       30.93 +/- 1.83     95.47%   -64.53%    64.53%    101,770               0
frozen           31.07 +/- 4.06     58.40%   -27.33%    27.33%      1,290               0
shuffled         10.00 +/- 0.00      7.33%     2.67%     2.93%      8,000           5,611

brain - mlp final accuracy: -8.67 points (combined 95% CI 0.26)
  -> FALSIFIED: backprop is statistically no worse (A >= B) on this benchmark.
  brain - brain_nodend :  +0.00 points (ablation makes no difference -> claim NOT supported)
  brain - brain_noplast:  +0.00 points (ablation makes no difference -> claim NOT supported)
  brain - shuffled     :  +0.00 points (ablation makes no difference -> claim NOT supported)
```

The brain arm scores **10.00%**, not 36.40%. Every arm that differs from the
documented table (frozen 31.07 vs 8.53) differs. I then bisected the cause.

### C3-defect-2: the documented result requires non-default flags

`brain/cortex.py:75-85` defines three flags whose defaults are:

```
plasticity        = True
readout_learn     = True
readout_rule      = "delta"      <-- default since commit c749be7
reset_between_samples = True     <-- default since commit e8542af
```

I ran the full 3-seed protocol varying only what `_make_arm` passes:

```
rule=binary  reset=False : per-seed=[0.380, 0.416, 0.296] mean=36.40
   DOCUMENTED:             per-seed=[0.380, 0.416, 0.296] mean=36.40
```

**Exact match — but only with `readout_rule="binary"` and
`reset_between_samples=False`.** Neither is the current default. `experiments/continual.py`
does not pass either flag, so it uses the defaults. The stored JSON corresponds to a
superseded revision of `brain/cortex.py`.

### C3-defect-3: the current default `readout_rule="delta"` diverges numerically

This is the most serious code defect I found. Per-sample weight-magnitude trace
over one task (80 samples, `seed=0`, `n_neurons=800`):

```
rule=delta   reset=True : |W|max = 0.00e+00  1.73e-01  1.16e+00  3.58e+01  1.23e+06  2.31e+13   task0 acc=0.200
rule=delta   reset=False: |W|max = 0.00e+00  1.39e+00  1.61e+04  2.77e+11  5.37e+24   nan       task0 acc=0.200
rule=binary  reset=True : |W|max = 0.00e+00  9.42e-02  2.02e-01  4.25e-01  5.19e-01  7.87e-01   task0 acc=0.920
rule=binary  reset=False: |W|max = 0.00e+00  1.61e-01  2.49e-01  3.49e-01  8.39e-01  9.43e-01   task0 acc=0.300
```

The `delta` rule grows the readout by ~13 orders of magnitude and reaches `nan`.
NumPy reports it:

```
brain/cortex.py:218: RuntimeWarning: invalid value encountered in add
  self.W += cfg.readout_lr * np.outer(cn, err).astype(np.float32)
brain/cortex.py:227: RuntimeWarning: overflow encountered in matmul
  return self.W.T @ cn
```

The cause is visible in `brain/cortex.py:204-219`: the substrate code is
binary spike-counts in `[0, t_ms]` but `_normalise` divides by a running standard
deviation that starts near zero, so `cn` is enormous for the first samples; the
graded `delta` error then writes a huge weight, which inflates the next output, and
the loop runs away. The binarised `binary` rule is self-limiting
(`|err| <= 1`). A `delta` rule needs a bounded normalisation or a step-size limit;
it has neither.

Under `delta` the full 5-task/3-seed protocol gives `brain 10.00 +/- 0.00` — this is
not a tuning difference, it is total failure, and it is the *default*.

### C3-defect-4: the within-document inconsistency

`docs/RESULTS.md` §2 (lines 92-98) uses the new numbers. But §1.3 (lines 53-57) and
the caveats bullets (lines 191, 195-196) still carry the **old** numbers:

| location | text | current JSON |
|---|---|---|
| `docs/RESULTS.md:191` | "~10,667 active SynOps/sample" | `11,157` |
| `docs/RESULTS.md:195` | "(96.0% vs 63.3%)" | diag `62.13` |
| `docs/RESULTS.md:196` | "(76.7% forgetting vs 26.3%)" | `77.33%` / `25.73%` |
| `docs/RESULTS.md:147` | shuffled "0.14/0.10/0.11/0.17/0.10" | final row `[0.067, 0.173, 0.133, 0.153, 0.120]` |
| `docs/RESULTS.md:151` | `brain_noplast` "0.47/0/0/0/0" | final row `[0.500, 0, 0, 0, 0]` |

(§1.3 lines 54-57 themselves were updated to 36.40/10.00/8.53/12.93 before I
finished; they changed during the review. The four rows above were still stale at
the pinned revision.)

**C3 verdict: SUPPORTED as arithmetic, NOT SUPPORTED as reproducible provenance.**
The CIs are computed correctly; the numbers are real for *some* revision; they are
not the numbers the current code produces, and the two are mixed inside one
document.

---

## C4 — the null controls

### The precise chance level — the math the claim rests on

Task: 5 tasks × 2 classes = 10 global classes, class-incremental, no task ID at test.

- A **uniform-random** predictor over 10 classes: `1/10 = 10%` mean.
- A **degenerate** predictor emitting one class always: 50% on the task that owns
  that class, 0% on the other four, `(0.5 + 0 + 0 + 0 + 0)/5 = 10%` mean.

Both score 10%. The mean cannot distinguish them. `docs/RESULTS.md` §2.2 (lines
134-156) states this correctly — that section is good work and I confirm it.

### I verified the arms against their matrices, not their means

```
shuffled  mean matrix:
 [[0.08  0.    0.    0.    0.   ]
  [0.093 0.16  0.    0.    0.   ]
  [0.14  0.167 0.213 0.    0.   ]
  [0.08  0.133 0.147 0.193 0.   ]
  [0.067 0.173 0.133 0.153 0.12 ]]
  final-row [0.067, 0.173, 0.133, 0.153, 0.120]   spread across all 5 tasks

brain_noplast mean matrix:
 [[0.5 0.  0.  0.  0. ]
  [0.5 0.  0.  0.  0. ]
  [0.5 0.  0.  0.  0. ]
  [0.5 0.  0.  0.  0. ]
  [0.5 0.  0.  0.  0. ]]
  final-row [0.5, 0.0, 0.0, 0.0, 0.0]             single-class collapse
```

So:

- `shuffled` — **genuinely at chance**, spread uniformly. Valid control. ✓
- `frozen` — 8.53%, spread thinly, at/below chance. Valid control. ✓ (though see below:
  it is the wrong control for the question being asked)
- `brain_noplast` — **NOT at chance.** It is a degenerate collapse: 50/0/0/0/0. The
  mean coincidentally equals 10%. The original claim "all three null controls land at
  chance (9.33%)" was already flagged and corrected in §2.2 — but `README.md:212`
  and `docs/RESULTS.md` §1.3 still assert the incorrect version.

### C4-defect-1: `brain_noplast` was not a control at all in the run that produced the JSON

In the revision that generated the stored numbers, `CortexClassifier.fit_task`
gated the **readout update** on `cfg.plasticity`:

```python
if cfg.plasticity:                      # brain/cortex.py (superseded revision, line ~175)
    out = self.W.T @ cn
    ...
    self.W += cfg.readout_lr * np.outer(cn, err)
```

I confirmed the consequence directly:

```
plasticity=False: W changed = False, readout_updates=0, |W|max=0.000000
plasticity=True : W changed = True,  readout_updates=12665, |W|max=5.599369
```

With `plasticity=False` the model is **frozen in both places**: no substrate
plasticity *and* no readout learning. `W` stays at exactly zero, so `argmax` returns
class 0 for every input forever. That is precisely the 50/0/0/0/0 matrix above.

`brain_noplast` therefore did not test "is anything being learned" (the question the
arm is documented to answer in `experiments/continual.py:8` and `README.md:169`). It
tested a network with *no learning anywhere*, which scores 10% by construction. The
inference drawn from it — "the effect is attributable to the substrate's plasticity
rather than the readout" — does not follow.

`brain/cortex.py:69-75` now documents this and separates the flags
(`readout_learn`), which is the right fix. Commits `e8542af` and `afaf81f` are
honest about it.

### C4-defect-2: the corrected control reverses the conclusion

Because the flag is now separated, the control the authors *intended* can finally be
run: frozen substrate (`plasticity=False`) with a **trained** readout
(`readout_learn=True`), everything else identical. I ran this through the repo's own
`run_arm` with the same protocol, using the configuration that reproduces the
documented numbers (`rule="binary"`, `reset=False`):

```
brain (plastic, readout learns)        rule=binary  plasticity=True : [0.448, 0.324] mean=38.60%
brain_noplast (frozen, readout learns) rule=binary  plasticity=False: [0.272, 0.384] mean=32.80%
```

And for contrast, with the current default rule:

```
brain (plastic, readout learns)        rule=delta   plasticity=True : [0.100, 0.100] mean=10.00%
brain_noplast (frozen, readout learns) rule=delta   plasticity=False: [0.100, 0.100] mean=10.00%
```

With a working readout rule, a **frozen** spiking substrate plus a trained linear
readout reaches **32.8%** against the plastic substrate's **38.6%**. About 85% of
the brain arm's accuracy is available without any substrate plasticity — the
substrate's contribution is roughly 6 points, not the ~28 points the null-control
argument implies.

This also answers the question the task set me: **is the advantage the readout or
the substrate?** The substrate changes the features, so a frozen-substrate/trained-readout
arm is the only way to separate them. That arm was specified in the design
(`docs/PARADIGM.md` §6 arm F, "is plasticity causal?") but `frozen` as implemented is
`FrozenFeaturesReadout` — a `tanh` of a random projection **of the raw pixels**
(`brain/baselines.py:573-635`), which never touches the substrate at all. It controls
for "random pixel features are useless", not for "the substrate's features are
sufficient". The control that isolates substrate plasticity was not in the run.

`docs/SUBSTRATE_CONTRIBUTION.md` (added during this review, after my probes) reaches
the same conclusion independently: ablation of the recurrence and plasticity moves
the number by less than seed noise, and a matched static random ReLU projection
*beats* the spiking substrate. That document is correct and it contradicts
`README.md:212` ("removing plasticity destroys learning, so the substrate's
plasticity is doing real work"). One of those two statements must go.

**C4 verdict: NOT SUPPORTED.** `shuffled` is a valid control; `frozen` is a valid
but off-target control; `brain_noplast` as stored was a void control (no parameter
changed anywhere), and as corrected it does not collapse to chance — the frozen
substrate retains most of the performance.

---

## C5 — is the dendrite ablation better, and is it a genuine negative result?

### What I could confirm: the ablation is really wired

The task asked me to check three things. All three pass:

```
use_dendrite=True  -> dend_mode=dcaap
use_dendrite=False -> dend_mode=linear
```

```
use_dendrite=True : v_dend max=1.7462 mean=0.8864 nonzero=0.980  code_sum=236  code_nonzero=236
use_dendrite=False: v_dend max=1.7462 mean=0.8864 nonzero=0.980  code_sum=285  code_nonzero=285
codes bit-identical: False
L1 difference: 359.0
```

`CortexConfig.use_dendrite` (`brain/cortex.py:68`) does set `dend_mode`
(`brain/cortex.py:110`), `v_dend` is non-zero in the intact case, and the two arms
produce different spike codes for the same input. **D5b#9 is genuinely fixed** —
this is no longer the "inert dendrite, bit-identical arms" situation. Good.

### What fails: the numbers

`docs/RESULTS.md:201-202` states:

> "`brain_nodend` (41.87 +/- 3.46) beats the intact `brain` (36.40 +/- 6.97) by
> **+5.47 points**, with disjoint CIs."

The difference is right; **"disjoint CIs" is false**:

```
brain          CI = [29.43, 43.37]
brain_nodend   CI = [38.41, 45.32]
overlap = True
```

`README.md:180` correctly says the *split* result is not a win over replay, but
neither document states the dendrite comparison is a non-significant difference.
The correct statement is "+5.47 points, intervals overlapping, not statistically
distinguishable at n=3". The claim that the dendrite is *harmful* is not supported;
what the data supports is that it is *not detectably helpful* on this benchmark.

`README.md:217-218` and `docs/RESULTS.md:286` quote 41.87/35.90 vs 36.40/22.10,
which do match the current JSON, so the README version of this claim is current even
though the RESULTS.md sentence containing "disjoint CIs" is not.

### What fails harder: the ablation is confounded with activity

The entire justification for having this arm (`docs/DECISIONS.md` D3) is Iyer et al.
2022: do not confound dendritic structure with activation sparsity. But the ablation
changes sparsity. Measured on the same 30 training images, same seed and config:

```
use_dendrite=True : mean active/step = 175.3   synops/sample = 10,240
use_dendrite=False: mean active/step = 225.3   synops/sample = 16,512
```

The ablated arm fires **29% more neurons** and pays **61% more SynOps**. So the
comparison "41.87 vs 36.40" is not a clean dendrite-vs-linear contrast: it varies
dendritic nonlinearity *and* active-set size *and* operation count at once. The
confound the arm was built to avoid is present in the arm itself. An
activity-matched comparison (e.g. comparing at equal mean active fraction, or
reporting each arm's sparsity beside its accuracy) would be needed before any
dendritic conclusion is drawn.

This is a design-level finding, not a coding error, and it cuts *against* the
authors' own careful negative-result framing — but it means the negative result is
less clean than reported.

**C5 verdict: NOT SUPPORTED as written.** The wiring claim is SUPPORTED and
reproduced (D5b#9 genuinely fixed). The "ablation is better, with disjoint CIs"
claim fails on the CI statement, the +7.87/+5.47 framing is stale between files, and
the ablation is confounded with activity level.

---

## C6 — the nine defects in D5/D5b

I reproduced four as historical bugs and re-verified six as fixed. Live results:

### D5#1 — lazy-evaluation artefact — **not independently reproduced**

I did not re-derive this one. What I verified is that the guard exists
(`bench/bench_scale.py:112` `force_eval`, called at lines 289, 326, 330, 351, 362)
and that `bench/results_scale.json` records
`lazy_eval_guard: {dispatch_only: 0.00298 ms/step, evalled: 0.18872 ms/step, ratio 63.3x}`,
plus `ms_per_step_above_plausible_floor` rejecting < 0.05 ms. That is a well-designed
guard, but it is a stored artifact, not something I reproduced. **UNVERIFIABLE by me.**
(My own measurement of the live path: 3.222 ms wall per ms of biological time at
800 neurons, with `force_eval` semantics via `be.to_numpy` reads — consistent with a
genuine per-step cost, not a dispatch artifact.)

### D5#2 — inverted LTP sign — **FIXED, reproduced**

```
numpy: coincident dw = +0.01000001  (LTP) ; post-before-pre dw = -0.01199999  (LTD)
```

Directions correct, magnitudes match `a_plus=0.010` / `a_minus=0.012`
(`brain/plasticity.py:51-52`) and the correct pre-update post-trace ordering is
enforced inside `apply` rather than left to callers (`brain/plasticity.py:128-160`).
Tests assert both directions for both eligibility modes
(`tests/test_plasticity.py:54-90`).

### D5#3 — silent spike truncation — **FIXED, reproduced**

```
numpy: capacity=5  max_spikes/step=256  total=768  overflow=753
mlx:   capacity=5  max_spikes/step=256  total=768  overflow=753
```

Overflow is counted and surfaced (`brain/simulator.py:135`, `total_overflow`), and
`tests/test_simulator.py:102-141` covers it. Note the overflow figure (753) is three
orders of magnitude larger than the buffer (5), so in this stress configuration the
simulation is almost entirely truncation — correctly reported rather than hidden.

### D5#4 — pathological synchrony — **PARTIALLY fixed / mitigated only**

`docs/DECISIONS.md:89-92` says the volley was removed by "per-step membrane noise
plus dispersed initial potentials". That is accurate as a description of the
*default* configuration, but not a fix. With the heterogeneity deliberately removed
(`init_noise=0.0`, `noise_std=0.0`), the volley is still there:

```
numpy: capacity=25 max_spikes=256 total=865 overflow=625
       first 25 steps: [0,0,0,0,0,0,0,0,0,0,0,0,0,256,0,0,0,0,0,0,0,0,0,0,0]
mlx:   capacity=25 max_spikes=256 total=859 overflow=651
```

All 256 identical neurons fire on the same step, overflowing a capacity-25 buffer.
The mitigation is real, the fix is not: the substrate cannot represent a
synchronous volley. Worth stating as "mitigated by required heterogeneity" rather
than "removed".

### D5b#5 — phantom drive — **FIXED, reproduced both ways**

This is the defect the task specifically asked about. Current code
(`brain/simulator.py:180`) gathers only real spikes.

**Current code, exactly the protocol SCALING.md describes:**

```
n=100,000 k=64:  spikes/step = 0.0
n=1,000,000 k=64: spikes/step = 0.0
```

Not approximately zero — exactly zero, on both backends, over 100 quiet steps with
`pending = 0.000000` and `x_pre[0] = 0.000000`:

```
numpy: 100 quiet steps -> total spikes=0, pending sum=0.000000, x_pre[0]=0.000000
mlx:   100 quiet steps -> total spikes=0, pending sum=0.000000, x_pre[0]=0.000000
```

**Historical reproduction.** I reverted the single line that constitutes the fix
(`spike_idx = spike_buf[:n_spk]` -> `spike_idx = spike_buf.astype(be.idx_dtype)`)
and re-ran the same protocol:

```
line present: True
BUGGY numpy: 100 quiet steps -> spikes=0,   pending=1266.537, x_pre[0]=119.290
BUGGY mlx:   100 quiet steps -> spikes=6,   pending=1437.374, x_pre[0]=117.831
```

The pathological signature the changelog describes is exactly reproduced: a quiet
network schedules >1200 units of phantom synaptic current and drives neuron 0's pre-trace
to ~118 in 100 steps, with no neuron having spiked. **D5b#5 is genuinely fixed and
the historical bug is real.** This is the strongest verification result in the review.

### D5b#6 — k-WTA ranked the wrong neurons — **PARTIALLY fixed** (see C2-defect-2)

Ranking source fixed (`v_pre_reset`, `brain/neurons.py:140`, used at
`brain/simulator.py:145`); tie handling still broken; test still failing with a stale
reason; bound needed on 804/900 steps of the real experiment.

### D5b#7 — `depressed` never incremented — **FIXED, reproduced**

```
{'potentiated': 364.0, 'depressed': 1583.0, 'weight_updates': 1947.0}
```

Both directions increment, and the comparison is now against the sign-corrected `dw`
(`brain/plasticity.py:196-204`).

### D5b#8 — SynOps overstated — **FIXED, reproduced**

```
spikes=10 k_out=8: synops_active=80  (= spikes*k_out)      ✓
                   synops_executed=48 (= capacity*k_out)   ✓
```

Both quantities are reported separately (`brain/simulator.py:207-213`) exactly as the
changelog claims. Honest caveat that belongs next to this: the *executed* count is
what the machine actually pays, and at the experiment scale it is larger than the
dense baseline's MAC count:

```
per sample (15 ms): real spikes=415
  synops_active   (spikes*k_out)          =     26,560
  synops_executed (actual gather rows)    =    230,400
  mlp dense MACs (n_params)               =    101,770
  active/mlp   = 0.26  -> brain 3.8x cheaper   (the reported number)
  executed/mlp = 2.26  -> brain 2.26x MORE expensive (the measured cost)
```

`docs/RESULTS.md` §"what this does NOT support" bullet 3 does disclose this
("the measured scaling in section 1.4 shows this implementation's cost is dominated
by the O(N) per-neuron state update and is largely insensitive to sparsity"). Good —
but the headline efficiency number in RESULTS.md:191 and README is the algorithmic
one, and the ratio quoted from the current revision is 11,157, not 10,667.

### D5b#9 — dendrite inert by construction — **FIXED, reproduced** (see C5)

### Summary of C6

**Six of nine defects are genuinely fixed and I reproduced the fix** (#2, #3, #5, #7,
#8, #9). **One is partially fixed** (#6). **One is mitigated rather than fixed** (#4).
**One I could not independently reproduce** (#1, though its guard is present and
well-designed).

So the changelog's flat claim that all nine are "fixed" is **overstated for two of
them** (#4 and #6), and one of those (#6) is documented in the codebase as fixed while
the test suite still demonstrates the failure — which is the most misleading of the
nine.

---

## Defects and discrepancies the authors did not list

### N1. `Brain.run(record_spikes=True)` crashes (real bug, public API)

```
run(record_spikes=True) RAISED NameError: name 'mask' is not defined
```

`brain/simulator.py:240` references `mask`, which is a local of `Brain.step` and is
not in scope in `run`. The `record_spikes` path has never been executed. It is
documented in the `run` docstring as recording spike history and it is dead. No test
covers `record_spikes` (grep over `tests/` returns nothing). This is a small bug but
it is exactly the class of defect the changelog at D5#1 claims to have eliminated
by "a deliberate check rather than by luck" — this one was never checked.

### N2. `docs/SCALING.md` describes the phantom-drive defect as UNFIXED, in a file that

also asserts its measurements are contaminated by it. `docs/SCALING.md:144-165`:

> "The zero-input control ... **still emits** spikes: [table: 32.0 / 32.0 / 511.5 /
> 128.0 spikes/step] ... **Consequence: activity in this substrate sits on a floor of
> order `k_out/2` spikes per step**, so spike-rate and event-count figures above are
> contaminated by that floor. This is a defect report, not a fix — `brain/` is
> outside this task's ownership."

`docs/DECISIONS.md` D5b#5 says the same defect was fixed and "the zero-input control
is now exactly silent". I ran SCALING.md's exact protocol on current code and got
**0.0 spikes/step** for both configurations in its table. So SCALING.md's defect
section is stale, describing a state that no longer exists — and it simultaneously
disclaims the throughput numbers in its own §2 as contaminated. The two documents
cannot both describe the same revision.

### N3. Documented scale numbers come from a different revision than the current code

`bench/results_scale.json` records the hashes of the substrate it measured:

```
revision_after_sweep: {"simulator.py": "f8daecd215c8a5d8", "cortex.py": "70d5a10b16b01bef",
                       "neurons.py": "4aa58f52ad3a828f", "plasticity.py": "0f44e14e4c798ac1", ...}
```

Current code (same hash function, `sha256[:16]`):

```
simulator.py: 7c7d28c7c016dbc3      (was f8daecd215c8a5d8)
cortex.py:    2c9cfcf34434cbe5      (was 70d5a10b16b01bef)
neurons.py:   be8fd0bbc5e50c798     (was 4aa58f52ad3a828f)
plasticity.py:8ef01efc9434665d     (was 0f44e14e4c798ac1)
```

Every substrate file has changed since the scale sweep. The JSON itself notes
"brain/ is under active concurrent development". The scale table in
`README.md:132-139` and `docs/RESULTS.md:66-73` is therefore attributable to a
superseded revision. The 12 bytes/synapse storage figure is a property of the
`connectivity.py` layout (int32 targets + fp32 weights + int32 delays), which I did
not see change, so that specific number is probably still valid — but the *timings*
are not attributable to current code.

### N4. Three different scale tables from two different JSON files, never reconciled

- `README.md:132-139` — 1e5/256: 1.31 ms; 1e6/256: 8.38 ms; 4e6/256: 40.24 ms.
  These are from `bench/lead_independent_scale.json` (1.3054, 8.3773, 40.2398).
- `docs/RESULTS.md:66-73` — 1e6/256: 6.19 ms; 1e6/1024: 19.81; 4e6/256: 29.88.
  These are from `bench/results_scale.json` (6.187, 19.805, 29.877).
- `docs/SCALING.md` — a third presentation, same `results_scale.json`.

README's "MLX is ~28x faster than NumPy" and `docs/DECISIONS.md:34`'s "MLX ~28x
faster" are not in either JSON as a documented field I could locate; the
`lazy_eval_guard` ratio (63.3x) is a different quantity. Three documents, two JSON
files, no cross-references; a reader cannot tell which run the README table came
from. This is a reproducibility problem for claim (g)/(h) in the parent task.

### N5. `xfail_strict` is not configured anywhere

Covered under C2-defect-1. Stated separately here because it is a one-line fix with
large consequences: without `xfail_strict = true`, every xfail in this repo is a
silent regression hole rather than a test.

### N6. New modules are untested

`brain/readout.py`, `brain/hierarchy.py`, `brain/afferents.py` — zero references in
`tests/`. `docs/SUBSTRATE_CONTRIBUTION.md` and `experiments/readout_diagnostic.py`
(the current diagnosis of *why* the paradigm fails) rest on these modules. The
diagnosis is the most valuable negative result in the repo and it is the least
tested code.

### N7. Claim/artifact mismatches still live at the pinned revision

| file:line | claims | measured |
|---|---|---|
| `docs/RESULTS.md:202` | "disjoint CIs" for dendrite ablation | intervals overlap |
| `docs/RESULTS.md:191` | "~10,667 active SynOps/sample" | 11,157 |
| `docs/RESULTS.md:195-196` | diag 63.3%, forgetting 76.7% / 26.3% | 62.13%, 77.33% / 25.73% |
| `docs/RESULTS.md:147` | shuffled final row 0.14/0.10/0.11/0.17/0.10 | 0.067/0.173/0.133/0.153/0.120 |
| `docs/SCALING.md:147-165` | phantom drive unfixed, 32.0 spikes/step | fixed, 0.0 spikes/step |
| `README.md:212` | "removing plasticity destroys learning" | frozen substrate + trained readout = 32.8% vs 38.6% |
| `docs/SCALING.md:8` etc. | scale numbers | substrate hashes have all changed since |

### N8. `docs/RESULTS.md:84` documents a command with `--seeds 1` for a 3-seed table

Cosmetic (a line-continuation slip) but it means the published reproduction command
would produce a 1-seed run if pasted, with different CIs.

---

## Is the physics/biophysics sound?

Independent checks on the current revision, beyond what each claim asked:

- **LTP/LTD direction**: correct (above).
- **Dale's principle**: enforced by construction (`brain/connectivity.py:74-83`, sign
  per pre-neuron) and sign-corrected in the weight write (`brain/plasticity.py:175-183`).
  The sign correction is a real correctness requirement and it is present.
- **Determinism**: reproduced. Same seed twice, both backends:

```
numpy: spike traj identical=True  weight sum identical=True (-28.31713867)  x_pre identical=True
mlx:   spike traj identical=True  weight sum identical=True (-34.48443604)  x_pre identical=True
```

  (numpy and mlx differ from each other, which is expected — different RNG streams —
  but each is internally bit-identical, satisfying "deterministic given a seed".)
- **Spike compaction**: verified exact against `np.flatnonzero` including the
  all-spike and overflow cases (`tests/test_simulator.py:38-99`), and the compaction
  is genuinely order-preserving.
- **Membrane/dendritic kinetics**: `phi(z) = z*exp(1-z)` peaks at `z=1` with value 1
  and decays; I verified this numerically across `[0, 0.5, 1, 1.5, 2, 3]` and it
  matches the documented table. Refractory behaviour is enforced
  (`brain/neurons.py:145-155`) and I confirmed it blocks spiking within the window.

The biophysics is not where the problems are. The problems are in the experimental
control design and in the drift between documents and artifacts.

---

## Answering the task's specific "also look for" questions

**Stale numbers between README / RESULTS / JSON?** Yes — six live mismatches (N7).
At session start the chasm was total (docs said 37.07/44.93/9.33/12.40; JSON said
36.40/41.87/10.00/12.93); the authors then updated most of §2 while leaving §1.3
adjacent text and the caveats bullets behind.

**Claims in RESULTS not supported by the JSON?** Yes: "disjoint CIs" (line 202),
the SynOps figure (line 191), the just-trained/forgetting figures (lines 195-196),
and the shuffled final row (line 147).

**Code paths that would silently produce wrong numbers?** Yes, three:
(i) `readout_rule="delta"` diverges to `nan` — silently, since warnings are the only
signal and a caller in a library context may not see them — and it is the default;
(ii) `brain_noplast`'s readout gate made a null control void while still producing a
plausible-looking 10%;
(iii) `Brain.run(record_spikes=True)` raises rather than silently mis-recording, so
that one at least fails loudly.

**Is the `shuffled` control still not a valid control?** It **is** valid now. Labels
are randomised over the full 10-class space (`experiments/continual.py:116-140`),
which guarantees unlearnability, and the resulting matrix is spread across all five
tasks rather than degenerate. The earlier permutation-based version was the flawed
one, and the fix is real. Minor note: 12.93% is ~2.1 standard errors above the 10%
chance rate for 750 test samples, which is marginal, not clearly "at chance" — but
with 5 tasks × 2 classes the effective chance rate is not exactly 10% per task and I
would not call this a defect without a proper per-task null model.

**Could the brain arm's advantage be the readout rather than the substrate?** This is
the right question and the answer is **yes, largely**. See C4-defect-2: a frozen
substrate with only the readout trained reaches 32.8% where the plastic substrate
reaches 38.6%. To separate "better features" from "more trainable readout" you must
hold the readout rule and capacity fixed and vary only substrate plasticity — which
is exactly the arm I ran. The repo's `frozen` arm does not do this (it bypasses the
substrate entirely, operating on `tanh` of raw pixels). `docs/SUBSTRATE_CONTRIBUTION.md`
was added during this review and reaches the same conclusion.

---

## What would make me revise these verdicts

- **C3**: pass `readout_rule="binary"`, `reset_between_samples=False` explicitly in
  `experiments/continual.py` (or fix `delta` to be bounded), re-run, and re-pin the
  docs to the JSON by hash. I would then call C3 SUPPORTED outright.
- **C4**: report the frozen-substrate/trained-readout arm alongside the others, with
  activity matched, and either delete the `brain_noplast`-based inference or restate
  it as "a network with no learning at all cannot learn" (which is trivially true).
- **C5**: report the dendrite comparison with its activity/SynOps difference and
  without the word "disjoint".
- **C2/C6**: remove the xfail markers that now pass, turn the padding test into a
  hard assertion, add `xfail_strict = true`, and either fix the k-WTA tie handling or
  change the marker reason to describe the tie failure.

## Bottom line

The mechanical, biological and numerical core of this project is in good shape: XOR
from a tuned dendrite holds and the impossibility argument against the linear
ablation survives an adversarial sweep; LTP/LTD directions, Dale compliance,
determinism, compaction, overflow accounting and the phantom-drive fix all verify
under re-execution; and the phantom-drive bug is genuinely, exactly gone. Six of the
nine documented defects are genuinely fixed and I reproduced both the bug and the fix.

The experimental claims are the weak part. The continual-learning table is
arithmetically correct but was produced by a superseded revision and is **not
reproducible from the current default configuration**, which silently collapses to
10%. The central control argument fails: `brain_noplast` as stored learned nothing
anywhere, and the corrected version of that control shows a frozen substrate
retaining most of the accuracy. The dendrite ablation's "disjoint CIs" claim is
false and the ablation is confounded with activity. And `docs/SCALING.md` still
asserts a correctness-critical defect is unfixed that `docs/DECISIONS.md` says is
fixed, while disclaiming its own measurements as contaminated by it.

The most valuable thing in the repo is the negative result — including the newly
added `docs/SUBSTRATE_CONTRIBUTION.md`, which is honest, well-controlled, and
directly contradicts a claim still standing in `README.md`. That direction is right.
The remaining work is to let the negative result propagate to the parts of the
documentation that have not caught up, and to stop the docs and the JSON from
drifting apart.

---

## Reproduction appendix

All probes are in `/tmp/rev/`; nothing outside `docs/` was modified in the repo.

| probe | what it does |
|---|---|
| `t_phantom.py` | phantom drive, current code, both backends, 100 quiet steps |
| `t_defects.py` / `t_probe2.py` | k-WTA ties vs distinct, SynOps identity, overflow, synchrony |
| `t_ltp.py` | LTP/LTD direction, both eligibility modes |
| `t_xor_edge.py` / `t_xor_linear.py` | XOR at varied `tau_soma`/`n_ms`; full threshold sweep for `linear` |
| `t_which_config.py` | bisect which flags reproduce the documented 36.40/0.416/0.296 |
| `t_divergence.py` | `delta` vs `binary` weight-magnitude trajectories |
| `t_nulls.py` | frozen substrate + trained readout vs plastic substrate |
| `t_iso2.py` | repo's own `run_arm` with a swapped arm (apples-to-apples) |
| `t_batch.py` | determinism, dendrite wiring, activity matching |
| `cur2/` | pinned copy of `brain/ experiments/ data/ tests/` at the reviewed revision |

Key commands whose output is quoted above:

```bash
python3 experiments/xor_neuron.py
python3 -m pytest tests/ -q
python3 -m pytest tests/ -q --runxfail
python3 -m pytest tests/ -q -rX
python3 experiments/continual.py --tasks 5 --seeds 3 --epochs 1 --neurons 800 \
        --train-per-task 80 --test-per-task 50 --lr 1e-2
```

Time note: the full 5-task/7-arm protocol takes ~4-6 minutes per seed-dependent run
on this machine; the cortex path costs ~3.2 ms wall per ms of simulated biological
time at 800 neurons.
