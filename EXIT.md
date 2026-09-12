# EXIT.md — established, retracted, and still open

A truthful snapshot for the next researcher. Every number below is copied from a
committed artifact in this repo (JSON, a run command, or a doc carrying its own
measurement). Nothing is extrapolated. Where a claim is retracted, it says so.

Short version: the mechanism-level work is real, narrow and reproducible. The
paradigm claim was never tested where it matters (§4), and the headline performance
claims were falsified or retracted (§2). Read §4 before quoting §1.

---

## 1. What is established

**1.1 A single neuron computes XOR via a non-monotonic dendritic nonlinearity; the
linear ablation provably cannot.** Source: `experiments/xor_neuron.py` (run here;
matches `docs/REVIEW.md` C1).

| input magnitude | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| `dcaap` drive | 0.0 | 0.8244 | 1.0 | 0.9098 | 0.7358 | 0.406 |
| `linear` drive | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |

`dcaap` emits `[0,1,1,0]` for `(0,0),(0,1),(1,0),(1,1)` — exactly XOR — at
threshold 0.8679. A linear drive is monotonic, so any threshold firing on `(1,0)`
also fires on `(1,1)`; an adversarial sweep of thresholds 0.05–3.00 in 0.05 steps
with the real spiking neuron found **none** producing `[0,1,1,0]`. Caveats: the
margin is thin (0.8679 vs 0.7358), the pattern breaks at `tau_soma=3.0`
(`[0,1,1,1]`) and at `n_ms<=20` (all zeros), and the threshold was chosen on the
same table it is tested against. This shows a separating threshold *exists*; it is
not a learning result.

**1.2 Working memory across a silent delay is a dendritic trace.** Source:
`docs/WORKING_MEMORY.md`. Sample 40 ms → silent delay D → readout window 40 ms with
input identically zero; 6-way, chance 0.167. `dend_scale` was recalibrated per arm
so every one sits on the tuning peak (`z=1`), removing the drive-strength confound:

| `tau_dend` | calibrated `dend_scale` | d=0 | 50 ms | 100 ms | 300 ms | 600 ms |
|---|---|---|---|---|---|---|
| 10 ms | 5.91 | 1.000 | 0.160 | 0.260 | 0.180 | 0.100 |
| 50 ms | 3.33 | 1.000 | **1.000** | 0.260 | 0.180 | 0.100 |
| 200 ms | 1.09 | 1.000 | **1.000** | **1.000** | 0.220 | 0.100 |

Zeroing the entire recurrent weight matrix leaves accuracy **bit-identical at every
delay**. `tau_adapt` 100→3000 ms did nothing; `tau_soma` 20→2000 ms was *worse*
(delay-0 collapsed 1.000→0.170). The window tracks `tau_dend` then saturates: no arm
reaches 300 ms, and 600 ms is at or below chance. That ceiling is the honest limit —
a passive decaying trace is not a trainable store.

**1.3 Plasticity directions are correct and Dale-compliant.** Coincident pre+post →
LTP `+0.010`; post-before-pre → LTD `-0.0114` (`docs/DECISIONS.md` D5#2), reproduced
as `+0.01000001` / `-0.01199999` (`docs/REVIEW.md` C6). Inhibitory synapses flip
sign as required (`+0.11`/`-0.11`). Dale violations: **0**.

**1.4 Determinism given a seed.** `bench/results_scale.json`:
`spike_trajectories_identical: true` at 20,000 neurons / 25 steps / seed 0. Only the
discrete train is exact; `max_abs_diff_v_soma = 1.19e-07`, because MLX
`scatter_add` accumulates with atomics.

**1.5 Measured local scale with sparse activity.** Two committed runs, which
disagree in detail (§3.19):

| source | config | ms/step | RAM | active | events/s |
|---|---|---|---|---|---|
| `bench/lead_independent_scale.json` | 4.0e6 neurons / 1.024e9 syn | 40.24 | 11.58 GiB | 3.85% | 9.80e8 |
| `bench/lead_independent_scale.json` | 1.0e6 neurons / 1.024e9 syn | 28.42 | 11.48 GiB | 5.61% | 2.02e9 |
| `bench/results_scale.json` (headline) | 1.0e6 neurons / 1.024e9 syn | 19.81 | 12.43 GB | 3.98% | 2.056e9 |

Storage is exactly **12 bytes/synapse**. This is the ≥1M-neuron sparse-activity
demonstration; the ≥5 GB budget is demonstrated at 11.58 GiB. Caveat that kills the
energy reading: cost is **not** sparsity-driven — an undriven network fires ~1100x
less and still costs ~80% as much per step, because the O(N) per-neuron state update
dominates.

**1.6 Tests currently pass.** `python3 -m pytest tests/ -q` → **118 passed in
141.58 s** today. (`README.md:208` says 114; `docs/REVIEW.md` saw 114 passed /
4 xfailed at an earlier revision.)

**1.7 Gap to the human brain.** Neurons ~86.1e9 vs 4.0e6 measured → **~1e4.33x**.
Synapses ~1e14–1e15 vs 1.024e9 → **~1e5.0–6.0x**. Storage ~1 PB vs 11.6 GiB →
**~1e5x**. Wall clock by linear extrapolation → **~1e6.3–7.3x**. A full replica is out
of reach by ~5 orders of magnitude and will not be closed by tuning.

---

## 2. What is falsified or retracted

Every item below was, at some point, a claim of this project.

1. **Permuted-MNIST continual learning — FALSIFIED.** Substrate **22.10%** vs
   parameter-matched dense MLP **68.30%**, a **−46.20 point** loss, combined CI 9.80
   (`experiments/results/continual_permuted.json`; `docs/RESULTS.md` §3).
2. **"The substrate beats backprop" on Split-MNIST — RETRACTED as a claim.** The
   +17.73 point win (36.40% vs 18.67%) is against a baseline retaining **0.00** on
   every task except the newest; it measures the baseline's pathology, not the
   substrate's strength.
3. **"The substrate beats replay" — RETRACTED.** `brain` 36.40 ± 6.97% CI
   [29.43, 43.37] vs `mlp_replay` 30.93 ± 1.83% CI [29.10, 32.76]; intervals
   **overlap**. The earlier +8.67 point claim came from a run predating the
   stratified-subsampling fix and does not survive.
4. **The LM memory arm beat a transformer — RETRACTED.** The comparison stopped at
   500 steps while the attention baseline was still at the bigram floor. Floor =
   **3.5806 bpc**; the attention baseline scored **3.6664 bpc**, i.e. **0.0858 bpc
   worse than a bigram**, which cannot serve as a baseline for anything. Trained
   properly, attention reaches **2.2354 bpc at 2000 steps**, better than the gated
   arm's 2.7086 at 500 steps. The "multiscale" arm was also `banks=1` (808,448
   params), not 4 banks (1,399,808), so it was never parameter-matched
   (`docs/EFFICIENCY.md` §1–4). **No claim that anything here beats a transformer
   survives, and none is made in this document.**
5. **The afferents hypothesis (convergence) — CAUSE CORRECTED, effect real.** The
   projection fix is large: identity **36.17%** → best mixed **71.67%**. But the
   stated cause was convergence (**+1.83 pts**); the real driver was receptive-field
   **polarity at +27.83 pts**, with centering at −2.83. A no-convergence projection
   (one pixel per neuron, signed) scores **64.00%**
   (`experiments/results/afferents.json`, `docs/AFFERENTS.md` §3).
6. **The projection does not rescue continual learning — CONFIRMED NEGATIVE.** Best
   projection arm 22.87% vs MLP 68.13% on Permuted-MNIST, still **45.27 points
   behind** (`afferents.json` `verdict.q3_consequence`).
7. **The quadratic-readout anomaly did not reproduce — RETRACTED as the diagnosis.**
   Mean linear→quadratic gain: **+0.0050** (identity) and **+0.0004** (mixed), i.e.
   none (`afferents.json` `quadratic_check`). The earlier "quadratic doubles linear"
   story (0.333 vs 0.157) does not survive this protocol.
8. **"Substrate self-organisation contributes" — RETRACTED.** Identical cached codes:
   FULL **0.702 ± 0.023**; recurrence zeroed **0.715**; plasticity off **0.715**; both
   off **0.710** — ablation moves the number *up*, inside seed noise. A static random
   ReLU projection at matched sparsity (29%) scores **0.765 ± 0.045** and a dense one
   **0.803 ± 0.033**; ridge on the raw 784 pixels scores **0.670**. A training-free
   control beats the spiking substrate (`docs/SUBSTRATE_CONTRIBUTION.md`).
9. **The input-wiring effect was overstated ~3x.** A single-seed comparison with
   inconsistent regularisation and mismatched standardisation reported **+29.5
   points**; the controlled effect is **+6 to +10 points** (0.396→0.456 standardised,
   0.466→0.570 unstandardised).
10. **"Removing plasticity destroys learning" — corrected.** The knock-out arm disabled
    the readout too (§3.5). The corrected control — frozen substrate, trained readout —
    reaches **32.8%** against the plastic substrate's **38.6%**, so ~**85%** of the
    brain arm's accuracy needs no substrate plasticity at all (`docs/REVIEW.md`
    C4-defect-2).
11. **"The dendritic nonlinearity hurts" — NOT SUPPORTED as written.** The split
    difference is +5.47 points with **overlapping CIs** ([29.43, 43.37] vs
    [38.41, 45.32]). The ablation is also confounded with activity: it fires **29%
    more neurons** (225.3 vs 175.3 active/step) and pays **61% more SynOps** (16,512
    vs 10,240) — the exact confound the arm was built to avoid (`docs/REVIEW.md` C5).
12. **"Predictive coding is local, therefore new" — rejected before it was claimed**
    (Whittington & Bogacz 2017). Recorded so it is not re-proposed.

---

## 3. Bugs in our own code that corrupted published numbers

At least eight; several changed a headline. Sources: `docs/DECISIONS.md` D5/D5b,
`docs/REVIEW.md` C2–C6, `docs/READOUT.md`.

1. **Phantom drive — contaminated every earlier result.** `_compact` padded unused buffer slots with index 0 and `step` gathered the whole padded buffer, so padding re-delivered neuron 0's real targets and weights. A network with **no input** emitted **28.4 spikes/step**; reverting the one-line fix reproduces it (100 quiet steps → `pending=1266.537`, `x_pre[0]=119.290`). Corrected: exactly **0** spikes over 100 quiet steps on both backends. Every pre-fix result sat on a floor of ~`k_out/2` spikes/step.
2. **Binarised readout error.** `err = target − 1[out > 0]` saturates at zero for a class already scoring positive — no margin, learning stops. Worse, the **published** split/permuted numbers used this rule *and* `reset_between_samples=False`, neither the current default, so the stored JSON is not reproducible from committed defaults. Corrected on identical code: `lin_perceptron` 37.33 / `lin_softmax` 46.40 / `lin_ridge` 62.80 (split); 25.20 / 25.67 / 18.33 (permuted).
3. **Graded rule diverged to `nan` — and it was the default.** `|W|max` grew 1.73e-1 → 1.16e0 → 3.58e1 → 1.23e6 → 2.31e13 → **nan** within one task; the full 5-task/3-seed protocol under the default gives `brain 10.00 ± 0.00`. An unbounded standardised code fed an unbounded update; the raw rule needs `lr < 2/||code||² ≈ 4e-5` against a default of 0.05. Fixed with NLMS normalisation plus a code clip.
4. **NumPy buffered fancy-index trap.** `xty[:, y[idx]] += f.T` keeps only the **last write per class column** with repeated labels, so each class column held one sample's features. This wrecked **every** `frozen` baseline — it scored *below chance* (8.5–14%) and was written up as "the substrate doubles a random projection". Fixed with `np.add.at` (`brain/baselines.py:711–717`).
5. **`brain_noplast` disabled the readout — the control tested nothing.** `plasticity=False` also gated the readout update: `|W|max = 0.000000`, `readout_updates = 0`, so `argmax` returned class 0 forever and the arm scored 10% by construction (matrix 0.50/0/0/0/0). Corrected by separating `readout_learn`; the corrected control reverses the conclusion (§2.10).
6. **k-WTA admitted 20 winners against `k=8`.** `score >= cut` keeps every neuron tied with the k-th value; measured `kept=20` for `k=8` on both backends, so it did not bound sparsity. In the real experiment raw candidates exceeded `k_wta` on **804 of 900 steps** (mean 77.8, p95 96.0, max 113 vs 64), so the bound was load-bearing and held only because potentials happened to be distinct. Fixed by rank-based selection (`brain/simulator.py:_kwta_mask`).
7. **The zero-spike operating-point trap.** At `dend_scale=1.0` with drive 6.0, `phi(6.0)=0.040` attenuates input 25x and the population emits **0 spikes**; at `dend_scale=6.0` the same input emits **173**. Failure is total silence with no error, read downstream as "the model is bad". A calibrated retest emitted **13,680 spikes in both arms** and scored 0.150 against chance 0.167. `tau_dend=200 ms` at `dend_scale=6.0` also emitted 0 spikes.
8. **SynOps overstated up to 10x.** `step` charged `capacity * k_out` (the *buffer*) rather than `spikes * k_out`. Now `synops_active` and `synops_executed` are reported separately: 415 spikes → 26,560 active / 230,400 executed vs the MLP's 101,770 dense MACs, i.e. **0.26x** algorithmically but **2.26x MORE expensive** as executed. The ratio at `RESULTS.md:191` (10,667) is stale against the table's 11,157.
9. **The dendrite was inert by construction.** `_code` injected the image into the soma, so `v_dend ≡ 0`, `phi(0)=0`, and intact and ablated arms produced **bit-identical** results (both 41.47%). The honest reading was "the ablation was a no-op", not "dendrites do not matter".
10. **Lazy-evaluation artefact.** An early benchmark reported **0.02 ms/step** at 1e6 neurons because nothing was forced with `mx.eval`. The guard now measures dispatch-only vs evaluated at **63.3x** (0.00298 vs 0.18872 ms/step); the real figure is ~1.5 ms/step.
11. **Silent spike truncation** dropped spikes beyond capacity without reporting it (`capacity=5`, `total=768`, `overflow=753`). Now counted and surfaced.
12. **Inverted LTP sign**: coincident pre+post tripped the LTP *and* LTD terms, so with `a_minus > a_plus` correlated pairs **depressed**; fixed by evaluating against the pre-update post-synaptic trace. Separately, **`depressed` was never incremented**, so it read 0 forever; now `potentiated 364 / depressed 1583`.
13. **Pathological synchrony, mitigated not fixed.** With heterogeneity removed, 256 identical neurons still fire one volley overflowing a capacity-25 buffer (`overflow=625`). The substrate cannot represent a synchronous volley.
14. **Public API crash on an untested path.** `Brain.run(record_spikes=True)` raises `NameError: name 'mask' is not defined`; documented, dead, no test.
15. **Result-file overwrite destroyed data.** A reduced `--arms` run silently overwrote the full results JSON the docs depended on; partial runs now write distinct filenames.
16. **Head-slice subsampling skewed labels** (5–16 samples per class in an 80-row permuted slice; now stratified). This fix alone changed the replay comparison and killed the "beats replay" claim.
17. **Green suite guarded nothing.** Reverting the phantom-drive fix left the suite at **114 passed, exit code 0**, because the regression test was an `xfail` and `xfail_strict` was never configured. Both xfails are now plain tests.
18. **Still-open reproducibility defects.** `readout_diagnostic.py` could not complete a fresh run (`NameError: cap_rows`, since fixed); 132 of 492 control rows cannot be produced by the committed arm-running path; the documented oracle arms were never run; `meta.wall_seconds_total = 0.0437 s` is a re-analysis time, not a run time.
19. **Three unreconciled scale tables and stale substrate hashes.** `README.md`, `docs/RESULTS.md` and `docs/SCALING.md` present three different tables from two JSON files; every substrate hash changed after the scale sweep (`simulator.py` `f8daecd215c8a5d8` → `7c7d28c7c016dbc3`), and `SCALING.md` still calls the phantom-drive defect unfixed while disclaiming its own numbers as contaminated.

## 4. The honest status of the central hypothesis

This is the most important section in the file.

The repo's paradigm claim is **credit assignment via a broadcast scalar with no
backward pass** (`docs/PARADIGM.md` §4). Precisely:

- **It was never applied to a hidden layer.** The afferent projection runs at a
  **constant `M = 1`** — pure unsupervised Hebbian self-organisation. The code says
  so: `brain/cortex.py:8` ("three-factor plasticity enabled and a constant
  neuromodulator (M = 1), i.e. pure unsupervised Hebbian learning"), repeated for
  layer 1 of the hierarchy at `brain/hierarchy.py:26`. A constant third factor is a
  multiplicative constant, not an error signal; it assigns no credit.
- **The only error-driven layer was the readout: a single linear layer trained by the
  delta rule.** `brain/readout.py:49–53` records that this is LMS/Widrow-Hoff, i.e.
  **exact gradient descent on that layer**. No hidden layer in the measured results
  sat under error modulation.
- **So the model under test was a frozen projection + a fixed nonlinearity + one
  linear layer.** The Permuted-MNIST falsification (§2.1) and the split result
  measured *that*, not three-factor credit assignment through a network. Where a
  hidden layer did exist, ablating the substrate moved the probe 0.702 → 0.710 —
  inside seed noise — and a static random ReLU projection at matched sparsity scored
  0.765 (§2.8).
- **Consequence.** "Three-factor local plasticity beats backprop on continual
  learning" was never the experiment that was run. What was run is a linear readout on
  a fixed near-random projection, and that comparison loses to backprop by 46.20
  points on the harder benchmark, ties replay on the easier one, and is beaten by a
  training-free control on at least one benchmark (`docs/READOUT.md` §3: split
  `code_gap` **−14.67**, control 80.00 vs best local 65.33).

**On `experiments/results/threefactor_hidden.json`: that file does not exist at the
time of writing.** The experiment that would test the paradigm where it matters — an
error-driven, non-constant third factor applied to a hidden layer's local plasticity —
is **running**. Its result is unknown and this document does not anticipate it. When
it lands, rewrite §4 from its verdict rather than around it. If it fails, the paradigm
claim is dead as stated; if it succeeds, §4 becomes the first section of a new
document, not a footnote to this one.

---

## 5. What is genuinely untested

The single most important open question, as a falsifiable experiment.

**Question:** does a *non-constant* broadcast scalar applied to a **hidden layer's**
local plasticity produce a capability that neither a frozen random projection nor
parameter-matched dense backprop can match?

**Experiment.** Three tasks that provably defeat a linear decoder on the substrate's
own code — an XOR-class parity task, and the §1.2 delayed 6-way task with distractors
added (its own named next test) — plus one continual stream. Arms, all
parameter-matched, reporting `synops_active` and `synops_executed` separately:

- **A** — hidden layer plastic with **`M = 1`** (what every result here measured);
- **B** — hidden layer plastic with **`M` = the readout's error**, the paradigm as
  advertised;
- **C** — frozen random hidden layer + trained linear readout, matched width and
  sparsity;
- **D** — dense backprop MLP of the same parameter count.

**What falsifies the paradigm:** B fails to exceed **max(A, C)** by more than seed
noise on a task where C is at chance. That is the direct test of whether the broadcast
scalar carries credit into the hidden layer at all. The instant C matches B, the
mechanism collapses into "a random projection with a linear head" — the outcome §2.8
already found on a static task.

**Cheaper, and worth doing first:** re-run the linear-vs-quadratic readout comparison
on *identical cached codes* with a **local** rule, not a closed-form ridge solve. That
was the repo's only actionable rescue hypothesis and it did not survive re-measurement
(§2.7: +0.0050 / +0.0004). Until a local rule reproduces a real gap, "the information
is in the code but not linearly decodable" should be treated as unestablished.

---

## 6. Four habits, each paid for by a bug above

1. **Check the floor before comparing.** A bigram table is five lines of NumPy and would have caught the transformer retraction in the first minute (§2.4). Print unigram and bigram floors beside every sequence arm; label any arm at or above the floor as uninformative.
2. **A control that changes nothing is not a control.** Verify the intervention moved a parameter (`|W|max`, `readout_updates`, spike counts) before inferring from it. `brain_noplast` (§3.5) and the inert dendrite (§3.9) both failed this.
3. **Confirm the neuron is firing before blaming the model.** Zero spikes is an operating-point fault, not a learning result (§3.7). Assert non-zero activity in every arm and fail loudly.
4. **A green suite is not a guard.** Without `xfail_strict`, an xfail silently converts a regression into a pass (§3.17). Re-introduce the bug deliberately and confirm the suite goes red — the only proof a regression test works.
