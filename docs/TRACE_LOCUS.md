# Where does the dendritic trace live? Per-neuron, not per-synapse

**Verdict: `PER-NEURON`.**

In this substrate the dendritic working-memory trace is a **single scalar per
neuron**. It is a leaky integral of the neuron's own drive history, and it is
mathematically the diagonal state-space recurrence that S4 / Mamba / RWKV
already implement. It does **not** carry presynaptic-input identity, so it
cannot implement addressed recall.

The result is stronger than a failed comparison. Under the discriminating
manipulation the substrate's accuracy is **exactly 0.500 at every sweep point and
every seed — 12/12 points** — and that exactness is an algebraic consequence of
the design, not a statistical near-miss (§2.1). A trace that could see *which*
afferent fired would break that tie; this one provably cannot.

```
python3 experiments/trace_locus.py
python3 experiments/trace_locus.py --seeds 5 --gains 4 --cross-backend
```

Writes `experiments/results/trace_locus.json`. Runtime 121 s (3 seeds × 4 gains,
NumPy, M4 Max).

---

## 1. The question, and why it matters beyond this repo

`docs/WORKING_MEMORY.md` established that working memory here is a dendritic
trace, not a recurrent weight: zeroing the entire recurrent weight matrix left
accuracy bit-identical. That ablation is silent on **locus**, and the two
candidates are not equivalent:

| hypothesis | trace is keyed by | effective gain depends on |
|---|---|---|
| **per-neuron** | the neuron | *how much* input arrived recently |
| **per-synapse** | the neuron **and the input line** | *which specific* partner was recently active |

The consequence is architectural:

* A per-neuron trace **is** a diagonal state-space model. Its computation is
  already covered by S4 / Mamba / RWKV, so citing biology adds nothing.
* A per-synapse trace is **addressed recall** — addressable by input pathway —
  which is a materially different computation, and is the biological
  justification proposed for a same-token-chain gated scan.

So this experiment decides whether the biology is load-bearing for that
architecture or merely decorative.

## 2. The discriminator

Input is split into two disjoint groups of `M = 64` afferent lines. Each trial:

```
[ SAMPLE 20 ms ]  ->  [ SILENT DELAY 20 ms ]  ->  [ PROBE 20 ms ]
```

The sample drives `K = 16` lines in group A (class 0) or **the same 16 lines** in
group B (class 1). The probe always drives those lines in group A. The class is
never present in the probe input, so the answer must come from state left by the
sample.

Two arms, differing **only** in the frozen projection `(128, 256)`:

* **`aliased`** — group B rows are a **bit-identical tile** of group A. Because
  both classes use the same 16 line indices, the two classes' *entire drive
  sequences are bit-identical*. A per-neuron trace is a function of the drive
  sequence, so it is **provably constant across classes**. A per-synapse trace
  keys on the line index, which does differ, so it can separate them.
* **`structured`** — group B is an independent draw with matched statistics.
  Both loci can solve this; it is the positive control that the task, operating
  point and readout are decodable at all.

### 2.1 Why exact chance, not approximate chance

Trials are generated in **pairs** — one class-0 and one class-1 trial on the same
16 lines — and the train/test split is made at the **pair** level, so every test
pair is complete.

Under the aliased projection both members of a test pair present bit-identical
features but carry opposite labels. A deterministic readout predicts one class
for both, getting exactly one right and one wrong. Accuracy is therefore **exactly
0.500**, for any readout, any dataset size, any seed. That converts the headline
result from "we failed to find an effect" into "the manipulation makes the effect
impossible by construction."

This is confirmed independently and not just asserted: **all five
`ref_per_neuron` beta settings also score exactly 0.500** (min = max = 0.500), as
they must if a per-neuron trace is really a function of the drive sequence alone.

## 3. Results

3 seeds × 4 input-rate sweep points, 80 train / 40 held-out test trials per arm
per point. All figures from `experiments/results/trace_locus.json`.

### 3.1 Headline

| arm | accuracy (mean ± CI95) | range | reading |
|---|---|---|---|
| **`aliased_substrate`** (discriminator) | **0.500 ± 0.000** | 0.500–0.500 | exactly chance at 12/12 points |
| `structured_substrate` (positive control) | 0.996 ± 0.006 | 0.975–1.000 | task fully decodable |
| `ref_per_neuron` (best of 5 betas) | 0.500 | 0.500–0.500 | ties with the substrate |
| `ref_per_synapse` (best of 5 betas) | **0.742 ± 0.049** | 0.575–0.875 | breaks the tie, clears the floor |

The per-synapse reference spans 0.650 / 0.654 / 0.665 / 0.740 / 0.742 across betas
0.5 / 1 / 2 / 4 / 8. Its best point clears the 0.62 power floor, so the task is
**demonstrably solvable by a line-keyed mechanism**. The substrate does not do
it, and instead matches the per-neuron reference exactly. That combination is the
verdict.

Decoding is a ridge readout on train-standardised spike counts (no drifting
normaliser, no quadratic expansion), so no arm is advantaged by feature space.

### 3.2 Rate sweep — all points spiking, genuinely varying

The sweep is a measured ladder, not a guessed one. `phi(z) = z*exp(1-z)` is
non-monotonic, so a ladder is only usable where the population actually fires;
and `dend_scale` is calibrated **once at a fixed scale**, because calibrating it
per gain would cancel the gain and make every point dynamically identical.

| gain | spike rate (per neuron/ms) | spikes per trial | aliased accuracy |
|---|---|---|---|
| 3.0 | 3.36e-03 | 48–53 | 0.500 |
| 6.0 | 1.11e-02 | 163–173 | 0.500 |
| 16.0 | 2.31e-02 | 347–359 | 0.500 |
| 32.0 | 2.38e-02 | 361–368 | 0.500 |

Rate spread max/min = **6.8×**, asserted at runtime to be ≥ 2× so a degenerate
sweep cannot pass unnoticed. `dend_scale = 5.1049` (fixed), calibrated at gain
6.0 from a measured peak dendritic potential of 7.147.

### 3.3 Matched-activity control

A difference that vanishes when total spike count is equated is a **rate** effect,
not a locus effect. Here the control is exact rather than approximate:

* **Within the aliased arm**, class-0 and class-1 trials are bit-identical, so
  their spike counts are equal **by construction**. Measured: 234.575 vs 234.575
  spikes/trial, relative difference **0.00e+00**, at every one of the 12 sweep
  points. This is not tuning — it is the same code path.
* **Across arms**, the relative spike-count difference at each gain is
  20.5% / 8.8% / 1.6% / 1.9% (gains 3 / 6 / 16 / 32), computed against the
  structured control at the *same* gain.
* The reference emulations necessarily inject additional drive by design
  (they modulate gain), so their spike counts run higher (250–305 vs 235 per
  trial) and are labelled as references throughout.

### 3.4 Spiking is asserted, not assumed

`docs/NEURON_OPERATING_POINT.md` documents the trap: `dend_scale=1.0` with a
large drive attenuates the input ~25× via `phi(z) = z*exp(1-z)`, the population
emits **zero** spikes, and the experiment silently compares silence to silence.

`assert_active()` therefore **raises** — it does not warn — if any arm has zero
spikes, a rate below `rate_lo = 1e-3`, or any trial with no spike at all. Ladder
rungs must additionally clear `rate_lo × 2.5`, because a rung sitting exactly on
the floor passes calibration and then dips below it under a different trial
sample.

This was not hypothetical: the first full-sweep attempt **died on exactly this
assertion** (rate 2.686e-04 at gain 1.0), and two further attempts died at
gain 1.5 and gain 2.0 until the ladder was selected from a measured rate curve
instead of guessed. Achieved values, all measured:

| quantity | measured |
|---|---|
| total spikes, aliased arm, all runs | 112,956 |
| spikes per trial (test set) | 235.3 |
| fraction of trials with ≥1 spike | **1.000** (minimum over all arms and points) |
| peak dendritic potential | 62.41 |

### 3.5 Pre-flight checks (asserted before the sweep)

| check | result |
|---|---|
| aliased class-0 vs class-1 *sample drive* bit-identical | **True**, max abs diff 0.0e+00 |
| per-neuron reference probe drive ties across classes | **True** |
| per-synapse reference probe drive ties across classes | **False** (max diff 78.6) — i.e. it really is line-keyed |
| full-run spike rasters, substrate, class 0 vs 1 | **bit-identical**, 315 / 315 spikes |
| full-run spike rasters, per-neuron reference | **bit-identical**, 138 / 138 |
| full-run spike rasters, per-synapse reference | **differs**, 276 / 305 |
| determinism (same seed, rerun) | **bit-identical** |
| zero-recurrent-weight codes | **bit-identical**, 0 cells changed |

Two of these are load-bearing: the substrate's rasters are bit-identical across
classes (so a per-neuron trace is not merely implied by the drives but confirmed
through the whole simulation), and the per-synapse reference's are not (so the
discriminator has teeth). If either flipped, the script raises.

## 4. Structural evidence (independent of the behavioural result)

The behavioural result is corroborated by reading the code. The trace is
per-neuron in the type system, before any experiment runs:

| fact | source |
|---|---|
| `NeuronState.v_dend` has shape `(n,)` — one scalar per neuron | `brain/neurons.py` |
| `update(i_soma, i_dend)` drives the dendrite **only** from `i_dend`, and `i_dend` is built **only** from `external_dend` | `brain/neurons.py`, `brain/simulator.py` |
| Recurrent synaptic delivery goes to `i_soma`, so **spiking synapses cannot reach the dendritic trace at all** | `brain/simulator.py` |
| `Synapses.targets` is `(n_pre, k_out)`, indexed by **presynaptic** neuron — a postsynaptic neuron has no representable in-edge list | `brain/connectivity.py` |
| A per-synapse `elig` matrix `(n_pre, k_out)` **does** exist, but it gates *weight writes* (only when `neuromod != 0`), not the dendritic trace | `brain/plasticity.py` |

There is no code path by which "which afferent was active" could reach `v_dend`.
The experiment confirms what the types already imply.

## 5. What this rules in, and what it rules out

**Rules in**

* The per-neuron (diagonal-state) reading of the dendritic trace: it is a
  function of the neuron's own drive history only.
* Treating this substrate's dendritic memory as mathematically equivalent to the
  diagonal recurrence in S4 / Mamba / RWKV **at this operating point**.
* Building working memory as one scalar per neuron, and looking for novelty
  elsewhere — recurrence, plasticity, the dendritic nonlinearity itself — rather
  than in the trace's addressing.

**Rules out**

* Any claim that this substrate performs presynaptic-input-keyed (**addressed**)
  recall via its dendritic trace.
* Using this substrate's trace as the biological justification for a
  same-token-chain gated scan: the trace carries no afferent identity, so as
  implemented it cannot support that computation.
* Any claim that dendritic traces are a novel memory mechanism relative to
  diagonal state-space models. At this locus, they are the same computation.

This is a **negative** result for the paradigm's novelty claim, and it is
reported as such rather than softened. Note the scope: it rules out the *current
implementation* as an addressed-recall substrate. It does not rule out
addressed recall as an architectural idea, and it does not touch the substrate's
other mechanisms (adaptive threshold, dCaAP nonlinearity, three-factor
plasticity), which this experiment did not test.

## 6. What would change the verdict

The experiment is not merely a failure to detect an effect; it is a bound. It
would be overturned by any of:

1. **A code path that indexes the trace by input line.** e.g. a per-inbound-edge
   current or trace in `NeuronState`/`Brain.step`. A minimal version: give the
   dendrite an in-edge list (CSR keyed by postsynaptic neuron) alongside
   `Synapses.targets`, and let each edge carry its own `e_ij` multiplying its own
   weight *before* summation into `v_dend`. The aliased manipulation would then
   no longer tie, and this script — unchanged — would report the aliased arm
   above chance.
2. **Sub-threshold dendritic compartmentalisation** where distinct input clusters
   sum non-linearly before reaching the soma. Currently `i_dend` is a single
   summed current, so all input lines are interchangeable at the dendrite by
   construction.
3. **A demonstration that the aliased tie fails on the real substrate** — i.e.
   the same drive sequence producing different probe codes. That would mean
   hidden pathway-dependent state exists; §3.5 asserts the opposite and would
   fire first.

The cheap discriminating test for follow-up: implement (1) **inside** the
simulator loop, then rerun the identical aliased arm. This script already
contains that emulation as a *reference* (`ref_per_synapse`); if the in-simulator
version clears chance where the current substrate does not, the locus is
per-synapse in the new code and the verdict flips.

## 7. Caveats

* **One substrate, one operating point family.** The conclusion is about `brain/`
  as currently implemented, at `dcaap` dendritic mode with a fixed calibrated
  `dend_scale` and a 60 ms trial. A different neuron model could differ.
* **The reference emulations are not the substrate.** `ref_per_neuron` and
  `ref_per_synapse` inject gain modulation through `external_dend`; they
  establish that the *task* is solvable by a line-keyed mechanism, not that the
  substrate contains one. They are labelled as references throughout, and their
  gain modulation clamps against ±60 (recorded per arm as `clamp_hits`; the per-neuron reference binds it far more often than the
  per-synapse one, 4.9–6.6M vs 2.0–9.5M,
  yet still lands on exactly 0.500 — clamping cannot rescue a per-neuron
  trace, because its drive is bit-identical across classes before the clamp),
  whereas the substrate arms apply **no clamp at all** — verified finite to gain 64, and
  reported with zero clamp hits.
* **Firing is dominated by spontaneous activity**, which is a real property of
  this operating point and worth stating plainly. At gain 6, the sample phase
  contributes ~13 spikes/trial, the *silent* delay phase ~62, and the probe phase
  ~94. So the stimulus-locked component is a minority of total spikes, and the
  task is decodable because the structured control reaches 0.996 — not because
  the stimulus dominates the population. A reader who wants a stimulus-dominated
  regime should lower `noise_std`/`init_noise` and re-run; the confound this
  guards against (a rate difference masquerading as a locus difference) is
  controlled exactly regardless, since class-0 and class-1 spike counts are
  identical to 0.00e+00.
* **The `structured` control sits at ceiling** (0.996), so it establishes
  decodability but does not calibrate partial difficulty.
* **`beta` ladders are coarse** (5 fixed values). The per-synapse reference's
  best point comes from that grid; a finer sweep might raise it further, which
  would only strengthen the negative result.
* **Cross-backend codes are not bit-identical, and the reason is pre-existing.**
  `brain/connectivity.py` draws `targets`, `delays` and `weights` from the
  *backend* RNG (`Backend.randint` / `Backend.normal`), and `numpy.random
  .default_rng` and MLX's generator differ — so the two backends build different
  recurrent networks from the same seed. Measured on the aliased arm at gain 16
  with *zero* membrane noise and zero init noise: NumPy 313 spikes vs MLX 316,
  7 of 256 cells differing by one spike in their probe-window count, identical
  peak `v_dend`. The projection this experiment actually varies is drawn in
  NumPy and is identical across backends, so the manipulation under test is
  unaffected; only the background recurrent draw differs. `--cross-backend`
  records this in the JSON rather than papering over it. Note this also means
  the sweep results above are backend-specific in their recurrent noise; the
  aliased arm is at exactly 0.500 regardless, because that follows from the
  drive tie and not from the recurrent draw.
* **NumPy is ~24× faster than MLX at this size**, so the default backend is
  NumPy. MLX's advantage in `docs/SCALING.md` is for far larger populations,
  where dispatch amortises; at 256 neurons it does not.
* **Recurrence was bit-inert here** (zero-recurrent-weight codes identical, 0
  cells changed), consistent with the short-window result in
  `docs/WORKING_MEMORY.md`. This is a diagnostic, not part of the claim: the
  dendritic trace receives no synaptic input under any condition.
