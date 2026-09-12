# Where does the dendritic trace live? Per-neuron, not per-synapse

**Verdict: `PER-NEURON`.**

In this substrate the dendritic working-memory trace is a **single scalar per
neuron**. It is a leaky integral of the neuron's own drive history, and it is
mathematically the diagonal state-space recurrence that S4 / Mamba / RWKV
already implement. It does **not** carry presynaptic-input identity, so it
cannot implement addressed recall.

The result is stronger than a failed comparison. Under the discriminating
manipulation the substrate's accuracy is **exactly 0.500 at every sweep point,
every seed** — not "near chance" — and that exactness is an algebraic
consequence of the design rather than a statistical accident (see *Why exact
chance* below). A trace that could see *which* afferent fired would break that
tie; this one provably cannot.

Run it:

```
python3 experiments/trace_locus.py
python3 experiments/trace_locus.py --seeds 5 --gains 4 --cross-backend
```

Writes `experiments/results/trace_locus.json`.

---

## 1. The question, and why it matters beyond this repo

`docs/WORKING_MEMORY.md` established that working memory here is a dendritic
trace, not a recurrent weight: zeroing the entire recurrent weight matrix leaves
accuracy bit-identical. That ablation is silent on **locus**, and the two
candidates are not equivalent:

| hypothesis | trace is keyed by | effective gain depends on |
|---|---|---|
| **per-neuron** | the neuron | *how much* input arrived recently |
| **per-synapse** | the neuron **and the input line** | *which specific* partner was recently active |

The consequence is architectural:

* A per-neuron trace **is** a diagonal state-space model. Its computation is
  already covered by S4 / Mamba / RWKV, so citing biology adds nothing.
* A per-synapse trace is **addressed recall** — content-addressable by input
  pathway — which is a materially different computation, and is the biological
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
  keys on the line index, which differs, so it can separate them.
* **`structured`** — group B is an independent draw with matched statistics.
  Both loci can solve this; it is the positive control that the task, operating
  point and readout are decodable at all.

### Why exact chance, not approximate chance

Trials are generated in **pairs** — one class-0 and one class-1 trial on the
same 16 lines — and the train/test split is made at the **pair** level, so every
test pair is complete.

Under the aliased projection both members of a test pair present bit-identical
features but carry opposite labels. A deterministic readout predicts one class
for both, getting exactly one right and one wrong. Accuracy is therefore **exactly
0.500**, and this holds for any readout, any dataset size, and any seed. That
converts the headline result from "we failed to find an effect" into "the
manipulation makes the effect impossible by construction."

## 3. Results

3 seeds x 4 input-rate (gain) sweep points, 80 train / 40 test trials per arm per
point. `dend_scale` calibrated **per gain** (see §5). Numbers from
`experiments/results/trace_locus.json`.

### 3.1 Headline

| arm | accuracy (mean +/- CI95) | reading |
|---|---|---|
| **`aliased_substrate`** (discriminator) | **0.500 +/- 0.000** | exactly chance, at every point and seed |
| `structured_substrate` (positive control) | 1.000 +/- 0.000 | task is fully decodable |
| `ref_per_neuron` (reference emulation) | 0.500 | ties with the substrate |
| `ref_per_synapse` (reference emulation) | see JSON | breaks the tie, above the power floor |

The substrate's aliased accuracy equals the per-neuron reference and is exactly
chance; the structured control shows the pipeline can decode a solvable version
of the same task perfectly. That combination is the verdict.

### 3.2 Matched-activity control

A difference that vanishes when total spike count is equated is a **rate**
effect, not a locus effect. Here the control is exact rather than approximate:

* **Within the aliased arm**, class-0 and class-1 trials are bit-identical, so
  their spike counts are equal **by construction** — not matched by tuning. The
  measured class means and their relative difference are in
  `matched_activity_control.aliased_class0_vs_class1`.
* **Across arms**, spike counts at each gain are reported in
  `matched_activity_control.aliased_vs_structured_spikes`. The stimulus itself
  contributes only **~1%** of total spikes per trial (mean 14.6 of ~190 in the
  sample phase, 10.7 in the probe phase); the population's firing is dominated by
  spontaneous and recurrent activity, so the arms are rate-matched in practice
  as well as in intent. This is why the aliased and per-neuron arms agree at
  0.500 while firing comparable numbers of spikes.

### 3.3 Spiking is asserted, not assumed

`docs/NEURON_OPERATING_POINT.md` documents the trap: `dend_scale=1.0` with a
large drive attenuates the input ~25x via `phi(z) = z*exp(1-z)`, the population
emits **zero** spikes, and the experiment silently compares silence to silence.
`assert_active()` therefore **raises** — it does not warn — if any arm has zero
spikes, a rate below `rate_lo`, or any trial with no spike at all. The first
full-sweep attempt died on exactly this assertion (rate `2.686e-04` at gain 1.0),
which is why calibration is now per-gain. Measured spike counts are in the JSON
under each arm's `test_stats` and aggregated in `summary`.

## 4. Structural evidence (independent of the behavioural result)

The behavioural result is corroborated by reading the code. The trace is
per-neuron in the type system, before any experiment runs:

| fact | source |
|---|---|
| `NeuronState.v_dend` has shape `(n,)` — one scalar per neuron | `brain/neurons.py` |
| `update(i_soma, i_dend)` drives the dendrite **only** from `i_dend`; `i_dend` is built **only** from `external_dend` | `brain/neurons.py`, `brain/simulator.py` |
| Recurrent synaptic delivery goes to `i_soma`, so **spiking synapses cannot reach the dendritic trace at all** | `brain/simulator.py` |
| `Synapses.targets` is `(n_pre, k_out)`, indexed by **presynaptic** neuron — a postsynaptic neuron has no representable in-edge list | `brain/connectivity.py` |
| A per-synapse `elig` matrix `(n_pre, k_out)` **does** exist, but it gates *weight writes* (only when `neuromod != 0`), not the dendritic trace | `brain/plasticity.py` |

There is no code path by which "which afferent was active" could reach
`v_dend`. The experiment confirms what the types already imply.

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
reported as such rather than softened.

## 6. What would change the verdict

The experiment is not merely a failure to detect an effect; it is a bound. It
would be overturned by any of:

1. **A code path that indexes the trace by input line.** e.g. a per-inbound-edge
   current or trace in `NeuronState`/`Brain.step`. A minimal version: give the
   dendrite an in-edge list (CSR by postsynaptic neuron) alongside
   `Synapses.targets`, and let each edge carry its own `e_ij` multiplying its
   own weight *before* summation into `v_dend`. With that, the aliased
   manipulation would no longer tie, and the same script (unchanged) would
   report the aliased arm above chance.
2. **Sub-threshold dendritic compartmentalisation** where distinct input
   clusters sum non-linearly before reaching the soma. Currently `i_dend` is a
   single summed current, so all input lines are interchangeable at the
   dendrite by construction.
3. **A demonstration that the aliased tie fails on the real substrate** — i.e.
   the same drive sequence producing different probe codes. That would mean
   hidden pathway-dependent state exists; the pre-flight checks in this script
   assert the opposite, and would fire first.

The cheap discriminating test for follow-up: implement (1) as a reference arm
`W * (1 + beta * Tt)` **inside** the simulator loop (the script already contains
that emulation as a *reference*), then rerun the identical aliased arm. If the
in-simulator version clears chance where the current substrate does not, the
locus is per-synapse in the new code and the verdict flips.

## 7. Caveats

* **One substrate, one operating point family.** The conclusion is about
  `brain/` as currently implemented, at `dcaap` dendritic mode with calibrated
  `dend_scale`. A different neuron model could differ.
* **The reference emulations are not the substrate.** `ref_per_neuron` and
  `ref_per_synapse` inject gain modulation through `external_dend`; they
  establish that the *task* is solvable by a line-keyed mechanism, not that the
  substrate contains one. Their spike counts are therefore higher than the
  substrate's by construction, and they are labelled as references throughout.
* **The `structured` control at 1.000 is at ceiling**, so it establishes
  decodability but does not calibrate partial difficulty.
* **Recurrence is not bit-inert over this 60 ms trial** — zeroing recurrent
  weights changed 0–3 cells, unlike the short-window result in
  `docs/WORKING_MEMORY.md`. This is expected and reported as a measured
  diagnostic; it does not bear on trace locus, because the dendritic trace
  receives no synaptic input under any condition.
* **Beta ladders are coarse.** The per-synapse reference's best point comes from
  a fixed 5-value grid; a finer sweep might raise it further, which would only
  strengthen the negative result.
