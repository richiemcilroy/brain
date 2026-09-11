# Decision Record

## D1 — Build a mechanism testbed, not a brain replica

**Decision.** Do not attempt a full human-brain replica. Build a
biologically-grounded cortical substrate at the largest scale this machine
sustains, and use it to test a specific learning-mechanism hypothesis.

**Reason.** The replica is arithmetically out of reach (~1e5x neurons, ~1e5x
memory) and has already been attempted at scale: the Human Brain Project spent
~EUR 607M over 10 years and concluded a full replica was neither achievable nor
of clear practical use. Repeating that on a laptop would be strictly worse.

**Unresolved uncertainty.** Whether mechanisms validated at 1e6 neurons transfer
to 1e11 is genuinely unknown. The scaling literature is not encouraging about
naive transfer, which is exactly why the roadmap in `docs/SCALING.md` is staged
and each stage carries its own question.

## D2 — MLX-only, rejecting the panel's hybrid CPU-core recommendation

**Decision.** Compute runs on MLX (Metal GPU) with a NumPy fallback. Rejected:
the recommendation from the "challenge the assumptions" specialist to build a
separate event-driven CPU core in Rust/C++ with pybind.

**Reason — this was settled by measurement, not argument.** The specialist
argued MLX lacks native sparse ops and that Metal is weakest at the irregular
gather/scatter that spike propagation consists of, extrapolating only
~5-50 ms of biological time per wall-second at 1e6 neurons. Measured on this
machine:

| Quantity | Specialist estimate | Measured |
|---|---|---|
| 1e6 neurons, K=256, 2% active | 5-50 ms bio / wall-s | **~660 ms bio / wall-s (0.66x real-time)** |
| MLX vs NumPy, identical sparse workload | — | **MLX ~28x faster** |
| Active SynOps throughput | — | **4-9e9 / s** |

The specialist's caution was reasonable but its numbers were explicitly
labelled unmeasured extrapolation, and the measurement contradicts them by
1-2 orders of magnitude. Building and maintaining a Rust extension would have
cost significant time for a path that is *slower* than the one already working.
Its other conclusions (§D3, §D4) were accepted.

**Unresolved uncertainty.** The benchmark drives activity at a forced rate to
measure the cost of a *given* activity level; it does not measure emergent
network dynamics. Real dynamics may sit at a different operating point. This
caveat is stated wherever throughput is reported.

## D3 — Adopt the specialist's four falsification controls

**Decision.** The experiment carries controls the panel supplied, which I had
not originally specified: backprop with 10% replay as the *strong* baseline
(not fine-tuning), frozen random features + linear readout to test whether
plasticity is causal at all, a dendrite-ablation arm to separate dendritic
computation from mere sparsity, and a shuffled-label arm.

**Reason.** Iyer et al. 2022 (Front Neurorobot 16:846219) is specifically
criticised for confounding dendritic structure with activation sparsity by
comparing k-WTA spiking models against dense ReLU MLPs. Two independent
specialists converged on this. A result that cannot survive these controls is
not a result.

## D4 — Treat "predictive coding is local, therefore new" as a dead end

**Decision.** Do not rest the novelty claim on predictive coding or on local
learning being novel in itself.

**Reason.** Whittington & Bogacz 2017 proved predictive coding with local
Hebbian updates converges to backprop under stated conditions. "Local" is not
automatically "new paradigm". The claim is narrowed to the specific composition
and its measured continual-learning behaviour under matched budgets.

## D5 — Bugs found and fixed during construction

Recorded because each was found by a deliberate check rather than by luck, and
one of them invalidated a headline number.

1. **Lazy-evaluation artefact (invalidated a measurement).** An early benchmark
   reported 0.02 ms/step for 1e6 neurons. The graph was never forced with
   `mx.eval`, so nothing was computed. The real figure is ~1.5 ms/step. Every
   timing path now forces evaluation.
2. **Inverted LTP sign (would have silently corrupted all learning).** A
   coincident pre+post spike triggered the LTP term *and* the LTD term, so with
   `a_minus > a_plus` perfectly correlated pairs *depressed*. Fixed by
   evaluating the rule against the pre-update post-synaptic trace. Verified:
   coincident -> +0.010, post-before-pre -> -0.0114.
3. **Silent spike truncation (would have corrupted large runs).** The spike
   compaction buffer dropped spikes beyond its capacity without reporting it.
   Overflow is now counted and surfaced; zero under normal sparse activity.
4. **Pathological synchrony.** Identical neurons receiving a step input fired a
   single synchronous volley at t=0, overflowing capacity. Real neurons are
   heterogeneous; per-step membrane noise plus dispersed initial potentials
   removed it.

### D5b — Additional defects found by independent workers

Four more were found after D5 was written. Three were found by workers reviewing
*my* code, not their own, which is the point of running them.

5. **Phantom drive (correctness-critical).** `_compact` padded unused buffer
   slots with index 0 and `step` gathered the *whole* padded buffer, so every
   padding slot re-delivered neuron 0's genuine targets and weights. A network
   with no input, no membrane noise and no initial noise still emitted **28.4
   spikes/step**. My in-code comment asserting "garbage multiplied by zero
   weights" was simply false, and the worker disproved it with a zero-input
   control rather than taking the comment at face value. Fixed by gathering only
   `spike_buf[:n_spk]`; the zero-input control is now exactly silent. Every
   result in this repo predating this fix was contaminated at a floor of
   ~k_out/2 spikes/step and has been re-run.
6. **k-WTA ranked the wrong neurons.** Winners were selected by post-reset
   `v_soma`, so neurons that had just fired (reset to rest) looked like the
   *weakest* competitors and were suppressed first — approximately inverting
   the intended selection. Fixed by exposing `v_pre_reset`.
7. **`depressed` was never incremented.** The plasticity stats compared the
   sign-corrected `dw` against the pre-update weight, so `depressed` stayed at 0
   forever. Fixed; both directions are now counted correctly.
8. **SynOps overstated by up to 10x.** `step` charged `capacity * k_out` (the
   *buffer* size) instead of `spikes * k_out`. On a sparsity claim this is the
   difference between a real result and a fabricated one. Both quantities are
   now reported separately: `synops_active` (spikes x fan-out, the algorithmic
   claim) and `synops_executed` (the dense padded gather actually performed, the
   wall-clock truth).
9. **The dendrite was inert by construction.** `CortexClassifier._code` injected
   the image into the soma, so `v_dend` was identically 0, `phi(0) = 0`, and the
   intact and dendrite-ablated arms produced **bit-identical** results (both
   41.47%). The honest reading of that run is "the ablation was a no-op", not
   "dendrites do not matter" — a distinction that would have been easy to get
   wrong in the direction that suited the hypothesis. Fixed by routing
   feedforward input to the dendrite, with a dendritic gain above threshold
   (the non-monotonic `phi` peaks at 1.0, so at `dend_gain=1.0` peak drive
   equals threshold and the neuron can *never* fire).

## D6 — Verifying rather than trusting the compaction primitive

**Decision.** The cumsum-based spike-index compaction (needed because MLX has no
`nonzero`) is checked against `np.flatnonzero` for exactness in the test-suite.

**Reason.** An intermediate check appeared to show the compaction was inexact.
Investigation showed the test was wrong (it compared arrays of different
lengths) and the compaction is exact. The episode is a reminder that a failing
check is not automatically a code bug — but it must be resolved, never assumed.

## D7 — Orthogonal check on the scatter primitive

**Decision.** Trust MLX `.at[idx].add` for duplicate-index accumulation.

**Reason.** Verified independently against `np.add.at`: duplicate indices
accumulate correctly (max relative error ~2e-7, consistent with fp32 summation
order), negatives subtract correctly, and both int32 and uint32 index dtypes
work. An orthogonal theory check agreed: with 5.12e6 random synaptic events
delivered over 1e6 neurons, the fraction of neurons receiving at least one
event should be `1 - exp(-5.12)` = 99.4%, and the measured value was 99.4%.
Agreement between an independent analytic prediction and the observed value is
stronger evidence than either alone.

## D8 — Two claims that did NOT survive, reported as negative results

> **UPDATE — the central hypothesis failed outright.** Everything in D8 was
> written from the Split-MNIST results. Running the harder Permuted-MNIST
> benchmark reversed the headline: `brain` 22.10% vs `mlp` 68.30%, a **-46.20
> point loss** to backprop with a combined CI of 9.80. The substrate's
> Split-MNIST "win" only appears on the benchmark where the baseline
> self-destructs (0.00 retention on all but the newest task), and it inverts as
> soon as the benchmark is made harder in a way a dense network can still fit
> sequentially. See `docs/RESULTS.md` §3. **The project's central claim is
> retracted.** The dendritic negative result below, by contrast, is now
> *strengthened*: two independent benchmarks agree the intact dendrite is worse
> than the ablation.

**The dendritic nonlinearity does not help at this scale — it hurts.** On
task 0 of Split-MNIST (200 training samples, 4 epochs), the intact `dcaap`
dendrite scored 0.67-0.73 while the linear ablation scored 0.85-0.87. The
ablation *beat* the mechanism it was supposed to justify. The most likely
explanation is that the non-monotonicity attenuates the strongest inputs, which
is precisely the information a readout wants; at this scale, with this gain and
this code, linear drive carries more signal. This is a genuine negative result
against the dendritic claim in the continual-learning setting and is reported as
such. It does **not** contradict Gidon et al. 2020 — the single-neuron XOR result
in `experiments/xor_neuron.py` still holds, verified with real spikes. The
correct statement is narrow: *tuned dendrites enable a computation a point neuron
cannot perform, but that does not make them a better feature extractor on this
task.*

**Replay closes the gap.** In the first clean continual-learning run, the
substrate reached 41.47% final accuracy against 19.47% for naive backprop — but
41.20% for backprop with a 10% replay buffer. The substrate's entire advantage
over naive backprop is matched by a well-known baseline that is cheap to
implement. Any claim that this substrate offers a *unique* solution to
catastrophic forgetting is not supported by these numbers. The honest framing is
that a no-replay mechanism reaching replay-level retention is interesting, but
that it has not been shown to be *better* than replay.
