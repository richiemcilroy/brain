# The neuron is a transient onset detector, and its operating point is fragile

**Result: the dCaAP neuron in this substrate fires only while its dendrite ramps
*through* the peak of the tuning curve. It is an onset detector, not a
rate-integrator. Its input scale must be matched to the tuning peak or it is
silent — and when it is silent, nothing else in the pipeline can tell you why.**

This single mechanism explains four separate observations that were previously
recorded as unrelated oddities.

## The mechanism

The dendritic nonlinearity is `phi(u) = z·exp(1−z)` with `z = u / dend_scale`.
It peaks at `z = 1` with value 1.0 and decays for larger input. The dendrite
charges as `dv_dend/dt = (−v_dend + i_dend) / tau_dend`, so for a step input of
magnitude `i`, `v_dend` ramps from 0 toward `i` with time constant `tau_dend`.

The neuron therefore fires in a **burst during the ramp**, while `z` is passing
through 1, and falls silent again once `z` climbs past the peak. It does not
sustain firing under sustained input.

## Evidence: mismatch makes the neuron completely silent

120 active input channels, `gain 6.0`, 80 timesteps, 400 neurons, no plasticity:

| `dend_scale` | drive `u` | `phi(u)` | spikes emitted |
|---|---|---|---|
| 1.0 | 6.0 | 0.040 | **0** |
| 6.0 | 6.0 | 1.000 | **173** |

At `dend_scale=1.0` the peak-normalised input is `z=6`, far past the peak, so
`phi` attenuates the drive 25× and the soma never reaches threshold. Not a
single spike in the entire population. At `dend_scale=6.0` the same input lands
exactly on the peak and the population fires normally.

`k_wta` inhibition is **not** responsible — spike counts are identical with and
without it (173 vs 173), which was worth checking because a silent network
under a winner-take-all rule looks exactly like a broken winner-take-all rule.

## What this explains

1. **Why neurons fire exactly once, at ms 12–14 of a 15 ms window.** A separate
   measurement found `count>1 among active spikes = 0.000` at every gain tested,
   with first-spike latency p10/50/90 = 12/13/15 ms. That is not a tuning
   artefact to be fixed; it is the tuning curve doing exactly what it says.
2. **Why recurrence contributes nothing.** Recurrent spikes arrive after the
   burst is over. Zeroing the entire recurrent weight matrix changes 0.5–1.1%
   of cells in a 15 ms window and leaves accuracy bit-identical. The window was
   too short for the burst to be *over* before recurrent input landed, so
   recurrence was structurally excluded rather than shown to be useless.
3. **Why the "gain sweep" was never a gain sweep.** Changing `gain` at fixed
   `dend_scale` moves the operating point along the tuning curve. Measured, that
   curve is non-monotonic: gain 6.0 scored 0.690 and gain 12.0 scored 0.580.
   That apparent optimum is the peak of `phi`, not a property of the network.
4. **Why the substrate ties a random projection on static tasks.** An onset
   detector responds to *transients*. A static image presented as a constant
   input produces exactly one burst, so the "spike-count code" carries at most
   one bit per neuron and is a binarised random projection by construction.

## Consequences for the design

- **`dend_scale` is not a free parameter to leave at 1.0.** It must be set from
  the input distribution, or calibrated. Defaults that happen to work for one
  gain silently fail for another, and the failure mode is total silence with no
  error — which reads downstream as "the model is bad", not "the model is off".
- **A transient detector wants transient inputs.** Feeding it a static image and
  reading a rate code discards the mechanism's entire advantage. The natural
  input is a change, and the natural readout is *which* neurons fired and
  *when* — a latency code, which is what the working-memory result in
  `docs/WORKING_MEMORY.md` turned out to be using.
- **This is a coherent design, not a defect.** An onset detector is what you
  want for a change-detecting sensory front end. It is the wrong unit for a
  static classifier, and every static benchmark in this repo has been measuring
  the second thing.

## Caveats

- Single seed, 120 channels, one stimulus pattern; the silence/misfire contrast
  is 0 vs 173 spikes, which is decisive for the mechanism but not a calibration
  study.
- The onset interpretation is inferred from the equations plus the measured
  spike latenties and counts. It is not a direct recording of `v_dend` during a
  burst, though `v_dend` was verified to reach exactly 6.00 in both arms.
- `dend_scale` sweeps at intermediate values were not run, so the width of the
  usable band around `z=1` is not pinned down.
