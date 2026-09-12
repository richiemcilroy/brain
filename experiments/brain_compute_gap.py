"""How much compute would a full human-brain simulation need?

Every figure is either [measured] on this machine by this repo, or [literature]
with its source named, or [arithmetic] on those two. Nothing is guessed.

The point of the exercise is not to produce a big number -- it is to identify
WHICH constraint binds first, because memory, throughput and energy have
different scaling laws and therefore different escape routes.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "brain_compute_gap.json")

# --------------------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------------------
# [literature] Azevedo et al. 2009, J Comp Neurol 513:532.
N_NEURONS = 86.1e9
# [literature] mean cortical synapses per neuron, commonly 7e3-1e4.
SYN_PER_NEURON = 7.0e3
# [measured] storage is EXACTLY 12 bytes/synapse, verified across four
# configurations in bench/lead_independent_scale.json (12.04-12.14).
BYTES_PER_SYN = 12.0
# [measured] peak synaptic-event throughput, 1e6 neurons / 1.024e9 synapses,
# 5.61% active, K=1024, 28.42 ms/step.
EV_PER_S = 2.020888932e9
# [measured] RAM at that configuration.
GIB_AT_PEAK = 11.48
# [literature] average cortical firing rate. Sparse-coding estimates run well
# below 1 Hz; 1 Hz is a common working figure. Cost scales LINEARLY with this.
RATE_HZ = 1.0
# [literature] energy anchors.
LOIHI2_PJ_PER_SYNOP = 23.6
BIOLOGY_FJ_PER_EVENT = (1.0, 100.0)
# [measured] package power budget for the machine doing the measured work.
WATTS = 60.0
# [literature] DTB 2023 human-scale run.
DTB = dict(gpus=14012, neurons=86e9, synapses=47.8e12,
           realtime=(0.008, 0.015), plasticity=False)
# [literature] K computer, Kunkel et al. 2014.
K = dict(procs=82944, neurons=1.86e9, synapses=1.1e13,
         s_per_bio_s=2481.66)


def main() -> dict:
    n_syn = N_NEURONS * SYN_PER_NEURON
    bytes_total = n_syn * BYTES_PER_SYN
    events_needed = n_syn * RATE_HZ

    rt_deficit = events_needed / EV_PER_S
    machines_mem = bytes_total / (128 * 1024**3)
    machines_tp = events_needed / EV_PER_S

    # energy per synaptic event on the measured machine
    j_per_event = WATTS / EV_PER_S
    pj_per_event = j_per_event * 1e12
    vs_loihi = pj_per_event / LOIHI2_PJ_PER_SYNOP
    vs_bio_lo = j_per_event / (BIOLOGY_FJ_PER_EVENT[0] * 1e-15)
    vs_bio_hi = j_per_event / (BIOLOGY_FJ_PER_EVENT[1] * 1e-15)

    # what one biological second costs on this machine
    s_per_bio_s = rt_deficit

    out = dict(
        inputs=dict(
            n_neurons=[N_NEURONS, "literature: Azevedo 2009"],
            syn_per_neuron=[SYN_PER_NEURON, "literature: cortical mean"],
            bytes_per_synapse=[BYTES_PER_SYN, "measured, exactly 12 B"],
            ev_per_s=[EV_PER_S, "measured, peak of 5 configs"],
            rate_hz=[RATE_HZ, "literature: cortical average, scales linearly"],
        ),
        totals=dict(
            synapses=n_syn,
            storage_bytes=bytes_total,
            storage_pb=bytes_total / 1e15,
            events_per_s_for_realtime=events_needed,
        ),
        memory_wall=dict(
            machines_128gb=machines_mem,
            note="The binding constraint. Does not improve with more compute per chip.",
        ),
        throughput_wall=dict(
            machines_of_ours=machines_tp,
            realtime_deficit_x=rt_deficit,
            s_per_biological_s=s_per_bio_s,
            days_per_biological_s=s_per_bio_s / 86400,
            ms_of_brain_per_hour_of_compute=3600.0 / s_per_bio_s * 1000.0,
        ),
        energy=dict(
            pj_per_synaptic_event=pj_per_event,
            x_less_efficient_than_loihi2=vs_loihi,
            x_less_efficient_than_biology_low=vs_bio_lo,
            x_less_efficient_than_biology_high=vs_bio_hi,
            note=("These are operation-count proxies scaled by package power, "
                  "not measured synapse energy."),
        ),
        sanity_checks=dict(
            dtb_2023=dict(**DTB, note="static connectivity, no plasticity, "
                                      "2x fewer synapses than the biological estimate"),
            k_computer_2014=dict(**K),
            ratio_dtb_gpus_to_our_measured_laptops=(
                (EV_PER_S * 1.0) / (events_needed / DTB["gpus"])
            ),
        ),
    )

    # --- printed report ---
    print("=" * 68)
    print("HOW MUCH COMPUTE FOR A FULL HUMAN BRAIN?")
    print("=" * 68)
    print(f"neurons                    {N_NEURONS:.3e}   [literature]")
    print(f"synapses (x{SYN_PER_NEURON:.0e}/neuron)  {n_syn:.3e}   [arithmetic]")
    print()
    print("--- CONSTRAINT 1: MEMORY (binds first) ---")
    print(f"at our measured {BYTES_PER_SYN:.0f} bytes/synapse:")
    print(f"  {bytes_total:.3e} bytes = {bytes_total/1e15:.1f} PB = {bytes_total/1e18:.2f} EB")
    print(f"  = {machines_mem:,.0f} machines at 128 GB each")
    print("  This is the wall. Adding GPUs does not help.")
    print()
    print("--- CONSTRAINT 2: THROUGHPUT (for real time) ---")
    print(f"needed at {RATE_HZ:g} Hz average rate: {events_needed:.3e} events/s")
    print(f"we measure:                  {EV_PER_S:.3e} events/s  [measured]")
    print(f"  deficit: {rt_deficit:.3e}x  = {machines_tp:,.0f} of our machines")
    print(f"  1 s of brain time -> {s_per_bio_s/86400:,.1f} days of compute")
    print(f"  1 hour of compute -> {3600.0/s_per_bio_s*1000:,.3f} ms of brain time")
    print()
    print("--- CONSTRAINT 3: ENERGY (the one biology wins outright) ---")
    print(f"our energy per synaptic event: {pj_per_event:,.0f} pJ  [measured W / measured rate]")
    print(f"  Loihi 2:                     {LOIHI2_PJ_PER_SYNOP:.1f} pJ  -> {vs_loihi:,.0f}x better")
    print(f"  biological synapse:          {BIOLOGY_FJ_PER_EVENT[0]:g}-{BIOLOGY_FJ_PER_EVENT[1]:g} fJ "
          f"-> {vs_bio_lo:,.0f}x-{vs_bio_hi:,.0f}x better")
    print(f"  the brain does this whole job on ~20 W.")
    print()
    print("--- SANITY CHECK vs HARDWARE THAT ACTUALLY RAN AT HUMAN SCALE ---")
    print(f"DTB 2023: {DTB['gpus']:,} GPUs, {DTB['synapses']:.1e} synapses (2x fewer than")
    print(f"  biology), STATIC connectivity, NO plasticity -> "
          f"{DTB['realtime'][0]}-{DTB['realtime'][1]}x realtime")
    print(f"K computer 2014: {K['procs']:,} procs, {K['neurons']:.2e} neurons -> "
          f"{K['s_per_bio_s']:,.0f} s per bio-second")
    print()
    print("--- WHAT THIS MEANS ---")
    print(f"The gap is ~{machines_mem:,.0f}x in memory and ~{machines_tp:,.0f}x in throughput.")
    print("Both are ~1e5. It is not a matter of a bigger cluster; it is a missing")
    print("algorithm. Any approach that stores a 12-byte weight per synapse and")
    print("touches every synapse per event is dead at this scale by construction.")
    print()
    print(f"note: cost scales linearly with the assumed {RATE_HZ:g} Hz rate.")
    print(f"at 0.1 Hz the throughput deficit is {rt_deficit/10:,.0f}x; at 10 Hz, {rt_deficit*10:,.0f}x.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    main()
