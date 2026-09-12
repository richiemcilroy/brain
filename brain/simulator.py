"""Event-driven cortical simulator.

Design notes
------------
The simulator is *clock-driven for state, spike-driven for communication*: every
neuron integrates once per millisecond, but synaptic work is performed only for
neurons that actually spiked. That combination is what makes sparse activity
pay off, because synaptic delivery scales with the number of spikes rather than
with the number of synapses.

Spike compaction. MLX has no ``nonzero``/``argwhere``, so indices of active
neurons are extracted with a cumulative-sum rank plus a scatter
(``pos = cumsum(mask) - 1``). This is exact and order-preserving; it is verified
against ``np.flatnonzero`` in the test-suite. The output buffer is sized to
``spike_capacity_frac * n_neurons`` and any overflow is *counted and reported*
rather than silently dropped, because silent truncation would quietly corrupt a
simulation.

Energy accounting. We report *active synaptic operations* (SynOps). This is an
operation count, not a joule measurement. That distinction matters: a
biological synapse costs ~1-100 fJ while a neuromorphic SynOp on Loihi costs
~23.6 pJ, so "spiking is cheap" is not a free claim. Counting SynOps is the
honest, hardware-independent proxy (Attwell & Laughlin 2001 place ~59% of
brain energy budget in synaptic signalling).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from typing import Any

from .backend import Backend, get_backend
from .connectivity import DelayBuffer, SynapseConfig, Synapses
from .neurons import NeuronConfig, NeuronState
from .plasticity import PlasticityConfig, ThreeFactorPlasticity


@dataclass
class SimConfig:
    n_neurons: int = 4000
    k_out: int = 64
    inhibition: str = "none"  # "none" | "kwta"
    k_wta: int = 0
    spike_capacity_frac: float = 0.10
    init_noise: float = 0.30
    seed: int = 0
    dt: float = 1.0
    excitatory_frac: float = 0.8
    syn: SynapseConfig | None = None
    neu: NeuronConfig | None = None
    plas: PlasticityConfig | None = None

    def __post_init__(self) -> None:
        if self.inhibition not in ("none", "kwta"):
            raise ValueError("inhibition must be none|kwta")
        if self.inhibition == "kwta" and self.k_wta <= 0:
            raise ValueError("kwta inhibition requires k_wta > 0")
        if self.n_neurons % 2:
            raise ValueError("n_neurons must be even (E/I split)")


@dataclass
class StepStats:
    step: int
    spikes: int
    overflow: int
    active_frac: float
    synops_active: int
    synops_executed: int
    mean_v: float


class Brain:
    """A spiking cortical population with local plasticity."""

    def __init__(self, cfg: SimConfig, backend: Backend | str | None = None):
        self.cfg = cfg
        be = backend if isinstance(backend, Backend) else get_backend(backend, seed=cfg.seed)
        self.be = be
        be.seed(cfg.seed)

        self.neuron_cfg = cfg.neu or NeuronConfig(n=cfg.n_neurons, dt=cfg.dt)
        self.neuron_cfg.n = cfg.n_neurons
        self.syn_cfg = cfg.syn or SynapseConfig(
            n_pre=cfg.n_neurons, n_post=cfg.n_neurons, k_out=cfg.k_out,
            excitatory_frac=cfg.excitatory_frac, seed=cfg.seed,
            delay_max=4,
        )
        self.plas_cfg = cfg.plas or PlasticityConfig(dt=cfg.dt)

        self.neurons = NeuronState(self.neuron_cfg, be, init_noise=cfg.init_noise)
        self.synapses = Synapses(self.syn_cfg, be)
        self.plasticity = ThreeFactorPlasticity(
            self.plas_cfg, cfg.n_neurons, cfg.n_neurons, self.syn_cfg.k_out, be
        )
        self.delay = DelayBuffer(cfg.n_neurons, self.syn_cfg.delay_max, be)

        self.capacity = max(1, int(cfg.spike_capacity_frac * cfg.n_neurons))
        if cfg.inhibition == "kwta":
            self.capacity = max(self.capacity, cfg.k_wta)
        self._spike_buf = be.zeros((self.capacity,))
        self._arange = be.arange(cfg.n_neurons)
        self._krow = be.arange(self.syn_cfg.k_out, dtype=be.idx_dtype).reshape((1, -1))
        self._time = 0
        self.total_spikes = 0
        self.total_overflow = 0
        self.total_synops = 0
        self.total_synops_executed = 0
        self.wall_step_ms: list[float] = []
        # Exposed so downstream consumers (readouts, decoders, analyses) can see
        # which neurons fired. Kept as a boolean mask, and also as compacted
        # indices, since gathering with indices is what avoids an O(N) scan.
        self.last_spike_mask: Any = None
        self.last_spike_idx: Any = None

    # ------------------------------------------------------------------ spikes
    def _compact(self, mask: Any) -> tuple[Any, int]:
        """Exact, order-preserving extraction of active neuron indices.

        Returns the full padded buffer plus the true spike count. Callers MUST
        use only ``buf[:n_spk]``: slots beyond ``n_spk`` hold a filler value, and
        feeding them to the synaptic gather makes the network re-deliver one
        real neuron's synapses on every padding slot. That produced ~28 phantom
        spikes per step in a network with no input whatsoever, so it is a
        correctness requirement, not an optimisation.
        """
        be = self.be
        pos = be.cumsum(be.astype(mask, be.float_dtype)) - 1.0
        clamped = be.clip(pos, 0.0, float(self.capacity - 1)).astype(be.idx_dtype)
        keep = be.logical_and(mask, pos < float(self.capacity))
        vals = be.where(keep, self._arange, be.zeros(()) + 0.0)
        buf = be.scatter_add(be.zeros((self.capacity,)), clamped, vals)
        n_spk = int(be.to_numpy(be.sum(be.astype(mask, be.float_dtype))))
        overflow = max(0, n_spk - self.capacity)
        return buf, overflow

    def _kwta_mask(self, mask: Any) -> Any:
        """Keep at most ``k_wta`` winners, by soma potential, among spikers.

        Selection is by RANK, not by thresholding at the k-th score. An earlier
        version returned ``mask & (score >= cut)``, which keeps every neuron
        TIED with the k-th value. Because the soma resets to ``v_reset`` after
        spiking and ``v_thresh`` is reached exactly, ties are common: measured,
        ``k_wta=8`` admitted up to 20 winners, so k-WTA did not bound sparsity
        at all — which silently invalidates any sparsity or SynOps claim made
        with inhibition enabled.

        Ties are broken by index, so the result is deterministic and exactly
        ``k`` neurons are ever kept.
        """
        be, k = self.be, self.cfg.k_wta
        if k <= 0 or k >= self.cfg.n_neurons:
            return mask
        n_spk = int(be.to_numpy(be.sum(be.astype(mask, be.float_dtype))))
        if n_spk <= k:
            return mask
        # rank by the PRE-reset potential; see NeuronState.update
        score = be.where(mask, self.neurons.v_pre_reset,
                         be.full((self.cfg.n_neurons,), -1e9))
        # MLX's argsort takes no ``kind`` and has no ``scatter_add``, so this is
        # written portably: order, take the first k, then build the mask by
        # adding 1 at those indices. That is deterministic given the backend's
        # argsort, and ties are broken by whatever order argsort returns.
        order = be.xp.argsort(-score)
        chosen = np.asarray(be.to_numpy(order[:k])).astype(np.int64)
        keep = np.zeros((self.cfg.n_neurons,), dtype=bool)
        keep[chosen] = True
        return be.array(keep)

    # -------------------------------------------------------------- main step
    def step(self, external_soma: Any = None, external_dend: Any = None,
             neuromod: float = 0.0) -> StepStats:
        be = self.be
        t0 = time.perf_counter()

        # Trace decay is O(n_neurons) per step. Skip it entirely when plasticity
        # is off so that large-scale dynamics benchmarks measure the simulator
        # rather than bookkeeping for a feature that is disabled.
        if self.plas_cfg.enabled:
            self.plasticity.decay()

        i_soma = self.delay.read_and_clear(self._time)
        i_dend = be.zeros((self.cfg.n_neurons,))
        if external_soma is not None:
            i_soma = i_soma + external_soma
        if external_dend is not None:
            i_dend = i_dend + external_dend

        mask = self.neurons.update(i_soma, i_dend)
        if self.cfg.inhibition == "kwta":
            mask = self._kwta_mask(mask)

        spike_buf, overflow = self._compact(mask)
        n_spk = int(be.to_numpy(be.sum(be.astype(mask, be.float_dtype))))

        # CRITICAL: take only the real spikes. Passing the whole padded buffer
        # here is the "phantom drive" bug - every unused slot would gather
        # neuron 0's genuine targets and weights and inject them for real.
        spike_idx = spike_buf[:n_spk].astype(be.idx_dtype)
        self.last_spike_mask = mask
        self.last_spike_idx = spike_idx
        self.total_overflow += overflow

        targets = be.take(self.synapses.targets, spike_idx)
        weights = be.take(self.synapses.weights, spike_idx)
        delays = be.take(self.synapses.delays, spike_idx)
        if self.plas_cfg.enabled:
            # apply() handles pre-trace, rule evaluation, weight write and
            # post-trace ordering internally; do not reorder those here.
            self.plasticity.apply(spike_idx, targets, mask, self.synapses, neuromod)
            self.plasticity.maybe_homeostasis(self.synapses)

        self.delay.schedule(self._time, targets, delays, weights)

        # Two different quantities, both reported because conflating them is
        # how spiking networks acquire fake efficiency numbers:
        #
        #   synops_active    = spikes x fan-out. The ALGORITHMIC cost, i.e. the
        #                      sparsity claim. This is what should be compared
        #                      against a dense baseline's MAC count.
        #   synops_executed  = the dense rectangular gather this implementation
        #                      actually performs (buffer_capacity x fan-out).
        #                      Padded slots multiply garbage by zero weights, so
        #                      the wall-clock cost is capacity-based until the
        #                      gather is made genuinely sparse.
        #
        # Reporting only the former would overstate efficiency by up to ~10x.
        synops_active = n_spk * self.syn_cfg.k_out
        synops_executed = self.capacity * self.syn_cfg.k_out
        self.total_spikes += n_spk
        self.total_synops += synops_active
        self.total_synops_executed += synops_executed
        self._time += 1
        self.wall_step_ms.append((time.perf_counter() - t0) * 1e3)
        if len(self.wall_step_ms) > 2000:
            self.wall_step_ms = self.wall_step_ms[-1000:]

        return StepStats(
            step=self._time, spikes=n_spk, overflow=overflow,
            active_frac=n_spk / self.cfg.n_neurons,
            synops_active=synops_active, synops_executed=synops_executed,
            mean_v=float(be.to_numpy(be.sum(self.neurons.v_soma))) / self.cfg.n_neurons,
        )

    def run(self, steps: int, *, drive: Any = None, neuromod: float = 0.0,
            record_spikes: bool = False) -> dict[str, Any]:
        """Run ``steps`` ms. Returns aggregate statistics."""
        be = self.be
        spikes_per_step: list[int] = []
        spike_history: list[Any] = []
        rates: list[float] = []
        for _ in range(steps):
            st = self.step(external_soma=drive, neuromod=neuromod)
            spikes_per_step.append(st.spikes)
            if record_spikes:
                # store the actual spike mask. An earlier version stored
                # ``v_soma > -1e8``, which is True for essentially every neuron
                # and so recorded nothing about spiking at all. The mask must be
                # read from the step that just ran: an earlier revision here
                # referenced an undefined ``mask``, so ``record_spikes=True``
                # raised NameError and the public API was simply dead.
                spike_history.append(
                    be.to_numpy(self.last_spike_mask).astype(bool))
            rates.append(st.mean_v)
        return {
            "steps": steps,
            "total_spikes": int(sum(spikes_per_step)),
            "mean_active_frac": float(sum(spikes_per_step)) / steps / self.cfg.n_neurons,
            "total_synops_active": self.total_synops,
            "total_synops_executed": self.total_synops_executed,
            "overflow": self.total_overflow,
            "spikes_per_step": spikes_per_step,
            "spike_history": spike_history,
        }

    # -------------------------------------------------------------- readouts
    def population_vector(self, n_groups: int) -> Any:
        """Mean soma potential in ``n_groups`` contiguous groups (readout input)."""
        be = self.be
        g = self.cfg.n_neurons // n_groups
        v = self.neurons.v_soma.reshape((n_groups, g))
        return be.sum(v, axis=1) / float(g)

    def spike_rates(self) -> Any:
        be = self.be
        return self.neurons.spike_count / max(1, self._time)

    # ------------------------------------------------------------- utilities
    @property
    def n_synapses(self) -> int:
        return self.synapses.n_synapses

    def measured_throughput(self) -> dict[str, float]:
        if not self.wall_step_ms:
            return {}
        recent = self.wall_step_ms[-500:]
        mean_ms = sum(recent) / len(recent)
        return {
            "n_neurons": float(self.cfg.n_neurons),
            "n_synapses": float(self.n_synapses),
            "ms_per_step": mean_ms,
            "bio_ms_per_wall_s": 1000.0 / mean_ms if mean_ms else 0.0,
            "realtime_factor": (1.0 / mean_ms) if mean_ms else 0.0,
            "synops_executed_per_second":
                self.capacity * self.syn_cfg.k_out / (mean_ms / 1e3),
            "overflow_total": float(self.total_overflow),
        }

    def memory_bytes(self) -> dict[str, int]:
        def nb(x: Any) -> int:
            try:
                return int(x.nbytes)
            except AttributeError:
                return int(self.be.to_numpy(x).nbytes)

        return {
            "synapse_targets": nb(self.synapses.targets),
            "synapse_weights": nb(self.synapses.weights),
            "synapse_delays": nb(self.synapses.delays),
            "neuron_state": sum(nb(v) for v in (
                self.neurons.v_soma, self.neurons.v_dend,
                self.neurons.adapt, self.neurons.refrac)),
            "eligibility": nb(self.plasticity.elig) if self.plasticity.elig is not None else 0,
            "delay_buffer": nb(self.delay.flat),
        }

    def reset(self) -> None:
        self.neurons.reset()
        self.plasticity.reset_traces()
        self.delay.flat = self.be.zeros((self.delay.depth * self.delay.n_post,))
        self._time = 0
        self.total_spikes = 0
        self.total_overflow = 0
        self.total_synops = 0
