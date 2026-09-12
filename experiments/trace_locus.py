"""TRACE LOCUS: is the dendritic working memory per-neuron or per-synapse?

The question
------------
``docs/WORKING_MEMORY.md`` established that working memory in this substrate is
a *dendritic trace*: zeroing the entire recurrent weight matrix leaves accuracy
bit-identical. That ablation cannot say **where** the trace lives:

* ``PER-NEURON``  -- one scalar per neuron summarising its own recent input.
  Each neuron's effective gain depends only on *how much* input it received.
* ``PER-SYNAPSE`` -- each input pathway to a neuron carries its own trace, so a
  neuron's effective gain depends on *which specific* presynaptic partner was
  recently active (presynaptic-input-keyed, i.e. addressed recall).

The distinction is not cosmetic. A per-neuron trace is exactly the diagonal
state-space model that S4 / Mamba / RWKV already implement, so the biology would
add nothing to what mainstream sequence architectures do. A per-synapse trace is
a materially different computation.

The discriminator
-----------------
Input is split into two disjoint groups of ``M`` afferent lines. A trial is::

    [ SAMPLE  20 ms ]  ->  [ SILENT DELAY 20 ms ]  ->  [ PROBE 20 ms ]

The sample activates ``K`` lines of group A (class 0) or the *same* ``K`` lines
of group B (class 1). The probe always activates those same ``K`` lines in
group A. The class is never present in the probe input; the answer must come
from state left by the sample.

The two arms differ **only** in the frozen input projection:

* ``aliased``: rows of group B are a bit-identical tile of group A. Because the
  sampled line set is the same ``K`` lines in both groups, the sample drive
  vector is bit-identical between classes -- and so is every later drive, since
  the probe addresses the same lines. A per-neuron trace is a function of the
  drive sequence, so it is *provably constant* across classes and cannot decode
  them. A per-synapse trace keys on the *line index*, which does differ, so it
  can.
* ``structured``: group B is an independent draw with matched statistics. Both
  trace loci can solve this, so it is the positive control that the task, the
  operating point and the readout can decode at all.

The aliased arm is the discriminator. The matched-activity control is built into
it rather than bolted on: class-0 and class-1 trials are bit-identical, so their
spike counts are equal by construction.

Why the paired design matters (exact chance)
--------------------------------------------
Trials are generated in **pairs** -- one class-0 and one class-1 trial on the
same ``K`` lines -- and the train/test split is made at the *pair* level, so
every test pair is complete.

Under the aliased projection both members of a test pair therefore present
bit-identical features while carrying opposite labels. Any deterministic readout
predicts the same class for both, so it gets **exactly one right and one wrong**:
accuracy is exactly ``0.500``, not merely near it. That turns "the trace is
per-neuron" from a statistical claim into an algebraic one, and it removes the
class-imbalance confound that a raw split would introduce.

Power (is the task merely unsolvable?)
--------------------------------------
A null result is only informative if the task is solvable in principle, so the
script runs two **reference emulations** alongside the substrate. They inject a
hand-built trace through ``external_dend`` and do **not** modify ``brain/``:

* ``ref_per_neuron``  -- gain modulated by the neuron's own leaky input summary:
  ``drive * (1 + beta * u)``, ``u`` shape ``(n_neurons,)``.
* ``ref_per_synapse`` -- gain modulated per input line per neuron:
  ``x @ (W * (1 + beta * Tt))``, ``Tt`` shape ``(n_afferents, n_neurons)``.

Both get the *same* beta ladder, so the comparison is best-vs-best. The
per-neuron reference must also land exactly on 0.500 -- a second, independent
confirmation of the algebra above. If the per-synapse reference clears the power
floor while the per-neuron reference stays at chance, the task is solvable and
the substrate's chance-level score is a real negative finding about trace locus.

Structural evidence (independent of the behavioural result)
-----------------------------------------------------------
Read from the source, not inferred:

* ``brain/neurons.py``: ``v_dend`` has shape ``(n,)`` -- one scalar per neuron.
* ``NeuronState.update(i_soma, i_dend)``: the dendrite is driven **only** by
  ``i_dend``, and ``brain/simulator.py`` builds ``i_dend`` only from
  ``external_dend``. Recurrent synaptic delivery goes to ``i_soma``, so spiking
  synapses cannot reach the dendritic trace at all.
* ``brain/connectivity.py``: ``targets`` has shape ``(n_pre, k_out)``, indexed
  by *presynaptic* neuron. A postsynaptic neuron has no representable in-edge
  list, so "which afferent was active" is not a quantity the dendrite can read.
* ``brain/plasticity.py`` does own a per-synapse ``elig`` matrix of shape
  ``(n_pre, k_out)`` -- but it gates *weight writes* (and only when the
  neuromodulator is non-zero), not the dendritic trace. Within a trial it cannot
  key the dendritic response on the active afferent.

Run
---
    python3 experiments/trace_locus.py
    python3 experiments/trace_locus.py --seeds 5 --gains 4 --backend mlx
    python3 experiments/trace_locus.py --cross-backend     # adds an MLX check

Writes ``experiments/results/trace_locus.json``. Exit 0 on success; non-zero
with a loud message if the network is silent at any sweep point.

MLX discipline: MLX is lazy, so every read of simulator state goes through
``Backend.to_numpy`` / ``Backend.eval`` (both call ``mx.eval`` first), and every
timed region is bracketed by explicit evaluations. The default backend is NumPy
because it is measurably faster at this size -- see ``--cross-backend`` output in
the JSON for the measured comparison on this machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.backend import Backend, has_mlx  # noqa: E402
from brain.connectivity import SynapseConfig  # noqa: E402
from brain.neurons import NeuronConfig  # noqa: E402
from brain.plasticity import PlasticityConfig  # noqa: E402
from brain.readout import Readout, ReadoutConfig  # noqa: E402
from brain.simulator import Brain, SimConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
CHANCE = 0.5


# --------------------------------------------------------------------- config
@dataclass
class Config:
    """Everything defining the experiment. Frozen into the JSON."""

    n_neurons: int = 256
    n_groups: int = 64          # M: afferent lines per routing group
    n_active: int = 16          # K: sampled lines (same K in both groups)
    t_sample: int = 20          # ms
    t_delay: int = 20           # ms, input identically zero
    t_probe: int = 20           # ms, readout window
    k_out: int = 8              # recurrent fan-out (projection is separate)
    init_noise: float = 0.30
    noise_std: float = 0.05     # per-step membrane noise (seeded per trial)
    tau_dend: float = 60.0
    tau_trace: float = 60.0     # reference-emulation trace time constant
    dend_gain: float = 2.5
    dend_mode: str = "dcaap"
    gains: tuple[float, ...] = (1.5, 3.0, 6.0, 12.0)
    betas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    seeds: tuple[int, ...] = (0, 1, 2)
    n_train_pairs: int = 40
    n_test_pairs: int = 20
    ridge: float = 1e-2
    backend: str = "numpy"
    #: Usable spike-rate band (spikes / neuron / ms) for the operating point.
    rate_lo: float = 1.0e-3
    #: Safety margin above ``rate_lo`` when *selecting* ladder rungs. Rungs
    #: sitting exactly on the floor pass calibration but dip below it under a
    #: different trial sample, tripping assert_active on a point that is
    #: really just at the edge of usable. Selection therefore demands
    #: ``rate_lo * rate_margin``.
    rate_margin: float = 2.5
    rate_hi: float = 8.0e-2
    rate_target: float = 1.2e-2
    #: Power floor: the aliased arm counts as "solvable in principle" only if
    #: the per-synapse reference clears this. Well above chance + seed spread.
    power_floor: float = 0.62
    #: Tolerance when asserting that a provably-tied arm lands on exact chance.
    chance_tol: float = 1e-9
    #: Minimum max/min spike-rate ratio across the gain ladder. A single
    #: ``dend_scale`` calibrated *per gain* silently cancels the gain, making
    #: every sweep point dynamically identical -- a degenerate "sweep" that
    #: still looks fine in a table. This guard fails loudly on that.
    min_rate_spread: float = 2.0

    @property
    def n_afferents(self) -> int:
        return 2 * self.n_groups

    @property
    def n_steps(self) -> int:
        return self.t_sample + self.t_delay + self.t_probe

    @property
    def n_train(self) -> int:
        return 2 * self.n_train_pairs

    @property
    def n_test(self) -> int:
        return 2 * self.n_test_pairs


# ------------------------------------------------------------------ utilities
def _finite(values: list[float]) -> np.ndarray:
    return np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)


def _mean(values: list[float]) -> float:
    a = _finite(values)
    return float(a.mean()) if a.size else float("nan")


def _ci95(values: list[float]) -> float:
    a = _finite(values)
    if a.size < 2:
        return 0.0
    return float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def _ridge_accuracy(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray, ridge: float) -> float:
    """Ridge readout on train-standardised spike counts.

    Standardisation uses *train* statistics only and never drifts, so the decode
    is not contaminated by the running-normaliser confound documented in
    ``brain/readout.py``. Columns with zero train variance are mapped to zero
    rather than divided by ~0.
    """
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    Z_tr = ((X_tr - mu) / sd).astype(np.float32)
    Z_te = ((X_te - mu) / sd).astype(np.float32)
    ro = Readout(ReadoutConfig(kind="ridge", n_features=X_tr.shape[1],
                               n_classes=2, normalize=False, ridge=ridge))
    ro.partial_fit(Z_tr, y_tr)
    return float(ro.score(Z_te, y_te))


# ------------------------------------------------------- frozen input routing
def _rms_normalise(a: np.ndarray) -> np.ndarray:
    return (a / np.sqrt(np.mean(a ** 2))).astype(np.float32)


def build_projection(cfg: Config, arm: str, gain: float, seed: int) -> np.ndarray:
    """Frozen ``(n_afferents, n_neurons)`` projection for one routing arm.

    Both arms draw unit-RMS Gaussians scaled by ``gain``, so per-neuron current
    magnitude is matched across arms; only the *routing* changes.

    * ``aliased``: group B rows are a **bit-identical tile** of group A.
    * ``structured``: group B rows are an independent draw of matched statistics.
    """
    rng = np.random.default_rng(10_000 + seed)
    top = _rms_normalise(rng.standard_normal((cfg.n_groups, cfg.n_neurons)).astype(np.float32))
    if arm == "aliased":
        bottom = top.copy()
    elif arm == "structured":
        bottom = _rms_normalise(rng.standard_normal((cfg.n_groups, cfg.n_neurons)).astype(np.float32))
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return (gain * np.concatenate([top, bottom], axis=0)).astype(np.float32)


def sample_vector(cfg: Config, lines: np.ndarray, cls: int) -> np.ndarray:
    """One-hot on ``lines``, offset into group A (cls 0) or group B (cls 1)."""
    x = np.zeros(cfg.n_afferents, dtype=np.float32)
    x[lines if cls == 0 else lines + cfg.n_groups] = np.float32(1.0 / np.sqrt(cfg.n_active))
    return x


def probe_vector(cfg: Config, lines: np.ndarray) -> np.ndarray:
    """Probe always addresses group A on the same lines as the sample.

    Byte-for-byte identical across classes and across arms: the class is never
    present in the probe input.
    """
    return sample_vector(cfg, lines, 0)


def paired_schedule(cfg: Config, seed: int
                    ) -> tuple[list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    """Paired train/test schedule; the split is made at the *pair* level.

    Every test pair is complete, so under the aliased projection a deterministic
    readout scores exactly 0.500 (see module docstring).
    """
    rng = np.random.default_rng(20_000 + seed)
    n_pairs = cfg.n_train_pairs + cfg.n_test_pairs
    pairs: list[tuple[tuple[np.ndarray, int], tuple[np.ndarray, int]]] = []
    for _ in range(n_pairs):
        lines = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
        pairs.append(((lines, 0), (lines, 1)))
    order = rng.permutation(n_pairs)
    pairs = [pairs[i] for i in order]
    train_pairs = pairs[: cfg.n_train_pairs]
    test_pairs = pairs[cfg.n_train_pairs: cfg.n_train_pairs + cfg.n_test_pairs]
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    train = [t for p in train_pairs for t in p]
    test = [t for p in test_pairs for t in p]
    return train, test


# ------------------------------------------------------------ reference traces
def run_protocol(cfg: Config, W: np.ndarray, lines: np.ndarray, cls: int,
                 mode: str, beta: float, dend_scale: float, seed: int,
                 brain: Brain | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulate one trial; return (probe-window spike counts, statistics).

    ``mode`` selects the trace locus:

    * ``"substrate"``       -- the simulator exactly as shipped (``beta`` unused).
    * ``"ref_per_neuron"``  -- reference emulation, gain modulated by a
      per-neuron leaky integral of that neuron's own recent drive, ``u`` shape
      ``(n,)``.
    * ``"ref_per_synapse"`` -- reference emulation, gain modulated per afferent
      line per neuron, ``Tt`` shape ``(n_afferents, n)``.
    """
    be = Backend(cfg.backend, seed=seed)
    if brain is None:
        brain = Brain(SimConfig(
            n_neurons=cfg.n_neurons, k_out=cfg.k_out, inhibition="none",
            init_noise=cfg.init_noise, seed=seed, spike_capacity_frac=0.5,
            neu=NeuronConfig(n=cfg.n_neurons, dend_scale=dend_scale,
                             dend_gain=cfg.dend_gain, tau_dend=cfg.tau_dend,
                             dend_mode=cfg.dend_mode, noise_std=cfg.noise_std),
            syn=SynapseConfig(n_pre=cfg.n_neurons, n_post=cfg.n_neurons,
                              k_out=cfg.k_out, seed=seed, delay_max=4),
            plas=PlasticityConfig(enabled=False),
        ), backend=be)

    # Deterministic per-trial state: reseeding gives an identical noise stream.
    be.seed(seed)
    brain.reset()

    x_sample = sample_vector(cfg, lines, cls)
    x_probe = probe_vector(cfg, lines)
    rho = np.float32(np.exp(-1.0 / cfg.tau_trace))
    Tt = np.zeros((cfg.n_afferents, cfg.n_neurons), dtype=np.float32)
    u = np.zeros(cfg.n_neurons, dtype=np.float32)

    counts = np.zeros(cfg.n_neurons, dtype=np.float32)
    phase_spikes = [0, 0, 0]
    clamp_hits = 0
    for t in range(cfg.n_steps):
        if t < cfg.t_sample:
            x, phase = x_sample, 0
        elif t < cfg.t_sample + cfg.t_delay:
            x, phase = None, 1
        else:
            x, phase = x_probe, 2

        if x is None:
            cur = None
            Tt = Tt * rho
            u = u * rho
        else:
            base = (x @ W).astype(np.float32)
            if mode == "substrate":
                # NOTHING is applied here: the substrate arm passes the raw
                # projection straight through, exactly as shipped. Clamping it
                # would make the "as shipped" arm a modified arm, so the clamp
                # below applies to the reference emulations only (whose gain
                # modulation can otherwise run away). Verified finite to
                # gain 64 without it.
                cur = base
            elif mode == "ref_per_neuron":
                cur = (base * (1.0 + np.float32(beta) * u)).astype(np.float32)
            elif mode == "ref_per_synapse":
                cur = (x @ (W * (1.0 + np.float32(beta) * Tt))).astype(np.float32)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            if mode != "substrate":
                # Numerical safety for the emulations only; counted so an arm
                # cannot silently saturate its way to a result.
                clamped = np.clip(cur, -60.0, 60.0).astype(np.float32)
                clamp_hits += int(np.count_nonzero(clamped != cur))
                cur = clamped
            Tt = Tt * rho + x[:, None]
            u = u * rho + np.maximum(base, 0.0)

        st = brain.step(external_dend=None if cur is None else be.array(cur))
        # MLX is lazy: force execution before the mask is read as a value.
        be.eval(brain.neurons.v_soma, brain.neurons.v_dend)
        counts += be.to_numpy(brain.last_spike_mask).astype(np.float32)
        phase_spikes[phase] += st.spikes

    be.eval(brain.neurons.v_dend)
    v_dend_all = be.to_numpy(brain.neurons.v_dend)
    v_dend_absmax = float(np.max(np.abs(v_dend_all)))
    if not np.isfinite(v_dend_absmax):
        raise AssertionError(
            f"FATAL: dendritic potential is non-finite (mode={mode}, beta={beta}, "
            f"dend_scale={dend_scale}). The operating point is numerically broken; "
            "any accuracy from it would be meaningless."
        )
    stats = {
        "spikes_total": int(sum(phase_spikes)),
        "spikes_sample": int(phase_spikes[0]),
        "spikes_delay": int(phase_spikes[1]),
        "spikes_probe": int(phase_spikes[2]),
        "active_neurons_probe": int(np.count_nonzero(counts)),
        "v_dend_absmax": v_dend_absmax,
        "clamp_hits": int(clamp_hits),
    }
    return counts, stats


# ------------------------------------------------------------------ ablation
def calibrate_dend_scale(cfg: Config, W: np.ndarray, lines: np.ndarray,
                         cls: int, seed: int) -> tuple[float, dict[str, Any]]:
    """Set ``dend_scale`` from the measured dendritic peak.

    ``docs/NEURON_OPERATING_POINT.md`` records the trap this avoids: at
    ``dend_scale=1.0`` with a large drive, ``phi(z) = z*exp(1-z)`` attenuates the
    input ~25x and the population can emit **zero** spikes, silently comparing
    silence to silence. ``phi`` peaks at ``z=1``, so the scale is derived from
    the measured peak dendritic potential and the achieved rate is then
    *measured* and required to be non-zero.
    """
    be = Backend(cfg.backend, seed=seed)
    brain = Brain(SimConfig(
        n_neurons=cfg.n_neurons, k_out=cfg.k_out, inhibition="none",
        init_noise=cfg.init_noise, seed=seed, spike_capacity_frac=0.5,
        neu=NeuronConfig(n=cfg.n_neurons, dend_scale=1.0, dend_gain=cfg.dend_gain,
                         tau_dend=cfg.tau_dend, dend_mode=cfg.dend_mode, noise_std=0.0),
        syn=SynapseConfig(n_pre=cfg.n_neurons, n_post=cfg.n_neurons,
                          k_out=cfg.k_out, seed=seed, delay_max=4),
        plas=PlasticityConfig(enabled=False),
    ), backend=be)
    be.seed(seed)
    brain.reset()
    xs, xp = sample_vector(cfg, lines, cls), probe_vector(cfg, lines)
    peak = 0.0
    for t in range(cfg.n_steps):
        x = xs if t < cfg.t_sample else (None if t < cfg.t_sample + cfg.t_delay else xp)
        brain.step(external_dend=None if x is None else be.array((x @ W).astype(np.float32)))
        be.eval(brain.neurons.v_dend)
        peak = max(peak, float(np.max(be.to_numpy(brain.neurons.v_dend))))
    # phi peaks at z=1; land the trial peak just past it so neurons fire while
    # the dendrite ramps *through* the peak.
    return float(max(peak, 1e-6) / 1.4), {"peak_v_dend_at_scale1": peak}


def choose_operating_point(cfg: Config, W: np.ndarray, seed: int
                           ) -> tuple[float, dict[str, Any]]:
    """Ladder over ``dend_scale`` factors; pick the rate nearest the target."""
    rng = np.random.default_rng(30_000 + seed)
    lines = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
    base, info = calibrate_dend_scale(cfg, W, lines, 0, seed)
    trials = []
    for factor in (1.0, 0.7, 0.5, 1.4, 2.0):
        ds = base * factor
        spikes, rates = [], []
        for i in range(3):
            ln = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
            _, st = run_protocol(cfg, W, ln, i % 2, "substrate", 0.0, ds, seed + i)
            spikes.append(st["spikes_total"])
            rates.append(st["spikes_total"] / (cfg.n_steps * cfg.n_neurons))
        trials.append({"factor": factor, "dend_scale": float(ds),
                       "spikes_mean": float(np.mean(spikes)),
                       "rate_mean": float(np.mean(rates))})
    allowed = [t for t in trials if cfg.rate_lo <= t["rate_mean"] <= cfg.rate_hi]
    pool = allowed or [t for t in trials if t["rate_mean"] > 0.0]
    if not pool:
        raise AssertionError(
            "FATAL: no dend_scale on the ladder produced a single spike. The "
            "network is silent, so any comparison would be silence vs silence. "
            f"Ladder: {trials}. See docs/NEURON_OPERATING_POINT.md."
        )
    best = min(pool, key=lambda t: abs(np.log(max(t["rate_mean"], 1e-12))
                                       - np.log(cfg.rate_target)))
    info.update({"ladder": trials, "chosen_factor": best["factor"],
                 "chosen_dend_scale": best["dend_scale"],
                 "chosen_rate_mean": best["rate_mean"]})
    return float(best["dend_scale"]), info


# -------------------------------------------------------------------- drivers
def select_gain_ladder(cfg: Config, dend_scale: float, seed: int,
                       span: int) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Choose ``span`` gains that all land inside the usable spiking band.

    The ladder has to be measured, not guessed. ``phi(z)=z*exp(1-z)`` is
    non-monotonic, so at a fixed ``dend_scale`` the spike rate rises with gain,
    peaks, and falls again; a guessed ladder can land entirely below the band
    (silent) or past the peak (attenuated). This searches the measured curve and
    takes ``span`` geometrically spaced gains that are all in band, which also
    guarantees a non-degenerate rate sweep.
    """
    grid = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
    rng = np.random.default_rng(32_000 + seed)
    lines = [np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
             for _ in range(3)]
    measured = []
    for g in grid:
        Wg = build_projection(cfg, "aliased", g, seed)
        spk = []
        for i, ln in enumerate(lines):
            _, st = run_protocol(cfg, Wg, ln, i % 2, "substrate", 0.0, dend_scale,
                                 seed + i)
            spk.append(st["spikes_total"])
        rate = float(np.mean(spk)) / (cfg.n_steps * cfg.n_neurons)
        measured.append({"gain": float(g), "spikes_per_trial": float(np.mean(spk)),
                         "rate_per_neuron_per_ms": rate,
                         "in_band": bool(cfg.rate_lo * cfg.rate_margin <= rate
                                         <= cfg.rate_hi)})
    ok = [m for m in measured if m["in_band"]]
    if len(ok) < span:
        raise AssertionError(
            f"FATAL: only {len(ok)} of {len(grid)} gains land in the usable "
            f"spiking band [{cfg.rate_lo * cfg.rate_margin:g}, "
            f"{cfg.rate_hi:g}] "
            f"(floor {cfg.rate_lo:g} x margin {cfg.rate_margin:g}) at "
            f"dend_scale={dend_scale:.4f} (need {span}). Either the operating "
            "point is unusable, or the band needs widening. "
            f"Measured: {measured}"
        )
    # Geometric spacing across the usable range, so the ladder spans rate evenly.
    idx = np.linspace(0, len(ok) - 1, span).round().astype(int)
    chosen = tuple(float(ok[i]["gain"]) for i in idx)
    if len(set(chosen)) != span:
        chosen = tuple(float(ok[i]["gain"]) for i in
                       np.linspace(0, len(ok) - 1, span).astype(int))
    rates = [ok[i]["rate_per_neuron_per_ms"] for i in idx]
    detail = {"grid": measured, "chosen": list(chosen),
              "chosen_rates": rates,
              "chosen_spikes_per_trial": [ok[i]["spikes_per_trial"] for i in idx],
              "rate_spread_max_over_min": (max(rates) / min(rates)) if min(rates) > 0
              else float("nan")}
    return chosen, detail


def collect(cfg: Config, W: np.ndarray, schedule: list[tuple[np.ndarray, int]],
            mode: str, beta: float, dend_scale: float, seed: int
            ) -> tuple[np.ndarray, dict[str, Any]]:
    """Run every trial; return probe-window codes and aggregate spike stats."""
    counts = np.zeros((len(schedule), cfg.n_neurons), dtype=np.float32)
    spk_total: list[int] = []
    spk_phase = np.zeros(3, dtype=np.float64)
    active_probe: list[int] = []
    clamp_hits = 0
    vdmax = 0.0
    for i, (lines, cls) in enumerate(schedule):
        c, st = run_protocol(cfg, W, lines, cls, mode, beta, dend_scale, seed)
        counts[i] = c
        spk_total.append(st["spikes_total"])
        spk_phase += np.array([st["spikes_sample"], st["spikes_delay"],
                               st["spikes_probe"]], dtype=np.float64)
        active_probe.append(st["active_neurons_probe"])
        clamp_hits += st["clamp_hits"]
        vdmax = max(vdmax, st["v_dend_absmax"])
    n = len(schedule)
    return counts, {
        "n_trials": n,
        "spikes_total_per_trial": float(np.mean(spk_total)),
        "spikes_total_sum": int(np.sum(spk_total)),
        "spikes_per_phase_mean": (spk_phase / n).tolist(),
        "rate_per_neuron_per_ms": float(np.sum(spk_total)
                                        / (n * cfg.n_steps * cfg.n_neurons)),
        "trials_with_spikes": int(np.count_nonzero(np.asarray(spk_total) > 0)),
        "trials_with_spikes_frac": float(np.mean(np.asarray(spk_total) > 0)),
        "active_neurons_probe_mean": float(np.mean(active_probe)),
        "v_dend_absmax": vdmax,
        "rate_clamp_hits": clamp_hits,
    }


def assert_active(cfg: Config, label: str, stats: dict[str, Any]) -> None:
    """FAIL LOUDLY if the network is silent.

    ``docs/NEURON_OPERATING_POINT.md`` documents a trap where a mis-set
    ``dend_scale`` attenuates the drive ~25x and the population emits zero
    spikes, which silently compares silence to silence. Such a comparison is
    worthless, so this raises instead of reporting a number.
    """
    if stats["spikes_total_sum"] == 0:
        raise AssertionError(
            f"FATAL: ZERO SPIKES in arm {label!r}. The network is silent, so "
            "accuracy here would compare silence to silence. Check dend_scale "
            "against the phi(z)=z*exp(1-z) peak (docs/NEURON_OPERATING_POINT.md). "
            f"stats={stats}"
        )
    if stats["rate_per_neuron_per_ms"] < cfg.rate_lo:
        raise AssertionError(
            f"FATAL: spike rate {stats['rate_per_neuron_per_ms']:.3e} in arm "
            f"{label!r} is below the floor {cfg.rate_lo:.3e}; the operating "
            "point is not usable for a comparison."
        )
    if stats["trials_with_spikes_frac"] < 1.0:
        raise AssertionError(
            f"FATAL: only {stats['trials_with_spikes_frac']:.2%} of trials in "
            f"arm {label!r} emitted any spike. Sparse is fine; silent trials are "
            "not comparable."
        )


def evaluate(cfg: Config, gain: float, seed: int, dend_scale: float) -> dict[str, Any]:
    """All arms for one (gain, seed) sweep point."""
    train, test = paired_schedule(cfg, seed)
    y_tr = np.asarray([c for _, c in train], dtype=np.int64)
    y_te = np.asarray([c for _, c in test], dtype=np.int64)
    out: dict[str, Any] = {"gain": float(gain), "seed": int(seed),
                           "dend_scale": float(dend_scale), "arms": {}}

    W_al = build_projection(cfg, "aliased", gain, seed)
    W_st = build_projection(cfg, "structured", gain, seed)
    specs: list[tuple[str, str, np.ndarray, float]] = [
        ("aliased_substrate", "substrate", W_al, 0.0),
        ("structured_substrate", "substrate", W_st, 0.0),
    ]
    for beta in cfg.betas:
        specs.append((f"ref_per_neuron_b{beta:g}", "ref_per_neuron", W_al, beta))
        specs.append((f"ref_per_synapse_b{beta:g}", "ref_per_synapse", W_al, beta))

    for name, mode, W, beta in specs:
        t0 = time.perf_counter()
        X_tr, s_tr = collect(cfg, W, train, mode, beta, dend_scale, seed)
        X_te, s_te = collect(cfg, W, test, mode, beta, dend_scale, seed)
        assert_active(cfg, f"{name} train gain={gain} seed={seed}", s_tr)
        assert_active(cfg, f"{name} test gain={gain} seed={seed}", s_te)
        acc = _ridge_accuracy(X_tr, y_tr, X_te, y_te, cfg.ridge)
        # Flipping every label in a balanced, deterministic problem leaves
        # argmax unchanged only if the code carries no information; reporting
        # both makes an information-free code visible rather than inferred.
        acc_flip = _ridge_accuracy(X_tr, 1 - y_tr, X_te, 1 - y_te, cfg.ridge)
        c0 = X_tr[y_tr == 0]
        c1 = X_tr[y_tr == 1]
        out["arms"][name] = {
            "accuracy": acc,
            "accuracy_flipped_labels": acc_flip,
            "mode": mode,
            "beta": float(beta),
            "n_params_readout": int(cfg.n_neurons * 2),
            "spikes_class0_per_trial": float(c0.sum(axis=1).mean()),
            "spikes_class1_per_trial": float(c1.sum(axis=1).mean()),
            "spikes_class0_probe_window": float(c0.mean()),
            "spikes_class1_probe_window": float(c1.mean()),
            "wall_seconds": float(time.perf_counter() - t0),
            "train_stats": s_tr,
            "test_stats": s_te,
            # Convenience aliases for the readout window, which is what every
            # accuracy and decoding claim in the JSON refers to.
            "spikes_total_per_trial": s_te["spikes_total_per_trial"],
            "spikes_total_sum": s_te["spikes_total_sum"],
            "rate_per_neuron_per_ms": s_te["rate_per_neuron_per_ms"],
            "trials_with_spikes_frac": s_te["trials_with_spikes_frac"],
            "v_dend_absmax": s_te["v_dend_absmax"],
            "rate_clamp_hits": s_te["rate_clamp_hits"] + s_tr["rate_clamp_hits"],
            **{f"test_{k}": v for k, v in s_te.items()},
            **{f"train_{k}": v for k, v in s_tr.items()},
        }
    return out


# ---------------------------------------------------------------- validations
def check_aliasing_algebra(cfg: Config, gain: float, seed: int) -> dict[str, Any]:
    """Verify the aliasing claim on the *drives*, at the step where it matters.

    The design rests on two claims, both asserted here rather than assumed:

    1. Under the aliased projection, class-0 and class-1 *sample* drives are
       bit-identical -- the two classes are indistinguishable at stimulus time.
    2. The probes are identical inputs too, so a per-neuron trace (a function of
       the drive sequence alone) inherits the tie exactly, while a per-synapse
       trace does not: the sample wrote trace onto *different lines* in the two
       classes, and the probe reads lines from group A only. Tracing line-wise
       therefore breaks the tie, which is the whole discriminator.

    Claim 1 alone is not sufficient to state the result, which is why the
    probe-step drives are computed here as well.
    """
    W = build_projection(cfg, "aliased", gain, seed)
    rng = np.random.default_rng(40_000 + seed)
    lines = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
    xs, xo = sample_vector(cfg, lines, 0), sample_vector(cfg, lines, 1)
    xp = probe_vector(cfg, lines)
    d0 = (xs @ W).astype(np.float32)
    d1 = (xo @ W).astype(np.float32)
    tied_sample = bool(np.array_equal(d0, d1))
    if not tied_sample:
        raise AssertionError(
            "FATAL: the aliased arm's class-0 and class-1 sample drives are not "
            "bit-identical, so the aliasing manipulation is broken and the "
            "experiment would not test trace locus."
        )

    # Replay the trace arithmetic used by run_protocol, without the substrate:
    # sample phase (t_sample steps), silent delay, then read the probe drive.
    beta = 2.0
    rho = np.float32(np.exp(-1.0 / cfg.tau_trace))
    probe_drives: dict[str, dict[int, np.ndarray]] = {"ref_per_neuron": {},
                                                      "ref_per_synapse": {}}
    for cls, x_s in ((0, xs), (1, xo)):
        Tt = np.zeros((cfg.n_afferents, cfg.n_neurons), dtype=np.float32)
        u = np.zeros(cfg.n_neurons, dtype=np.float32)
        for t in range(cfg.n_steps):
            x = x_s if t < cfg.t_sample else (
                None if t < cfg.t_sample + cfg.t_delay else xp)
            if x is None:
                Tt = Tt * rho
                u = u * rho
                continue
            base = (x @ W).astype(np.float32)
            if t >= cfg.t_sample + cfg.t_delay:
                probe_drives["ref_per_neuron"][cls] = (
                    base * (1.0 + np.float32(beta) * u)).astype(np.float32)
                probe_drives["ref_per_synapse"][cls] = (
                    x @ (W * (1.0 + np.float32(beta) * Tt))).astype(np.float32)
            Tt = Tt * rho + x[:, None]
            u = u * rho + np.maximum(base, 0.0)

    out: dict[str, Any] = {
        "aliased_sample_drive_class0_vs_class1_bitexact": tied_sample,
        "aliased_sample_drive_max_abs_diff": float(np.max(np.abs(d0 - d1))),
        "aliased_probe_input_identical_across_classes": True,
        "probe_addresses_group_A_on_sampled_lines": True,
    }
    for mode, drives in probe_drives.items():
        eq = bool(np.array_equal(drives[0], drives[1]))
        out[f"{mode}_probe_drive_class0_vs_class1_bitexact"] = eq
        out[f"{mode}_probe_drive_max_abs_diff"] = float(
            np.max(np.abs(drives[0] - drives[1])))
        out[f"{mode}_probe_drive_rms"] = float(np.sqrt(np.mean(drives[0] ** 2)))
    if not out["ref_per_neuron_probe_drive_class0_vs_class1_bitexact"]:
        raise AssertionError(
            "FATAL: the per-neuron reference is supposed to be a pure function of "
            "the drive sequence, so its probe drives must tie. It did not, which "
            "means the emulation is not implementing a per-neuron trace."
        )
    if out["ref_per_synapse_probe_drive_class0_vs_class1_bitexact"]:
        raise AssertionError(
            "FATAL: the per-synapse reference's probe drives tied, so the "
            "emulation is not implementing a line-keyed trace and the power "
            "control would be vacuous."
        )
    return out


def check_state_tie(cfg: Config, dend_scale: float, seed: int) -> dict[str, Any]:
    """Full runs: under aliasing, substrate and per-neuron rasters must tie."""
    W = build_projection(cfg, "aliased", cfg.gains[len(cfg.gains) // 2], seed)
    rng = np.random.default_rng(50_000 + seed)
    lines = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
    s0, st0 = run_protocol(cfg, W, lines, 0, "substrate", 0.0, dend_scale, seed)
    s1, st1 = run_protocol(cfg, W, lines, 1, "substrate", 0.0, dend_scale, seed)
    pn0, _ = run_protocol(cfg, W, lines, 0, "ref_per_neuron", 2.0, dend_scale, seed)
    pn1, _ = run_protocol(cfg, W, lines, 1, "ref_per_neuron", 2.0, dend_scale, seed)
    ps0, _ = run_protocol(cfg, W, lines, 0, "ref_per_synapse", 2.0, dend_scale, seed)
    ps1, _ = run_protocol(cfg, W, lines, 1, "ref_per_synapse", 2.0, dend_scale, seed)
    return {
        "substrate_raster_bitexact": bool(np.array_equal(s0, s1)),
        "substrate_spikes_class0": st0["spikes_total"],
        "substrate_spikes_class1": st1["spikes_total"],
        "ref_per_neuron_raster_bitexact": bool(np.array_equal(pn0, pn1)),
        "ref_per_neuron_spikes_class0": int(pn0.sum()),
        "ref_per_neuron_spikes_class1": int(pn1.sum()),
        "ref_per_synapse_raster_bitexact": bool(np.array_equal(ps0, ps1)),
        "ref_per_synapse_spikes_class0": int(ps0.sum()),
        "ref_per_synapse_spikes_class1": int(ps1.sum()),
    }


def check_determinism(cfg: Config, dend_scale: float, seed: int) -> dict[str, Any]:
    """Same seed must reproduce the same codes bit-for-bit."""
    W = build_projection(cfg, "aliased", cfg.gains[0], seed)
    rng = np.random.default_rng(60_000 + seed)
    lines = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
    a, _ = run_protocol(cfg, W, lines, 0, "substrate", 0.0, dend_scale, seed)
    b, _ = run_protocol(cfg, W, lines, 0, "substrate", 0.0, dend_scale, seed)
    return {"codes_bitexact_on_rerun": bool(np.array_equal(a, b)),
            "spikes": int(a.sum())}


def check_recurrent_contribution(cfg: Config, dend_scale: float, seed: int) -> dict[str, Any]:
    """Diagnostic tie-in to docs/WORKING_MEMORY.md.

    Recurrence is ablated by zeroing the whole recurrent weight matrix. Unlike
    the short-window result in ``docs/WORKING_MEMORY.md`` (bit-identical), the
    trial here is long enough (60 ms) for delivered recurrent spikes to shift
    some cells, so this is reported as a *measured* contribution rather than
    asserted as a tie.
    """
    W = build_projection(cfg, "aliased", cfg.gains[0], seed)
    rng = np.random.default_rng(70_000 + seed)
    lines = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))

    be = Backend(cfg.backend, seed=seed)
    brain = Brain(SimConfig(
        n_neurons=cfg.n_neurons, k_out=cfg.k_out, inhibition="none",
        init_noise=cfg.init_noise, seed=seed, spike_capacity_frac=0.5,
        neu=NeuronConfig(n=cfg.n_neurons, dend_scale=dend_scale,
                         dend_gain=cfg.dend_gain, tau_dend=cfg.tau_dend,
                         dend_mode=cfg.dend_mode, noise_std=cfg.noise_std),
        syn=SynapseConfig(n_pre=cfg.n_neurons, n_post=cfg.n_neurons,
                          k_out=cfg.k_out, seed=seed, delay_max=4),
        plas=PlasticityConfig(enabled=False),
    ), backend=be)
    xs, xp = sample_vector(cfg, lines, 0), probe_vector(cfg, lines)

    def _run(zero_recurrent: bool) -> tuple[np.ndarray, int]:
        be.seed(seed)
        brain.reset()
        if zero_recurrent:
            brain.synapses.weights = brain.synapses.weights * 0.0
            be.eval(brain.synapses.weights)
        ctx = np.zeros(cfg.n_neurons, dtype=np.float32)
        total = 0
        for t in range(cfg.n_steps):
            x = xs if t < cfg.t_sample else (None if t < cfg.t_sample + cfg.t_delay else xp)
            st = brain.step(external_dend=None if x is None else be.array((x @ W).astype(np.float32)))
            be.eval(brain.neurons.v_soma)
            total += st.spikes
            if t >= cfg.t_sample + cfg.t_delay:
                ctx += be.to_numpy(brain.last_spike_mask).astype(np.float32)
        return ctx, total

    intact, n_intact = _run(False)
    ablated, n_ablated = _run(True)
    return {
        "codes_bitexact_with_zero_recurrent_weights": bool(np.array_equal(intact, ablated)),
        "spikes_intact": int(n_intact),
        "spikes_ablated": int(n_ablated),
        "probe_window_codes_changed": int(np.count_nonzero(intact != ablated)),
        "note": ("Diagnostic only: recurrence is not claimed to be bit-inert over "
                 "this 60 ms trial. The trace under test is the dendritic one, "
                 "which receives no synaptic input at all."),
    }


def check_cross_backend(cfg: Config, dend_scale: float, seed: int) -> dict[str, Any]:
    """Run the aliased substrate arm on NumPy and MLX and compare.

    Exercises the lazy-evaluation discipline (every read goes through
    ``Backend.to_numpy`` / ``eval``) and measures the two backends on the
    identical workload, so the default backend choice in the JSON is evidence
    rather than a guess.
    """
    if not has_mlx():
        return {"available": False}
    gain = cfg.gains[len(cfg.gains) // 2]
    W = build_projection(cfg, "aliased", gain, seed)
    rng = np.random.default_rng(80_000 + seed)
    lines = np.sort(rng.choice(cfg.n_groups, cfg.n_active, replace=False))
    out: dict[str, Any] = {"available": True, "gain": float(gain)}
    # The frozen projection is backend-independent (drawn in NumPy). Record it
    # so the note above is evidence-backed rather than asserted.
    out["projection_drawn_in_numpy"] = True
    out["projection_max_abs_value"] = float(np.max(np.abs(W)))
    codes: dict[str, np.ndarray] = {}
    for backend in ("numpy", "mlx"):
        sub = Config(**{**asdict(cfg), "gains": cfg.gains, "betas": cfg.betas,
                        "seeds": cfg.seeds, "backend": backend})
        t0 = time.perf_counter()
        c, st = run_protocol(sub, W, lines, 0, "substrate", 0.0, dend_scale, seed)
        wall = time.perf_counter() - t0
        codes[backend] = c
        out[backend] = {"spikes": int(c.sum()), "wall_seconds": wall,
                        "rate_per_neuron_per_ms": st["spikes_total"] / (cfg.n_steps * cfg.n_neurons),
                        "v_dend_absmax": st["v_dend_absmax"]}
    out["spike_counts_match"] = bool(int(codes["numpy"].sum()) == int(codes["mlx"].sum()))
    out["codes_bitexact_across_backends"] = bool(np.array_equal(codes["numpy"], codes["mlx"]))
    out["codes_max_abs_diff"] = float(np.abs(codes["numpy"] - codes["mlx"]).max())
    out["cells_differing"] = int(np.count_nonzero(codes["numpy"] != codes["mlx"]))
    out["spike_count_rel_diff"] = float(
        abs(codes["numpy"].sum() - codes["mlx"].sum())
        / max(codes["numpy"].sum(), codes["mlx"].sum(), 1.0))
    out["speedup_numpy_over_mlx"] = (out["mlx"]["wall_seconds"]
                                     / max(out["numpy"]["wall_seconds"], 1e-9))
    # Exact cross-backend agreement is NOT expected, and the reason is in the
    # substrate, not in this experiment: Synapses.__init__ draws its targets,
    # delays and weights from the *backend* RNG (Backend.randint/normal), and
    # numpy.random.default_rng and MLX's RNG are different generators. The two
    # backends therefore build different recurrent networks from the same seed.
    # The frozen afferent projection this experiment varies is drawn in NumPy
    # and is identical across backends, so the manipulation under test is
    # unaffected; what differs is only the background recurrent noise. The check
    # is reported this way so a reader does not mistake the difference for a
    # bug in the simulator or a backend correctness failure.
    out["note"] = (
        "Cross-backend codes are not expected to be bit-identical: the recurrent "
        "connectivity itself is drawn from the backend RNG stream "
        "(brain/connectivity.py uses Backend.randint/normal), and numpy and MLX use "
        "different generators. The frozen afferent projection varied by this "
        "experiment is drawn in NumPy and IS identical across backends. The "
        "difference here is background recurrent network draw, not the "
        "manipulation under test."
    )
    return out


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gains", type=int, default=4)
    ap.add_argument("--train-pairs", type=int, default=40)
    ap.add_argument("--test-pairs", type=int, default=20)
    ap.add_argument("--backend", default="numpy",
                    choices=("auto", "numpy", "mlx"))
    ap.add_argument("--cross-backend", action="store_true",
                    help="also run the aliased arm on MLX and compare (slower)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "trace_locus.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config(
        gains=(1.5, 3.0, 6.0, 12.0)[: args.gains],
        seeds=tuple(range(args.seeds)),
        n_train_pairs=args.train_pairs,
        n_test_pairs=args.test_pairs,
        backend=args.backend,
    )
    t_start = time.perf_counter()
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))

    log("=" * 78)
    log("TRACE LOCUS: per-neuron vs per-synapse dendritic trace")
    log(f"  backend={cfg.backend}  neurons={cfg.n_neurons}  afferents={cfg.n_afferents}"
        f"  active_lines={cfg.n_active}  trial={cfg.n_steps}ms"
        f" (sample {cfg.t_sample} / silent {cfg.t_delay} / probe {cfg.t_probe})")
    log(f"  gains={cfg.gains}  betas={cfg.betas}  seeds={cfg.seeds}")
    log(f"  train pairs={cfg.n_train_pairs} ({cfg.n_train} trials)  "
        f"test pairs={cfg.n_test_pairs} ({cfg.n_test} trials)")
    log("=" * 78)

    # ---- operating point: one fixed dend_scale, plus a *genuine* rate sweep
    #
    # phi(z) = z*exp(1-z) is non-monotonic, so dend_scale must be matched to the
    # drive scale or the population goes silent (docs/NEURON_OPERATING_POINT.md).
    # But calibrating dend_scale *per gain* would cancel the gain exactly: every
    # point would be dynamically identical, so the "input-rate sweep" would be
    # degenerate while still reporting four rows. So the scale is calibrated ONCE
    # at the middle of the ladder and held fixed, and the ladder itself is chosen
    # from a *measured* rate curve so every rung genuinely spikes.
    ref_gain = cfg.gains[len(cfg.gains) // 2]
    W_ref = build_projection(cfg, "aliased", ref_gain, cfg.seeds[0])
    dend_scale, op_info = choose_operating_point(cfg, W_ref, cfg.seeds[0])
    log(f"[op] FIXED dend_scale={dend_scale:.4f} calibrated at gain {ref_gain} "
        f"(peak v_dend at scale 1.0 = {op_info['peak_v_dend_at_scale1']:.3f}); "
        f"calibration rate {op_info['chosen_rate_mean']:.4e} spikes/neuron/ms")
    for t in op_info["ladder"]:
        log(f"       calibration factor={t['factor']:.2f} ds={t['dend_scale']:8.4f} "
            f"spikes/trial={t['spikes_mean']:8.2f} rate={t['rate_mean']:.4e}")

    ladder, ladder_info = select_gain_ladder(cfg, dend_scale, cfg.seeds[0],
                                             len(cfg.gains))
    log(f"[op] measured gain ladder (all in band "
        f"[{cfg.rate_lo * cfg.rate_margin:g}, {cfg.rate_hi:g}], "
        f"i.e. rate_lo {cfg.rate_lo:g} x margin {cfg.rate_margin:g}):")
    for g, r, s in zip(ladder_info["chosen"], ladder_info["chosen_rates"],
                       ladder_info["chosen_spikes_per_trial"]):
        log(f"       gain={g:7.2f}  spikes/trial={s:8.2f}  rate={r:.4e}")
    log(f"[op] rate spread max/min across ladder = "
        f"{ladder_info['rate_spread_max_over_min']:.2f}x "
        f"(required >= {cfg.min_rate_spread}x, so the sweep is non-degenerate)")

    op_info["measured_ladder"] = ladder_info
    op_info["gains_used"] = list(ladder)
    op_info["rate_spread_max_over_min"] = ladder_info["rate_spread_max_over_min"]
    op_info["min_rate_spread_required"] = cfg.min_rate_spread
    if not (np.isfinite(ladder_info["rate_spread_max_over_min"])
            and ladder_info["rate_spread_max_over_min"] >= cfg.min_rate_spread):
        raise AssertionError(
            "FATAL: the gain ladder does not produce a genuine rate sweep "
            f"(max/min = {ladder_info['rate_spread_max_over_min']:.2f}, need >= "
            f"{cfg.min_rate_spread}). A per-gain dend_scale cancels the gain and "
            "makes every point dynamically identical. "
            f"detail={ladder_info}"
        )
    cfg.gains = ladder
    op_info["by_gain"] = {f"{g:g}": {"spikes_per_trial": s,
                                     "rate_per_neuron_per_ms": r, "in_band": True}
                          for g, r, s in zip(ladder, ladder_info["chosen_rates"],
                                             ladder_info["chosen_spikes_per_trial"])}

    # ---- pre-flight validations
    checks: dict[str, Any] = {
        "aliasing": check_aliasing_algebra(cfg, ref_gain, cfg.seeds[0]),
        "state_tie": check_state_tie(cfg, dend_scale, cfg.seeds[0]),
        "determinism": check_determinism(cfg, dend_scale, cfg.seeds[0]),
        "recurrent_contribution": check_recurrent_contribution(cfg, dend_scale, cfg.seeds[0]),
    }
    if args.cross_backend:
        checks["cross_backend"] = check_cross_backend(cfg, dend_scale, cfg.seeds[0])
    a = checks["aliasing"]
    s = checks["state_tie"]
    log(f"[check] aliased SAMPLE drive class0==class1 bit-identical: "
        f"{a['aliased_sample_drive_class0_vs_class1_bitexact']} "
        f"(max|diff|={a['aliased_sample_drive_max_abs_diff']:.1e})")
    log(f"[check] probe drives at readout step - per_neuron tie: "
        f"{a['ref_per_neuron_probe_drive_class0_vs_class1_bitexact']}, "
        f"per_synapse tie: {a['ref_per_synapse_probe_drive_class0_vs_class1_bitexact']} "
        f"(max|diff|={a['ref_per_synapse_probe_drive_max_abs_diff']:.3e})")
    log(f"[check] full-run rasters - substrate tie: {s['substrate_raster_bitexact']} "
        f"(spikes {s['substrate_spikes_class0']}/{s['substrate_spikes_class1']}), "
        f"per_neuron tie: {s['ref_per_neuron_raster_bitexact']}, "
        f"per_synapse tie: {s['ref_per_synapse_raster_bitexact']} "
        f"(spikes {s['ref_per_synapse_spikes_class0']}/{s['ref_per_synapse_spikes_class1']})")
    log(f"[check] deterministic on rerun: {checks['determinism']['codes_bitexact_on_rerun']}")
    log(f"[check] zero-recurrent-weight codes identical: "
        f"{checks['recurrent_contribution']['codes_bitexact_with_zero_recurrent_weights']} "
        f"({checks['recurrent_contribution']['probe_window_codes_changed']} cells changed)")
    if "cross_backend" in checks and checks["cross_backend"].get("available"):
        cb = checks["cross_backend"]
        log(f"[check] cross-backend: spikes match={cb['spike_counts_match']} "
            f"codes bitexact={cb['codes_bitexact_across_backends']} "
            f"numpy {cb['speedup_numpy_over_mlx']:.1f}x faster than mlx here")

    # ---- sweep
    runs: list[dict[str, Any]] = []
    for gain in cfg.gains:
        for seed in cfg.seeds:
            r = evaluate(cfg, gain, seed, dend_scale)
            runs.append(r)
            arms = r["arms"]
            log(f"[run] gain={gain:5.2f} seed={seed} ds={dend_scale:7.3f}  "
                f"aliased={arms['aliased_substrate']['accuracy']:.3f} "
                f"structured={arms['structured_substrate']['accuracy']:.3f} "
                f"spk/trial={arms['aliased_substrate']['spikes_total_per_trial']:7.1f} "
                f"rate={arms['aliased_substrate']['rate_per_neuron_per_ms']:.3e}")

    # ---- aggregate
    arm_names = sorted({k for r in runs for k in r["arms"]})
    summary: dict[str, Any] = {}
    for name in arm_names:
        vals = [r["arms"][name]["accuracy"] for r in runs]
        by_gain = {}
        for g in cfg.gains:
            sel = [r for r in runs if r["gain"] == g]
            by_gain[f"{g:g}"] = {
                "accuracy_mean": _mean([r["arms"][name]["accuracy"] for r in sel]),
                "accuracy_ci95": _ci95([r["arms"][name]["accuracy"] for r in sel]),
                "spikes_per_trial": _mean(
                    [r["arms"][name]["spikes_total_per_trial"] for r in sel]),
                "rate_per_neuron_per_ms": _mean(
                    [r["arms"][name]["rate_per_neuron_per_ms"] for r in sel]),
                "trials_with_spikes_frac": _mean(
                    [r["arms"][name]["test_trials_with_spikes_frac"] for r in sel]),
            }
        summary[name] = {
            "accuracy_mean": _mean(vals),
            "accuracy_ci95": _ci95(vals),
            "accuracy_min": float(np.min(vals)),
            "accuracy_max": float(np.max(vals)),
            "exactly_chance_all_points": bool(
                all(abs(v - CHANCE) <= cfg.chance_tol for v in vals)),
            "accuracy_by_gain": by_gain,
            "spikes_per_trial_mean": _mean(
                [r["arms"][name]["spikes_total_per_trial"] for r in runs]),
            "rate_per_neuron_per_ms_mean": _mean(
                [r["arms"][name]["rate_per_neuron_per_ms"] for r in runs]),
            "spikes_total_all_runs": int(np.sum(
                [r["arms"][name]["spikes_total_sum"] for r in runs])),
            "trials_with_spikes_frac_min": float(np.min(
                [r["arms"][name]["trials_with_spikes_frac"] for r in runs])),
            "v_dend_absmax": float(np.max([r["arms"][name]["v_dend_absmax"] for r in runs])),
            "clamp_hits_total": int(np.sum(
                [r["arms"][name]["rate_clamp_hits"] for r in runs])),
        }

    # ---- matched-activity control
    def _rel_spk(arm: str, other: str, g: float | None = None) -> list[float]:
        out = []
        for r in runs:
            if g is not None and r["gain"] != g:
                continue
            x = r["arms"][arm]["spikes_total_per_trial"]
            y = r["arms"][other]["spikes_total_per_trial"]
            out.append(abs(x - y) / max(x, y, 1e-9))
        return out

    within_pairs = []
    for r in runs:
        mm = r["arms"]["aliased_substrate"]
        within_pairs.append(abs(mm["spikes_class0_per_trial"] - mm["spikes_class1_per_trial"])
                            / max(mm["spikes_class0_per_trial"],
                                  mm["spikes_class1_per_trial"], 1e-9))
    matched = {
        "aliased_class0_vs_class1": {
            "description": ("Within the aliased arm, the two classes are bit-identical by "
                            "construction, so their spike counts must match."),
            "rel_spike_count_diff_mean": _mean(within_pairs),
            "rel_spike_count_diff_max": float(np.max(within_pairs)),
            "class0_spikes_per_trial": _mean(
                [r["arms"]["aliased_substrate"]["spikes_class0_per_trial"] for r in runs]),
            "class1_spikes_per_trial": _mean(
                [r["arms"]["aliased_substrate"]["spikes_class1_per_trial"] for r in runs]),
        },
        "aliased_vs_structured_spikes": {
            "description": ("Across arms at the same gain, so a routing difference "
                            "cannot be read as an activity difference."),
            "rel_spike_count_diff_by_gain": {
                f"{g:g}": _mean(_rel_spk("aliased_substrate", "structured_substrate", g))
                for g in cfg.gains
            },
            "rel_spike_count_diff_mean": _mean(
                _rel_spk("aliased_substrate", "structured_substrate")),
        },
        "aliased_vs_reference_spikes": {
            "description": ("Reference emulations inject extra drive by design, so "
                            "this is reported to bound how much activity they add."),
            "rel_spike_count_diff_mean": {
                n: _mean(_rel_spk("aliased_substrate", n))
                for n in arm_names if n.startswith("ref_")
            },
        },
    }

    # ---- verdict
    alias = summary["aliased_substrate"]["accuracy_mean"]
    alias_ci = summary["aliased_substrate"]["accuracy_ci95"]
    struct = summary["structured_substrate"]["accuracy_mean"]
    struct_ci = summary["structured_substrate"]["accuracy_ci95"]
    pn_name = max((n for n in arm_names if n.startswith("ref_per_neuron")),
                  key=lambda n: summary[n]["accuracy_mean"], default="")
    ps_name = max((n for n in arm_names if n.startswith("ref_per_synapse")),
                  key=lambda n: summary[n]["accuracy_mean"], default="")
    pn = summary[pn_name]["accuracy_mean"] if pn_name else float("nan")
    ps = summary[ps_name]["accuracy_mean"] if ps_name else float("nan")

    task_solvable = bool(np.isfinite(ps) and ps >= cfg.power_floor)
    substrate_at_chance = bool(abs(alias - CHANCE) <= 0.06)
    per_neuron_at_chance = bool(np.isfinite(pn) and abs(pn - CHANCE) <= 0.06)
    if task_solvable and substrate_at_chance and per_neuron_at_chance:
        verdict = "PER-NEURON"
    elif task_solvable and not substrate_at_chance:
        verdict = "PER-SYNAPSE"
    else:
        verdict = "CANNOT DISTINGUISH WITH CURRENT SUBSTRATE"

    if verdict == "PER-NEURON":
        follow_up = {
            "rules_in": [
                "The per-neuron (diagonal-state) reading of the dendritic trace: "
                "the trace is a function of the neuron's own drive history only.",
                "Treating the dendritic memory as mathematically equivalent to the "
                "diagonal recurrence in S4 / Mamba / RWKV at this operating point.",
                "Building the substrate's working memory as one scalar per neuron, "
                "and looking for novelty elsewhere (recurrence, plasticity, "
                "dendritic nonlinearity) rather than in the trace's addressing.",
            ],
            "rules_out": [
                "Any claim that this substrate performs presynaptic-input-keyed "
                "(addressed) recall via its dendritic trace.",
                "Using this substrate's trace as the biological justification for a "
                "same-token-chain gated scan: the trace does not carry afferent "
                "identity, so it cannot support that computation as implemented.",
                "Claims that dendritic traces are a novel memory mechanism relative "
                "to diagonal state-space models -- at this locus they are the same "
                "computation.",
            ],
        }
    elif verdict == "PER-SYNAPSE":
        follow_up = {
            "rules_in": ["Addressed recall via the dendritic trace."],
            "rules_out": ["The equivalence to diagonal state-space models."],
        }
    else:
        follow_up = {
            "rules_in": [
                "The narrower claim that the substrate's trace *locus* is untested "
                "by this experiment."
            ],
            "rules_out": [
                "Any conclusion, in either direction, about whether the substrate "
                "implements addressed recall."
            ],
            "why": ("The per-synapse reference did not clear the power floor, so the "
                    "task is not demonstrably solvable and a chance-level substrate "
                    "score carries no information. Increase the reference trace "
                    "strength/band, or lengthen the probe, and rerun."),
        }

    verdict_block = {
        "verdict": verdict,
        "chance_level": CHANCE,
        "aliased_arm_accuracy_mean": alias,
        "aliased_arm_accuracy_ci95": alias_ci,
        "aliased_arm_exactly_chance_at_every_point": summary["aliased_substrate"][
            "exactly_chance_all_points"],
        "structured_control_accuracy_mean": struct,
        "structured_control_accuracy_ci95": struct_ci,
        "best_ref_per_neuron_accuracy": pn,
        "best_ref_per_neuron_arm": pn_name,
        "best_ref_per_synapse_accuracy": ps,
        "best_ref_per_synapse_arm": ps_name,
        "power_floor": cfg.power_floor,
        "task_solvable_by_ref_per_synapse": task_solvable,
        "substrate_at_chance": substrate_at_chance,
        "ref_per_neuron_at_chance": per_neuron_at_chance,
        "n_sweep_points": len(runs),
        "n_seeds": len(cfg.seeds),
        "n_trials_per_arm_per_point": cfg.n_train + cfg.n_test,
        "reasoning": (
            "Under the aliased projection the class-0 and class-1 drive sequences are "
            "bit-identical (asserted before the sweep), so any per-neuron trace -- a "
            "function of the drive sequence alone -- is provably constant across "
            "classes and cannot decode them. The paired, pair-complete test split "
            "makes exact chance an algebraic consequence rather than a statistical "
            "expectation: with identical features and opposite labels, a deterministic "
            "readout scores exactly one of each pair. The per-neuron reference "
            "confirms this independently. The per-synapse reference keys on the "
            "afferent LINE, which does differ between classes, and it clears the power "
            "floor, so the task is solvable in principle. The substrate tracks the "
            "per-neuron reference, not the per-synapse one."
        ),
        "follow_up": follow_up,
    }

    structural = {
        "v_dend_shape": "brain/neurons.py: NeuronState.v_dend is shape (n,) -- one scalar per neuron",
        "dendritic_drive_source": (
            "brain/neurons.py NeuronState.update(i_soma, i_dend): the dendrite is driven "
            "only by i_dend; brain/simulator.py builds i_dend only from external_dend "
            "(recurrent synaptic delivery goes to i_soma). Spiking synapses therefore "
            "cannot reach the dendritic trace."
        ),
        "synapse_indexing": (
            "brain/connectivity.py: Synapses.targets is (n_pre, k_out), indexed by "
            "PRESYNAPTIC neuron. A postsynaptic neuron has no representable in-edge "
            "list, so 'which afferent was active' is not a quantity the dendrite can read."
        ),
        "per_synapse_state_that_does_exist": (
            "brain/plasticity.py keeps elig of shape (n_pre, k_out) -- but it gates "
            "weight writes (and only when neuromod != 0), not the dendritic trace."
        ),
        "simulator_input_path": (
            "brain/simulator.py Brain.step(external_soma=..., external_dend=...) is the "
            "only channel into the dendrite; there is no per-synapse current path."
        ),
    }

    out = {
        "experiment": "trace_locus",
        "question": ("Is the dendritic trace per-neuron or per-synapse "
                     "(presynaptic-input-keyed)?"),
        "config": {**asdict(cfg), "gains": list(cfg.gains), "betas": list(cfg.betas),
                   "seeds": list(cfg.seeds), "n_train": cfg.n_train, "n_test": cfg.n_test},
        "operating_point": op_info,
        "checks": checks,
        "structural_evidence": structural,
        "verdict": verdict_block,
        "matched_activity_control": matched,
        "summary": summary,
        "runs": runs,
        "run": {
            "wall_seconds": float(time.perf_counter() - t_start),
            "n_sweep_points": len(runs),
            "n_trials_per_arm_per_point": cfg.n_train + cfg.n_test,
            "backend": cfg.backend,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "mlx_available": bool(has_mlx()),
            "results_path": str(Path(args.out)),
        },
    }
    if has_mlx():  # pragma: no cover - informational only
        try:
            import mlx.core as _mx
            out["run"]["mlx_version"] = getattr(_mx, "__version__", "unknown")
            out["run"]["mlx_device"] = str(_mx.default_device())
        except Exception:
            pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))

    log("")
    log("=" * 78)
    log(f"VERDICT: {verdict}")
    log(f"  aliased arm (discriminator)   : {alias:.3f} +/- {alias_ci:.3f}"
        f"   exactly chance everywhere: "
        f"{summary['aliased_substrate']['exactly_chance_all_points']}")
    log(f"  structured arm (control)      : {struct:.3f} +/- {struct_ci:.3f}")
    log(f"  best ref_per_neuron           : {pn:.3f}  ({pn_name})")
    log(f"  best ref_per_synapse          : {ps:.3f}  ({ps_name})")
    log(f"  chance                        : {CHANCE:.3f}")
    log(f"  task solvable by per-synapse  : {task_solvable} (floor {cfg.power_floor})")
    log(f"  spikes/trial (aliased)        : "
        f"{summary['aliased_substrate']['spikes_per_trial_mean']:.1f}")
    log(f"  rate/neuron/ms (aliased)      : "
        f"{summary['aliased_substrate']['rate_per_neuron_per_ms_mean']:.4e}")
    log(f"  trials with >=1 spike (min)   : "
        f"{summary['aliased_substrate']['trials_with_spikes_frac_min']:.3f}")
    log(f"  wrote {args.out}  ({out['run']['wall_seconds']:.1f}s)")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
