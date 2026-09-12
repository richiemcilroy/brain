"""Is the multiscale `tau_dend` advantage real, or a feature-count artefact?

The claim under test
--------------------
An earlier run is recorded as: a **multiscale** ``tau_dend`` population (several
distinct dendritic time constants in one population) scored **0.388** on an
8-way time-since-event decoding task, against **0.287** for a **single**
``tau_dend`` population. That comparison is confounded: the multiscale arm used
1200 readout features and the single-tau arm used 300, so part or all of the
0.101 gap may be bought by the extra readout parameters rather than by the
multiscale dynamics.

This script settles it at **matched feature counts**.

The task, reproduced
--------------------
Each trial is one event pulse followed by a silent gap and then a readout
window::

    [ event pulse: 40 ms at amplitude u ] -> [ SILENT gap d_c ms ] -> [ readout: 40 ms ]

The readout sees **spike counts from the final 40 ms only**, during which the
input is identically zero. Class ``c`` is the length of the silent gap,
``d_c in {0, 25, 50, 100, 175, 275, 400, 550} ms`` -- i.e. "how long ago did the
event happen". Nothing in the present input identifies the class; accuracy above
chance (0.125) must come from state carried across the silent gap. That is the
task structure documented in ``docs/WORKING_MEMORY.md``, with the documented
6-way identification replaced by the 8-way class set this retest is about.

Arms, at identical feature counts and identical readout parameter counts
-----------------------------------------------------------------------
=========================  =====================================================
``single_50/100/200/400``  ``tau_dend`` set to one value for every feature neuron
``multiscale``             ``tau_dend`` drawn (per neuron) from a geometric
                           ladder, 15 -> 800 ms -- the designed multiscale set
``random_tau``             ``tau_dend`` i.i.d. log-uniform on the same range
                           (unstructured spread) -- the shuffled/random control
``no_memory``              ``tau_dend`` = 1 ms: the trace cannot outlive the
                           stimulus, so the readout window is empty by design
=========================  =====================================================

Every arm gets the same feature count and therefore the same readout parameter
count (``features x 8``). The single-tau control is given its best shot:
``single_best`` selects, at each feature count, the single-tau arm with the best
**validation** accuracy and reports that arm's test scores.

Each arm is one fixed population: its per-neuron ``tau_dend`` and stimulus
amplitude are drawn once per arm and reused on every trial and for every class,
exactly as the same neurons would be recorded across trials in an experiment.
Only the substrate's own somatic noise varies between trials. Resampling the
time constants per trial would make each feature index mean a different time
constant on every trial, which no fixed readout could learn -- it would
handicap the random arms, not the single-tau arms, and would be a bug.

Controls
--------
* ``raw_input`` (feature-count matched, in the sweep payload): the *same* ridge
  probe on the same population's **stimulus-window** spike counts. The stimulus
  is identical for every class, so this control must sit at chance. It shows the
  class information is not present in the input, only in the post-gap state.
* ``no_memory``: identical architecture with a 1 ms dendritic time constant, so
  the memory readout is empty. Any accuracy above chance here would mean class
  information is leaking in outside the intended path.
* ``linear_dend`` (auxiliary population): the same multiscale ladder with the
  tuned nonlinear transfer replaced by a linear one, to separate "a slow state
  exists" from "the tuned dendritic nonlinearity makes that state readable".
* ``recurrent_kout32`` (auxiliary): the multiscale arm with real recurrent
  synapses present (``k_out = 0`` in the primary sweep, because recurrence was
  already measured inert in this substrate; this rechecks that here).

Calibrated operating point (the documented trap)
------------------------------------------------
``docs/NEURON_OPERATING_POINT.md`` records that at ``dend_scale = 1.0`` with a
drive of 6.0 the dCaAP tuning curve attenuates the input ~25x and a population
emits **zero spikes**, so a comparison silently compares silence to silence.
The same document records that a slow dendrite at a fixed scale also emits zero
spikes, because it never charges to the tuning peak within the stimulus:
``tau_dend`` moves both the trace duration *and* the operating point.

Every neuron here is therefore calibrated individually:
``dend_scale = u * (1 - exp(-T_stim / tau_dend))``, the dendritic potential the
stimulus actually reaches, so each neuron sits exactly on the tuning peak
(``z = 1``) at stimulus offset regardless of its time constant. A self-test
reproduces the documented silence trap, asserts it is silent, asserts the
calibrated point spikes, and every memory arm asserts a non-zero measured
spike count in its readout windows. Zero spikes raise ``SystemExit`` rather than
producing a number.

Honesty notes
-------------
* Design constants (delays, ladder, feature counts, trial counts, ridge grid)
  were fixed from the documented task and operating point before the sweep was
  run. Nothing was tuned toward a positive result, and a negative result is
  reported in the same detail as a positive one.
* The readout is a **closed-form ridge probe** over the substrate's spike
  counts. It measures *where the information is*, not how a neuron would learn
  to use it.
* MLX is lazy: every value read goes through ``Backend.to_numpy`` (which calls
  ``mx.eval``), and a self-test asserts the same seed gives bit-identical
  features. The default backend is NumPy; ``--backend mlx`` runs the same
  protocol.
* Operation counts are not joules. The reported cost proxies are spike counts
  and simulated neuron counts. No wall-clock efficiency claim is made: the host
  was under heavy external load throughout, which the payload records.

Run:  python3 experiments/multiscale_fair.py
      python3 experiments/multiscale_fair.py --quick     # smoke test only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.backend import get_backend                     # noqa: E402
from brain.neurons import NeuronConfig                    # noqa: E402
from brain.plasticity import PlasticityConfig             # noqa: E402
from brain.readout import Readout, ReadoutConfig          # noqa: E402
from brain.simulator import Brain, SimConfig              # noqa: E402

RESULTS_DIR = ROOT / "experiments" / "results"

# ---------------------------------------------------------------- the design
#: Silent-gap lengths (ms) defining the eight classes ("time since event").
DELAYS_MS: tuple[int, ...] = (0, 25, 50, 100, 175, 275, 400, 550)
#: Geometric dendritic time-constant ladder (the designed multiscale set).
TAU_LADDER: tuple[float, ...] = tuple(
    float(t) for t in np.geomspace(15.0, 800.0, len(DELAYS_MS))
)
#: Single-tau arms; ``single_best`` = best of these per feature count, on validation.
SINGLE_TAUS: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0)
#: Matched feature counts. 300 and 1200 are the widths of the two arms of the
#: original confounded comparison, so the retest is run at the original
#: multiscale width as well as at small widths, where a curve can be resolved.
FEATURE_COUNTS: tuple[int, ...] = (8, 16, 32, 64, 128, 300, 1200)
#: Feature count the pre-registered verdict is stated at: the *multiscale* width
#: of the original comparison, with the single-tau arm widened to match it.
PRIMARY_FC: int = 1200
#: Seeds (>= 3 required); per-seed values are reported.
SEEDS: tuple[int, ...] = (0, 1, 2)
#: Trials per class, disjoint and balanced.
N_TRAIN, N_VAL, N_TEST = 20, 8, 15
N_TRIALS_PER_CLASS = N_TRAIN + N_VAL + N_TEST
#: Stimulus duration, readout-window duration, amplitude, dendritic gain, noise.
T_STIM_MS, T_WIN_MS = 40, 40
DRIVE, DEND_GAIN = 6.0, 2.5
NOISE_STD = 0.02
#: Relative per-neuron stimulus-amplitude jitter (identical in all arms).
AMP_JITTER = 0.15
#: Ridge penalties swept on validation; the winner refits on train + validation.
RIDGE_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
#: ``tau_dend`` of the memoryless control: far shorter than the 40 ms stimulus.
NOMEM_TAU = 1.0
#: The documented trap configuration (docs/NEURON_OPERATING_POINT.md).
TRAP_DRIVE, TRAP_DEND_SCALE, TRAP_DEND_GAIN = 6.0, 1.0, 1.0
#: Scope probe: the same comparison with the documented gap set stretched 3x.
#: At the documented delays a single 400 ms dendrite already spans the whole
#: range, so a ceiling is expected there; this asks where the ladder helps.
EXTENDED_DELAY_SCALE: int = 3


@dataclass(frozen=True)
class ArmSpec:
    """One experimental arm. ``kind`` selects the tau assignment."""

    name: str
    kind: str                        # single | ladder | random | nomem
    window_spikes_expected: bool     # False for the memoryless control
    in_single_best: bool = False     # eligible for the single_best control curve


PRIMARY_ARMS: tuple[ArmSpec, ...] = tuple(
    [ArmSpec(f"single_{int(t)}", "single", True, True) for t in SINGLE_TAUS]
    + [
        ArmSpec("multiscale", "ladder", True),
        ArmSpec("random_tau", "random", True),
        ArmSpec("no_memory", "nomem", False),
    ]
)
SINGLE_ARM_NAMES = tuple(s.name for s in PRIMARY_ARMS if s.in_single_best)


# ------------------------------------------------------------------ helpers
def calibrated_dend_scale(tau: np.ndarray, drive: float, t_stim: int) -> np.ndarray:
    """``dend_scale`` placing each neuron exactly on the tuning peak (``z`` = 1).

    ``v_dend`` after a step of amplitude ``drive`` lasting ``t_stim`` ms is
    ``drive * (1 - exp(-t_stim / tau))``. Setting ``dend_scale`` to that value
    makes ``z = v_dend / dend_scale = 1`` at stimulus offset independently of
    ``tau``, which removes the trace-duration / drive-strength confound recorded
    in ``docs/WORKING_MEMORY.md``.
    """
    tau = np.asarray(tau, np.float32)
    return (float(drive) * (1.0 - np.exp(-float(t_stim) / tau))).astype(np.float32)


def sample_ladder(n_features: int) -> np.ndarray:
    """``n_features`` values from the designed ladder, evenly stepped by index."""
    ladder = np.asarray(TAU_LADDER, np.float32)
    if n_features >= ladder.size:
        return np.resize(ladder, n_features).astype(np.float32)
    idx = np.round(np.linspace(0, ladder.size - 1, n_features)).astype(np.int64)
    return ladder[idx].astype(np.float32)


def assign_taus(spec: ArmSpec, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Dendritic time constants for one arm's fixed population of ``n_features``."""
    if spec.kind == "single":
        return np.full(n_features, float(spec.name.split("_", 1)[1]), np.float32)
    if spec.kind == "ladder":
        return sample_ladder(n_features)
    if spec.kind == "random":
        lo, hi = float(np.log(TAU_LADDER[0])), float(np.log(TAU_LADDER[-1]))
        return np.exp(rng.uniform(lo, hi, n_features)).astype(np.float32)
    if spec.kind == "nomem":
        return np.full(n_features, NOMEM_TAU, np.float32)
    raise ValueError(f"unknown arm kind {spec.kind!r}")


def jittered_amplitude(n: int, rng: np.random.Generator, relative: float) -> np.ndarray:
    """Per-neuron stimulus amplitude: bounded jitter around :data:`DRIVE`."""
    amp = np.float32(DRIVE) * (1.0 + relative * rng.standard_normal(n))
    return np.clip(amp, 0.25 * DRIVE, 3.0 * DRIVE).astype(np.float32)


# --------------------------------------------------------------- simulation
def simulate(*, feature_count: int, specs: tuple[ArmSpec, ...], seed: int,
             backend: str = "numpy", n_trials: int = N_TRIALS_PER_CLASS,
             delays: tuple[int, ...] = DELAYS_MS, amp_jitter: float = AMP_JITTER,
             tau_soma: float = 20.0, k_out: int = 0, dend_mode: str = "dcaap",
             dend_gain: float = DEND_GAIN) -> dict:
    """Simulate every (arm, trial, class) block of one population on one clock.

    At ``k_out = 0`` the neurons are independent, so all arms, trials and
    classes can be simulated as disjoint neuron blocks of a single population
    with per-neuron ``tau_dend`` / ``dend_scale`` and a per-block readout
    window. One clock step then advances every arm at once, which is what makes
    the matched comparison affordable. The block layout is an indexing choice,
    not a modelling one: no neuron influences any other.

    Block layout is ``[arm][trial][class]``, each block owning ``feature_count``
    contiguous neurons. Within an arm the per-neuron time constants and stimulus
    amplitudes are **fixed**: the same population is used for every trial and
    class, and only the substrate's own noise differs between trials.

    Returns readout-window spike counts per block, plus stimulus-window spike
    counts (the raw-input control, class-independent by construction).
    """
    n_classes = len(delays)
    n_blocks = len(specs) * n_trials * n_classes
    width = int(feature_count)
    n_neurons = n_blocks * width
    if n_neurons % 2:
        raise ValueError("population must be even (E/I split in SimConfig)")

    rng = np.random.default_rng((int(seed), width, int(k_out), 20240912))
    tau = np.empty(n_neurons, np.float32)
    amp = np.empty(n_neurons, np.float32)
    block_arm = np.empty(n_blocks, np.int64)
    block_class = np.empty(n_blocks, np.int64)
    block_window = np.empty(n_blocks, np.int64)

    # Per-arm fixed populations, then tiled across that arm's trial/class blocks.
    per_arm = []
    for spec in specs:
        per_arm.append((assign_taus(spec, width, rng),
                        jittered_amplitude(width, rng, amp_jitter)))

    b = 0
    for ai, _spec in enumerate(specs):
        tau_arm, amp_arm = per_arm[ai]
        for _trial in range(n_trials):
            for ci in range(n_classes):
                sl = slice(b * width, (b + 1) * width)
                tau[sl] = tau_arm
                amp[sl] = amp_arm
                block_arm[b] = ai
                block_class[b] = ci
                block_window[b] = T_STIM_MS + int(delays[ci])
                b += 1

    dend_scale = calibrated_dend_scale(tau, DRIVE, T_STIM_MS)

    be = get_backend(backend, seed=int(seed))
    neu = NeuronConfig(n=n_neurons, tau_dend=tau, dend_scale=dend_scale,
                       dend_gain=float(dend_gain), dend_mode=dend_mode,
                       noise_std=NOISE_STD, tau_soma=float(tau_soma))
    plas = PlasticityConfig(enabled=False, eligibility="off", homeostasis=False)
    brain = Brain(SimConfig(n_neurons=n_neurons, k_out=int(k_out), seed=int(seed),
                            neu=neu, plas=plas, inhibition="none",
                            spike_capacity_frac=1.0), be)
    drive = be.array(amp)
    be.eval(drive)

    steps = T_STIM_MS + int(max(delays)) + T_WIN_MS
    win_end = block_window + T_WIN_MS
    window_blocks = [np.flatnonzero((t >= block_window) & (t < win_end))
                     for t in range(steps)]

    counts = np.zeros((n_blocks, width), np.float32)        # the memory readout
    stim_counts = np.zeros((n_blocks, width), np.float32)   # raw-input control
    brain.reset()
    t0 = time.perf_counter()
    for t in range(steps):
        st = brain.step(external_dend=drive) if t < T_STIM_MS else brain.step()
        if st.overflow:
            raise SystemExit(
                f"FAIL LOUDLY: spike-buffer overflow ({st.overflow}) at step {t}; "
                "spike capacity must never truncate a measurement")
        if t < T_STIM_MS:
            m = be.to_numpy(brain.last_spike_mask).astype(np.float32)
            stim_counts += m.reshape(n_blocks, width)
        idx = window_blocks[t]
        if idx.size:
            m = be.to_numpy(brain.last_spike_mask).astype(np.float32)
            counts[idx] += m.reshape(n_blocks, width)[idx]
    wall = time.perf_counter() - t0

    return {
        "counts": counts, "stim_counts": stim_counts,
        "block_arm": block_arm, "block_class": block_class, "tau": tau,
        "n_trials": n_trials, "n_classes": n_classes, "steps": steps,
        "n_neurons": n_neurons, "n_blocks": n_blocks, "width": width,
        "total_spikes": int(brain.total_spikes), "wall_seconds": wall,
        "bio_ms": steps, "k_out": int(k_out), "backend": backend,
        "dend_mode": dend_mode, "dend_gain": float(dend_gain),
        "synops_active": int(brain.total_spikes) * int(k_out),
    }


# ------------------------------------------------------------------ readout
def _standardiser(train: np.ndarray):
    """Frozen per-feature z-score statistics; dead features become exactly 0."""
    mean = train.mean(axis=0)
    sd = train.std(axis=0)
    dead = np.flatnonzero(sd <= 1e-8)
    sd = np.where(sd <= 1e-8, 1.0, sd)

    def apply(x: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - mean.astype(np.float32)) / sd.astype(np.float32)
        if dead.size:
            z[:, dead] = 0.0
        return z.astype(np.float32)

    return apply, {"n_dead_features": int(dead.size)}


def decode(counts_arm: np.ndarray, *, seed: int, n_train: int = N_TRAIN,
           n_val: int = N_VAL, n_classes: int = len(DELAYS_MS)) -> dict:
    """Ridge probe over one arm's spike counts.

    ``counts_arm`` is ``(n_trials, n_classes, n_features)``. The penalty is
    chosen on the validation trials and the final model is refitted on
    train + validation; the test trials are scored once.
    """
    n_features = int(counts_arm.shape[2])
    X = counts_arm.reshape(-1, n_features).astype(np.float32)
    y = np.tile(np.arange(n_classes, dtype=np.int64), counts_arm.shape[0])
    per = n_classes
    tr = slice(0, n_train * per)
    va = slice(n_train * per, (n_train + n_val) * per)
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[(n_train + n_val) * per:], y[(n_train + n_val) * per:]

    zscore, zinfo = _standardiser(Xtr)
    Ztr = zscore(Xtr)

    def fit(lam: float, Z: np.ndarray, labels: np.ndarray) -> Readout:
        model = Readout(ReadoutConfig(kind="ridge", n_features=n_features,
                                      n_classes=n_classes, ridge=float(lam),
                                      normalize=False, seed=int(seed)))
        model.partial_fit(Z, labels)
        return model

    sweep, best_val, best_lam = [], -1.0, float(RIDGE_GRID[0])
    for lam in RIDGE_GRID:
        acc = float(fit(lam, Ztr, ytr).score(zscore(Xva), yva))
        sweep.append({"ridge": float(lam), "val_acc": acc})
        if acc > best_val:
            best_val, best_lam = acc, float(lam)

    Ztv = np.concatenate([Ztr, zscore(Xva)], axis=0)
    labels_tv = np.concatenate([ytr, yva], axis=0)
    final = fit(best_lam, Ztv, labels_tv)
    return {
        "test_acc": float(final.score(zscore(Xte), yte)),
        "train_acc": float(final.score(Ztv, labels_tv)),
        "val_acc": float(best_val), "ridge": best_lam, "ridge_sweep": sweep,
        "n_train": int(n_train), "n_val": int(n_val), "n_test": int(Xte.shape[0]),
        "n_features": n_features, "readout_params": int(final.n_params),
        "n_dead_features": zinfo["n_dead_features"],
        "mean_feature_value": float(X.mean()),
    }


# ------------------------------------------------------------------- checks
def operating_point_self_test(backend: str) -> dict:
    """Reproduce the documented silence trap and assert the calibrated fix.

    The trap is reproduced at the configuration ``docs/NEURON_OPERATING_POINT.md``
    used (drive 6.0, ``dend_scale`` 1.0, unit dendritic gain, 200 ms dendrite);
    the fix is the per-neuron calibration every arm of this experiment applies.
    """
    out = {}
    configs = (
        ("documented_trap", TRAP_DEND_SCALE, TRAP_DEND_GAIN, TRAP_DRIVE),
        ("trap_at_this_experiments_gain", TRAP_DEND_SCALE, DEND_GAIN, TRAP_DRIVE),
        ("slow_dendrite_at_unit_scale", TRAP_DEND_SCALE, DEND_GAIN, TRAP_DRIVE),
        ("calibrated", None, DEND_GAIN, DRIVE),
    )
    for label, scale, gain, drive_v in configs:
        be = get_backend(backend, seed=0)
        tau = np.full(64, 200.0, np.float32)
        ds = (calibrated_dend_scale(tau, drive_v, T_STIM_MS) if scale is None
              else np.full(64, scale, np.float32))
        cfg = NeuronConfig(n=64, tau_dend=tau, dend_scale=ds, dend_gain=float(gain),
                           noise_std=NOISE_STD, tau_soma=20.0)
        plas = PlasticityConfig(enabled=False, eligibility="off", homeostasis=False)
        brain = Brain(SimConfig(n_neurons=64, k_out=0, seed=0, neu=cfg, plas=plas,
                                spike_capacity_frac=1.0), be)
        drive = be.array(np.full(64, float(drive_v), np.float32))
        spikes = sum(brain.step(external_dend=drive).spikes for _ in range(400))
        out[label] = {"dend_scale": float(np.asarray(ds).reshape(-1)[0]),
                      "dend_gain": float(gain), "drive": float(drive_v),
                      "spikes_over_400ms": int(spikes)}
    if out["documented_trap"]["spikes_over_400ms"] != 0:
        raise SystemExit(
            "FAIL LOUDLY: the documented silence trap did not reproduce "
            f"(expected 0 spikes). detail={out['documented_trap']}")
    if not out["calibrated"]["spikes_over_400ms"] > 0:
        raise SystemExit(
            "FAIL LOUDLY: the CALIBRATED operating point emits zero spikes, so every "
            f"number below would compare silence to silence. detail={out}")
    return out


def determinism_self_test(backend: str) -> dict:
    """Same seed -> bit-identical features; an unevaluated lazy graph would break this."""
    kw = dict(feature_count=8, specs=PRIMARY_ARMS[:1], backend=backend,
              n_trials=4, delays=(0, 100))
    a = simulate(seed=7, **kw)
    b = simulate(seed=7, **kw)
    if not (np.array_equal(a["counts"], b["counts"])
            and np.array_equal(a["stim_counts"], b["stim_counts"])):
        raise SystemExit(
            "FAIL LOUDLY: the same seed produced different features. Either an "
            "unseeded RNG or an unevaluated (lazy) array is in the path.")
    return {"bit_identical": True, "n_values": int(a["counts"].size),
            "total_spikes": int(a["total_spikes"])}


def check_spiking(counts: np.ndarray, stim_counts: np.ndarray, sim: dict,
                  specs: tuple[ArmSpec, ...]) -> dict:
    """Assert measured spike rates are non-zero, loudly, before anything is trusted.

    For arms expected to carry memory, the readout-window (memory) spike count
    must be strictly positive. For the memoryless control the window is expected
    to be empty by construction, so the assertion there is that the population
    is nevertheless alive -- it must spike during the stimulus -- otherwise
    "no memory" would be indistinguishable from "dead population".
    """
    wide = counts.reshape(len(specs), sim["n_trials"], sim["n_classes"], sim["width"])
    wide_stim = stim_counts.reshape(len(specs), sim["n_trials"], sim["n_classes"],
                                    sim["width"])
    report = {}
    for ai, spec in enumerate(specs):
        window = int(wide[ai].sum())
        stimulus = int(wide_stim[ai].sum())
        report[spec.name] = {"window_spikes": window, "stimulus_spikes": stimulus,
                             "population_spikes": int(sim["total_spikes"])}
        if spec.window_spikes_expected and window <= 0:
            raise SystemExit(
                f"FAIL LOUDLY: arm {spec.name!r} emitted ZERO spikes in its readout "
                "windows. It would be comparing silence to silence; the operating "
                f"point is wrong. detail={report[spec.name]}")
        if not spec.window_spikes_expected and stimulus <= 0:
            raise SystemExit(
                f"FAIL LOUDLY: the memoryless control {spec.name!r} emitted zero "
                "stimulus spikes, so its population is dead, not memoryless. "
                f"detail={report[spec.name]}")
    if int(sim["total_spikes"]) <= 0:
        raise SystemExit("FAIL LOUDLY: the population emitted zero spikes over the "
                         "entire run; the simulation is not spiking.")
    return report


# --------------------------------------------------------------- sweep driver
def evaluate_sweep(args, specs, feature_counts, seeds, *, tag, amp_jitter=None,
                   k_out=0, tau_soma=20.0, dend_mode="dcaap",
                   dend_gain=DEND_GAIN, delays=DELAYS_MS) -> dict:
    """All (feature count, seed) populations for one arm set."""
    jitter = AMP_JITTER if amp_jitter is None else float(amp_jitter)
    per_seed: dict[str, dict] = {}
    spikes: dict[str, dict[str, int]] = {s.name: {} for s in specs}
    wall_total = 0.0
    for fc in feature_counts:
        for seed in seeds:
            sim = simulate(feature_count=fc, specs=specs, seed=seed,
                           backend=args.backend, amp_jitter=jitter, k_out=k_out,
                           tau_soma=tau_soma, dend_mode=dend_mode,
                           dend_gain=dend_gain, delays=delays)
            wall_total += sim["wall_seconds"]
            check = check_spiking(sim["counts"], sim["stim_counts"], sim, specs)
            for name, info in check.items():
                spikes[name][f"{fc}:{seed}"] = info["window_spikes"]
            wide = sim["counts"].reshape(len(specs), sim["n_trials"],
                                         sim["n_classes"], fc)
            wide_stim = sim["stim_counts"].reshape(len(specs), sim["n_trials"],
                                                   sim["n_classes"], fc)
            for ai, spec in enumerate(specs):
                res = decode(wide[ai], seed=seed, n_classes=len(delays))
                raw = decode(wide_stim[ai], seed=seed, n_classes=len(delays))
                res.update({
                    "raw_input_control_test_acc": raw["test_acc"],
                    "population_neurons": sim["n_neurons"],
                    "population_spikes": sim["total_spikes"],
                    "stimulus_spikes": int(wide_stim[ai].sum()),
                    "window_spikes": int(wide[ai].sum()),
                    "bio_ms": sim["steps"], "wall_seconds": sim["wall_seconds"],
                    "synops_active": sim["synops_active"], "k_out": k_out,
                    "dend_mode": sim["dend_mode"], "dend_gain": sim["dend_gain"],
                })
                per_seed.setdefault(str(fc), {}).setdefault(spec.name, {})[str(seed)] = res
            print(f"  [{tag}] features={fc:<4d} seed={seed}  "
                  f"neurons={sim['n_neurons']:>7d} spikes={sim['total_spikes']:>9d} "
                  f"{sim['wall_seconds']:6.1f}s", flush=True)
    return {"tag": tag, "per_seed": per_seed, "spikes_by_arm": spikes,
            "wall_seconds": wall_total, "feature_counts": list(feature_counts),
            "seeds": list(seeds), "arms": [s.name for s in specs],
            "delays_ms": list(delays),
            "window_spikes_expected": {s.name: s.window_spikes_expected for s in specs}}


def write_payload(path: Path, payload: dict) -> None:
    """Write results atomically, so an interrupted run cannot leave a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def checkpoint(path: Path, stage: str, **sections) -> None:
    """Persist whatever has been computed so far, tagged with the stage."""
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.setdefault("meta", {}).update({"last_stage": stage,
                                            "partial": True})
    existing.update(sections)
    write_payload(path, existing)
    print(f"  [checkpoint] wrote {stage} to {path}", flush=True)


def summarise(values) -> dict:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {"mean": None, "ci95": None, "n": 0, "values": []}
    ci = float(1.96 * a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "ci95": ci, "n": int(a.size),
            "values": [float(v) for v in a]}


def paired_delta(a: dict, b: dict) -> dict:
    """Per-seed ``a - b``; paired because both arms share the trial structure."""
    out = summarise([float(x) - float(y) for x, y in zip(a["values"], b["values"])])
    out["per_seed"] = list(out["values"])
    return out


def arm_metric(sweep: dict, fc, arm: str, seeds, metric: str) -> dict:
    return summarise([sweep["per_seed"][str(fc)][arm][str(s)][metric] for s in seeds])


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="numpy", choices=("numpy", "mlx"))
    p.add_argument("--out", default=str(RESULTS_DIR / "multiscale_fair.json"))
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke-test config; does not produce the claim")
    p.add_argument("--no-aux", action="store_true", help="skip auxiliary arms")
    args = p.parse_args(argv)

    t_start = time.perf_counter()
    seeds = (0,) if args.quick else SEEDS
    fcs = (8, 16) if args.quick else FEATURE_COUNTS
    primary_fc = int(fcs[-1]) if args.quick else PRIMARY_FC

    checks = {
        "operating_point": operating_point_self_test(args.backend),
        "determinism": determinism_self_test(args.backend),
    }
    print(f"[check] operating point: {checks['operating_point']}")
    print(f"[check] determinism:     {checks['determinism']}")
    print(f"[sweep] arms={[s.name for s in PRIMARY_ARMS]} feature_counts={fcs} "
          f"seeds={seeds}")

    out_path = Path(args.out)
    primary = evaluate_sweep(args, PRIMARY_ARMS, fcs, seeds, tag="primary")
    checks["spiking_primary"] = primary["spikes_by_arm"]
    checkpoint(out_path, "primary", checks=checks, sweep=primary)

    aux: dict[str, dict] = {}
    if not args.no_aux:
        print("[aux]   no amplitude jitter, primary arms")
        aux["no_jitter"] = evaluate_sweep(args, PRIMARY_ARMS, (primary_fc,), seeds,
                                          tag="no_jitter", amp_jitter=0.0)
        print("[aux]   linear dendritic transfer, multiscale ladder")
        aux_fcs = (16,) if args.quick else (16, 128, 1200)
        aux["linear_dend"] = evaluate_sweep(
            args, (ArmSpec("multiscale_linear", "ladder", True),), aux_fcs,
            seeds, tag="linear_dend", dend_mode="linear")
        print("[aux]   recurrence present (k_out=32), multiscale only")
        aux["recurrent_kout32"] = evaluate_sweep(
            args, (ArmSpec("multiscale", "ladder", True),), (32,), seeds,
            tag="recurrent", k_out=32)
        print(f"[aux]   extended delay range (x{EXTENDED_DELAY_SCALE})")
        ext_delays = tuple(int(d) * EXTENDED_DELAY_SCALE for d in DELAYS_MS)
        ext_fcs = (32,) if args.quick else (64, 300)
        aux["extended_delay_range"] = evaluate_sweep(
            args, PRIMARY_ARMS, ext_fcs, seeds, tag="extended", delays=ext_delays)
        aux["extended_delay_range"]["delays_ms"] = list(ext_delays)
        checkpoint(out_path, "aux", checks=checks, sweep=primary, aux=aux)

    # ------------------------------------------------------------ summaries
    arms = [s.name for s in PRIMARY_ARMS]
    summary = {str(fc): {a: arm_metric(primary, fc, a, seeds, "test_acc")
                         for a in arms} for fc in fcs}
    raw_control = {str(fc): {a: arm_metric(primary, fc, a, seeds,
                                           "raw_input_control_test_acc")
                             for a in arms} for fc in fcs}
    window_spikes = {str(fc): {a: arm_metric(primary, fc, a, seeds, "window_spikes")
                               for a in arms} for fc in fcs}
    single_curve = {str(fc): {a: arm_metric(primary, fc, a, seeds, "test_acc")
                              for a in SINGLE_ARM_NAMES} for fc in fcs}

    single_best = {}
    for fc in fcs:
        val_means = {a: float(np.mean([primary["per_seed"][str(fc)][a][str(s)]["val_acc"]
                                       for s in seeds])) for a in SINGLE_ARM_NAMES}
        winner = max(val_means, key=val_means.get)
        single_best[str(fc)] = {
            "arm": winner, "val_means": val_means,
            "test": arm_metric(primary, fc, winner, seeds, "test_acc"),
            "raw_input_control": arm_metric(primary, fc, winner, seeds,
                                            "raw_input_control_test_acc"),
            "selection": "best mean VALIDATION accuracy among the single-tau arms",
        }

    deltas = {}
    for fc in fcs:
        k = str(fc)
        deltas[k] = {
            "multiscale_minus_single_best": paired_delta(summary[k]["multiscale"],
                                                         single_best[k]["test"]),
            "multiscale_minus_random_tau": paired_delta(summary[k]["multiscale"],
                                                        summary[k]["random_tau"]),
            "multiscale_minus_no_memory": paired_delta(summary[k]["multiscale"],
                                                       summary[k]["no_memory"]),
            "random_tau_minus_single_best": paired_delta(summary[k]["random_tau"],
                                                         single_best[k]["test"]),
            "multiscale_minus_raw_input_control": paired_delta(
                summary[k]["multiscale"], raw_control[k]["multiscale"]),
        }

    # ------------------------------------------- how much the confound can buy
    confound = {}
    for fc in fcs:
        wide = 4 * fc
        if wide in fcs:
            confound[f"{fc}_vs_{wide}"] = {
                "matched_gain_multiscale_minus_single_best":
                    summary[str(wide)]["multiscale"]["mean"]
                    - single_best[str(wide)]["test"]["mean"],
                "confounded_gain_multiscale_wide_minus_single_small":
                    summary[str(wide)]["multiscale"]["mean"]
                    - single_best[str(fc)]["test"]["mean"],
                "multiscale_gain_from_width_alone":
                    summary[str(wide)]["multiscale"]["mean"]
                    - summary[str(fc)]["multiscale"]["mean"],
                "single_best_gain_from_width_alone":
                    single_best[str(wide)]["test"]["mean"]
                    - single_best[str(fc)]["test"]["mean"],
                "single_best_arm_small": single_best[str(fc)]["arm"],
                "single_best_arm_wide": single_best[str(wide)]["arm"],
            }

    # ---------------------------------------------- pre-registered verdict
    k = str(primary_fc)
    d = deltas[k]["multiscale_minus_single_best"]
    lo = (d["mean"] - d["ci95"]) if d["mean"] is not None else None
    hi = (d["mean"] + d["ci95"]) if d["mean"] is not None else None
    survives = bool(d["mean"] is not None and lo is not None
                    and d["mean"] > 0 and lo > 0)
    single_mean = single_best[k]["test"]["mean"]
    ratio = (summary[k]["multiscale"]["mean"] / single_mean) if single_mean else None
    verdict = {
        "primary_feature_count": primary_fc,
        "decision_rule": ("SURVIVES iff, at the primary matched feature count, the paired "
                          "multiscale-minus-single_best mean test accuracy is positive AND "
                          "its 95% CI excludes zero; otherwise REFUTED."),
        "chance": 1.0 / len(DELAYS_MS),
        "multiscale_mean": summary[k]["multiscale"]["mean"],
        "multiscale_values": summary[k]["multiscale"]["values"],
        "single_best_arm": single_best[k]["arm"],
        "single_best_mean": single_mean,
        "single_best_values": single_best[k]["test"]["values"],
        "delta_mean": d["mean"], "delta_ci95": d["ci95"],
        "delta_ci_low": lo, "delta_ci_high": hi, "delta_per_seed": d["per_seed"],
        "ratio": ratio,
        "random_tau_mean": summary[k]["random_tau"]["mean"],
        "random_tau_values": summary[k]["random_tau"]["values"],
        "no_memory_mean": summary[k]["no_memory"]["mean"],
        "no_memory_window_spikes": window_spikes[k]["no_memory"]["mean"],
        "raw_input_control_mean": raw_control[k]["multiscale"]["mean"],
        "survives_feature_count_matching": survives,
        "original_claim": {"multiscale": 0.388, "single_tau": 0.287,
                           "feature_counts": [1200, 300], "delta": 0.101},
    }
    verdict["text"] = (
        f"{primary_fc} matched features: multiscale={summary[k]['multiscale']['mean']:.3f} "
        f"vs single_best({single_best[k]['arm']})={single_mean:.3f}, paired delta="
        f"{d['mean']:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]); "
        f"random-tau={summary[k]['random_tau']['mean']:.3f}, "
        f"no-memory={summary[k]['no_memory']['mean']:.3f}, "
        f"raw-input-control={raw_control[k]['multiscale']['mean']:.3f}, "
        f"chance={1/len(DELAYS_MS):.3f}. "
        + ("SURVIVES feature-count matching." if survives else
           "REFUTED: no multiscale advantage once feature count is matched.")
    )
    if ratio is not None:
        verdict["text"] += f" Ratio {ratio:.3f}x."

    # ------------------------------ extended-range scope probe (auxiliary)
    extended = None
    if "extended_delay_range" in aux:
        ext = aux["extended_delay_range"]
        ext_fcs = ext["feature_counts"]
        ext_summary, ext_single_best, ext_deltas = {}, {}, {}
        for fc in ext_fcs:
            kk = str(fc)
            ext_summary[kk] = {a: arm_metric(ext, fc, a, ext["seeds"], "test_acc")
                               for a in ext["arms"]}
            vm = {a: float(np.mean([ext["per_seed"][kk][a][str(s0)]["val_acc"]
                                    for s0 in ext["seeds"]]))
                  for a in SINGLE_ARM_NAMES}
            win = max(vm, key=vm.get)
            ext_single_best[kk] = {
                "arm": win, "val_means": vm,
                "test": arm_metric(ext, fc, win, ext["seeds"], "test_acc")}
            d_ext = paired_delta(ext_summary[kk]["multiscale"],
                                 ext_single_best[kk]["test"])
            lo_e = d_ext["mean"] - d_ext["ci95"]
            ext_deltas[kk] = {
                "multiscale_minus_single_best": d_ext,
                "multiscale_minus_random_tau": paired_delta(
                    ext_summary[kk]["multiscale"], ext_summary[kk]["random_tau"]),
                "multiscale_beats_best_single": bool(
                    d_ext["mean"] > 0 and lo_e > 0),
            }
        extended = {
            "delays_ms": ext["delays_ms"], "feature_counts": ext_fcs,
            "summary": ext_summary, "single_best": ext_single_best,
            "deltas": ext_deltas, "spikes_by_arm": ext["spikes_by_arm"],
        }
        verdict["extended_delay_scope"] = {
            "delays_ms": ext["delays_ms"],
            "text": ("Where the documented delay set saturates a single tau, the "
                     "comparison was repeated with the gap set stretched "
                     f"{EXTENDED_DELAY_SCALE}x."),
            "detail": ext_deltas,
        }

    payload = {
        "meta": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "backend": args.backend,
            "wall_seconds_total": time.perf_counter() - t_start,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "quick": bool(args.quick),
            "host_note": ("The host ran under heavy external load throughout; wall-clock "
                          "figures are indicative only and no throughput claim is made "
                          "from them."),
        },
        "design": {
            "task": ("8-way time-since-event: event pulse (40 ms) -> silent gap d_c -> "
                     "silent readout window (40 ms); the readout sees only window spikes"),
            "delays_ms": list(DELAYS_MS), "chance": 1.0 / len(DELAYS_MS),
            "tau_ladder_ms": [round(float(t), 3) for t in TAU_LADDER],
            "single_taus_ms": list(SINGLE_TAUS), "feature_counts": list(fcs),
            "primary_feature_count": primary_fc,
            "seeds": list(seeds), "n_train_per_class": N_TRAIN,
            "n_val_per_class": N_VAL, "n_test_per_class": N_TEST,
            "n_trials_per_class": N_TRIALS_PER_CLASS,
            "t_stim_ms": T_STIM_MS, "t_window_ms": T_WIN_MS,
            "drive": DRIVE, "dend_gain": DEND_GAIN, "noise_std": NOISE_STD,
            "amp_jitter": AMP_JITTER, "ridge_grid": list(RIDGE_GRID),
            "k_out_primary": 0, "inhibition": "none", "n_mem_tau": NOMEM_TAU,
            "readout": ("closed-form ridge probe on readout-window spike counts "
                        "(brain/readout.py); penalty chosen on validation"),
            "operating_point": ("per-neuron dend_scale = u*(1-exp(-T_stim/tau_dend)), so "
                                "z = 1 at stimulus offset "
                                "(docs/NEURON_OPERATING_POINT.md)"),
            "matched": ("every arm gets the same feature count and therefore the same "
                        "readout parameter count (features x n_classes)"),
            "population_model": ("one fixed population per arm: per-neuron tau_dend and "
                                 "stimulus amplitude are drawn once per arm and reused on "
                                 "every trial and class"),
            "arms": {s.name: {"kind": s.kind,
                              "window_spikes_expected": s.window_spikes_expected,
                              "in_single_best": s.in_single_best}
                     for s in PRIMARY_ARMS},
        },
        "checks": checks,
        "sweep": primary,
        "aux": aux,
        "summary": summary,
        "raw_input_control": raw_control,
        "window_spikes": window_spikes,
        "single_tau_curve": single_curve,
        "single_best": single_best,
        "deltas": deltas,
        "extended_delay_range": extended,
        "confounded_replication": confound,
        "verdict": verdict,
        "caveats": [
            "The readout is a closed-form ridge probe, not a biologically plausible "
            "learning rule: it measures where information is, not how a neuron would "
            "learn to use it.",
            "The primary sweep runs at k_out=0 (no recurrent synapses) because "
            "recurrence was already measured inert in this substrate "
            "(docs/WORKING_MEMORY.md); an auxiliary arm rechecks that here.",
            "no_memory (tau_dend = 1 ms) is expected to emit no readout-window spikes: "
            "the trace cannot outlive the stimulus. It is a leak control, so its "
            "accuracy should sit at chance; above chance would mean class information "
            "is entering outside the intended path.",
            "The raw_input control uses stimulus-window counts from the same simulation. "
            "The stimulus is identical for every class, so it must sit at chance.",
            "At the two widest feature counts the ridge probe has more parameters than "
            "training samples (224 train+val for 1200 features); the penalty is chosen "
            "on validation in every arm, and both arms face the same regime.",
            "Operation counts are not joules; spikes and simulated neuron counts are "
            "reported as cost proxies and no wall-clock claim is made.",
            "Confidence intervals are across seeds only (95% CI of the mean, "
            "1.96*SEM); test-set sampling noise is not included in them.",
            "Delays, ladder, feature counts, trial counts and ridge grid were fixed from "
            "the documented task before the sweep; no post-hoc tuning was done.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_payload(out, payload)

    def fmt(stat):
        return (f"{stat['mean']:.3f}+/-{stat['ci95']:.3f}"
                if stat["mean"] is not None else "n/a")

    print("\n" + "=" * 108)
    print("accuracy vs MATCHED feature count  (mean +/- 95% CI over seeds)")
    print("=" * 108)
    print(f"{'features':>8s}" + "".join(f"{a:>18s}" for a in arms)
          + f"{'single_best':>18s}{'raw_input':>18s}")
    for fc in fcs:
        print(f"{fc:>8d}" + "".join(f"{fmt(summary[str(fc)][a]):>18s}" for a in arms)
              + f"{fmt(single_best[str(fc)]['test']):>18s}"
              + f"{fmt(raw_control[str(fc)]['multiscale']):>18s}")
    print("-" * 108)
    print(verdict["text"])
    print(f"\nwrote {out} ({out.stat().st_size / 1e3:.0f} kB) in "
          f"{payload['meta']['wall_seconds_total']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
