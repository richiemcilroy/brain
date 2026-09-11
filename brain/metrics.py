"""Continual-learning and energy-proxy metrics.

This module is the measurement half of the head-to-head experiment: it scores
the spiking substrate and the parameter-matched dense backprop baseline with
*identical* code, so a difference between the arms cannot come from a
difference in how they were measured. Everything here is pure NumPy (the arrays
are tiny - a few dozen floats), deterministic, and free of any dependency on
the simulator, MLX, or a random seed.

Continual-learning protocol
---------------------------
Sequential training over ``n_tasks`` tasks with no rehearsal and no access to
earlier data. After finishing task ``i`` we evaluate on every task ``j <= i``
and store the result in row ``i``::

    acc_matrix[i, j] = accuracy on task j after training through task i

so the matrix is lower triangular. Entries with ``j > i`` were never measured
and are stored as ``0.0`` (they carry no meaning - do not take raw means of the
whole matrix; use :meth:`ContinualCurve.final_average`). The whole model state
is scored at each row, not a per-task head, which is what makes forgetting
visible.

Summary statistics
------------------
* ``final_average``      = ``mean_j acc[n-1, j]`` - mean accuracy on all tasks
  at the end of the stream.
* ``backward_transfer``  = ``mean_j (acc[n-1, j] - acc[j, j])`` (BWT, Lopez-Paz
  & Ranzato 2017, "Gradient Episodic Memory for Continual Learning", NeurIPS).
  Negative BWT means later learning hurt earlier tasks.
* ``forgetting``         = ``mean_j (max_{i>=j} acc[i, j] - acc[n-1, j])``
  (Chaudhry et al. 2018, "On Tiny Episodic Memories in Continual Learning"),
  the mean drop from the best accuracy ever observed on a task to its accuracy
  at the end. Note this is a *max-drop*, so it is not simply ``-BWT``: a task
  can lose accuracy after having first improved. The self-check asserts that
  distinction with a hand-computed example.

Normalisation caveat (deliberate, per project spec). Both statistics are
averaged over all ``n`` tasks here. GEM averages BWT over its first ``T-1``
tasks (its last term is identically zero), and some implementations average
forgetting over ``T-1`` tasks as well, so published numbers may differ from
these by a factor of ``n/(n-1)``. Sign and arm-vs-arm ordering are unaffected;
report ``n`` alongside any number.

Energy / SynOp accounting
-------------------------
The comparison against the dense baseline is in *active synaptic operations*
(SynOps) - an operation count, never a joule measurement:

    SynOp  ``= spikes x fan-out``  (one accumulate into a post-synaptic target,
                                   with no multiply, since a spike is 0/1)
    MAC    ``= multiply + accumulate``; a dense (coincident) layer pays one MAC
                                   per parameter per sample, active or not.

**Operation counts are not joules, and spiking is not a free energy win.** A
SynOp on Intel's Loihi costs ~23.6 pJ (Davies et al. 2018, IEEE Micro 38(1):
82-99), roughly five orders of magnitude *more* than the ~1e-16 J (0.1 fJ)
often quoted for a biological synaptic event (Attwell & Laughlin 2001 estimate
the same order; sources vary by 1-2 orders of magnitude). And the brain arm's
SynOps here are derived from spike counts x fan-out, not measured on hardware:
the MLX implementation in this repo executes dense gathers, so the proxy is an
*algorithmic* sparsity claim about the substrate, not a wall-clock or energy
result on this machine. Report both arms' counts, state the unit, and never
present the ratio as a joules saving.

Public surface
--------------
``accuracy``, ``ContinualCurve``, ``summarise``, ``synop_comparison``,
``coincident_dense_macs``, plus ``self_check`` (run as ``python3
brain/metrics.py`` from the repo root).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

# ~23.6 pJ per synaptic operation on Intel Loihi (Davies et al. 2018, IEEE
# Micro 38(1):82-99). Cited round figure, not a measurement of this substrate.
LOIHI_PJ_PER_SYNOP: float = 23.6

# Order of magnitude for a biological synaptic event, in joules.
BIOLOGICAL_J_PER_SYNAPTIC_EVENT: float = 1e-16

_ENERGY_CAVEAT = (
    "OPERATION COUNTS, NOT JOULES. A SynOp is one accumulate (spike x fan-out); "
    "a dense MAC is a multiply-accumulate. On Loihi a SynOp costs ~23.6 pJ "
    "(Davies et al. 2018, IEEE Micro 38(1):82-99) vs ~1e-16 J for a biological "
    "synaptic event, so sparsity is not a free energy win. Brain-arm SynOps are "
    "derived from spike counts x fan-out, not measured on hardware; this "
    "implementation executes dense gathers, so the ratio is an algorithmic "
    "sparsity claim, not a joule or wall-clock claim."
)

__all__ = [
    "accuracy",
    "ContinualCurve",
    "summarise",
    "synop_comparison",
    "coincident_dense_macs",
    "self_check",
    "LOIHI_PJ_PER_SYNOP",
    "BIOLOGICAL_J_PER_SYNAPTIC_EVENT",
]


# --------------------------------------------------------------------- accuracy
def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of samples for which ``y_pred`` selects the true class.

    Interpretation rules, applied in this order and *never* guessed from
    magnitudes:

    * ``y_pred.ndim >= 2`` -> treated as scores/logits, reduced with
      ``argmax`` over the last axis.
    * ``y_pred.ndim == 1`` -> treated as class labels. A 1-D vector of
      probabilities is **not** detected and **not** thresholded; pass an
      ``(n, 2)`` score array, or threshold it yourself, instead.
    * ``y_true.ndim == 2`` with ``y_true.shape[0] == len(y_pred)`` and more than
      one column -> decoded with ``argmax`` (one-hot labels).
    * anything else -> flattened to 1-D.

    Empty inputs return ``nan`` rather than raising, so a zero-length eval split
    cannot take down a long experiment run.
    """
    preds = np.asarray(y_pred)
    truth = np.asarray(y_true)
    if preds.size == 0 or truth.size == 0:
        return float("nan")

    if preds.ndim >= 2:
        preds = np.argmax(preds.reshape(preds.shape[0], -1), axis=1)
    else:
        preds = preds.reshape(-1)

    if truth.ndim == 2 and truth.shape[1] > 1 and truth.shape[0] == preds.shape[0]:
        truth = np.argmax(truth, axis=1)
    else:
        truth = truth.reshape(-1)

    if truth.shape[0] != preds.shape[0]:
        raise ValueError(
            f"accuracy: y_pred and y_true disagree on sample count "
            f"({preds.shape[0]} vs {truth.shape[0]}) after reduction"
        )
    return float(np.mean(preds == truth))


# ------------------------------------------------------------------- the curve
@dataclass(eq=False)
class ContinualCurve:
    """Accuracy matrix and the standard continual-learning summary statistics.

    ``acc`` is ``(n_tasks, n_tasks)`` lower triangular for the full protocol,
    where ``acc[i, j]`` is the accuracy on task ``j`` measured right after
    training through task ``i``; entries above the diagonal are never measured
    and must be ``0.0`` or ``nan``. A rectangular ``(n_tasks, n_eval_points)``
    matrix is also accepted (a cheaper, banded protocol): ``final_average`` and
    ``per_task_final`` still work, while ``backward_transfer`` and
    ``forgetting`` require the square protocol and say so with a ``ValueError``.
    """

    acc: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.acc, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(
                f"acc must be 2-D (n_tasks, n_eval_points); got shape {a.shape}"
            )
        self.acc = a

    # ------------------------------------------------------------------ shapes
    @property
    def n_tasks(self) -> int:
        """Number of rows: tasks trained, in stream order."""
        return int(self.acc.shape[0])

    @property
    def n_eval_points(self) -> int:
        """Number of columns: evaluation points per row (== n_tasks when square)."""
        return int(self.acc.shape[1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinualCurve):
            return NotImplemented
        return self.acc.shape == other.acc.shape and bool(np.array_equal(self.acc, other.acc))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ContinualCurve(n_tasks={self.n_tasks}, n_eval_points={self.n_eval_points})"

    def _require_square(self, name: str) -> int:
        """Validate the full protocol and return ``n``, else raise."""
        n_rows, n_cols = self.acc.shape
        if n_rows != n_cols:
            raise ValueError(
                f"{name} is only defined for a square (n_tasks, n_tasks) "
                f"lower-triangular matrix from the full protocol; got shape "
                f"{self.acc.shape}. The rectangular banded protocol has no "
                f"per-task column identity."
            )
        upper = self.acc[np.triu_indices(n_rows, k=1)]
        filled = upper[~np.isnan(upper) & (upper != 0.0)]
        if filled.size:
            raise ValueError(
                f"{name}: {filled.size} entr{'y' if filled.size == 1 else 'ies'} "
                f"above the diagonal are non-zero, so this is not a "
                f"lower-triangular accuracy matrix (entries with j > i are "
                f"never measured)."
            )
        return n_rows

    # --------------------------------------------------------------- summaries
    def final_average(self) -> float:
        """Mean accuracy on all tasks at the end of the stream (last row)."""
        if self.acc.size == 0:
            return float("nan")
        return float(np.nanmean(self.acc[-1]))

    def backward_transfer(self) -> float:
        """BWT = ``mean_j (acc[n-1, j] - acc[j, j])`` (Lopez-Paz & Ranzato 2017).

        How much the *later* tasks changed accuracy on the earlier ones, from
        the moment each was learned to the end. Negative = interference,
        positive = later tasks helped earlier ones. Averaged over all ``n``
        tasks, so GEM's over-``T-1`` value is this one times ``n/(n-1)``.
        """
        n = self._require_square("backward_transfer")
        if n == 0:
            return float("nan")
        return float(np.mean(self.acc[n - 1, :] - np.diag(self.acc)))

    def forgetting(self) -> float:
        """F = ``mean_j (max_{i>=j} acc[i, j] - acc[n-1, j])`` (Chaudhry et al. 2018).

        Mean over tasks of the drop from the *best* accuracy ever observed on
        that task to its accuracy at the end. Every task is counted, including
        the last (whose drop is 0 by construction).
        """
        n = self._require_square("forgetting")
        if n == 0:
            return float("nan")
        best = np.array([np.nanmax(self.acc[j:, j]) for j in range(n)], dtype=float)
        return float(np.mean(best - self.acc[n - 1, :]))

    def per_task_final(self) -> dict[int, float]:
        """``{task_id: accuracy at the end}``, read off the last row."""
        if self.acc.size == 0:
            return {}
        return {j: float(v) for j, v in enumerate(self.acc[-1])}

    def summary(self) -> dict[str, float]:
        """All three statistics plus the task count, keyed for logging."""
        out = {
            "final_average": self.final_average(),
            "n_tasks": float(self.n_tasks),
        }
        try:
            out["backward_transfer"] = self.backward_transfer()
            out["forgetting"] = self.forgetting()
        except ValueError:
            out["backward_transfer"] = float("nan")
            out["forgetting"] = float("nan")
        return out


# ----------------------------------------------------------- suite duck-typing
_TRAIN_HOOKS = ("train_task", "train", "fit")
_EVAL_HOOKS = ("evaluate_task", "eval_task", "evaluate", "eval")
_DATA_HOOKS = ("eval_batches", "eval_data", "test_batches", "test_data")

EvalHook = Callable[..., Any]


def _first_callable(obj: Any, names: Iterable[str]) -> tuple[str | None, EvalHook | None]:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return name, fn
    return None, None


def _accepts_n_positional(fn: Callable[..., Any], n: int) -> bool:
    """True if ``fn`` can be called with exactly ``n`` positional arguments.

    Only used to decide whether an optional argument (the task id, the
    predictor) can be handed to a user-supplied hook. Unknown signatures
    (C builtins) are treated as not accepting it, which is the safe default.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    n_positional = 0
    n_required = 0
    has_var_args = False
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n_positional += 1
            if p.default is p.empty:
                n_required += 1
        elif p.kind is p.VAR_POSITIONAL:
            has_var_args = True
    if has_var_args:
        return True
    return n_required <= n <= n_positional


def _call_trimmed(fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn`` with the longest prefix of ``args`` it accepts."""
    for k in range(len(args), -1, -1):
        if _accepts_n_positional(fn, k):
            return fn(*args[:k])
    n_given = len(args)
    raise TypeError(
        f"{fn!r} does not accept {n_given} positional argument(s) or fewer "
        f"(supported hook signatures pass at most the task id and the predictor)"
    )


class _SuiteAdapter:
    """Normalises the plausible shapes of a task suite into ``train``/``evaluate``.

    The experiment runner owns the suite; this module only orchestrates. Two
    shapes are supported and dispatched on structure, never on guesses:

    ``suite`` object
        Exposes ``n_tasks`` (or ``__len__``) plus one training hook named
        ``train_task``/``train``/``fit``, and either

        * an evaluation hook ``evaluate_task``/``eval_task``/``evaluate``/
          ``eval`` returning a float or a mapping with ``accuracy``/``acc``, or
        * a data hook ``eval_batches``/``eval_data``/``test_batches``/
          ``test_data`` yielding ``(x, y)`` pairs (or a dict with
          ``x``/``inputs`` and ``y``/``labels``/``targets``). In that case
          ``summarise`` calls ``pred_fn`` itself and scores with
          :func:`accuracy`, so the suite only has to supply data.

        Hooks called with a task id may be written as ``hook(i)`` or ``hook(i,
        pred_fn)``; the signature decides, and ``pred_fn`` is the caller's
        predictor for the *current* model state.

    sequence of task objects
        Each task exposes its own ``train``/``fit`` (called with ``pred_fn`` if
        it accepts one) and ``evaluate``/``eval`` or a data hook as above.
    """

    def __init__(self, suite: Any):
        self.suite = suite
        if isinstance(suite, (list, tuple)) or (
            hasattr(suite, "__getitem__")
            and hasattr(suite, "__len__")
            and _first_callable(suite, _TRAIN_HOOKS + _EVAL_HOOKS + _DATA_HOOKS) == (None, None)
        ):
            self.mode = "sequence"
            self._tasks = list(suite)
            if not self._tasks:
                raise ValueError("summarise: the task sequence is empty")
            self.n_tasks = len(self._tasks)
        else:
            self.mode = "suite"
            self._tasks = []
            self.n_tasks = _suite_n_tasks(suite)
            _, self._train = _first_callable(suite, _TRAIN_HOOKS)
            _, self._eval = _first_callable(suite, _EVAL_HOOKS)
            self._data = None
            if self._train is None:
                raise TypeError(
                    "summarise: suite has no training hook; expected one of "
                    f"{_TRAIN_HOOKS} on {type(suite).__name__}"
                )
            if self._eval is None:
                _, self._data = _first_callable(suite, _DATA_HOOKS)
                if self._data is None:
                    raise TypeError(
                        "summarise: suite has no evaluation hook; expected one of "
                        f"{_EVAL_HOOKS} or a data hook {_DATA_HOOKS} on "
                        f"{type(suite).__name__}"
                    )

    # ------------------------------------------------------------------ train
    def train(self, i: int, pred_fn: Callable[..., Any]) -> Any:
        if self.mode == "sequence":
            task = self._tasks[i]
            _, fn = _first_callable(task, _TRAIN_HOOKS)
            if fn is None:
                raise TypeError(
                    f"summarise: task {i} ({type(task).__name__}) has no train hook"
                )
            return _call_trimmed(fn, pred_fn)
        return _call_trimmed(self._train, i, pred_fn)

    # --------------------------------------------------------------- evaluate
    def evaluate(self, i: int, pred_fn: Callable[..., Any]) -> float:
        if self.mode == "sequence":
            task = self._tasks[i]
            _, eval_fn = _first_callable(task, _EVAL_HOOKS)
            data_fn = None
        else:
            task = self.suite
            eval_fn = self._eval
            data_fn = None if eval_fn is not None else self._data

        if eval_fn is not None:
            if self.mode == "sequence":
                value = _call_trimmed(eval_fn, pred_fn)
            else:
                value = _call_trimmed(eval_fn, i, pred_fn)
            return _as_accuracy(value, i)

        if data_fn is None:
            _, data_fn = _first_callable(task, _DATA_HOOKS)
        if data_fn is None:  # pragma: no cover - guarded in __init__
            raise TypeError(f"summarise: task {i} exposes no evaluation hook")
        if self.mode == "sequence":
            data = _call_trimmed(data_fn, pred_fn)
        else:
            data = _call_trimmed(data_fn, i, pred_fn)

        total = 0
        correct = 0.0
        for x, y in _iter_batches(data):
            y_true = np.asarray(y)
            acc = accuracy(_call_trimmed(pred_fn, x), y_true)
            n = int(y_true.shape[0]) if y_true.ndim else 1
            total += n
            correct += acc * n
        if total == 0:
            return float("nan")
        return correct / total


def _suite_n_tasks(suite: Any) -> int:
    n = getattr(suite, "n_tasks", None)
    if n is None:
        try:
            n = len(suite)
        except TypeError:
            n = None
    if n is None:
        raise TypeError(
            "summarise: cannot determine the number of tasks; expose `n_tasks` "
            f"or `__len__` on {type(suite).__name__}"
        )
    n = int(n)
    if n < 0:
        raise ValueError(f"summarise: negative task count {n}")
    return n


def _as_accuracy(value: Any, i: int) -> float:
    """Coerce an evaluation hook's return value to a float accuracy."""
    if isinstance(value, Mapping):
        for key in ("accuracy", "acc", "top1", "top_1"):
            if key in value:
                return float(value[key])
        raise ValueError(
            f"summarise: evaluation hook for task {i} returned a mapping without "
            f"an accuracy key (looked for accuracy/acc/top1); got keys {sorted(value)}"
        )
    return float(value)


def _is_arraylike(x: Any) -> bool:
    return isinstance(x, (np.ndarray, list, tuple, int, float, np.generic))


def _iter_batches(data: Any) -> Iterable[tuple[Any, Any]]:
    """Yield ``(x, y)`` from the several shapes a data hook might return."""
    if data is None:
        return
    if isinstance(data, Mapping):
        x = data.get("x", data.get("inputs", data.get("features")))
        y = data.get("y", data.get("labels", data.get("targets")))
        if x is None or y is None:
            raise ValueError(
                "summarise: data mapping must contain x/inputs and y/labels; "
                f"got keys {sorted(data)}"
            )
        yield x, y
        return
    if (
        isinstance(data, tuple)
        and len(data) == 2
        and _is_arraylike(data[0])
        and _is_arraylike(data[1])
    ):
        yield data
        return
    for batch in data:
        if isinstance(batch, Mapping):
            yield from _iter_batches(batch)
        else:
            x, y = batch
            yield x, y


# ------------------------------------------------------------------- summarise
def summarise(
    pred_fn: Callable[..., Any],
    suite: Any,
    *,
    eval_tasks: int | None = None,
) -> ContinualCurve:
    """Run the sequential protocol and return the :class:`ContinualCurve`.

    For each task ``i`` in stream order: train on task ``i``, then evaluate on
    every task ``j <= i`` and store ``acc[i, j]``. Tasks are never revisited for
    training, and earlier data is never supplied again - that is what makes the
    forgetting numbers meaningful.

    Parameters
    ----------
    pred_fn
        The predictor for the current model state, i.e. ``pred_fn(x) ->
        predictions``. It is handed to the suite's hooks (so a suite that owns
        its own evaluation loop still scores the arm you think it does) and is
        called directly by this function when the suite exposes data instead.
        A zero-argument ``pred_fn`` is also accepted for the data path.
    suite
        Either a suite object or a sequence of per-task objects; see
        :class:`_SuiteAdapter` for the exact hook names and signatures.
    eval_tasks
        Budget knob: run and record only the first ``eval_tasks`` tasks of the
        suite (default: all of them). Truncating the stream changes the
        protocol, so a truncated curve is not comparable with a full one -
        report the value you used.

    Returns
    -------
    ContinualCurve
        Square and lower triangular, with entries above the diagonal left at
        ``0.0`` (they are never measured).
    """
    if eval_tasks is not None and int(eval_tasks) < 1:
        raise ValueError(f"summarise: eval_tasks must be >= 1, got {eval_tasks!r}")

    adapter = _SuiteAdapter(suite)
    n = adapter.n_tasks if eval_tasks is None else min(int(eval_tasks), adapter.n_tasks)
    acc = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        adapter.train(i, pred_fn)
        for j in range(i + 1):
            acc[i, j] = adapter.evaluate(j, pred_fn)
    return ContinualCurve(acc=acc)


# --------------------------------------------------------------- SynOp / energy
def coincident_dense_macs(n_params: int, n_samples: int) -> int:
    """Multiply-accumulate count of a dense (coincident) arm.

    Every parameter participates in one multiply-accumulate per sample, whether
    or not its input is active: ``n_params * n_samples``. This is the honest
    denominator for the SynOp comparison - a *higher* count than a sparse arm
    can ever post, because it charges for dead weights too.

    Unit caveat: a MAC is a multiply *and* an add, while a SynOp is a single
    accumulate (a spike is 0/1, so no multiply is needed). The two are not the
    same unit; the ratio is a first-order operation-count comparison, so report
    which arm used which unit. This function counts arithmetic, not joules.
    """
    n_params = int(n_params)
    n_samples = int(n_samples)
    if n_params < 0 or n_samples < 0:
        raise ValueError(
            f"coincident_dense_macs: counts must be non-negative, got "
            f"n_params={n_params}, n_samples={n_samples}"
        )
    return n_params * n_samples


def _as_float(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _per_sample(stats: Any, arm: str, notes: list[str]) -> tuple[float | None, float | None]:
    """Resolve ``synops_per_sample`` and the total, recording what was missing."""
    if not isinstance(stats, Mapping):
        notes.append(f"{arm}: expected a mapping of stats, got {type(stats).__name__}")
        return None, None

    per_sample = _as_float(stats.get("synops_per_sample"))
    total = _as_float(stats.get("total_synops"))
    n_samples = _as_float(stats.get("n_samples"))

    if per_sample is None and total is not None:
        if n_samples is not None and n_samples > 0:
            per_sample = total / n_samples
        else:
            notes.append(
                f"{arm}: has total_synops={total:g} but no usable n_samples, so "
                f"per-sample SynOps cannot be derived"
            )
    if per_sample is not None and total is None and n_samples is not None and n_samples > 0:
        total = per_sample * n_samples
    if per_sample is None:
        notes.append(f"{arm}: no synops_per_sample and no (total_synops, n_samples) pair")
    return per_sample, total


def synop_comparison(brain_stats: dict, mlp_stats: dict) -> dict:
    """Operation-count comparison between the spiking and dense arms.

    Parameters
    ----------
    brain_stats, mlp_stats
        Mappings that report either ``synops_per_sample``, or ``total_synops``
        together with ``n_samples`` (the total is divided by the sample count).
        ``mlp_stats`` may instead report ``n_params``, in which case the dense
        arm's coincident MACs per sample (``n_params``) are used and flagged in
        ``mlp_metric``. Missing or nonsensical keys are *not* an error: the
        affected field comes back as ``None`` and ``warnings`` explains why.

    Returns
    -------
    dict
        ``brain_synops_per_sample``, ``mlp_synops_per_sample``, ``mlp_metric``,
        ``brain_to_mlp_ratio`` (< 1 favours the brain arm), ``mlp_to_brain_ratio``
        (the saving factor), ``synops_saved_per_sample``, totals where
        derivable, a clearly-labelled Loihi energy projection for the SynOp arm,
        ``warnings`` and ``caveat``. Counts only - see the module docstring:
        these are not joules and must not be reported as an energy win.
    """
    notes: list[str] = []
    brain_per, brain_total = _per_sample(brain_stats, "brain", notes)

    mlp_per, mlp_total = _per_sample(mlp_stats, "mlp", notes)
    mlp_metric: str | None = "synops" if mlp_per is not None else None
    if mlp_per is None and isinstance(mlp_stats, Mapping):
        n_params = _as_float(mlp_stats.get("n_params"))
        if n_params is not None:
            mlp_per = n_params
            n_samples_mlp = int(_as_float(mlp_stats.get("n_samples")) or 0)
            mlp_total = coincident_dense_macs(int(n_params), n_samples_mlp)
            mlp_metric = "coincident_dense_macs"
            notes.append(
                "mlp: no synops reported; used coincident dense MACs per sample "
                "= n_params (unit mismatch with SynOps: MAC = multiply + accumulate)"
            )

    ratio = ratio_inv = saved = None
    if brain_per is not None and mlp_per is not None and mlp_per > 0:
        ratio = brain_per / mlp_per
        ratio_inv = mlp_per / brain_per
        saved = mlp_per - brain_per
    elif brain_per is not None and mlp_per is not None:
        notes.append("mlp operation count is 0, so no ratio is defined")

    brain_pj = None if brain_per is None else brain_per * LOIHI_PJ_PER_SYNOP
    mlp_pj = None if mlp_per is None or mlp_metric != "synops" else mlp_per * LOIHI_PJ_PER_SYNOP
    if mlp_metric == "coincident_dense_macs" and mlp_per is not None:
        notes.append(
            "no Loihi energy projection for the mlp arm: Loihi executes SynOps, "
            "not dense MACs, so the pJ constant does not apply to it"
        )

    return {
        "brain_synops_per_sample": brain_per,
        "brain_total_synops": brain_total,
        "mlp_synops_per_sample": mlp_per,
        "mlp_total_synops": mlp_total,
        "mlp_metric": mlp_metric,
        "brain_to_mlp_ratio": ratio,
        "mlp_to_brain_ratio": ratio_inv,
        "synops_saved_per_sample": saved,
        "loihi_pj_per_synop": LOIHI_PJ_PER_SYNOP,
        "brain_pj_per_sample_loihi_projection": brain_pj,
        "mlp_pj_per_sample_loihi_projection": mlp_pj,
        "biological_j_per_synaptic_event": BIOLOGICAL_J_PER_SYNAPTIC_EVENT,
        "warnings": notes,
        "caveat": _ENERGY_CAVEAT,
    }


# ------------------------------------------------------------------ self-check
class _ToySuite:
    """Suite-mode toy: analytic per-task accuracies supplied through a table."""

    def __init__(self, table: np.ndarray):
        self.table = np.asarray(table, dtype=float)
        self.n_tasks = int(self.table.shape[0])
        self.after = 0
        self.train_log: list[int] = []
        self.eval_log: list[tuple[int, int]] = []

    def train_task(self, i, pred_fn):  # noqa: ANN001 - duck-typed hook
        self.train_log.append(i)
        self.after = i

    def evaluate_task(self, j, pred_fn):  # noqa: ANN001 - duck-typed hook
        self.eval_log.append((self.after, j))
        return self.table[self.after, j]


class _ToyStream:
    """Sequence-mode toy: per-task objects that expose *data*, not accuracy.

    Each task yields two uneven batches, so the score only comes out right if
    ``summarise`` weights batches by sample count (6 and 4 samples): a task
    scores 1.0 while it is the task just learned and 0.8 afterwards, and the
    weighted mean of (1.0 x 6, 0.5 x 4) is 0.8, whereas an unweighted mean
    would give 0.75.
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self.after = 0
        self._tasks = [_ToyTask(self, j) for j in range(n_tasks)]

    def __len__(self) -> int:
        return self.n_tasks

    def __getitem__(self, i: int) -> "_ToyTask":
        return self._tasks[i]

    def batch_b(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """The second batch: 4 samples, 2 correct unless task ``j`` is current."""
        x = np.zeros((4, 1), dtype=np.float64)
        y = np.zeros(4, dtype=np.int64) if self.after == j else np.array([0, 0, 1, 1])
        return x, y


class _ToyTask:
    """One task of :class:`_ToyStream`; exposes train + a data hook."""

    def __init__(self, stream: _ToyStream, j: int):
        self.stream = stream
        self.j = j

    def train(self, pred_fn):  # noqa: ANN001 - duck-typed hook
        self.stream.after = self.j

    def eval_batches(self):
        yield np.zeros((6, 1), dtype=np.float64), np.zeros(6, dtype=np.int64)
        yield self.stream.batch_b(self.j)


def _check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(f"self-check failed: {label}")


def _check_close(got: float, want: float, label: str) -> None:
    if not np.isclose(got, want, rtol=1e-12, atol=1e-15):
        raise AssertionError(f"self-check failed: {label}: got {got!r}, expected {want!r}")


def _brute_force(matrix: np.ndarray) -> tuple[float, float, float]:
    """Independent triple-loop reference for the three statistics (small n)."""
    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    final = sum(a[n - 1, j] for j in range(n)) / n
    bwt = sum(a[n - 1, j] - a[j, j] for j in range(n)) / n
    frg = 0.0
    for j in range(n):
        best = max(a[i, j] for i in range(j, n))
        frg += best - a[n - 1, j]
    return final, bwt, frg / n


def self_check(*, verbose: bool = True) -> dict[str, float]:
    """Hand-computed assertions for every metric. Returns the values checked.

    ``python3 brain/metrics.py`` runs this from a clean checkout; it is also
    importable so the test-suite can call ``self_check(verbose=False)``.
    """
    results: dict[str, float] = {}

    # ---------------------------------------------------------------- accuracy
    _check_close(accuracy(np.array([0, 1, 1]), np.array([0, 1, 0])), 2.0 / 3.0, "labels")
    _check_close(accuracy(np.array([0.0, 1.0]), np.array([0, 1])), 1.0, "float labels")
    _check_close(
        accuracy(np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([0, 1])), 1.0, "argmax scores"
    )
    _check_close(
        accuracy(np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[1, 0], [0, 1]])), 1.0, "one-hot"
    )
    _check_close(
        accuracy(np.array([[0.9, 0.1]]), np.array([[0], [1]])[0]), 1.0, "2-D truth col"
    )
    _check(np.isnan(accuracy(np.array([]), np.array([]))), "empty -> nan")
    try:
        accuracy(np.array([0, 1]), np.array([0, 1, 0]))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("self-check failed: length mismatch must raise")
    results["accuracy_labels_2of3"] = 2.0 / 3.0

    # ------------------------------------- hand-computed BWT and forgetting (A)
    # acc = [[1.0, ...], [0.9, 1.0, ...], [0.6, 0.7, 1.0]]   (only j <= i measured)
    # final_average = (0.6 + 0.7 + 1.0) / 3 = 0.7666666666666667
    # BWT = ((0.6-1.0) + (0.7-1.0) + (1.0-1.0)) / 3 = -0.7/3 = -0.23333333333333334
    # forgetting = ((max(1.0,0.9,0.6)-0.6) + (max(1.0,0.7)-0.7) + (1.0-1.0)) / 3
    #            = (0.4 + 0.3 + 0.0) / 3 = 0.23333333333333334
    hand_a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 1.0, 0.0],
            [0.6, 0.7, 1.0],
        ]
    )
    curve_a = ContinualCurve(acc=hand_a)
    _check_close(curve_a.final_average(), 0.7666666666666667, "A final_average")
    _check_close(curve_a.backward_transfer(), -0.23333333333333334, "A BWT")
    _check_close(curve_a.forgetting(), 0.23333333333333334, "A forgetting")
    _check(curve_a.per_task_final() == {0: 0.6, 1: 0.7, 2: 1.0}, "A per_task_final")
    results["A_backward_transfer"] = curve_a.backward_transfer()
    results["A_forgetting"] = curve_a.forgetting()

    # ------------------------------------- hand-computed max-drop, case B
    # acc = [[0.8, 0, 0], [0.9, 1.0, 0], [0.7, 0.6, 1.0]]
    # BWT = ((0.7-0.8) + (0.6-1.0) + 0) / 3 = -0.5/3 = -0.16666666666666666
    # forgetting = ((max(0.8,0.9,0.7)-0.7) + (max(1.0,0.6)-0.6) + 0) / 3
    #            = (0.2 + 0.4 + 0.0) / 3 = 0.2
    # NOTE j=0: the *best* accuracy (0.9) came after the diagonal (0.8), so
    # "diagonal - final" would give 0.1 and would be wrong; the correct
    # max-drop forgetting is 0.2. This is the discriminating case.
    hand_b = np.array(
        [
            [0.8, 0.0, 0.0],
            [0.9, 1.0, 0.0],
            [0.7, 0.6, 1.0],
        ]
    )
    curve_b = ContinualCurve(acc=hand_b)
    _check_close(curve_b.backward_transfer(), -0.16666666666666666, "B BWT")
    _check_close(curve_b.forgetting(), 0.2, "B forgetting (max-drop, not diag-final)")
    _check_close(curve_b.final_average(), 0.7666666666666667, "B final_average")
    results["B_forgetting"] = curve_b.forgetting()

    # ------------------------------------------- no-drift control: BWT = F = 0
    hand_c = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    curve_c = ContinualCurve(acc=hand_c)
    _check_close(curve_c.backward_transfer(), 0.0, "C BWT zero")
    _check_close(curve_c.forgetting(), 0.0, "C forgetting zero")
    _check_close(curve_c.final_average(), 1.0, "C final_average")

    # ------------------------- randomised cross-check against explicit loops
    rng = np.random.default_rng(0)
    n = 7
    rand = np.tril(rng.random((n, n)))
    curve_r = ContinualCurve(acc=rand)
    want_final, want_bwt, want_frg = _brute_force(rand)
    _check_close(curve_r.final_average(), want_final, "random final_average")
    _check_close(curve_r.backward_transfer(), want_bwt, "random BWT")
    _check_close(curve_r.forgetting(), want_frg, "random forgetting")

    # single-task and empty edge cases
    _check_close(ContinualCurve(acc=np.array([[0.5]])).backward_transfer(), 0.0, "1-task BWT")
    _check_close(ContinualCurve(acc=np.array([[0.5]])).forgetting(), 0.0, "1-task forgetting")
    _check(np.isnan(ContinualCurve(acc=np.zeros((0, 0))).final_average()), "empty final")
    # upper-triangle values are a protocol violation, not silently used
    try:
        ContinualCurve(acc=np.array([[1.0, 0.5], [0.9, 1.0]])).backward_transfer()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("self-check failed: filled upper triangle must raise")
    # rectangular (banded) matrices: final row works, BWT/forgetting refuse
    rect = ContinualCurve(acc=np.array([[1.0], [0.8], [0.6]]))
    _check_close(rect.final_average(), 0.6, "rect final_average")
    _check(rect.per_task_final() == {0: 0.6}, "rect per_task_final")
    try:
        rect.backward_transfer()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("self-check failed: rectangular BWT must raise")

    # ---------------------------------------------------------------- summarise
    table = np.array(
        [
            [0.9, 0.0, 0.0],
            [0.7, 0.8, 0.0],
            [0.4, 0.5, 1.0],
        ]
    )
    suite = _ToySuite(table)
    curve = summarise(lambda x: x, suite)
    _check(np.array_equal(curve.acc, table), f"summarise matrix: {curve.acc!r}")
    _check(suite.train_log == [0, 1, 2], f"train order: {suite.train_log}")
    _check(suite.eval_log == [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)], "eval order")
    want_final_s, want_bwt_s, want_frg_s = _brute_force(table)
    _check_close(curve.backward_transfer(), want_bwt_s, "summarise BWT")
    _check_close(curve.forgetting(), want_frg_s, "summarise forgetting")
    _check_close(curve.final_average(), want_final_s, "summarise final_average")
    # eval_tasks budget truncates the stream (first k tasks only)
    small = summarise(lambda x: x, _ToySuite(table), eval_tasks=2)
    _check(small.acc.shape == (2, 2), "eval_tasks shape")
    _check_close(small.final_average(), 0.75, "eval_tasks final_average")

    # data-path suite: uneven batches must be weighted by sample count.
    # accuracy[i, j] = 1.0 if i == j else 0.8  ->  [[1,0,0],[.8,1,0],[.8,.8,1]]
    stream = _ToyStream(3)
    curve_d = summarise(lambda x: np.zeros(len(x), dtype=np.int64), stream)
    expected_d = np.array([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [0.8, 0.8, 1.0]])
    _check(np.array_equal(curve_d.acc, expected_d), f"data-path matrix: {curve_d.acc!r}")
    _check_close(curve_d.final_average(), 0.8666666666666667, "data-path final_average")
    _check_close(curve_d.backward_transfer(), -0.13333333333333333, "data-path BWT")
    _check_close(curve_d.forgetting(), 0.13333333333333333, "data-path forgetting")
    # predict-then-score path must not be fooled by an unweighted batch mean
    _check(
        not np.isclose(curve_d.final_average(), (1.0 + 0.75 + 0.75) / 3.0),
        "batch weighting",
    )

    # ------------------------------------------------------------ SynOp / MACs
    _check(coincident_dense_macs(1000, 10) == 10_000, "coincident macs")
    try:
        coincident_dense_macs(-1, 10)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("self-check failed: negative MAC count must raise")

    cmp_full = synop_comparison(
        {"synops_per_sample": 1000.0, "total_synops": 500_000, "n_samples": 500},
        {"synops_per_sample": 8000.0},
    )
    _check_close(cmp_full["brain_to_mlp_ratio"], 0.125, "synop ratio")
    _check_close(cmp_full["mlp_to_brain_ratio"], 8.0, "synop inverse ratio")
    _check_close(cmp_full["synops_saved_per_sample"], 7000.0, "synops saved")
    _check(cmp_full["warnings"] == [], f"synop warnings: {cmp_full['warnings']}")
    _check_close(
        cmp_full["brain_pj_per_sample_loihi_projection"],
        1000.0 * LOIHI_PJ_PER_SYNOP,
        "loihi projection",
    )
    cmp_derived = synop_comparison(
        {"total_synops": 500_000, "n_samples": 500},
        {"n_params": 1000, "n_samples": 500},
    )
    _check_close(cmp_derived["brain_synops_per_sample"], 1000.0, "derived from totals")
    _check(cmp_derived["mlp_metric"] == "coincident_dense_macs", "macs fallback flagged")
    _check_close(cmp_derived["mlp_synops_per_sample"], 1000.0, "macs per sample == n_params")
    _check(cmp_derived["mlp_pj_per_sample_loihi_projection"] is None, "no pJ for MAC arm")
    # missing / junk input must never raise
    for brain_stats, mlp_stats in (
        ({}, {}),
        (None, None),
        ({"total_synops": 10}, {"synops_per_sample": 10}),
        ({"synops_per_sample": 10}, {"total_synops": 5, "n_samples": 0}),
        ({"synops_per_sample": 10}, {"synops_per_sample": 0}),
    ):
        out = synop_comparison(brain_stats, mlp_stats)
        _check(isinstance(out, dict), "graceful synop_comparison")
        _check(isinstance(out["warnings"], list) and out["caveat"], "caveat present")
    _check(
        "not joules" in synop_comparison({}, {})["caveat"].lower()
        or "NOT JOULES" in synop_comparison({}, {})["caveat"],
        "caveat states operation counts",
    )
    _check(isinstance(__doc__, str) and "23.6 pJ" in __doc__, "module doc cites Loihi pJ")
    results["synop_ratio_1000_over_8000"] = 0.125

    # determinism: identical inputs, identical outputs
    _check(
        np.array_equal(summarise(lambda x: x, _ToySuite(table)).acc, curve.acc),
        "deterministic summarise",
    )

    # curio log line for a human reading the output
    if verbose:
        print(f"accuracy(2 of 3 correct)      = {2.0 / 3.0:.12f}")
        print(f"A  final_average              = {curve_a.final_average():.12f}  (2.3/3)")
        print(f"A  backward_transfer (BWT)    = {curve_a.backward_transfer():.12f}  (-0.7/3)")
        print(f"A  forgetting                 = {curve_a.forgetting():.12f}  (0.7/3)")
        print(f"B  backward_transfer (BWT)    = {curve_b.backward_transfer():.12f}  (-0.5/3)")
        print(f"B  forgetting (max-drop)      = {curve_b.forgetting():.12f}  (0.6/3)")
        print(f"R  random 7-task cross-check  = BWT {curve_r.backward_transfer():.12f}, "
              f"F {curve_r.forgetting():.12f} vs loops OK")
        print(f"summarise() square matrix     = {curve.acc.tolist()}")
        print(f"summarise() data path (inc. weighted batches) OK")
        print(f"synop ratio (1000 vs 8000)    = {cmp_full['brain_to_mlp_ratio']:.6f} "
              f"(brain/mlp; < 1 favours the brain arm)")
        print("ALL METRIC SELF-CHECKS PASSED")
    return results


if __name__ == "__main__":
    self_check()
