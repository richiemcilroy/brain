"""Class-incremental continual-learning tasks on MNIST.

Why these two benchmarks
------------------------
Quantifying catastrophic forgetting needs a task sequence in which learning a
later task *can* overwrite an earlier one. The two standard MNIST constructions
stress different failure modes:

* **Split-MNIST** (``permute=False``) partitions the ten digit classes into
  ``n_tasks`` disjoint groups. Labels stay global (a task predicts digit ids,
  not local indices), but a learner can still dodge interference by never
  reusing a readout unit across tasks, so this is the easier benchmark.
* **Permuted-MNIST** (``permute=True``) gives every task all ten classes and
  reorders the 784 pixels with a task-specific fixed permutation. The input
  distribution is unchanged and only the input->label mapping moves, so readout
  weights must be reused and interference is unavoidable. That is the benchmark
  where local, weight-protecting plasticity should pay off, and the one this
  repo treats as primary.

Both suites keep a single **global** label space, which is what "class
incremental" means: task ``t`` of Split-MNIST predicts class ids 0/1, not 0/1
relative to its own group. Data is read from a local cache
(``data/mnist.npz``); no network access happens at runtime. Every stochastic
choice comes from a locally seeded ``numpy.random.Generator`` (or, for encoder
sampling, the array backend's seeded RNG), so an experiment is reproducible from
one integer seed.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # normal package import
    from .backend import Backend, get_backend, has_mlx
except ImportError:  # pragma: no cover - direct execution: `python3 brain/tasks.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from brain.backend import Backend, get_backend, has_mlx

N_CLASSES = 10
N_INPUT = 784
DEFAULT_MNIST_PATH = "data/mnist.npz"

# Permuted-MNIST reuses the whole dataset once per task in the literature, which
# would cost ~190 MB of float32 per task and make the local benchmark slow to
# build and to train. The suite therefore takes a balanced stratified subset and
# records the cap in ``TaskSuite.meta``. The brain substrate and the dense
# baseline are always handed exactly the same samples, so comparisons are fair.
PERMUTED_TRAIN_PER_CLASS = 1000
PERMUTED_TEST_PER_CLASS = 100

# Distinct SeedSequence streams so that subsetting, ordering and per-task pixel
# permutations cannot accidentally share a random stream.
_SUBSET_STREAM = 0xBEEF
_ORDER_STREAM = 0x1000

__all__ = [
    "Task",
    "TaskSuite",
    "RateEncoder",
    "load_mnist",
    "split_mnist",
    "pixel_permutation",
    "N_CLASSES",
    "N_INPUT",
    "DEFAULT_MNIST_PATH",
]


# --------------------------------------------------------------------- helpers
def _as_images(x: Any, what: str) -> np.ndarray:
    """Validate/convert an image batch to contiguous float32 in ``[0, 1]``."""
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[1:] == (28, 28):
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape[1] != N_INPUT:
        raise ValueError(f"{what} must have shape (n, {N_INPUT}); got {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"{what} must be float images in [0, 1]; got integer dtype {arr.dtype} "
            "(scale raw MNIST with .astype(np.float32) / 255.0 first)"
        )
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.size:
        lo, hi = float(arr.min()), float(arr.max())
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            raise ValueError(f"{what} must lie in [0, 1]; got range [{lo}, {hi}]")
    return arr


def _as_labels(y: Any, what: str) -> np.ndarray:
    """Validate/convert labels to a flat contiguous int64 array."""
    arr = np.asarray(y)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{what} must be an integer label array; got dtype {arr.dtype}")
    return np.ascontiguousarray(arr, dtype=np.int64).reshape(-1)


def _raw_images(raw: Any, what: str) -> np.ndarray:
    """Convert raw cached images (uint8 0-255 or float) to float32 in [0, 1]."""
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.integer):
        scale = 255.0 if (arr.size and int(arr.max()) > 1) else 1.0
        arr = arr.astype(np.float32) / np.float32(scale)
    return _as_images(arr, what)


def _rng_for(seed: int, stream: int) -> np.random.Generator:
    """Deterministic generator keyed by ``(seed, stream)``.

    Seeds are normalised into the 32-bit range SeedSequence accepts so that
    negative or very large user seeds still work.
    """
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(stream)])


def _balanced_subset(y: np.ndarray, rng: np.random.Generator, per_class: int) -> np.ndarray:
    """Stratified index sample with ``per_class`` examples of every class."""
    picks = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if idx.size < per_class:
            raise ValueError(f"class {int(c)} has {idx.size} samples, need {per_class}")
        picks.append(rng.permutation(idx)[:per_class])
    return np.concatenate(picks)


# ----------------------------------------------------------------------- tasks
@dataclass
class Task:
    """One task in a continual-learning sequence.

    Attributes:
        name: Identifier, unique within its suite.
        x_train: ``(n, 784)`` float32 images in ``[0, 1]``.
        y_train: ``(n,)`` int64 global class ids in ``[0, n_classes)``.
        x_test: ``(m, 784)`` float32 test images in ``[0, 1]``.
        y_test: ``(m,)`` int64 global test labels.
        classes: Global class ids present in the task's training set, ascending.
    """

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: list[int]

    def __post_init__(self) -> None:
        self.x_train = _as_images(self.x_train, f"{self.name}.x_train")
        self.x_test = _as_images(self.x_test, f"{self.name}.x_test")
        self.y_train = _as_labels(self.y_train, f"{self.name}.y_train")
        self.y_test = _as_labels(self.y_test, f"{self.name}.y_test")
        self.classes = sorted({int(c) for c in self.classes})

        if self.x_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(
                f"{self.name}: x_train has {self.x_train.shape[0]} rows but y_train "
                f"has {self.y_train.shape[0]}"
            )
        if self.x_test.shape[0] != self.y_test.shape[0]:
            raise ValueError(
                f"{self.name}: x_test has {self.x_test.shape[0]} rows but y_test "
                f"has {self.y_test.shape[0]}"
            )
        if self.x_train.shape[0] == 0 or self.x_test.shape[0] == 0:
            raise ValueError(f"{self.name}: train and test sets must both be non-empty")
        if self.y_train.min() < 0 or self.y_test.min() < 0:
            raise ValueError(f"{self.name}: class ids must be non-negative")
        if not self.classes:
            raise ValueError(f"{self.name}: classes must not be empty")
        present = sorted(int(c) for c in np.unique(self.y_train))
        if present != self.classes:
            raise ValueError(
                f"{self.name}: classes {self.classes} do not match the labels present in "
                f"y_train ({present})"
            )
        extra = sorted(set(int(c) for c in np.unique(self.y_test)) - set(self.classes))
        if extra:
            raise ValueError(f"{self.name}: test labels {extra} are not in classes")

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.x_test.shape[0])


@dataclass
class TaskSuite:
    """An ordered class-incremental task sequence over a shared label space."""

    name: str
    n_classes: int
    n_input: int
    tasks: list[Task]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("a TaskSuite needs at least one task")
        if self.n_classes <= 0 or self.n_input <= 0:
            raise ValueError("n_classes and n_input must be positive")
        names = [t.name for t in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate task names {names}")
        for t in self.tasks:
            if t.x_train.shape[1] != self.n_input:
                raise ValueError(
                    f"{t.name}: expected {self.n_input} inputs, got {t.x_train.shape[1]}"
                )
            bad = [c for c in t.classes if not 0 <= c < self.n_classes]
            if bad:
                raise ValueError(f"{t.name}: class ids {bad} outside [0, {self.n_classes})")

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def memory_bytes(self) -> int:
        """Total bytes held by the task arrays (float32 inputs dominate)."""
        return int(sum(
            t.x_train.nbytes + t.y_train.nbytes + t.x_test.nbytes + t.y_test.nbytes
            for t in self.tasks
        ))

    def describe(self) -> str:
        """Multi-line summary used by the self-check and experiment logs."""
        lines = [
            f"{self.name}: {len(self.tasks)} tasks, {self.n_classes} classes, "
            f"{self.n_input} inputs, {self.memory_bytes() / 2 ** 20:.1f} MiB"
        ]
        lines.append(f"  {'task':<24}{'train':>7}{'test':>7}  classes")
        for t in self.tasks:
            classes = ",".join(str(c) for c in t.classes)
            lines.append(f"  {t.name:<24}{t.n_train:>7}{t.n_test:>7}  {classes}")
        lines.append(
            f"  {'total':<24}{sum(t.n_train for t in self.tasks):>7}"
            f"{sum(t.n_test for t in self.tasks):>7}"
        )
        return "\n".join(lines)


# ------------------------------------------------------------------ data loading
def load_mnist(
    path: str = DEFAULT_MNIST_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the local MNIST cache (no download is ever attempted).

    Args:
        path: ``.npz`` file with keys ``x_train``, ``y_train``, ``x_test`` and
            ``y_test``. Relative paths resolve against the current working
            directory, so run from the repository root.

    Returns:
        ``(x_train, y_train, x_test, y_test)`` with images as ``(n, 784)``
        float32 in ``[0, 1]`` and labels as ``(n,)`` int64.

    Raises:
        FileNotFoundError: if the cache is missing.
        KeyError: if the cache lacks one of the expected keys.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"MNIST cache not found at {p!s}: this package reads a local file and never "
            "downloads data (relative paths are resolved from the repo root)."
        )
    with np.load(p) as data:
        required = ("x_train", "y_train", "x_test", "y_test")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"{p!s} is missing keys {missing}; found {sorted(data.files)}")
        x_train = _raw_images(data["x_train"], "x_train")
        y_train = _as_labels(data["y_train"], "y_train")
        x_test = _raw_images(data["x_test"], "x_test")
        y_test = _as_labels(data["y_test"], "y_test")
    return x_train, y_train, x_test, y_test


def pixel_permutation(seed: int, task_index: int, n_input: int = N_INPUT) -> np.ndarray:
    """Pixel order applied to task ``task_index`` in Permuted-MNIST.

    Returns an ``int64`` array ``p`` such that the permuted batch is
    ``x_permuted = x[:, p]`` (and ``x = x_permuted[:, np.argsort(p)]``). The
    permutation depends only on ``(seed, task_index)``, so it is stable across
    runs and independently checkable by a reviewer.
    """
    return np.asarray(_rng_for(seed, task_index).permutation(int(n_input)), dtype=np.int64)


def split_mnist(
    n_tasks: int = 5,
    *,
    seed: int = 0,
    permute: bool = False,
    path: str = DEFAULT_MNIST_PATH,
) -> TaskSuite:
    """Build a class-incremental continual-learning suite from MNIST.

    Args:
        n_tasks: Number of tasks. Must divide 10 (1, 2, 5 or 10) so split mode
            gives every task exactly ``10 / n_tasks`` classes; in permuted mode
            it is simply the number of task-specific pixel permutations.
        seed: Seed for subset selection, per-task shuffling and (permuted mode)
            pixel permutations. Repeating a call reproduces identical arrays.
        permute: ``False`` -> Split-MNIST (disjoint contiguous class groups,
            untouched pixels). ``True`` -> Permuted-MNIST (all ten classes in
            every task, one fixed pixel permutation per task).
        path: Location of the local MNIST ``.npz`` cache.

    Returns:
        A :class:`TaskSuite` whose tasks share one global label space.

    Raises:
        ValueError: if ``n_tasks`` does not divide 10.
    """
    if n_tasks < 1 or N_CLASSES % n_tasks:
        raise ValueError(
            f"n_tasks must divide {N_CLASSES} (1, 2, 5 or 10); got {n_tasks}"
        )
    x_train, y_train, x_test, y_test = load_mnist(path)
    if permute:
        return _permuted_suite(n_tasks, seed, path, x_train, y_train, x_test, y_test)
    return _split_suite(n_tasks, seed, path, x_train, y_train, x_test, y_test)


def _split_suite(
    n_tasks: int,
    seed: int,
    path: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> TaskSuite:
    """Split-MNIST: contiguous disjoint class groups, global labels."""
    per_task = N_CLASSES // n_tasks
    tasks: list[Task] = []
    for t in range(n_tasks):
        classes = list(range(t * per_task, (t + 1) * per_task))
        rng = _rng_for(seed, t)
        train_idx = rng.permutation(np.flatnonzero(np.isin(y_train, classes)))
        test_idx = rng.permutation(np.flatnonzero(np.isin(y_test, classes)))
        tasks.append(Task(
            name=f"task{t}_classes{classes[0]}-{classes[-1]}",
            x_train=x_train[train_idx], y_train=y_train[train_idx],
            x_test=x_test[test_idx], y_test=y_test[test_idx],
            classes=classes,
        ))
    return TaskSuite(
        name=f"split-mnist-{n_tasks}",
        n_classes=N_CLASSES,
        n_input=N_INPUT,
        tasks=tasks,
        meta={
            "mode": "split-mnist",
            "seed": int(seed),
            "n_tasks": int(n_tasks),
            "classes_per_task": per_task,
            "source": str(path),
            "notes": "contiguous disjoint class groups; pixels unpermuted; labels global",
        },
    )


def _permuted_suite(
    n_tasks: int,
    seed: int,
    path: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> TaskSuite:
    """Permuted-MNIST: identical samples, task-specific pixel permutations.

    Every task holds the same balanced subset in the same order and differs
    *only* by its pixel permutation. That control matters experimentally: any
    measured interference is attributable to the changed input->label mapping,
    not to a different sample draw. Labels stay global.
    """
    picker = _rng_for(seed, _SUBSET_STREAM)
    train_idx = _balanced_subset(y_train, picker, PERMUTED_TRAIN_PER_CLASS)
    test_idx = _balanced_subset(y_test, picker, PERMUTED_TEST_PER_CLASS)
    order = _rng_for(seed, _ORDER_STREAM)
    train_idx = order.permutation(train_idx)
    test_idx = order.permutation(test_idx)
    base_x_train, base_y_train = x_train[train_idx], y_train[train_idx]
    base_x_test, base_y_test = x_test[test_idx], y_test[test_idx]

    classes = list(range(N_CLASSES))
    tasks: list[Task] = []
    for t in range(n_tasks):
        p = pixel_permutation(seed, t, N_INPUT)
        tasks.append(Task(
            name=f"task{t}_perm",
            x_train=base_x_train[:, p], y_train=base_y_train.copy(),
            x_test=base_x_test[:, p], y_test=base_y_test.copy(),
            classes=classes,
        ))
    return TaskSuite(
        name=f"permuted-mnist-{n_tasks}",
        n_classes=N_CLASSES,
        n_input=N_INPUT,
        tasks=tasks,
        meta={
            "mode": "permuted-mnist",
            "seed": int(seed),
            "n_tasks": int(n_tasks),
            "train_per_class": PERMUTED_TRAIN_PER_CLASS,
            "test_per_class": PERMUTED_TEST_PER_CLASS,
            "source": str(path),
            "notes": "all 10 classes per task; one fixed pixel permutation per task",
        },
    )


# -------------------------------------------------------------------- encoding
@dataclass
class RateEncoder:
    """Encode one normalized image as an injection current for the simulator.

    The deterministic path maps pixel intensity to a steady-state current,
    ``I_i = gain * x_i``. With ``NeuronConfig.v_thresh = 1`` and a soma time
    constant of tens of ms, only pixels brighter than ``1 / gain`` (about 0.83
    at the default gain) can drive a neuron to threshold under sustained input,
    which keeps the population code sparse - the regime in which event-driven
    simulation is cheap.

    ``poisson=True`` emits a binary population code instead of a graded current:
    each neuron fires with probability ``clip(gain * x_i, 0, 1)``, so the mean
    output tracks the deterministic current in its linear range and saturates at
    1.0 for very bright pixels (a rate ceiling, not a bug). Sampling uses the
    array backend's seeded uniform RNG, drawn *once* at first use and then
    reused. Two consequences are deliberate:

    * the same ``(image, seed)`` always produces the same current, so a training
      run is bit-reproducible without replaying the RNG stream;
    * the encoder never resets the caller's RNG stream. (MLX keeps a single
      global RNG, so the one-time draw advances it once; NumPy uses a private
      generator and leaves the caller's stream untouched.) Reseeding per call
      would instead freeze the simulator's own noise and quietly break dynamics.

    The price is no trial-to-trial variability for a repeated identical input;
    variability comes from the heterogeneous per-neuron thresholds, which is
    enough to exercise a stochastic population code.
    """

    n_input: int = N_INPUT
    gain: float = 1.2
    poisson: bool = False
    seed: int = 0
    _threshold: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.n_input <= 0:
            raise ValueError("n_input must be positive")
        if not self.gain > 0.0:
            raise ValueError("gain must be positive")

    @property
    def n_neurons_out(self) -> int:
        """Length of the returned current vector (one neuron per input pixel)."""
        return int(self.n_input)

    def current(self, x: np.ndarray, be: Backend) -> Any:
        """Encode one image as an injection current vector.

        Args:
            x: ``(n_input,)`` image in ``[0, 1]``; a ``(28, 28)`` array is
                accepted and flattened.
            be: Array backend used for the returned array.

        Returns:
            Backend array of shape ``(n_input,)``: ``gain * x`` normally, or a
            ``{0, 1}`` Bernoulli mask when ``poisson`` is set.

        Raises:
            ValueError: if ``x`` has the wrong shape or lies outside ``[0, 1]``.
        """
        pixels = self._as_pixels(x)
        drive = be.array(pixels) * float(self.gain)
        if not self.poisson:
            return drive
        thresholds = self._sample_thresholds(be)
        prob = be.clip(drive, 0.0, 1.0)
        return be.astype(thresholds < prob, be.float_dtype)

    def _as_pixels(self, x: Any) -> np.ndarray:
        """Validate a single image and return it as flat contiguous float32."""
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape == (28, 28):
            arr = arr.reshape(-1)
        if arr.ndim != 1 or arr.shape[0] != self.n_input:
            raise ValueError(
                f"x must be a single image of shape ({self.n_input},); got {arr.shape}"
            )
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            raise ValueError(f"x must lie in [0, 1]; got range [{lo}, {hi}]")
        return arr

    def _sample_thresholds(self, be: Backend) -> Any:
        """Fixed uniform variates (one per neuron), drawn once and memoised."""
        if self._threshold is None:
            rng = Backend(be.name, seed=self.seed)
            draw = rng.uniform((self.n_neurons_out,))
            self._threshold = np.asarray(be.to_numpy(draw), dtype=np.float32)
        return be.array(self._threshold)


# -------------------------------------------------------------------- self-check
def _self_check() -> None:
    """Interface/consistency check run by ``python3 brain/tasks.py``."""
    t0 = time.perf_counter()
    x_train, y_train, x_test, y_test = load_mnist()
    print(
        f"MNIST cache {DEFAULT_MNIST_PATH}: x_train={x_train.shape} y_train={y_train.shape} "
        f"x_test={x_test.shape} dtype={x_train.dtype}"
    )

    suite = split_mnist(5, seed=0)
    print(suite.describe())
    seen: dict[int, str] = {}
    for task in suite.tasks:
        assert task.x_train.dtype == np.float32 and task.y_train.dtype == np.int64
        assert set(np.unique(task.y_train).tolist()) == set(task.classes)
        assert task.x_train.max() <= 1.0 and task.x_train.min() >= 0.0
        for c in task.classes:
            assert c not in seen, f"class {c} appears in {seen[c]} and {task.name}"
            seen[c] = task.name
    assert sorted(seen) == list(range(N_CLASSES)), "every class must appear exactly once"
    print(f"split-mode class disjointness: OK ({len(seen)}/{N_CLASSES} classes, one task each)")

    permuted = split_mnist(5, seed=0, permute=True)
    print(permuted.describe())
    for task in permuted.tasks:
        assert task.classes == list(range(N_CLASSES))
        assert np.bincount(task.y_train, minlength=N_CLASSES).min() == PERMUTED_TRAIN_PER_CLASS
    p0, p1 = pixel_permutation(0, 0), pixel_permutation(0, 1)
    assert not np.array_equal(p0, p1), "each task needs its own pixel permutation"
    base = permuted.tasks[0].x_train[:, np.argsort(p0)]
    assert np.array_equal(base[:, p1], permuted.tasks[1].x_train), "permutation not applied"
    print("permuted-mode checks: 10 classes/task, per-task permutation applied, balanced subsets")

    del permuted
    again = split_mnist(5, seed=0)
    assert np.array_equal(again.tasks[0].x_train, suite.tasks[0].x_train)
    assert np.array_equal(again.tasks[4].y_test, suite.tasks[4].y_test)
    assert np.array_equal(pixel_permutation(0, 3), pixel_permutation(0, 3))
    print("determinism: repeated builds with seed=0 are identical (OK)")
    del again, x_train, y_train, x_test, y_test

    be = get_backend("numpy", seed=0)
    image = suite.tasks[0].x_train[0]
    encoder = RateEncoder()
    out = encoder.current(image, be)
    assert tuple(out.shape) == (N_INPUT,), f"unexpected shape {out.shape}"
    assert np.allclose(np.asarray(out), encoder.gain * image, atol=1e-6)
    poisson = RateEncoder(poisson=True, seed=0)
    a = np.asarray(poisson.current(image, be))
    b = np.asarray(poisson.current(image, be))
    c = np.asarray(poisson.current(suite.tasks[1].x_train[0], be))
    assert a.shape == (N_INPUT,) and a.dtype == np.float32
    assert np.array_equal(a, b), "poisson encoding must be reproducible"
    assert not np.array_equal(a, c), "poisson encoding must depend on the input"
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    assert 0.0 < float(a.mean()) < 1.0
    print("RateEncoder: shape (784,), gain*x path exact, poisson reproducible and input-dependent")

    if has_mlx():
        mx_be = get_backend("mlx", seed=0)
        mx_out = np.asarray(mx_be.to_numpy(RateEncoder().current(image, mx_be)))
        assert mx_out.shape == (N_INPUT,)
        assert np.allclose(mx_out, encoder.gain * image, atol=1e-5)
        mx_p = RateEncoder(poisson=True, seed=0)
        first = np.asarray(mx_be.to_numpy(mx_p.current(image, mx_be)))
        second = np.asarray(mx_be.to_numpy(mx_p.current(image, mx_be)))
        assert np.array_equal(first, second)
        print("mlx backend: RateEncoder.current OK on Metal")

    print(f"all checks passed in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
