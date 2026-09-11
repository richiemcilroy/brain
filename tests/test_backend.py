"""Backend contract: duplicate-index scatter-add, gather, and NumPy/MLX parity.

These tests are independent of the simulator: they check the array adapter that
every other module builds on, against ``np.add.at`` as the reference for
duplicate-index accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.backend import Backend, has_mlx

DUPLICATE_IDX = np.array([0, 1, 1, 3, 3, 3, 2, 0, 5], dtype=np.int64)
DUPLICATE_VAL = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)


def test_scatter_add_accumulates_duplicate_indices_like_np_add_at(backend_name):
    """Duplicate indices must sum, not overwrite (compare with np.add.at)."""
    be = Backend(backend_name, seed=0)
    expected = np.zeros(6, dtype=np.float32)
    np.add.at(expected, DUPLICATE_IDX, DUPLICATE_VAL)

    out = be.scatter_add(
        be.zeros((6,)),
        be.array(DUPLICATE_IDX, dtype=be.idx_dtype),
        be.array(DUPLICATE_VAL, dtype=be.float_dtype),
    )
    got = be.to_numpy(out)

    assert got.shape == expected.shape, f"scatter_add changed shape: {got.shape} != {expected.shape}"
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-5,
        atol=1e-6,
        err_msg="scatter_add disagrees with np.add.at on duplicate indices",
    )
    assert got[1] == pytest.approx(5.0), "duplicate index 1 did not accumulate (2+3)"
    assert got[3] == pytest.approx(15.0), "triplicate index 3 did not accumulate (4+5+6)"
    assert got[0] == pytest.approx(9.0), "duplicate index 0 did not accumulate (1+8)"
    assert got[4] == 0.0, "scatter_add wrote a value to an untouched index"


def test_scatter_add_empty_index_is_a_noop(backend_name):
    be = Backend(backend_name, seed=0)
    before = be.zeros((4,))
    out = be.scatter_add(
        before,
        be.array(np.empty(0, dtype=np.int64), dtype=be.idx_dtype),
        be.array(np.empty(0, dtype=np.float32), dtype=be.float_dtype),
    )
    np.testing.assert_array_equal(
        be.to_numpy(out), np.zeros(4, dtype=np.float32),
        err_msg="empty scatter_add changed the buffer",
    )


def test_take_gathers_rows_in_requested_order(backend_name):
    """take must reproduce NumPy fancy indexing, duplicates and order included."""
    be = Backend(backend_name, seed=0)
    rows = np.arange(12, dtype=np.float32).reshape(4, 3)
    idx = np.array([2, 0, 2, 3, 1], dtype=np.int64)

    got = be.to_numpy(be.take(be.array(rows), be.array(idx, dtype=be.idx_dtype)))
    np.testing.assert_array_equal(
        got, rows[idx], err_msg="take did not gather rows in the requested order"
    )


def test_take_on_vector_preserves_duplicates(backend_name):
    be = Backend(backend_name, seed=0)
    vec = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    idx = np.array([3, 3, 0], dtype=np.int64)
    got = be.to_numpy(be.take(be.array(vec), be.array(idx, dtype=be.idx_dtype)))
    np.testing.assert_array_equal(
        got, vec[idx], err_msg="1-D take lost duplicate/ordering semantics"
    )


@pytest.mark.skipif(
    not has_mlx(), reason="MLX is not importable on this machine; MLX/NumPy parity check skipped"
)
def test_mlx_and_numpy_backends_agree_on_core_ops():
    """Same inputs through both backends must give numerically equal outputs."""
    np_be = Backend("numpy", seed=0)
    mlx_be = Backend("mlx", seed=0)

    a = np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(6, 4)
    b = np.linspace(1.0, 3.0, 24, dtype=np.float32).reshape(6, 4)
    idx = np.array([3, 0, 3, 5], dtype=np.int64)
    sparse_idx = np.array([0, 1, 1, 3, 3, 3, 2, 0, 5], dtype=np.int64)
    sparse_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)

    checks = {
        "take": lambda be: be.take(be.array(a), be.array(idx, dtype=be.idx_dtype)),
        "scatter_add": lambda be: be.scatter_add(
            be.zeros((6,)),
            be.array(sparse_idx, dtype=be.idx_dtype),
            be.array(sparse_val, dtype=be.float_dtype),
        ),
        "cumsum": lambda be: be.cumsum(be.array(a)),
        "clip": lambda be: be.clip(be.array(a), -0.5, 0.5),
        "where": lambda be: be.where(be.array(a) > 0.0, be.array(a), be.array(b)),
        "exp": lambda be: be.exp(be.array(a)),
        "tanh": lambda be: be.tanh(be.array(a)),
        "abs": lambda be: be.abs(be.array(a)),
        "maximum": lambda be: be.maximum(be.array(a), be.array(b)),
        "minimum": lambda be: be.minimum(be.array(a), be.array(b)),
        "sum_axis1": lambda be: be.sum(be.array(a), axis=1),
    }

    for name, fn in checks.items():
        reference = np_be.to_numpy(fn(np_be))
        candidate = mlx_be.to_numpy(fn(mlx_be))
        np.testing.assert_allclose(
            candidate,
            reference,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"MLX and NumPy backends disagree on op {name!r}",
        )


def test_seed_makes_random_draws_reproducible(backend_name):
    be = Backend(backend_name, seed=123)
    first = be.to_numpy(be.uniform((16,)))
    be.seed(123)
    second = be.to_numpy(be.uniform((16,)))
    be.seed(124)
    third = be.to_numpy(be.uniform((16,)))

    np.testing.assert_array_equal(
        first, second, err_msg="same seed produced different random draws"
    )
    assert not np.array_equal(first, third), "different seeds produced identical draws"


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        Backend("quantum")


def test_backend_reports_its_name(backend_name):
    be = Backend(backend_name)
    assert be.name == backend_name, f"Backend.name is {be.name!r}, expected {backend_name!r}"
    assert be.is_mlx == (backend_name == "mlx")
