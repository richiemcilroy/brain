"""Shared fixtures for the human-brain validation suite.

The path bootstrap keeps ``python3 -m pytest tests/ -q`` working from a clean
checkout regardless of how pytest computes ``rootdir``: the suite imports the
repo's own ``brain`` package and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain.backend import Backend, has_mlx  # noqa: E402

BACKENDS: list[str] = ["numpy"] + (["mlx"] if has_mlx() else [])

requires_mlx = pytest.mark.skipif(
    not has_mlx(), reason="MLX is not importable on this machine; MLX-only checks skipped"
)


@pytest.fixture(params=BACKENDS)
def backend_name(request) -> str:
    """Parametrised over every backend available on this machine."""
    return request.param


@pytest.fixture
def be(backend_name):
    """A freshly seeded backend."""
    return Backend(backend_name, seed=0)
