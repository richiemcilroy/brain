"""The benchmark must distinguish active KV positions from allocated capacity."""
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from oss_model_benchmark import cache_bytes, cache_storage_bytes  # noqa: E402
from streaming_memory import MemoryCache  # noqa: E402


def test_active_kv_accounting_excludes_spare_capacity():
    keys = mx.zeros((2, 4, 256, 8), dtype=mx.bfloat16)
    values = mx.zeros((2, 4, 256, 8), dtype=mx.bfloat16)
    kv = SimpleNamespace(keys=keys, values=values, offset=129)
    memory = MemoryCache()
    memory.h = mx.zeros((2, 1, 32), dtype=mx.float32)

    expected_active = (2 * 4 * 129 * 8 * 2) * 2 + memory.h.nbytes
    expected_storage = keys.nbytes + values.nbytes + memory.h.nbytes
    assert cache_bytes([kv, memory]) == expected_active
    assert cache_storage_bytes([kv, memory]) == expected_storage
    assert expected_active < expected_storage
