"""Cached memory must compute the same function as one complete prefix."""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from llm_efficiency import GatedMemory  # noqa: E402
from streaming_memory import MemoryCache, StreamingGatedMemoryCarrier  # noqa: E402


class Carrier:
    mode = "transfer"
    out_gain = 0.3
    d = 8

    def __init__(self, banks):
        self.mem = GatedMemory(8, banks=banks, chunk=16)


@pytest.mark.parametrize("banks", [1, 4])
def test_segmented_prefix_matches_full_prefix(banks):
    mx.random.seed(19)
    rng = np.random.default_rng(19)
    x = mx.array(rng.normal(size=(2, 65, 8)).astype(np.float32))
    layer = StreamingGatedMemoryCarrier(Carrier(banks))
    full = np.asarray(layer(x))
    cache = MemoryCache()
    segments = [1, 7, 16, 3, 38]
    parts, start = [], 0
    for length in segments:
        parts.append(np.asarray(layer(x[:, start : start + length], cache=cache)))
        start += length
    split = np.concatenate(parts, axis=1)
    assert cache.offset == x.shape[1]
    assert cache.state.shape == (2, banks, 8)
    assert np.max(np.abs(full - split)) < 2e-4


def test_wrong_cache_type_fails_loudly():
    layer = StreamingGatedMemoryCarrier(Carrier(1))
    with pytest.raises(TypeError, match="MemoryCache"):
        layer(mx.zeros((1, 1, 8)), cache=object())
