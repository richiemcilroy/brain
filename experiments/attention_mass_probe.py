"""Diagnose the 1B layer-8 information missing from a 64-token local window.

The teacher supplies genuine layer-8 input activations. We measure how much
causal attention mass each sampled query places before its most recent 64
tokens, then compare the full attention output to the exact local output and
to the transplanted gated trace on those same activations. This is an untimed
diagnostic, not a model-quality or throughput claim.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from local_window_hybrid import install_local_window_hybrid  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "attention_mass_probe.json"
POSITIONS = (127, 255, 511, 1023, 1535, 2047)
LAYER = 8
WINDOW = 64


def save(result: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(OUTPUT)


def capture_input(model, ids: np.ndarray) -> mx.array:
    layer = model.layers[LAYER]
    original = layer.self_attn
    box = {}

    class Capture:
        def __call__(self, x, mask=None, cache=None):
            box["x"] = x
            return original(x, mask=mask, cache=cache)

    layer.self_attn = Capture()
    try:
        hidden = model.model(mx.array(ids[None, :]))
        mx.eval(hidden, box["x"])
        return box["x"]
    finally:
        layer.self_attn = original


def outside_attention_mass(original, x: mx.array) -> list[dict]:
    B, T, _ = x.shape
    q = original.q_proj(x).reshape(B, T, original.n_heads,
                                    original.head_dim).transpose(0, 2, 1, 3)
    k = original.k_proj(x).reshape(B, T, original.n_kv_heads,
                                    original.head_dim).transpose(0, 2, 1, 3)
    q = original.rope(q).astype(mx.float32)
    k = original.rope(k).astype(mx.float32)
    k = mx.repeat(k, original.n_heads // original.n_kv_heads, axis=1)
    rows = []
    for pos in POSITIONS:
        scores = (q[:, :, pos:pos + 1, :] * k[:, :, :pos + 1, :]).sum(axis=-1)
        probs = mx.softmax(scores * original.scale, axis=-1)
        older_count = max(0, pos - WINDOW + 1)
        mass = np.asarray(probs[..., :older_count].sum(axis=-1))
        values = mass.reshape(-1).astype(float).tolist()
        rows.append({
            "query_position": pos,
            "older_than_window_positions": older_count,
            "outside_mass_per_head": values,
            "outside_mass_mean": float(np.mean(values)),
            "outside_mass_median": float(np.median(values)),
            "outside_mass_max": float(np.max(values)),
        })
    return rows


def output_fit(model, x: mx.array) -> dict:
    original = model.layers[LAYER].self_attn
    full = original(x, mask="causal", cache=None).astype(mx.float32)
    mx.random.seed(0)
    installed = install_local_window_hybrid(
        model, layer=LAYER, decay=0.7, memory_gain=0.0)
    try:
        local = installed.hybrid._local(x, "causal", cache=None).astype(mx.float32)
        trace = installed.hybrid.memory(x, cache=None).astype(mx.float32)
        residual = full - local
        local_mse = float(mx.mean(residual * residual))
        trace_norm = float(mx.sum(trace * trace))
        dot = float(mx.sum(residual * trace))
        best_scalar = dot / trace_norm if trace_norm > 0 else 0.0
        after = residual - best_scalar * trace
        return {
            "full_attention_rms": float(mx.sqrt(mx.mean(full * full))),
            "local_attention_rms": float(mx.sqrt(mx.mean(local * local))),
            "missing_output_rms": float(mx.sqrt(mx.mean(residual * residual))),
            "missing_output_mse": local_mse,
            "trace_rms": float(mx.sqrt(mx.mean(trace * trace))),
            "best_scalar_trace_gain_for_missing_output": best_scalar,
            "mse_after_best_scalar_trace": float(mx.mean(after * after)),
            "fraction_of_missing_mse_recovered_by_scalar_trace":
                1.0 - float(mx.mean(after * after)) / local_mse
                if local_mse > 0 else None,
            "fit_note": "scalar least-squares diagnostic on this window; not a trained or deployable gain",
            "conversion": installed.conversion,
        }
    finally:
        installed.restore()


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline snapshot missing: {DEFAULT_SNAPSHOT}")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    windows = [("selection", int(0.6 * len(ids))),
               ("validation", int(0.9 * len(ids)))]
    result = {
        "status": "running",
        "meta": {
            "revision": DEFAULT_SNAPSHOT.name,
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_window_source_sha256": hashlib.sha256(
                (HERE / "local_window_hybrid.py").read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "layer": LAYER,
            "prefix_tokens": 2048,
            "local_window": WINDOW,
            "sampled_query_positions": list(POSITIONS),
            "started_epoch": time.time(),
        },
        "windows": {},
    }
    save(result)
    for name, start in windows:
        segment = ids[start:start + 2048]
        if len(segment) != 2048:
            raise ValueError(f"{name} token window is short")
        x = capture_input(model, segment)
        original = model.layers[LAYER].self_attn
        result["windows"][name] = {
            "token_window": [start, start + 2048],
            "token_ids_sha256": hashlib.sha256(segment.tobytes()).hexdigest(),
            "outside_attention_mass": outside_attention_mass(original, x),
            "output_fit": output_fit(model, x),
        }
        save(result)
        last = result["windows"][name]["outside_attention_mass"][-1]
        fit = result["windows"][name]["output_fit"]
        print(f"{name}: last-query outside mass {last['outside_mass_median']:.3f}; "
              f"scalar-trace residual MSE recovery "
              f"{fit['fraction_of_missing_mse_recovered_by_scalar_trace']:.3f}",
              flush=True)
    result["status"] = "complete"
    result["meta"]["finished_epoch"] = time.time()
    save(result)


if __name__ == "__main__":
    main()
