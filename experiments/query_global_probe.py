"""Test untrained query-addressable bounded states on genuine 1B activations.

This is an attention-output diagnostic, not an OSS-model conversion. The
teacher supplies layer-8 inputs on disjoint 2,048-token corpus windows.
For six sampled queries, exact full softmax and exact 64-token local outputs
are compared with a global read from a constant-size feature-key/value state.
One scalar per map is fitted on the 60% selection window, then applied without
change to untouched 75%/85% windows and the already-inspected 90% window.

Local + feature-global attention is prior art (LoLCATs); this probe only
decides whether an untrained MLX baseline is worth taking into full-model
quality, generation and cost tests.
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

from attention_mass_probe import capture_input  # noqa: E402
from local_window_hybrid import install_local_window_hybrid  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "query_global_probe.json"
LAYER = 8
WINDOW = 64
PREFIX = 2048
POSITIONS = (127, 255, 511, 1023, 1535, 2047)
WINDOW_PERCENTAGES = {
    "selection": 60,
    "fresh_75": 75,
    "fresh_85": 85,
    "inspected_90": 90,
}
MAPS = ("elu_plus_one", "relu_plus_one",
        "identity_hedgehog_temp1", "identity_hedgehog_temp2")


def atomic_json(record: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(OUTPUT)


def feature_map(x: np.ndarray, kind: str) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if kind == "elu_plus_one":
        return np.where(x >= 0, x + 1, np.exp(np.minimum(x, 0))).astype(np.float32)
    if kind == "relu_plus_one":
        return np.maximum(x, 0) + 1
    if kind.startswith("identity_hedgehog_temp"):
        temperature = 1.0 if kind.endswith("temp1") else 2.0
        y = x * temperature
        positive = np.exp(y - y.max(axis=-1, keepdims=True))
        positive /= positive.sum(axis=-1, keepdims=True)
        negative = np.exp(-y - (-y).max(axis=-1, keepdims=True))
        negative /= negative.sum(axis=-1, keepdims=True)
        return np.concatenate((positive, negative), axis=-1)
    raise ValueError(kind)


def state_reads(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                kind: str) -> tuple[np.ndarray, int]:
    """Advance an eight-KV-head state only when keys exit the local window."""
    qf = feature_map(q, kind)
    kf = feature_map(k, kind)
    hq, _, head_dim = q.shape
    hkv = k.shape[0]
    if hq % hkv:
        raise ValueError("GQA query heads must divide into KV-head groups")
    group = np.arange(hq) // (hq // hkv)
    features = kf.shape[-1]
    state = np.zeros((hkv, features, head_dim), dtype=np.float32)
    normalizer = np.zeros((hkv, features), dtype=np.float32)
    reads = []
    incorporated = 0
    for pos in POSITIONS:
        older_count = max(0, pos - WINDOW + 1)
        if older_count > incorporated:
            new_keys = kf[:, incorporated:older_count, :]
            new_values = v[:, incorporated:older_count, :]
            state += np.matmul(new_keys.transpose(0, 2, 1), new_values)
            normalizer += new_keys.sum(axis=1)
            incorporated = older_count
        query = qf[:, pos, :]
        numerator = np.einsum("hf,hfd->hd", query, state[group])
        denominator = np.einsum("hf,hf->h", query, normalizer[group])
        reads.append(numerator / np.maximum(denominator[:, None], 1e-8))
    state_bytes = int(state.nbytes + normalizer.nbytes)
    return np.stack(reads).astype(np.float32), state_bytes


def outputs(model, segment: np.ndarray) -> dict:
    x = capture_input(model, segment)
    original = model.layers[LAYER].self_attn
    _, length, _ = x.shape
    if length != PREFIX:
        raise ValueError("probe prefix length changed")
    head_dim = original.head_dim
    q = original.q_proj(x).reshape(
        1, length, original.n_heads, head_dim).transpose(0, 2, 1, 3)
    k = original.k_proj(x).reshape(
        1, length, original.n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = original.v_proj(x).reshape(
        1, length, original.n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    q = np.asarray(original.rope(q).astype(mx.float32))[0]
    k = np.asarray(original.rope(k).astype(mx.float32))[0]
    v = np.asarray(v.astype(mx.float32))[0]
    full = original(x, mask="causal", cache=None)
    mx.random.seed(0)
    installed = install_local_window_hybrid(
        model, layer=LAYER, decay=0.7, memory_gain=0.0)
    try:
        local = installed.hybrid._local(x, "causal", cache=None)
        trace = installed.hybrid.memory(x, cache=None)
        mx.eval(full, local, trace)
        full = np.asarray(full.astype(mx.float32))[0, list(POSITIONS)]
        local = np.asarray(local.astype(mx.float32))[0, list(POSITIONS)]
        trace = np.asarray(trace.astype(mx.float32))[0, list(POSITIONS)]
    finally:
        installed.restore()

    global_outputs = {}
    state_bytes = {}
    for kind in MAPS:
        pre_o, state_bytes[kind] = state_reads(q, k, v, kind)
        flattened = pre_o.reshape(1, len(POSITIONS), -1)
        projected = original.o_proj(mx.array(flattened).astype(x.dtype))
        global_outputs[kind] = np.asarray(projected.astype(mx.float32))[0]
    return {
        "full": full, "local": local, "trace": trace,
        "global": global_outputs, "state_bytes": state_bytes,
    }


def scalar_fit(target: np.ndarray, direction: np.ndarray,
               minimum: float, maximum: float) -> dict:
    denom = float(np.sum(direction * direction, dtype=np.float64))
    raw = float(np.sum(target * direction, dtype=np.float64) / denom) if denom else 0.0
    return {"unconstrained_gain": raw,
            "selected_gain": float(np.clip(raw, minimum, maximum)),
            "allowed_gain_range": [minimum, maximum]}


def mse(a: np.ndarray) -> float:
    return float(np.mean(a * a, dtype=np.float64))


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {DEFAULT_SNAPSHOT}")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    starts = {name: int(pct * len(ids) / 100)
              for name, pct in WINDOW_PERCENTAGES.items()}
    if any(start + PREFIX > len(ids) for start in starts.values()):
        raise ValueError("probe window exceeds corpus")
    windows = {name: ids[start:start + PREFIX] for name, start in starts.items()}
    record = {
        "status": "running",
        "meta": {
            "revision": DEFAULT_SNAPSHOT.name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_window_source_sha256": hashlib.sha256(
                (HERE / "local_window_hybrid.py").read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "prefix": PREFIX,
            "layer": LAYER,
            "local_window": WINDOW,
            "sampled_positions": list(POSITIONS),
            "feature_maps": list(MAPS),
            "quality_note": "attention-output diagnostic on sampled genuine layer inputs; scalar map gains fit on 60% only; no next-token model score, training or timing",
            "started_epoch": time.time(),
        },
        "windows": {
            name: {"token_window": [start, start + PREFIX],
                   "token_ids_sha256": hashlib.sha256(windows[name].tobytes()).hexdigest()}
            for name, start in starts.items()
        },
        "selection_fits": {},
        "rows": {},
        "choice": None,
    }
    atomic_json(record)
    arrays = {}
    for name, segment in windows.items():
        arrays[name] = outputs(model, segment)
        record["rows"][name] = {
            "local_mse": mse(arrays[name]["full"] - arrays[name]["local"]),
            "state_bytes_by_map": arrays[name]["state_bytes"],
        }
        atomic_json(record)
        print(name, "local_mse", record["rows"][name]["local_mse"],
              flush=True)

    selected = arrays["selection"]
    target = selected["full"] - selected["local"]
    record["selection_fits"]["trace"] = scalar_fit(
        target, selected["trace"], -1.0, 1.0)
    for kind in MAPS:
        direction = selected["global"][kind] - selected["local"]
        record["selection_fits"][kind] = scalar_fit(
            target, direction, 0.0, 1.0)
    for name, row in record["rows"].items():
        sample = arrays[name]
        target = sample["full"] - sample["local"]
        local_mse = row["local_mse"]
        row["trace"] = {}
        trace_gain = record["selection_fits"]["trace"]["selected_gain"]
        trace_mse = mse(target - trace_gain * sample["trace"])
        row["trace"] = {
            "selected_gain": trace_gain,
            "mse": trace_mse,
            "fraction_of_local_mse_recovered": 1.0 - trace_mse / local_mse,
        }
        row["maps"] = {}
        for kind in MAPS:
            gain = record["selection_fits"][kind]["selected_gain"]
            direction = sample["global"][kind] - sample["local"]
            after = mse(target - gain * direction)
            row["maps"][kind] = {
                "selected_gain": gain,
                "mse": after,
                "fraction_of_local_mse_recovered": 1.0 - after / local_mse,
            }
        atomic_json(record)
        print(name, "recovery", {
            kind: round(x["fraction_of_local_mse_recovered"], 3)
            for kind, x in row["maps"].items()
        }, "trace", round(row["trace"]["fraction_of_local_mse_recovered"], 3),
              flush=True)
    best = max(MAPS, key=lambda kind:
               record["rows"]["selection"]["maps"][kind][
                   "fraction_of_local_mse_recovered"])
    record["choice"] = {
        "map": best,
        "criterion": "largest local residual MSE recovery on the 60% selection window",
        "selected_gain": record["selection_fits"][best]["selected_gain"],
        "fresh_75_recovery": record["rows"]["fresh_75"]["maps"][best][
            "fraction_of_local_mse_recovered"],
        "fresh_85_recovery": record["rows"]["fresh_85"]["maps"][best][
            "fraction_of_local_mse_recovered"],
    }
    record["status"] = "complete"
    record["meta"]["finished_epoch"] = time.time()
    atomic_json(record)
    print("selected", record["choice"], flush=True)


if __name__ == "__main__":
    main()
