"""Stage-1 teacher-attention transfer for one query-global Llama-3.2-1B layer.

Genuine layer-8 inputs and full softmax attention outputs are captured from the
offline teacher. Only zero-initialized per-head feature-map residual matrices
are trained; all pretrained q/k/v/o weights stay frozen. A deterministic
TinyShakespeare protocol uses disjoint training windows from the first 50%,
two selection windows at 55%/58%, and a fixed best-selection-MSE checkpoint.
It counts teacher capture, training and validation time separately.

This is a small, single-corpus research run, not a LoLCATs replication or a
general quality/cost claim. LoLCATs uses substantially more data and LoRA
after attention transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from attention_mass_probe import capture_input  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import DEFAULT_GAIN, LocalQueryGlobalAttention  # noqa: E402
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results" / "query_global_transfer.json"
DEFAULT_WEIGHTS = HERE / "results" / "query_global_transfer_weights.npz"
PRIMARY_CONTEXT = 512
PRIMARY_TRAIN_WINDOWS = 32
PRIMARY_STEPS = 200
PRIMARY_EVAL_EVERY = 25
LEARNING_RATE = 0.01
CLIP = 1.0
SEED = 0
LAYER = 8
LOCAL_WINDOW = 64


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_weights(path: Path, q: np.ndarray, k: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, delta_q=q, delta_k=k)
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    temp.replace(path)
    return digest


def mse_after_local(module, x: mx.array, target: mx.array,
                    local_window: int = LOCAL_WINDOW) -> mx.array:
    output = module(x)
    delta = (output[:, local_window:].astype(mx.float32) -
             target[:, local_window:].astype(mx.float32))
    return mx.mean(delta * delta)


def selection_mse(module, prepared) -> tuple[float, list[float]]:
    values = []
    for x, target in prepared:
        loss = mse_after_local(module, x, target)
        mx.eval(loss)
        values.append(float(loss))
    return statistics.mean(values), values


def prepare(model, original, segment: np.ndarray) -> tuple[mx.array, mx.array]:
    x = capture_input(model, segment)
    target = original(x, mask="causal", cache=None)
    mx.eval(x, target)
    return x, target


def main() -> None:
    opts = options()
    if not (opts.snapshot / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {opts.snapshot}")
    if opts.smoke:
        context, n_train, steps, eval_every, val_percentages = (
            128, 2, 2, 1, (55,))
        if opts.result == DEFAULT_RESULT:
            opts.result = HERE / "results" / "query_global_transfer_smoke.json"
        if opts.weights == DEFAULT_WEIGHTS:
            opts.weights = HERE / "results" / "query_global_transfer_smoke_weights.npz"
    else:
        context, n_train, steps, eval_every, val_percentages = (
            PRIMARY_CONTEXT, PRIMARY_TRAIN_WINDOWS, PRIMARY_STEPS,
            PRIMARY_EVAL_EVERY, (55, 58))
    started = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    load_s = time.perf_counter() - started
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    train_region = int(len(ids) * 0.5)
    spacing = train_region // n_train
    train_starts = [int((i + 0.5) * spacing) for i in range(n_train)]
    val_starts = [int(pct * len(ids) / 100) for pct in val_percentages]
    all_starts = train_starts + val_starts
    if (min(spacing, *(right - left for left, right in zip(
            sorted(all_starts), sorted(all_starts)[1:]))) < context or
            max(all_starts) + context > len(ids)):
        raise ValueError("training and validation windows overlap or exceed corpus")
    schedule = np.random.default_rng(SEED).integers(
        0, n_train, size=steps, dtype=np.int32)
    original = model.layers[LAYER].self_attn
    fixed = LocalQueryGlobalAttention(
        original, window=LOCAL_WINDOW, gain=DEFAULT_GAIN, chunk=64,
        inference_sync_blocks=0)
    module = TrainableQueryGlobalAttention(
        original, window=LOCAL_WINDOW, gain=DEFAULT_GAIN, chunk=64,
        inference_sync_blocks=0)
    record = {
        "status": "running",
        "meta": {
            "primary": not opts.smoke,
            "snapshot_revision": opts.snapshot.name,
            "config_sha256": hashlib.sha256(
                (opts.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (opts.snapshot / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "query_trainable_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "tokenized_corpus_length": int(len(ids)),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "layer": LAYER,
            "context": context,
            "local_window": LOCAL_WINDOW,
            "train_windows": n_train,
            "train_percent_region": [0, 50],
            "validation_percentages": list(val_percentages),
            "teacher_capture_unique_tokens": (n_train + len(val_starts)) * context,
            "training_token_exposures": steps * context,
            "steps": steps,
            "eval_every": eval_every,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW constant LR, zero weight decay",
            "grad_clip_norm": CLIP,
            "seed": SEED,
            "fixed_gain_from_output_probe": DEFAULT_GAIN,
            "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
            "limitations": "single Shakespeare corpus; vastly fewer unique tokens than LoLCATs; one layer only; no next-token or serving-quality verdict in this stage",
        },
        "windows": {
            "train": [
                {"token_window": [start, start + context],
                 "token_ids_sha256": hashlib.sha256(
                     ids[start:start + context].tobytes()).hexdigest()}
                for start in train_starts
            ],
            "selection": [
                {"token_window": [start, start + context],
                 "token_ids_sha256": hashlib.sha256(
                     ids[start:start + context].tobytes()).hexdigest()}
                for start in val_starts
            ],
        },
        "timing": {"load_s": load_s, "teacher_capture_s": None,
                   "train_s": None, "selection_s": 0.0},
        "trainable_keys": None,
        "trainable_parameter_count": None,
        "initial_fixed_vs_trainable_max_abs": None,
        "baseline_selection_mse": None,
        "curve": [],
        "best": None,
        "weights": {"path": str(opts.weights), "sha256": None},
        "peak_metal_bytes": None,
    }
    atomic_json(opts.result, record)

    capture_started = time.perf_counter()
    train_prepared = [
        prepare(model, original, ids[start:start + context])
        for start in train_starts
    ]
    val_prepared = [
        prepare(model, original, ids[start:start + context])
        for start in val_starts
    ]
    record["timing"]["teacher_capture_s"] = time.perf_counter() - capture_started
    atomic_json(opts.result, record)
    probe_x = val_prepared[0][0]
    difference = module(probe_x).astype(mx.float32) - fixed(probe_x).astype(mx.float32)
    record["initial_fixed_vs_trainable_max_abs"] = float(
        mx.max(mx.abs(difference)))
    if record["initial_fixed_vs_trainable_max_abs"] != 0.0:
        raise AssertionError("zero residual feature maps changed the fixed baseline")
    model.freeze()
    module.unfreeze(
        keys=["delta_q", "delta_k"], recurse=False, strict=True)
    trainable = sorted(k for k, _ in nn.utils.tree_flatten(
        module.trainable_parameters()))
    if trainable != ["delta_k", "delta_q"]:
        raise AssertionError(f"pretrained weights became trainable: {trainable}")
    record["trainable_keys"] = trainable
    record["trainable_parameter_count"] = sum(
        int(np.prod(value.shape)) for _, value in nn.utils.tree_flatten(
            module.trainable_parameters()))
    module.train()
    baseline, baseline_rows = selection_mse(module, val_prepared)
    record["baseline_selection_mse"] = {
        "mean": baseline, "by_window": baseline_rows}
    best = {
        "step": 0, "selection_mse": baseline,
        "delta_q": np.asarray(module.delta_q).copy(),
        "delta_k": np.asarray(module.delta_k).copy(),
    }
    record["weights"]["sha256"] = atomic_weights(
        opts.weights, best["delta_q"], best["delta_k"])
    record["best"] = {"step": 0, "selection_mse": baseline}
    atomic_json(opts.result, record)

    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=0.0)

    def loss_fn(candidate, x, target):
        return mse_after_local(candidate, x, target)

    value_and_grad = nn.value_and_grad(module, loss_fn)
    mx.reset_peak_memory()
    training_started = time.perf_counter()
    for step, chosen in enumerate(schedule, start=1):
        x, target = train_prepared[int(chosen)]
        loss, grads = value_and_grad(module, x, target)
        grads, grad_norm = optim.clip_grad_norm(grads, CLIP)
        optimizer.update(module, grads)
        mx.eval(module.parameters(), optimizer.state, loss, grad_norm)
        if step == 1 or step % eval_every == 0 or step == steps:
            selection_started = time.perf_counter()
            module.eval()
            selected, per_window = selection_mse(module, val_prepared)
            module.train()
            record["timing"]["selection_s"] += (
                time.perf_counter() - selection_started)
            row = {
                "step": step,
                "training_window_index": int(chosen),
                "train_mse": float(loss),
                "selection_mse": selected,
                "selection_mse_by_window": per_window,
                "gradient_norm_before_clip": float(grad_norm),
                "host_load_average": list(os.getloadavg()),
                "metal_peak_bytes": int(mx.get_peak_memory()),
            }
            record["curve"].append(row)
            if selected < best["selection_mse"]:
                best = {
                    "step": step, "selection_mse": selected,
                    "delta_q": np.asarray(module.delta_q).copy(),
                    "delta_k": np.asarray(module.delta_k).copy(),
                }
                record["best"] = {
                    "step": step, "selection_mse": selected}
                record["weights"]["sha256"] = atomic_weights(
                    opts.weights, best["delta_q"], best["delta_k"])
            atomic_json(opts.result, record)
            print(
                f"step={step} train_mse={float(loss):.6f} "
                f"select_mse={selected:.6f} best={best['selection_mse']:.6f}",
                flush=True)
    record["timing"]["train_s"] = time.perf_counter() - training_started
    record["peak_metal_bytes"] = int(mx.get_peak_memory())
    record["meta"]["host_load_at_end"] = list(os.getloadavg())
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(opts.result, record)
    print(
        "best", record["best"],
        "selection_recovery", 1.0 - best["selection_mse"] / baseline,
        "weights", opts.weights,
        flush=True)


if __name__ == "__main__":
    main()
