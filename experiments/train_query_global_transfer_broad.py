"""Balanced two-text teacher-attention transfer for one 1B Llama layer.

Warm-start the published Shakespeare-only layer-8 feature map. Capture genuine
layer inputs and full-attention outputs from disjoint Shakespeare and pinned
WikiText-2 raw train windows. Train only delta_q/delta_k, select by balanced
attention-output MSE on Shakespeare and WikiText validation, and leave fresh
next-token quality to the separately predeclared evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402

from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from query_global_attention import DEFAULT_GAIN  # noqa: E402
from query_global_broad_data import (  # noqa: E402
    DATASET_CARD, DATASET_URL, MEMBERS, ZIP_SHA256, percentage_starts,
    spaced_starts, wiki_raw, window_rows,
)
from query_global_trainable import TrainableQueryGlobalAttention  # noqa: E402
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as OLD_RESULT, DEFAULT_WEIGHTS as OLD_WEIGHTS,
    atomic_json, atomic_weights, mse_after_local, prepare, selection_mse,
)


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results" / "query_global_transfer_broad.json"
DEFAULT_WEIGHTS = HERE / "results" / "query_global_transfer_broad_weights.npz"
PROTOCOL = HERE.parent / "docs" / "QUERY_GLOBAL_TRANSFER_BROAD_PROTOCOL.md"
CONTEXT = 512
SHAKESPEARE_WINDOWS = 128
WIKI_TRAIN_WINDOWS = 256
SHAKESPEARE_STEPS = 300
WIKI_STEPS = 300
EVAL_EVERY = 50
LEARNING_RATE = 0.002
CLIP = 1.0
SCHEDULE_SEED = 31
SHAKESPEARE_SELECTION = (55, 58)
WIKI_SELECTION = (20, 40, 60, 80)
LAYER = 8
LOCAL_WINDOW = 64


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def draw_schedule(shake_count: int, wiki_count: int,
                  shake_steps: int, wiki_steps: int) -> np.ndarray:
    """Visit each disjoint training window, then sample quota extras."""
    if shake_steps < shake_count or wiki_steps < wiki_count:
        raise ValueError("schedule must visit every training window")
    rng = np.random.default_rng(SCHEDULE_SEED)
    shake = np.concatenate((
        np.arange(shake_count, dtype=np.int32),
        rng.integers(0, shake_count, size=shake_steps - shake_count,
                     dtype=np.int32)))
    wiki = np.concatenate((
        np.arange(wiki_count, dtype=np.int32),
        rng.integers(0, wiki_count, size=wiki_steps - wiki_count,
                     dtype=np.int32)))
    pairs = np.concatenate((
        np.column_stack((np.zeros(len(shake), np.int32), shake)),
        np.column_stack((np.ones(len(wiki), np.int32), wiki))))
    return rng.permutation(pairs)


def balanced_selection(module, prepared: dict) -> dict:
    rows = {}
    for domain in ("shakespeare", "wikitext_valid"):
        mean, by_window = selection_mse(module, prepared[domain])
        rows[domain] = {"mse": mean, "by_window": by_window}
    return {
        "score": 0.5 * rows["shakespeare"]["mse"] +
                 0.5 * rows["wikitext_valid"]["mse"],
        "domains": rows,
    }


def main() -> None:
    opts = options()
    if not (opts.snapshot / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {opts.snapshot}")
    old = json.loads(OLD_RESULT.read_text())
    old_result_sha = hashlib.sha256(OLD_RESULT.read_bytes()).hexdigest()
    old_weights_sha = hashlib.sha256(OLD_WEIGHTS.read_bytes()).hexdigest()
    if (old["status"] != "complete" or not old["meta"]["primary"]
            or old["weights"]["sha256"] != old_weights_sha):
        raise ValueError("complete published stage-1 warm-start required")
    with np.load(OLD_WEIGHTS) as checkpoint:
        old_q = checkpoint["delta_q"].astype(np.float32)
        old_k = checkpoint["delta_k"].astype(np.float32)
    if old_q.shape != (32, 64, 64) or old_k.shape != (8, 64, 64):
        raise ValueError("published stage-1 map has unexpected 1B head shapes")

    if opts.smoke:
        shake_windows, wiki_windows = 2, 2
        shake_steps, wiki_steps, eval_every = 2, 2, 1
        selection_shake, selection_wiki = (55,), (20,)
        if opts.result == DEFAULT_RESULT:
            opts.result = HERE / "results" / "query_global_transfer_broad_smoke.json"
        if opts.weights == DEFAULT_WEIGHTS:
            opts.weights = HERE / "results" / "query_global_transfer_broad_smoke_weights.npz"
    else:
        shake_windows, wiki_windows = SHAKESPEARE_WINDOWS, WIKI_TRAIN_WINDOWS
        shake_steps, wiki_steps, eval_every = (
            SHAKESPEARE_STEPS, WIKI_STEPS, EVAL_EVERY)
        selection_shake, selection_wiki = (
            SHAKESPEARE_SELECTION, WIKI_SELECTION)
    schedule = draw_schedule(
        shake_windows, wiki_windows, shake_steps, wiki_steps)
    if (len(schedule) != shake_steps + wiki_steps or
            int(np.sum(schedule[:, 0] == 0)) != shake_steps or
            int(np.sum(schedule[:, 0] == 1)) != wiki_steps or
            len(np.unique(schedule[schedule[:, 0] == 0, 1])) != shake_windows or
            len(np.unique(schedule[schedule[:, 0] == 1, 1])) != wiki_windows):
        raise AssertionError("balanced schedule quotas or full coverage changed")

    load_started = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    model.freeze()
    load_s = time.perf_counter() - load_started
    config_sha = hashlib.sha256(
        (opts.snapshot / "config.json").read_bytes()).hexdigest()
    if (config_sha != old["meta"]["config_sha256"] or
            (opts.snapshot / "model.safetensors").resolve().name !=
            old["meta"]["weight_blob_id"]):
        raise ValueError("warm-start and current checkpoint identity differ")
    read_started = time.perf_counter()
    ids = {
        "shakespeare": np.asarray(
            tokenizer.encode(CORPUS.read_text()), dtype=np.int32),
        "wikitext_train": np.asarray(
            tokenizer.encode(wiki_raw("train")), dtype=np.int32),
        "wikitext_valid": np.asarray(
            tokenizer.encode(wiki_raw("valid")), dtype=np.int32),
    }
    train_starts = {
        "shakespeare": spaced_starts(
            len(ids["shakespeare"]),
            region_end=int(len(ids["shakespeare"]) * 0.5),
            count=shake_windows, window=CONTEXT),
        "wikitext_train": spaced_starts(
            len(ids["wikitext_train"]),
            region_end=len(ids["wikitext_train"]),
            count=wiki_windows, window=CONTEXT),
    }
    selection_starts = {
        "shakespeare": percentage_starts(
            len(ids["shakespeare"]), selection_shake, CONTEXT),
        "wikitext_valid": percentage_starts(
            len(ids["wikitext_valid"]), selection_wiki, CONTEXT),
    }
    if (train_starts["shakespeare"][-1] + CONTEXT >
            min(selection_starts["shakespeare"].values())):
        raise ValueError("Shakespeare train and selection windows overlap")
    read_s = time.perf_counter() - read_started

    original = model.layers[LAYER].self_attn
    module = TrainableQueryGlobalAttention(
        original, window=LOCAL_WINDOW, gain=DEFAULT_GAIN, chunk=64,
        inference_sync_blocks=0)
    module.delta_q = mx.array(old_q)
    module.delta_k = mx.array(old_k)
    mx.eval(module.delta_q, module.delta_k)
    if (not np.array_equal(np.asarray(module.delta_q), old_q) or
            not np.array_equal(np.asarray(module.delta_k), old_k)):
        raise AssertionError("new map does not exactly warm-start old map")
    module.unfreeze(
        keys=["delta_q", "delta_k"], recurse=False, strict=True)
    trainable = sorted(k for k, _ in nn.utils.tree_flatten(
        module.trainable_parameters()))
    if trainable != ["delta_k", "delta_q"]:
        raise AssertionError(f"pretrained weights became trainable: {trainable}")
    trainable_count = sum(int(np.prod(value.shape)) for _, value in
                          nn.utils.tree_flatten(module.trainable_parameters()))
    if trainable_count != 163840:
        raise AssertionError("feature-map parameter count differs")

    record = {
        "status": "running",
        "meta": {
            "primary": not opts.smoke,
            "snapshot_revision": opts.snapshot.name,
            "config_sha256": config_sha,
            "weight_blob_id": (opts.snapshot / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "data_source_sha256": hashlib.sha256(
                (HERE / "query_global_broad_data.py").read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "trainable_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "capture_and_loss_source_sha256": hashlib.sha256(
                (HERE / "train_query_global_transfer.py").read_bytes()).hexdigest(),
            "old_result_sha256": old_result_sha,
            "old_weights_sha256": old_weights_sha,
            "shakespeare_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "wiki_zip_sha256": ZIP_SHA256,
            "wiki_member_sha256": {
                split: MEMBERS[split][1] for split in ("train", "valid")},
            "wiki_dataset_url": DATASET_URL,
            "wiki_dataset_card": DATASET_CARD,
            "token_counts": {name: len(value) for name, value in ids.items()},
            "token_ids_sha256": {
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in ids.items()},
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "layer": LAYER,
            "context": CONTEXT,
            "local_window": LOCAL_WINDOW,
            "fixed_gain": DEFAULT_GAIN,
            "trainable_keys": trainable,
            "trainable_parameter_count": trainable_count,
            "train_windows": {
                "shakespeare": shake_windows,
                "wikitext_train": wiki_windows},
            "training_distinct_input_positions": (
                shake_windows + wiki_windows) * CONTEXT,
            "teacher_capture_distinct_input_positions": (
                shake_windows + wiki_windows +
                len(selection_shake) + len(selection_wiki)) * CONTEXT,
            "training_input_token_exposures": len(schedule) * CONTEXT,
            "steps": len(schedule),
            "steps_by_domain": {
                "shakespeare": shake_steps,
                "wikitext_train": wiki_steps},
            "every_train_window_visited": True,
            "eval_every": eval_every,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW constant LR, zero weight decay",
            "grad_clip_norm": CLIP,
            "schedule_seed": SCHEDULE_SEED,
            "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "selection_percentages": {
                "shakespeare": list(selection_shake),
                "wikitext_valid": list(selection_wiki)},
            "selection_score": "0.5 * mean Shakespeare attention MSE + 0.5 * mean WikiText-valid attention MSE",
            "no_wikitext_test_training_or_selection": True,
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
        },
        "windows": {
            "train": {
                domain: window_rows(ids[domain], starts, CONTEXT)
                for domain, starts in train_starts.items()},
            "selection": {
                domain: window_rows(ids[domain], starts, CONTEXT)
                for domain, starts in selection_starts.items()},
        },
        "timing": {
            "load_s": load_s, "read_and_tokenize_s": read_s,
            "teacher_capture_s": None, "training_updates_s": 0.0,
            "selection_s": 0.0, "training_loop_s": None},
        "baseline_selection": None,
        "curve": [],
        "best": None,
        "weights": {"path": str(opts.weights), "sha256": None},
        "capture_peak_metal_bytes": None,
        "training_peak_metal_bytes": None,
    }
    atomic_json(opts.result, record)

    mx.reset_peak_memory()
    capture_started = time.perf_counter()
    prepared_train = {
        domain: [prepare(model, original, ids[domain][start:start + CONTEXT])
                 for start in starts]
        for domain, starts in train_starts.items()
    }
    prepared_selection = {
        domain: [prepare(model, original, ids[domain][start:start + CONTEXT])
                 for start in starts.values()]
        for domain, starts in selection_starts.items()
    }
    record["timing"]["teacher_capture_s"] = (
        time.perf_counter() - capture_started)
    record["capture_peak_metal_bytes"] = int(mx.get_peak_memory())
    atomic_json(opts.result, record)

    module.eval()
    baseline_started = time.perf_counter()
    baseline = balanced_selection(module, prepared_selection)
    record["timing"]["selection_s"] += (
        time.perf_counter() - baseline_started)
    record["baseline_selection"] = baseline
    best = {
        "step": 0, "selection": baseline,
        "delta_q": np.asarray(module.delta_q).copy(),
        "delta_k": np.asarray(module.delta_k).copy(),
    }
    record["weights"]["sha256"] = atomic_weights(
        opts.weights, best["delta_q"], best["delta_k"])
    record["best"] = {"step": 0, "selection": baseline}
    atomic_json(opts.result, record)

    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=0.0)
    value_and_grad = nn.value_and_grad(module, mse_after_local)
    module.train()
    mx.reset_peak_memory()
    train_started = time.perf_counter()
    for step, (domain_index, window_index) in enumerate(schedule, start=1):
        domain = "shakespeare" if domain_index == 0 else "wikitext_train"
        x, target = prepared_train[domain][int(window_index)]
        update_started = time.perf_counter()
        loss, grads = value_and_grad(module, x, target)
        grads, grad_norm = optim.clip_grad_norm(grads, CLIP)
        optimizer.update(module, grads)
        mx.eval(module.parameters(), optimizer.state, loss, grad_norm)
        record["timing"]["training_updates_s"] += (
            time.perf_counter() - update_started)
        if step == 1 or step % eval_every == 0 or step == len(schedule):
            select_started = time.perf_counter()
            module.eval()
            selected = balanced_selection(module, prepared_selection)
            module.train()
            record["timing"]["selection_s"] += (
                time.perf_counter() - select_started)
            record["curve"].append({
                "step": step,
                "training_domain": domain,
                "training_window_index": int(window_index),
                "training_mse": float(loss),
                "selection": selected,
                "gradient_norm_before_clip": float(grad_norm),
                "host_load_average": list(os.getloadavg()),
                "metal_peak_bytes": int(mx.get_peak_memory()),
            })
            if selected["score"] < best["selection"]["score"]:
                best = {
                    "step": step, "selection": selected,
                    "delta_q": np.asarray(module.delta_q).copy(),
                    "delta_k": np.asarray(module.delta_k).copy(),
                }
                record["best"] = {
                    "step": step, "selection": selected}
                record["weights"]["sha256"] = atomic_weights(
                    opts.weights, best["delta_q"], best["delta_k"])
            atomic_json(opts.result, record)
            print(
                f"step={step} {domain} train_mse={float(loss):.6f} "
                f"selection={selected['score']:.6f} "
                f"best={best['selection']['score']:.6f}",
                flush=True)
    record["timing"]["training_loop_s"] = (
        time.perf_counter() - train_started)
    record["training_peak_metal_bytes"] = int(mx.get_peak_memory())
    record["meta"]["host_load_at_end"] = list(os.getloadavg())
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(opts.result, record)
    print("best", record["best"], "weights", opts.weights, flush=True)


if __name__ == "__main__":
    main()
