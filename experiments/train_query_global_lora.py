"""Matched stage-2 rank-8 LoRA adjustment of three one-layer Llama arms.

Full attention, exact 64-token local attention and the selected query-state
conversion receive identical q/k/v/o LoRA parameter counts, initial A arrays,
zero B arrays, disjoint Shakespeare training windows, batch schedule, optimizer
and selection windows. The selected stage-1 query feature map stays frozen.
The first eight frozen layers are captured once; training differentiates the
exact remaining model and next-token loss. No held-out quality is used for
checkpoint selection. This tiny local protocol is not a LoLCATs replication.
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
from query_global_lora import (  # noqa: E402
    ARMS, INIT_SEED, PROJECTIONS, build_arm, canonical_hashes,
    canonical_lora, canonical_numpy, lora_core,
)
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "query_global_lora_transfer.json"
DEFAULT_WEIGHTS_DIR = HERE / "results"
LAYER = 8
CONTEXT = 512
TRAIN_WINDOWS = 32
STEPS = 300
EVAL_EVERY = 50
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0
SCHEDULE_SEED = 19
SELECTION_PERCENTAGES = (55, 58)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_weights(path: Path, module) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **canonical_numpy(module))
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(path)
    return digest


def frozen_prefix(model, inputs: mx.array) -> mx.array:
    """Output of layers 0..7, in the unchanged full-model order."""
    hidden = model.model.embed_tokens(inputs)
    mask = "causal" if hidden.shape[1] > 1 else None
    for layer in model.layers[:LAYER]:
        hidden = layer(hidden, mask, cache=None)
    mx.eval(hidden)
    return hidden


def suffix_logits(model, prefix: mx.array) -> mx.array:
    hidden = prefix
    mask = "causal" if hidden.shape[1] > 1 else None
    for layer in model.layers[LAYER:]:
        hidden = layer(hidden, mask, cache=None)
    hidden = model.model.norm(hidden)
    return (model.model.embed_tokens.as_linear(hidden)
            if model.args.tie_word_embeddings else model.lm_head(hidden))


def nll(model, prefix: mx.array, labels: mx.array) -> mx.array:
    logits = suffix_logits(model, prefix).astype(mx.float32)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        reduction="mean")


def selection_nll(model, prepared) -> tuple[float, list[float]]:
    values = []
    for prefix, labels in prepared:
        loss = nll(model, prefix, labels)
        mx.eval(loss)
        values.append(float(loss))
    return sum(values) / len(values), values


def projection_hashes(base) -> dict[str, str]:
    return {
        name: hashlib.sha256(np.asarray(
            getattr(base, name).weight.astype(mx.float32)).tobytes()).hexdigest()
        for name in PROJECTIONS
    }


def main() -> None:
    opts = options()
    if not (opts.snapshot / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {opts.snapshot}")
    stage1 = json.loads(STAGE1_RESULT.read_text())
    stage1_sha = hashlib.sha256(STAGE1_RESULT.read_bytes()).hexdigest()
    stage1_weight_sha = hashlib.sha256(STAGE1_WEIGHTS.read_bytes()).hexdigest()
    if (stage1["status"] != "complete" or not stage1["meta"]["primary"]
            or stage1["weights"]["sha256"] != stage1_weight_sha):
        raise ValueError("complete primary selected stage-1 feature map required")
    with np.load(STAGE1_WEIGHTS) as checkpoint:
        stage1_q = checkpoint["delta_q"].astype(np.float32)
        stage1_k = checkpoint["delta_k"].astype(np.float32)
    if stage1_q.shape != (32, 64, 64) or stage1_k.shape != (8, 64, 64):
        raise ValueError("selected stage-1 map shapes differ from 1B Llama")

    if opts.smoke:
        context, train_windows, steps, eval_every, selection_pcts = (
            128, 2, 2, 1, (55,))
        if opts.output == DEFAULT_OUTPUT:
            opts.output = HERE / "results" / "query_global_lora_smoke.json"
    else:
        context, train_windows, steps, eval_every, selection_pcts = (
            CONTEXT, TRAIN_WINDOWS, STEPS, EVAL_EVERY,
            SELECTION_PERCENTAGES)
    weight_paths = {
        arm: opts.weights_dir / (
            f"query_global_lora_{arm}{'_smoke' if opts.smoke else ''}_weights.npz")
        for arm in ARMS
    }
    started = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    model.freeze()
    load_s = time.perf_counter() - started
    ids = np.asarray(tokenizer.encode(CORPUS.read_text()), dtype=np.int32)
    spacing = int(len(ids) * 0.5) // train_windows
    train_starts = [int((index + 0.5) * spacing)
                    for index in range(train_windows)]
    selection_starts = [int(pct * len(ids) / 100)
                        for pct in selection_pcts]
    starts = sorted(train_starts + selection_starts)
    if (min(right - left for left, right in zip(starts, starts[1:]))
            < context + 1 or starts[-1] + context + 1 > len(ids)):
        raise ValueError("train and selection windows overlap or exceed corpus")
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(
        0, train_windows, size=steps, dtype=np.int32)
    original = model.layers[LAYER].self_attn
    base_hashes = projection_hashes(original)
    modules = {}
    arm_meta = {}
    for arm in ARMS:
        modules[arm], arm_meta[arm] = build_arm(
            original, model.args, arm, stage1_q=stage1_q,
            stage1_k=stage1_k, inference_sync_blocks=0)
    counts = {arm_meta[arm]["trainable_parameters"] for arm in ARMS}
    if len(counts) != 1 or counts != {106496}:
        raise AssertionError(f"rank-8 LoRA parameter count differs: {counts}")
    initial_hashes = {arm: canonical_hashes(module)
                      for arm, module in modules.items()}
    for key in canonical_lora(modules[ARMS[0]]):
        if key.endswith("lora_a") and len({initial_hashes[arm][key]
                                          for arm in ARMS}) != 1:
            raise AssertionError(f"LoRA A initializer differs at {key}")
        if key.endswith("lora_b") and any(np.count_nonzero(
                canonical_numpy(modules[arm])[key]) for arm in ARMS):
            raise AssertionError(f"LoRA B did not start at zero: {key}")

    record = {
        "status": "running",
        "meta": {
            "primary": not opts.smoke,
            "snapshot_revision": opts.snapshot.name,
            "config_sha256": hashlib.sha256(
                (opts.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (opts.snapshot / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lora_builder_source_sha256": hashlib.sha256(
                (HERE / "query_global_lora.py").read_bytes()).hexdigest(),
            "query_attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "stage1_result_sha256": stage1_sha,
            "stage1_weights_sha256": stage1_weight_sha,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "layer": LAYER,
            "context": context,
            "train_windows": train_windows,
            "train_percent_region": [0, 50],
            "selection_percentages": list(selection_pcts),
            "steps_per_arm": steps,
            "eval_every": eval_every,
            "teacher_prefix_unique_tokens":
                (train_windows + len(selection_pcts)) * context,
            "training_token_exposures_per_arm": steps * context,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW constant LR, zero weight decay",
            "grad_clip_norm": GRAD_CLIP,
            "schedule_seed": SCHEDULE_SEED,
            "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "lora_init_seed": INIT_SEED,
            "lora_zero_B": True,
            "lora_A_identical_across_arms": True,
            "selection_criterion": "lowest mean next-token NLL on fixed 55/58% windows per arm",
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
            "limitations": "one corpus, one converted layer, tiny local fine-tune; final quality and speed require separate tests",
        },
        "windows": {
            "train": [
                {"token_window": [start, start + context + 1],
                 "token_ids_sha256": hashlib.sha256(
                     ids[start:start + context + 1].tobytes()).hexdigest()}
                for start in train_starts
            ],
            "selection": [
                {"token_window": [start, start + context + 1],
                 "token_ids_sha256": hashlib.sha256(
                     ids[start:start + context + 1].tobytes()).hexdigest()}
                for start in selection_starts
            ],
        },
        "pretrained_projection_hashes": base_hashes,
        "initial_lora_hashes": initial_hashes,
        "arm_meta": arm_meta,
        "prefix_vs_full_teacher_max_abs_logit": None,
        "timing": {
            "load_s": load_s, "teacher_prefix_capture_s": None,
            "training_s_by_arm": {arm: 0.0 for arm in ARMS},
            "selection_s_by_arm": {arm: 0.0 for arm in ARMS},
        },
        "baseline_selection": {},
        "curve": {arm: [] for arm in ARMS},
        "best": {arm: None for arm in ARMS},
        "weights": {arm: {"path": str(weight_paths[arm]), "sha256": None}
                    for arm in ARMS},
        "peak_metal_bytes": None,
    }
    atomic_json(opts.output, record)

    # Prove the frozen-prefix shortcut is the same full teacher computation.
    probe = mx.array(ids[selection_starts[0]:selection_starts[0] +
                         min(context, 128)][None, :], dtype=mx.int32)
    whole = model(probe)
    split = suffix_logits(model, frozen_prefix(model, probe))
    difference = mx.max(mx.abs(whole.astype(mx.float32) -
                               split.astype(mx.float32)))
    mx.eval(difference)
    record["prefix_vs_full_teacher_max_abs_logit"] = float(difference)
    if record["prefix_vs_full_teacher_max_abs_logit"] != 0.0:
        raise AssertionError("frozen-prefix shortcut changed teacher logits")

    capture_started = time.perf_counter()
    def prepare(start: int):
        input_ids = mx.array(ids[start:start + context][None, :], dtype=mx.int32)
        labels = mx.array(ids[start + 1:start + context + 1][None, :],
                          dtype=mx.int32)
        return frozen_prefix(model, input_ids), labels

    train_prepared = [prepare(start) for start in train_starts]
    selection_prepared = [prepare(start) for start in selection_starts]
    record["timing"]["teacher_prefix_capture_s"] = (
        time.perf_counter() - capture_started)
    atomic_json(opts.output, record)

    optimizers = {
        arm: optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.0)
        for arm in ARMS
    }
    gradients = {}
    for arm, module in modules.items():
        model.layers[LAYER].self_attn = module
        module.eval()
        selected, by_window = selection_nll(model, selection_prepared)
        record["baseline_selection"][arm] = {
            "nll": selected, "by_window": by_window}
        record["best"][arm] = {"step": 0, "selection_nll": selected}
        record["weights"][arm]["sha256"] = atomic_weights(
            weight_paths[arm], module)
        record["curve"][arm].append({
            "step": 0, "selection_nll": selected,
            "selection_nll_by_window": by_window})

        def loss_fn(candidate, prefix, labels, *, expected=module):
            if candidate is not expected or model.layers[LAYER].self_attn is not candidate:
                raise AssertionError("LoRA arm differs from installed model layer")
            return nll(model, prefix, labels)

        gradients[arm] = nn.value_and_grad(module, loss_fn)
    atomic_json(opts.output, record)
    mx.reset_peak_memory()
    try:
        for step, chosen in enumerate(schedule, start=1):
            rotation = (step - 1) % len(ARMS)
            order = ARMS[rotation:] + ARMS[:rotation]
            prefix, labels = train_prepared[int(chosen)]
            latest = {}
            for arm in order:
                module = modules[arm]
                model.layers[LAYER].self_attn = module
                module.train()
                begun = time.perf_counter()
                loss, grads = gradients[arm](module, prefix, labels)
                if step == 1:
                    grad_by_key = {key: float(mx.sqrt(mx.sum(
                        value.astype(mx.float32) ** 2)))
                        for key, value in nn.utils.tree_flatten(grads)}
                    if not any(value > 0 for key, value in grad_by_key.items()
                               if key.endswith("lora_b")):
                        raise AssertionError(f"{arm}: zero initial LoRA B gradients")
                    arm_meta[arm]["initial_gradient_norms"] = grad_by_key
                grads, grad_norm = optim.clip_grad_norm(grads, GRAD_CLIP)
                optimizers[arm].update(module, grads)
                mx.eval(module.trainable_parameters(), optimizers[arm].state,
                        loss, grad_norm)
                record["timing"]["training_s_by_arm"][arm] += (
                    time.perf_counter() - begun)
                latest[arm] = {"train_nll": float(loss),
                               "gradient_norm_before_clip": float(grad_norm)}
            if step == 1 or step % eval_every == 0 or step == steps:
                for arm in ARMS:
                    module = modules[arm]
                    model.layers[LAYER].self_attn = module
                    module.eval()
                    begun = time.perf_counter()
                    selected, by_window = selection_nll(
                        model, selection_prepared)
                    record["timing"]["selection_s_by_arm"][arm] += (
                        time.perf_counter() - begun)
                    row = {
                        "step": step,
                        "train_window_index": int(chosen),
                        **latest[arm],
                        "selection_nll": selected,
                        "selection_nll_by_window": by_window,
                        "metal_peak_bytes": int(mx.get_peak_memory()),
                        "host_load_average": list(os.getloadavg()),
                    }
                    record["curve"][arm].append(row)
                    if selected < record["best"][arm]["selection_nll"]:
                        record["best"][arm] = {
                            "step": step, "selection_nll": selected}
                        record["weights"][arm]["sha256"] = atomic_weights(
                            weight_paths[arm], module)
                    print(f"step={step} {arm} train={float(latest[arm]['train_nll']):.5f} "
                          f"select={selected:.5f} best={record['best'][arm]['selection_nll']:.5f}",
                          flush=True)
                atomic_json(opts.output, record)
    finally:
        model.layers[LAYER].self_attn = original
    if projection_hashes(original) != base_hashes:
        raise AssertionError("frozen pretrained q/k/v/o weights changed")
    record["peak_metal_bytes"] = int(mx.get_peak_memory())
    record["meta"]["host_load_at_end"] = list(os.getloadavg())
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(opts.output, record)
    print("best", record["best"], "weights", record["weights"], flush=True)


if __name__ == "__main__":
    main()
