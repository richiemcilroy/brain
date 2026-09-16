"""Predeclared larger-data matched LoRA control for one 1B Llama layer.

Use disjoint Shakespeare and WikiText-2 raw train windows, separate
Shakespeare and WikiText validation windows, lower LR and the same seeded
rank-8 q/k/v/o LoRA schedule for full/local/query attention. The selected
stage-1 query map and every pretrained tensor stay frozen. This source and
its protocol must be published before the primary training outcome.
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
from query_global_broad_data import (  # noqa: E402
    DATASET_CARD, DATASET_URL, MEMBERS, ZIP_SHA256, percentage_starts,
    spaced_starts, wiki_raw, window_rows,
)
from query_global_lora import (  # noqa: E402
    ARMS, INIT_SEED, build_arm, canonical_hashes, canonical_lora,
    canonical_numpy,
)
from train_query_global_lora import (  # noqa: E402
    frozen_prefix, nll, projection_hashes, selection_nll, suffix_logits,
)
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as STAGE1_RESULT, DEFAULT_WEIGHTS as STAGE1_WEIGHTS,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "query_global_lora_broad.json"
DEFAULT_WEIGHTS_DIR = HERE / "results"
PROTOCOL = HERE.parent / "docs" / "QUERY_GLOBAL_LORA_BROAD_PROTOCOL.md"
CONTEXT = 512
SHAKESPEARE_WINDOWS = 96
WIKI_TRAIN_WINDOWS = 256
STEPS = 500
SHAKESPEARE_STEPS = 125
WIKI_STEPS = 375
EVAL_EVERY = 50
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
SCHEDULE_SEED = 23
SHAKESPEARE_SELECTION = (55, 58)
WIKI_SELECTION = (20, 40, 60, 80)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights-dir", type=Path,
                        default=DEFAULT_WEIGHTS_DIR)
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


def draw_schedule(shake_count: int, wiki_count: int,
                  shake_steps: int, wiki_steps: int) -> np.ndarray:
    rng = np.random.default_rng(SCHEDULE_SEED)
    domains = np.asarray(
        [0] * shake_steps + [1] * wiki_steps, dtype=np.int32)
    domains = rng.permutation(domains)
    indices = np.asarray([
        rng.integers(0, shake_count if domain == 0 else wiki_count)
        for domain in domains], dtype=np.int32)
    return np.column_stack((domains, indices))


def balanced_selection(model, prepared: dict) -> dict:
    domain_rows = {}
    for domain in ("shakespeare", "wikitext_valid"):
        mean, by_window = selection_nll(model, prepared[domain])
        domain_rows[domain] = {
            "nll": mean, "by_window": by_window}
    return {
        "score": 0.5 * domain_rows["shakespeare"]["nll"] +
                 0.5 * domain_rows["wikitext_valid"]["nll"],
        "domains": domain_rows,
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
        raise ValueError("complete primary selected stage-1 map required")
    with np.load(STAGE1_WEIGHTS) as checkpoint:
        stage1_q = checkpoint["delta_q"].astype(np.float32)
        stage1_k = checkpoint["delta_k"].astype(np.float32)
    if stage1_q.shape != (32, 64, 64) or stage1_k.shape != (8, 64, 64):
        raise ValueError("selected stage-1 map shapes differ from 1B Llama")

    if opts.smoke:
        shake_windows, wiki_windows, shake_steps, wiki_steps = 2, 2, 1, 1
        selection_shake, selection_wiki, eval_every = (55,), (20,), 1
        if opts.output == DEFAULT_OUTPUT:
            opts.output = HERE / "results" / "query_global_lora_broad_smoke.json"
    else:
        shake_windows, wiki_windows = SHAKESPEARE_WINDOWS, WIKI_TRAIN_WINDOWS
        shake_steps, wiki_steps = SHAKESPEARE_STEPS, WIKI_STEPS
        selection_shake, selection_wiki = (
            SHAKESPEARE_SELECTION, WIKI_SELECTION)
        eval_every = EVAL_EVERY
    steps = shake_steps + wiki_steps
    weight_paths = {
        arm: opts.weights_dir / (
            f"query_global_lora_broad_{arm}"
            f"{'_smoke' if opts.smoke else ''}_weights.npz")
        for arm in ARMS
    }

    begun = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    model.freeze()
    load_s = time.perf_counter() - begun
    data_started = time.perf_counter()
    ids = {
        "shakespeare": np.asarray(
            tokenizer.encode(CORPUS.read_text()), dtype=np.int32),
        "wikitext_train": np.asarray(
            tokenizer.encode(wiki_raw("train")), dtype=np.int32),
        "wikitext_valid": np.asarray(
            tokenizer.encode(wiki_raw("valid")), dtype=np.int32),
    }
    starts = {
        "shakespeare": spaced_starts(
            len(ids["shakespeare"]),
            region_end=int(len(ids["shakespeare"]) * 0.5),
            count=shake_windows, window=CONTEXT + 1),
        "wikitext_train": spaced_starts(
            len(ids["wikitext_train"]),
            region_end=len(ids["wikitext_train"]),
            count=wiki_windows, window=CONTEXT + 1),
    }
    selection_starts = {
        "shakespeare": percentage_starts(
            len(ids["shakespeare"]), selection_shake, CONTEXT + 1),
        "wikitext_valid": percentage_starts(
            len(ids["wikitext_valid"]), selection_wiki, CONTEXT + 1),
    }
    if (starts["shakespeare"][-1] + CONTEXT + 1 >
            min(selection_starts["shakespeare"].values())):
        raise ValueError("Shakespeare train and selection windows overlap")
    schedule = draw_schedule(
        shake_windows, wiki_windows, shake_steps, wiki_steps)
    if (len(schedule) != steps or
            int(np.sum(schedule[:, 0] == 0)) != shake_steps or
            int(np.sum(schedule[:, 0] == 1)) != wiki_steps):
        raise AssertionError("predeclared domain quotas changed")
    data_s = time.perf_counter() - data_started

    original = model.layers[8].self_attn
    base_hashes = projection_hashes(original)
    modules = {}
    arm_meta = {}
    for arm in ARMS:
        modules[arm], arm_meta[arm] = build_arm(
            original, model.args, arm, stage1_q=stage1_q,
            stage1_k=stage1_k, inference_sync_blocks=0)
    if {arm_meta[arm]["trainable_parameters"] for arm in ARMS} != {106496}:
        raise AssertionError("rank-8 LoRA parameter count differs")
    initial_hashes = {arm: canonical_hashes(module)
                      for arm, module in modules.items()}
    for key in canonical_lora(modules[ARMS[0]]):
        if key.endswith("lora_a") and len({
                initial_hashes[arm][key] for arm in ARMS}) != 1:
            raise AssertionError("LoRA A initializer differs across arms")
        if key.endswith("lora_b") and any(
                np.count_nonzero(canonical_numpy(modules[arm])[key])
                for arm in ARMS):
            raise AssertionError("LoRA B did not start at zero")

    record = {
        "status": "running",
        "meta": {
            "primary": not opts.smoke,
            "snapshot_revision": opts.snapshot.name,
            "config_sha256": hashlib.sha256(
                (opts.snapshot / "config.json").read_bytes()).hexdigest(),
            "weight_blob_id": (opts.snapshot / "model.safetensors").resolve().name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "data_source_sha256": hashlib.sha256(
                (HERE / "query_global_broad_data.py").read_bytes()).hexdigest(),
            "lora_builder_source_sha256": hashlib.sha256(
                (HERE / "query_global_lora.py").read_bytes()).hexdigest(),
            "prefix_and_loss_source_sha256": hashlib.sha256(
                (HERE / "train_query_global_lora.py").read_bytes()).hexdigest(),
            "stage1_result_sha256": stage1_sha,
            "stage1_weights_sha256": stage1_weight_sha,
            "shakespeare_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "wiki_zip_sha256": ZIP_SHA256,
            "wiki_member_sha256": {
                split: MEMBERS[split][1] for split in ("train", "valid")},
            "wiki_dataset_url": DATASET_URL,
            "wiki_dataset_card": DATASET_CARD,
            "token_counts": {domain: len(value)
                             for domain, value in ids.items()},
            "token_ids_sha256": {
                domain: hashlib.sha256(value.tobytes()).hexdigest()
                for domain, value in ids.items()},
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "hardware": platform.platform(),
            "dtype": str(model.model.embed_tokens.weight.dtype),
            "converted_layer": 8,
            "context": CONTEXT,
            "train_windows": {
                "shakespeare": shake_windows,
                "wikitext_train": wiki_windows},
            "training_window_distinct_input_positions": (
                shake_windows + wiki_windows) * CONTEXT,
            "teacher_prefix_capture_input_positions": (
                shake_windows + wiki_windows +
                len(selection_shake) + len(selection_wiki)) * CONTEXT,
            "scheduled_distinct_input_positions": (
                len(np.unique(schedule, axis=0)) * CONTEXT),
            "scheduled_distinct_windows_by_domain": {
                "shakespeare": len(np.unique(
                    schedule[schedule[:, 0] == 0, 1])),
                "wikitext_train": len(np.unique(
                    schedule[schedule[:, 0] == 1, 1])),
            },
            "steps_per_arm": steps,
            "domain_steps": {
                "shakespeare": shake_steps,
                "wikitext_train": wiki_steps},
            "training_token_exposures_per_arm": steps * CONTEXT,
            "eval_every": eval_every,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW constant LR, zero weight decay",
            "grad_clip_norm": GRAD_CLIP,
            "schedule_seed": SCHEDULE_SEED,
            "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "lora_init_seed": INIT_SEED,
            "selection_percentages": {
                "shakespeare": list(selection_shake),
                "wikitext_valid": list(selection_wiki)},
            "selection_score": "0.5 * mean Shakespeare NLL + 0.5 * mean WikiText validation NLL",
            "no_wikitext_test_training_or_selection": True,
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
        },
        "windows": {
            "train": {
                domain: window_rows(ids[domain], positions, CONTEXT + 1)
                for domain, positions in starts.items()},
            "selection": {
                domain: window_rows(ids[domain], positions, CONTEXT + 1)
                for domain, positions in selection_starts.items()},
        },
        "timing": {
            "load_s": load_s, "read_and_tokenize_s": data_s,
            "teacher_prefix_capture_s": None,
            "training_s_by_arm": {arm: 0.0 for arm in ARMS},
            "selection_s_by_arm": {arm: 0.0 for arm in ARMS},
        },
        "pretrained_projection_hashes": base_hashes,
        "initial_lora_hashes": initial_hashes,
        "arm_meta": arm_meta,
        "prefix_vs_full_teacher_max_abs_logit": None,
        "baseline_selection": {},
        "curve": {arm: [] for arm in ARMS},
        "best": {arm: None for arm in ARMS},
        "weights": {arm: {"path": str(weight_paths[arm]), "sha256": None}
                    for arm in ARMS},
        "capture_peak_metal_bytes": None,
        "training_peak_metal_bytes": None,
    }
    atomic_json(opts.output, record)

    probe_start = min(selection_starts["shakespeare"].values())
    probe = mx.array(ids["shakespeare"][probe_start:probe_start + 128][
        None, :], dtype=mx.int32)
    whole = model(probe)
    split = suffix_logits(model, frozen_prefix(model, probe))
    difference = mx.max(mx.abs(
        whole.astype(mx.float32) - split.astype(mx.float32)))
    mx.eval(difference)
    record["prefix_vs_full_teacher_max_abs_logit"] = float(difference)
    if record["prefix_vs_full_teacher_max_abs_logit"] != 0.0:
        raise AssertionError("frozen-prefix shortcut changed teacher logits")

    def prepare(domain: str, start: int):
        corpus_ids = ids[domain]
        input_ids = mx.array(corpus_ids[start:start + CONTEXT][
            None, :], dtype=mx.int32)
        labels = mx.array(corpus_ids[start + 1:start + CONTEXT + 1][
            None, :], dtype=mx.int32)
        return frozen_prefix(model, input_ids), labels

    mx.reset_peak_memory()
    capture_started = time.perf_counter()
    train_prepared = {
        domain: [prepare(domain, start) for start in positions]
        for domain, positions in starts.items()}
    selection_prepared = {
        domain: [prepare(domain, start) for start in positions.values()]
        for domain, positions in selection_starts.items()}
    record["timing"]["teacher_prefix_capture_s"] = (
        time.perf_counter() - capture_started)
    record["capture_peak_metal_bytes"] = int(mx.get_peak_memory())
    atomic_json(opts.output, record)

    optimizers = {
        arm: optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.0)
        for arm in ARMS}
    gradients = {}
    for arm, module in modules.items():
        model.layers[8].self_attn = module
        module.eval()
        selected = balanced_selection(model, selection_prepared)
        record["baseline_selection"][arm] = selected
        record["best"][arm] = {
            "step": 0, "selection_score": selected["score"]}
        record["weights"][arm]["sha256"] = atomic_weights(
            weight_paths[arm], module)
        record["curve"][arm].append({"step": 0, "selection": selected})

        def loss_fn(candidate, prefix, labels, *, expected=module):
            if (candidate is not expected or
                    model.layers[8].self_attn is not candidate):
                raise AssertionError("LoRA arm differs from installed layer")
            return nll(model, prefix, labels)

        gradients[arm] = nn.value_and_grad(module, loss_fn)
    atomic_json(opts.output, record)
    mx.reset_peak_memory()
    try:
        for step, (domain_code, chosen) in enumerate(schedule, start=1):
            domain = "shakespeare" if domain_code == 0 else "wikitext_train"
            prefix, labels = train_prepared[domain][int(chosen)]
            rotation = (step - 1) % len(ARMS)
            order = ARMS[rotation:] + ARMS[:rotation]
            latest = {}
            for arm in order:
                module = modules[arm]
                model.layers[8].self_attn = module
                module.train()
                started = time.perf_counter()
                loss, grads = gradients[arm](module, prefix, labels)
                if step == 1:
                    grad_by_key = {key: float(mx.sqrt(mx.sum(
                        value.astype(mx.float32) ** 2)))
                        for key, value in nn.utils.tree_flatten(grads)}
                    if not any(value > 0 for key, value in grad_by_key.items()
                               if key.endswith("lora_b")):
                        raise AssertionError(f"{arm}: zero initial LoRA B gradient")
                    arm_meta[arm]["initial_gradient_norms"] = grad_by_key
                grads, grad_norm = optim.clip_grad_norm(grads, GRAD_CLIP)
                optimizers[arm].update(module, grads)
                mx.eval(module.trainable_parameters(), optimizers[arm].state,
                        loss, grad_norm)
                record["timing"]["training_s_by_arm"][arm] += (
                    time.perf_counter() - started)
                latest[arm] = {
                    "train_nll": float(loss),
                    "gradient_norm_before_clip": float(grad_norm)}
            if step == 1 or step % eval_every == 0 or step == steps:
                for arm in ARMS:
                    module = modules[arm]
                    model.layers[8].self_attn = module
                    module.eval()
                    started = time.perf_counter()
                    selected = balanced_selection(
                        model, selection_prepared)
                    record["timing"]["selection_s_by_arm"][arm] += (
                        time.perf_counter() - started)
                    row = {
                        "step": step,
                        "train_domain": domain,
                        "train_window_index": int(chosen),
                        **latest[arm],
                        "selection": selected,
                        "metal_peak_bytes": int(mx.get_peak_memory()),
                        "host_load_average": list(os.getloadavg()),
                    }
                    record["curve"][arm].append(row)
                    if selected["score"] < record["best"][arm][
                            "selection_score"]:
                        record["best"][arm] = {
                            "step": step,
                            "selection_score": selected["score"]}
                        record["weights"][arm]["sha256"] = atomic_weights(
                            weight_paths[arm], module)
                    print(f"step={step} {arm} {domain} train="
                          f"{latest[arm]['train_nll']:.5f} select="
                          f"{selected['score']:.5f} best="
                          f"{record['best'][arm]['selection_score']:.5f}",
                          flush=True)
                atomic_json(opts.output, record)
    finally:
        model.layers[8].self_attn = original
    if projection_hashes(original) != base_hashes:
        raise AssertionError("frozen pretrained projections changed")
    record["training_peak_metal_bytes"] = int(mx.get_peak_memory())
    record["meta"]["host_load_at_end"] = list(os.getloadavg())
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(opts.output, record)
    print("best", record["best"], "weights", record["weights"], flush=True)


if __name__ == "__main__":
    main()
