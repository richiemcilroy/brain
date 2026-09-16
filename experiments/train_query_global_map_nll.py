"""Direct next-token adjustment of the published one-layer query map.

Keep the pretrained 1B model frozen. Warm-start its Shakespeare-selected
delta_q/delta_k map, capture the frozen first-eight-layer prefixes once, then
differentiate genuine suffix next-token loss through only those two arrays.
Use disjoint balanced Shakespeare/WikiText train windows and a separate
balanced validation score. Fresh quality is evaluated by another script.
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
from train_query_global_lora import (  # noqa: E402
    frozen_prefix, nll, projection_hashes, selection_nll, suffix_logits,
)
from train_query_global_transfer import (  # noqa: E402
    DEFAULT_RESULT as OLD_RESULT, DEFAULT_WEIGHTS as OLD_WEIGHTS,
    atomic_json, atomic_weights,
)


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results" / "query_global_map_nll.json"
DEFAULT_WEIGHTS = HERE / "results" / "query_global_map_nll_weights.npz"
PROTOCOL = HERE.parent / "docs" / "QUERY_GLOBAL_MAP_NLL_PROTOCOL.md"
LAYER = 8
CONTEXT = 512
SHAKESPEARE_WINDOWS = 128
WIKI_TRAIN_WINDOWS = 192
SHAKESPEARE_STEPS = 200
WIKI_STEPS = 200
EVAL_EVERY = 50
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
SCHEDULE_SEED = 37
SHAKESPEARE_SELECTION = (55, 58, 61)
WIKI_SELECTION = (10, 20, 40, 60, 80, 90)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def draw_schedule(shake_count: int, wiki_count: int,
                  shake_steps: int, wiki_steps: int) -> np.ndarray:
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


def balanced_selection(model, prepared: dict) -> dict:
    rows = {}
    for domain in ("shakespeare", "wikitext_valid"):
        mean, by_window = selection_nll(model, prepared[domain])
        rows[domain] = {"nll": mean, "by_window": by_window}
    return {
        "score": 0.5 * rows["shakespeare"]["nll"] +
                 0.5 * rows["wikitext_valid"]["nll"],
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
        raise ValueError("complete selected original map required")
    with np.load(OLD_WEIGHTS) as checkpoint:
        old_q = checkpoint["delta_q"].astype(np.float32)
        old_k = checkpoint["delta_k"].astype(np.float32)
    if old_q.shape != (32, 64, 64) or old_k.shape != (8, 64, 64):
        raise ValueError("original map has wrong 1B head shapes")

    if opts.smoke:
        context = 128
        shake_windows, wiki_windows = 2, 2
        shake_steps, wiki_steps, eval_every = 2, 2, 1
        selection_shake, selection_wiki = (55,), (20,)
        if opts.result == DEFAULT_RESULT:
            opts.result = HERE / "results" / "query_global_map_nll_smoke.json"
        if opts.weights == DEFAULT_WEIGHTS:
            opts.weights = HERE / "results" / "query_global_map_nll_smoke_weights.npz"
    else:
        context = CONTEXT
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
        raise AssertionError("predeclared balanced schedule changed")

    load_started = time.perf_counter()
    model, tokenizer = load(str(opts.snapshot))
    model.eval()
    model.freeze()
    load_s = time.perf_counter() - load_started
    config_sha = hashlib.sha256(
        (opts.snapshot / "config.json").read_bytes()).hexdigest()
    blob_id = (opts.snapshot / "model.safetensors").resolve().name
    if (config_sha != old["meta"]["config_sha256"] or
            blob_id != old["meta"]["weight_blob_id"]):
        raise ValueError("warm-start and current 1B model identity differ")

    data_started = time.perf_counter()
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
            count=shake_windows, window=context + 1),
        "wikitext_train": spaced_starts(
            len(ids["wikitext_train"]),
            region_end=len(ids["wikitext_train"]),
            count=wiki_windows, window=context + 1),
    }
    selection_starts = {
        "shakespeare": percentage_starts(
            len(ids["shakespeare"]), selection_shake, context + 1),
        "wikitext_valid": percentage_starts(
            len(ids["wikitext_valid"]), selection_wiki, context + 1),
    }
    if (train_starts["shakespeare"][-1] + context + 1 >
            min(selection_starts["shakespeare"].values())):
        raise ValueError("Shakespeare training and selection windows overlap")
    data_s = time.perf_counter() - data_started

    original = model.layers[LAYER].self_attn
    base_hashes = projection_hashes(original)
    module = TrainableQueryGlobalAttention(
        original, window=64, gain=DEFAULT_GAIN, chunk=64,
        inference_sync_blocks=0)
    module.delta_q = mx.array(old_q)
    module.delta_k = mx.array(old_k)
    mx.eval(module.delta_q, module.delta_k)
    if (not np.array_equal(np.asarray(module.delta_q), old_q) or
            not np.array_equal(np.asarray(module.delta_k), old_k)):
        raise AssertionError("direct-NLL map did not exactly warm-start")
    module.freeze()
    module.unfreeze(keys=["delta_q", "delta_k"], recurse=False, strict=True)
    trainable = sorted(k for k, _ in nn.utils.tree_flatten(
        module.trainable_parameters()))
    count = sum(int(np.prod(value.shape)) for _, value in
                nn.utils.tree_flatten(module.trainable_parameters()))
    if trainable != ["delta_k", "delta_q"] or count != 163840:
        raise AssertionError("pretrained tensor became trainable")

    record = {
        "status": "running",
        "meta": {
            "primary": not opts.smoke,
            "snapshot_revision": opts.snapshot.name,
            "config_sha256": config_sha,
            "weight_blob_id": blob_id,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "data_source_sha256": hashlib.sha256(
                (HERE / "query_global_broad_data.py").read_bytes()).hexdigest(),
            "map_source_sha256": hashlib.sha256(
                (HERE / "query_global_trainable.py").read_bytes()).hexdigest(),
            "attention_source_sha256": hashlib.sha256(
                (HERE / "query_global_attention.py").read_bytes()).hexdigest(),
            "prefix_and_nll_source_sha256": hashlib.sha256(
                (HERE / "train_query_global_lora.py").read_bytes()).hexdigest(),
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
            "context": context,
            "train_windows": {
                "shakespeare": shake_windows,
                "wikitext_train": wiki_windows},
            "training_distinct_input_positions": (
                shake_windows + wiki_windows) * context,
            "teacher_prefix_capture_input_positions": (
                shake_windows + wiki_windows +
                len(selection_shake) + len(selection_wiki)) * context,
            "training_input_exposures": len(schedule) * context,
            "steps": len(schedule),
            "steps_by_domain": {
                "shakespeare": shake_steps,
                "wikitext_train": wiki_steps},
            "every_train_window_visited": True,
            "eval_every": eval_every,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW constant LR, zero weight decay",
            "grad_clip_norm": GRAD_CLIP,
            "schedule_seed": SCHEDULE_SEED,
            "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "selection_percentages": {
                "shakespeare": list(selection_shake),
                "wikitext_valid": list(selection_wiki)},
            "selection_score": "0.5 * mean Shakespeare NLL + 0.5 * mean WikiText-valid NLL",
            "trainable_keys": trainable,
            "trainable_parameter_count": count,
            "no_wikitext_test_or_alice_training_or_selection": True,
            "host_load_at_start": list(os.getloadavg()),
            "started_epoch": time.time(),
        },
        "windows": {
            "train": {
                domain: window_rows(ids[domain], starts, context + 1)
                for domain, starts in train_starts.items()},
            "selection": {
                domain: window_rows(ids[domain], starts, context + 1)
                for domain, starts in selection_starts.items()},
        },
        "pretrained_projection_hashes": base_hashes,
        "prefix_vs_full_teacher_max_abs_logit": None,
        "teacher_selection": None,
        "baseline_selection": None,
        "curve": [],
        "best": None,
        "weights": {"path": str(opts.weights), "sha256": None},
        "timing": {
            "load_s": load_s, "read_and_tokenize_s": data_s,
            "teacher_prefix_capture_s": None,
            "training_updates_s": 0.0,
            "selection_s": 0.0, "training_loop_s": None},
        "capture_peak_metal_bytes": None,
        "training_peak_metal_bytes": None,
    }
    atomic_json(opts.result, record)

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
        inputs = mx.array(corpus_ids[start:start + context][
            None, :], dtype=mx.int32)
        labels = mx.array(corpus_ids[start + 1:start + context + 1][
            None, :], dtype=mx.int32)
        return frozen_prefix(model, inputs), labels

    mx.reset_peak_memory()
    capture_started = time.perf_counter()
    prepared_train = {
        domain: [prepare(domain, start) for start in starts]
        for domain, starts in train_starts.items()}
    prepared_selection = {
        domain: [prepare(domain, start) for start in starts.values()]
        for domain, starts in selection_starts.items()}
    record["timing"]["teacher_prefix_capture_s"] = (
        time.perf_counter() - capture_started)
    record["capture_peak_metal_bytes"] = int(mx.get_peak_memory())
    atomic_json(opts.result, record)

    model.layers[LAYER].self_attn = original
    teacher_started = time.perf_counter()
    record["teacher_selection"] = balanced_selection(
        model, prepared_selection)
    record["timing"]["selection_s"] += (
        time.perf_counter() - teacher_started)
    model.layers[LAYER].self_attn = module
    module.eval()
    baseline_started = time.perf_counter()
    baseline = balanced_selection(model, prepared_selection)
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

    def loss_fn(candidate, prefix, labels):
        if (candidate is not module or
                model.layers[LAYER].self_attn is not candidate):
            raise AssertionError("trained map differs from installed layer")
        return nll(model, prefix, labels)

    value_and_grad = nn.value_and_grad(module, loss_fn)
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=0.0)
    mx.reset_peak_memory()
    loop_started = time.perf_counter()
    try:
        for step, (domain_code, chosen) in enumerate(schedule, start=1):
            domain = "shakespeare" if domain_code == 0 else "wikitext_train"
            prefix, labels = prepared_train[domain][int(chosen)]
            module.train()
            update_started = time.perf_counter()
            loss, grads = value_and_grad(module, prefix, labels)
            if step == 1:
                norm_by_key = {
                    key: float(mx.sqrt(mx.sum(value.astype(mx.float32) ** 2)))
                    for key, value in nn.utils.tree_flatten(grads)}
                if not all(value > 0 for value in norm_by_key.values()):
                    raise AssertionError("initial direct-map gradient vanished")
                record["initial_gradient_norms"] = norm_by_key
            grads, grad_norm = optim.clip_grad_norm(grads, GRAD_CLIP)
            optimizer.update(module, grads)
            mx.eval(module.trainable_parameters(), optimizer.state,
                    loss, grad_norm)
            record["timing"]["training_updates_s"] += (
                time.perf_counter() - update_started)
            if step == 1 or step % eval_every == 0 or step == len(schedule):
                module.eval()
                selection_started = time.perf_counter()
                selected = balanced_selection(model, prepared_selection)
                record["timing"]["selection_s"] += (
                    time.perf_counter() - selection_started)
                record["curve"].append({
                    "step": step,
                    "training_domain": domain,
                    "training_window_index": int(chosen),
                    "training_nll": float(loss),
                    "selection": selected,
                    "gradient_norm_before_clip": float(grad_norm),
                    "metal_peak_bytes": int(mx.get_peak_memory()),
                    "host_load_average": list(os.getloadavg()),
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
                    f"step={step} {domain} train={float(loss):.5f} "
                    f"selection={selected['score']:.5f} "
                    f"best={best['selection']['score']:.5f}",
                    flush=True)
    finally:
        model.layers[LAYER].self_attn = original
    record["timing"]["training_loop_s"] = time.perf_counter() - loop_started
    if projection_hashes(original) != base_hashes:
        raise AssertionError("frozen pretrained projections changed")
    record["training_peak_metal_bytes"] = int(mx.get_peak_memory())
    record["meta"]["host_load_at_end"] = list(os.getloadavg())
    record["meta"]["finished_epoch"] = time.time()
    record["status"] = "complete"
    atomic_json(opts.result, record)
    print("best", record["best"], "weights", opts.weights, flush=True)


if __name__ == "__main__":
    main()
