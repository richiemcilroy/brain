"""Real mlx-lm greedy generation with explicit caches for the 1B prototypes.

For each arm, four generated token IDs must equal a fresh-cache manual greedy
loop on the same 129-token prompt. The official `generate_step` path chunks
prefill at 64 tokens and evaluates every custom cache's `state`. This is a
functional smoke test, not a speed or quality benchmark.
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
from mlx_lm.generate import generate_step  # noqa: E402

from benchmark_local_hybrid import model_call  # noqa: E402
from llm_hybrid import transplant  # noqa: E402
from local_window_hybrid import install_local_window_hybrid, make_hybrid_cache  # noqa: E402
from oss_model_benchmark import CORPUS, DEFAULT_SNAPSHOT  # noqa: E402
from streaming_memory import MemoryCache, StreamingGatedMemoryCarrier  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "verify_generation.json"


def run_arm(model, prompt: np.ndarray, arm: str) -> dict:
    def cache():
        if arm in ("local", "hybrid", "signed"):
            return make_hybrid_cache(model)
        slots = model.make_cache()
        if arm == "trace":
            slots[8] = MemoryCache()
        return slots

    x = mx.array(prompt[None, :])
    # Untimed equivalence guard for this checkpoint's tied vocabulary head.
    direct = np.asarray(model(x)[:, -1, :].astype(mx.float32))
    optimized = np.asarray(model_call(model, x, cache=None).astype(mx.float32))
    head_max = float(np.abs(direct - optimized).max())
    head_top1 = bool(np.array_equal(direct.argmax(-1), optimized.argmax(-1)))

    prompt_cache = cache()
    generated = []
    sampled_logprobs = []
    for token, logprobs in generate_step(
        mx.array(prompt), model, max_tokens=4, prompt_cache=prompt_cache,
        prefill_step_size=64, kv_bits=None,
    ):
        generated.append(int(token))
        mx.eval(logprobs)
        sampled_logprobs.append(float(logprobs[int(token)]))

    manual_cache = cache()
    logits = model_call(model, x, manual_cache)
    manual = []
    for i in range(4):
        token = int(np.asarray(logits.astype(mx.float32)).argmax(-1).item())
        manual.append(token)
        if i < 3:
            logits = model_call(model, mx.array([[token]], dtype=mx.int32),
                                manual_cache)

    row = {
        "arm": arm,
        "generated_ids": generated,
        "manual_ids": manual,
        "ids_equal": generated == manual,
        "sampled_logprobs": sampled_logprobs,
        "logprobs_finite": bool(np.isfinite(sampled_logprobs).all()),
        "direct_vs_last_head_max_abs_logit": head_max,
        "direct_vs_last_head_top1_equal": head_top1,
        "generate_cache_offsets": [int(c.offset) for c in prompt_cache],
        "manual_cache_offsets": [int(c.offset) for c in manual_cache],
        "generate_cache_offsets_agree":
            len({int(c.offset) for c in prompt_cache}) == 1,
        "manual_cache_offsets_agree":
            len({int(c.offset) for c in manual_cache}) == 1,
    }
    if arm in ("local", "hybrid", "signed"):
        row["bounded_kv_tokens"] = int(prompt_cache[8].keys.shape[2])
        row["bounded_kv_ok"] = row["bounded_kv_tokens"] <= 63
    if not (row["ids_equal"] and row["logprobs_finite"] and head_top1
            and row["generate_cache_offsets_agree"]
            and row["manual_cache_offsets_agree"]
            and row.get("bounded_kv_ok", True)):
        raise AssertionError(f"real generation differs from manual greedy: {row}")
    return row


def main() -> None:
    if not (DEFAULT_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"offline checkpoint missing: {DEFAULT_SNAPSHOT}")
    model, tokenizer = load(str(DEFAULT_SNAPSHOT))
    model.eval()
    prompt = np.asarray(tokenizer.encode(CORPUS.read_text()[:2000])[:129],
                        dtype=np.int32)
    if len(prompt) != 129:
        raise ValueError("short prompt corpus")
    result = {
        "status": "running",
        "meta": {
            "revision": DEFAULT_SNAPSHOT.name,
            "weight_blob_id": (DEFAULT_SNAPSHOT / "model.safetensors").resolve().name,
            "config_sha256": hashlib.sha256(
                (DEFAULT_SNAPSHOT / "config.json").read_bytes()).hexdigest(),
            "prompt_ids_sha256": hashlib.sha256(prompt.tobytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_window_source_sha256": hashlib.sha256(
                (HERE / "local_window_hybrid.py").read_bytes()).hexdigest(),
            "streaming_memory_source_sha256": hashlib.sha256(
                (HERE / "streaming_memory.py").read_bytes()).hexdigest(),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "prompt_tokens": 129,
            "generated_tokens": 4,
            "prefill_step_size": 64,
            "greedy": True,
            "kv_bits": None,
            "custom_prompt_cache": True,
            "signed_memory_gain": -0.1,
            "started_epoch": time.time(),
        },
        "rows": [],
    }

    def save():
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temp = OUTPUT.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temp.replace(OUTPUT)

    save()
    layer = model.layers[8]
    original = layer.self_attn
    result["rows"].append(run_arm(model, prompt, "teacher"))
    save()

    mx.random.seed(0)
    installation = install_local_window_hybrid(
        model, layer=8, decay=0.7, memory_gain=0.0)
    result["conversion"] = installation.conversion
    result["rows"].append(run_arm(model, prompt, "local"))
    save()
    installation.hybrid.memory_gain = 0.05
    result["rows"].append(run_arm(model, prompt, "hybrid"))
    save()
    installation.hybrid.memory_gain = -0.1
    result["rows"].append(run_arm(model, prompt, "signed"))
    save()
    installation.restore()

    mx.random.seed(0)
    carrier, trace_conversion = transplant(
        model, 8, "transfer", original=original, decay=0.7)
    layer.self_attn = StreamingGatedMemoryCarrier(carrier)
    result["trace_conversion"] = trace_conversion
    result["rows"].append(run_arm(model, prompt, "trace"))
    layer.self_attn = original
    result["status"] = "complete"
    result["meta"]["finished_epoch"] = time.time()
    save()
    print([(r["arm"], r["generated_ids"], r["ids_equal"])
           for r in result["rows"]], flush=True)


if __name__ == "__main__":
    main()
