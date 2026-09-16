"""Paired time-to-quality gate for fused attention versus input-only memory.

This is a NEW run with its criterion fixed before it starts. The old headline
in docs/EFFICIENCY.md was retracted: its attention control had not beaten a
bigram. The current content_gate.py model supplies a properly tuned attention
control and a parameter-matched input-only gated recurrence. This file asks a
different question from a same-step final-loss comparison: how many training
tokens, estimated FLOPs, and synchronized seconds does each arm need to reach
the same held-out quality?

Primary protocol (changes are labelled diagnostic in the output):
    corpus TinyShakespeare, character-level, ctx=512, d=128, 2 layers
    fused causal A_attn versus B_match (input-only gate, extra input projection)
    seeds 0..4, exactly paired batch positions for each seed
    batch=16, AdamW lr=1e-3, 100-step warmup, clip=1, weight decay=.01
    1500 train steps maximum; full deterministic validation at step 0 and every
    100 steps; first observed full-validation bpc <= 2.4 is the quality gate

The true crossing lies between the preceding and first passing checkpoints.
The first passing checkpoint is an observed upper bound, not an interpolated
crossing time. A miss by step 1500 is right-censored. We report training time,
validation time, and setup-plus-total wall time separately. The FLOPs numbers
use content_gate.py's analytic forward+backward estimator; they exclude exact
softmax, optimizer and Metal-kernel costs and are NOT hardware counters.

The aspirational efficiency gate is predeclared as a >=2x paired median win in
both synchronized training seconds and estimated train FLOPs to 2.4 bpc,
with wins in at least four of five seed pairs. A shared-machine load warning
prevents a clean wall-clock verdict. Passing this tiny-corpus gate would justify
a larger, multi-corpus, realistic-width experiment, not an OSS-model claim.

Run: python3 experiments/train_time_to_quality.py
Quick diagnostic only: python3 experiments/train_time_to_quality.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_gate as cg  # noqa: E402
from llm_efficiency import CORPUS, load_corpus  # noqa: E402
from llm_tuned import floors  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "results" / "train_time_to_quality.json"
ARMS = ("A_attn", "B_match")
SEEDS = (0, 1, 2, 3, 4)
QUALITY_BPC = 2.4
CTX = 512
D = 128
LAYERS = 2
HEADS = 4
MLP = 4
BATCH = 16
MAX_STEPS = 1500
EVAL_EVERY = 100
LR = 1e-3
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MIN_SPEEDUP = 2.0
MIN_PAIR_WINS = 4


@dataclass(frozen=True)
class Protocol:
    seeds: tuple[int, ...] = SEEDS
    steps: int = MAX_STEPS
    eval_every: int = EVAL_EVERY
    max_val_windows: int = 0
    compiled: bool = True

    @property
    def primary(self) -> bool:
        return self == Protocol()


def parse_args() -> tuple[Protocol, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="2 steps, one seed, two val windows; no claim")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--max-val-windows", type=int, default=0,
                        help="0 uses the entire deterministic holdout")
    parser.add_argument("--eager", action="store_true",
                        help="use eager train steps; labelled diagnostic")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    if args.smoke:
        protocol = Protocol(seeds=(0,), steps=2, eval_every=1,
                            max_val_windows=2, compiled=False)
        out = args.out or DEFAULT_OUT.with_name("train_time_to_quality_smoke.json")
    else:
        protocol = Protocol(seeds=seeds, steps=args.steps,
                            eval_every=args.eval_every,
                            max_val_windows=args.max_val_windows,
                            compiled=not args.eager)
        out = args.out or DEFAULT_OUT
    if not protocol.seeds or len(protocol.seeds) != len(set(protocol.seeds)):
        parser.error("--seeds must name at least one distinct integer seed")
    if protocol.steps <= 0 or protocol.eval_every <= 0:
        parser.error("--steps and --eval-every must be positive")
    if protocol.max_val_windows < 0:
        parser.error("--max-val-windows must be nonnegative")
    return protocol, out


def sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def paired_schedule(train: np.ndarray, seed: int, steps: int) -> np.ndarray:
    """Both arms receive the exact same ordered contexts for each seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(train) - CTX - 1,
                        size=(steps, BATCH), dtype=np.int32)


def batch_at(train: np.ndarray, starts: np.ndarray) -> tuple[mx.array, mx.array]:
    offsets = np.arange(CTX, dtype=np.int32)
    positions = starts[:, None] + offsets[None, :]
    return mx.array(train[positions]), mx.array(train[positions + 1])


def load_snapshot() -> dict:
    """Record contention and process/Metal memory without confusing their peaks."""
    try:
        avg = [round(float(x), 2) for x in os.getloadavg()]
    except OSError:
        avg = None
    def mlx_bytes(name: str):
        fn = getattr(mx, name, None)
        return int(fn()) if fn is not None else None
    return dict(
        load_avg_1_5_15=avg,
        cpu_count=os.cpu_count(),
        mlx_active_bytes=mlx_bytes("get_active_memory"),
        mlx_cache_bytes=mlx_bytes("get_cache_memory"),
        mlx_peak_bytes_since_arm_reset=mlx_bytes("get_peak_memory"),
        # ru_maxrss is cumulative for the PROCESS and cannot be reset per arm.
        process_peak_rss_bytes_cumulative=int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def _loss(model: nn.Module, x: mx.array, y: mx.array, vocab: int) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits.reshape(-1, vocab),
                                  y.reshape(-1), reduction="mean")


def validate(model: nn.Module, windows, vocab: int) -> float:
    """Score ALL selected deterministic windows, just as content_gate.py does."""
    total, n = 0.0, 0
    for x, y in windows:
        loss = _loss(model, x, y, vocab)
        mx.eval(loss)
        total += float(loss) * int(y.size)
        n += int(y.size)
    return (total / n) / math.log(2)


def run_arm(arm: str, seed: int, train: np.ndarray, windows, vocab: int,
            schedule: np.ndarray, protocol: Protocol) -> dict:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    setup_start = time.perf_counter()
    mx.random.seed(seed)
    model = cg.build(arm, vocab, d=D, ctx=CTX, n_layer=LAYERS)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model,
                                     lambda mod, x, y: _loss(mod, x, y, vocab))

    def body(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, GRAD_CLIP)
        opt.update(model, grads)
        return loss

    if protocol.compiled:
        # Dynamic model/optimizer states are essential. Compiling a closure
        # without them freezes model parameters in this MLX harness.
        state = [model.state, opt.state]
        step_fn = mx.compile(body, state, state)
    else:
        step_fn = None
    setup_s = time.perf_counter() - setup_start
    flops_per_token = cg.flops_per_token(arm, vocab, d=D, ctx=CTX,
                                        n_layer=LAYERS, mlp=MLP)
    params = cg.n_params(model)
    val_tokens = sum(int(y.size) for _, y in windows)
    result = dict(arm=arm, seed=seed, params=params,
                  estimated_train_flops_per_token=flops_per_token,
                  schedule_sha256=sha256_bytes(schedule),
                  compiled=protocol.compiled, setup_wall_s=setup_s,
                  # The first synchronized step includes a one-time JIT compile
                  # when `compiled=True`; primary train timing includes it.
                  first_step_wall_s=None, checkpoints=[],
                  first_observed_quality=None, final_bpc=None,
                  training_wall_s=0.0, validation_wall_s=0.0,
                  after_setup_wall_s=0.0, total_train_tokens=0,
                  memory_start=load_snapshot(), memory_end=None)
    train_s = val_s = 0.0
    prior_above_step = None
    run_start = time.perf_counter()

    def checkpoint(step: int, train_loss: float | None):
        nonlocal val_s, prior_above_step
        start = time.perf_counter()
        bpc = validate(model, windows, vocab)
        val_s += time.perf_counter() - start
        seen = step * BATCH * CTX
        n_val = len(result["checkpoints"]) + 1
        row = dict(step=step, train_tokens=seen,
                   train_loss_nats=train_loss, val_bpc=bpc,
                   training_wall_s=train_s, validation_wall_s=val_s,
                   after_setup_wall_s=time.perf_counter() - run_start,
                   estimated_train_flops=seen * flops_per_token,
                   estimated_val_flops=n_val * val_tokens * flops_per_token / 3,
                   memory_load=load_snapshot())
        result["checkpoints"].append(row)
        if bpc > QUALITY_BPC:
            prior_above_step = step
        elif result["first_observed_quality"] is None:
            result["first_observed_quality"] = dict(
                first_passing_checkpoint_step=step,
                preceding_above_threshold_checkpoint_step=prior_above_step,
                crossing_step_interval=[prior_above_step, step]
                if prior_above_step is not None else None,
                train_tokens_observed_upper_bound=seen,
                estimated_train_flops_observed_upper_bound=seen * flops_per_token,
                estimated_val_flops_through_detection=
                    n_val * val_tokens * flops_per_token / 3,
                training_wall_s=train_s, validation_wall_s=val_s,
                after_setup_wall_s=row["after_setup_wall_s"],
                setup_plus_total_wall_s=setup_s + row["after_setup_wall_s"],
                bpc=bpc,
            )
        print(f"  {arm:<7} seed {seed} step {step:4d} | "
              f"val {bpc:.4f} bpc | train {train_s:6.1f}s "
              f"eval {val_s:5.1f}s | load {row['memory_load']['load_avg_1_5_15']}",
              flush=True)

    # An initial validation gives the first crossing a known lower checkpoint.
    checkpoint(0, None)
    for step in range(1, protocol.steps + 1):
        started = time.perf_counter()
        x, y = batch_at(train, schedule[step - 1])
        opt.learning_rate = LR * min(1.0, step / WARMUP)
        if step_fn is not None:
            loss = step_fn(x, y)
            mx.eval(loss, model.state, opt.state)
        else:
            loss = body(x, y)
            mx.eval(loss, model.parameters(), opt.state)
        elapsed = time.perf_counter() - started
        train_s += elapsed
        if step == 1:
            result["first_step_wall_s"] = elapsed
        if step % protocol.eval_every == 0 or step == protocol.steps:
            checkpoint(step, float(loss))

    result["final_bpc"] = result["checkpoints"][-1]["val_bpc"]
    result["training_wall_s"] = train_s
    result["validation_wall_s"] = val_s
    result["after_setup_wall_s"] = time.perf_counter() - run_start
    result["total_train_tokens"] = protocol.steps * BATCH * CTX
    result["memory_end"] = load_snapshot()
    result["right_censored_at_step"] = (protocol.steps
                                         if result["first_observed_quality"] is None
                                         else None)
    del model, opt, loss_and_grad, step_fn
    gc.collect()
    return result


def summary(out: dict) -> dict:
    """Pair by seed; never turn a right-censored miss into an artificial time."""
    pairs = []
    for seed in out["protocol"]["seeds"]:
        rows = out["runs"].get(str(seed), {})
        if not all(arm in rows for arm in ARMS):
            continue
        a, b = (rows[arm] for arm in ARMS)
        qa, qb = a["first_observed_quality"], b["first_observed_quality"]
        pair = dict(seed=seed, attention_hit=qa is not None,
                    memory_hit=qb is not None,
                    attention_final_bpc=a["final_bpc"],
                    memory_final_bpc=b["final_bpc"])
        if qa and qb:
            pair.update(
                attention_train_s=qa["training_wall_s"],
                memory_train_s=qb["training_wall_s"],
                attention_first_step_s=a["first_step_wall_s"],
                memory_first_step_s=b["first_step_wall_s"],
                attention_setup_plus_total_s=qa["setup_plus_total_wall_s"],
                memory_setup_plus_total_s=qb["setup_plus_total_wall_s"],
                attention_estimated_train_flops=
                    qa["estimated_train_flops_observed_upper_bound"],
                memory_estimated_train_flops=
                    qb["estimated_train_flops_observed_upper_bound"],
                train_speedup_a_over_b=qa["training_wall_s"] / qb["training_wall_s"]
                if qb["training_wall_s"] else None,
                total_speedup_a_over_b=qa["setup_plus_total_wall_s"] /
                    qb["setup_plus_total_wall_s"]
                if qb["setup_plus_total_wall_s"] else None,
                estimated_train_flops_speedup_a_over_b=
                    qa["estimated_train_flops_observed_upper_bound"] /
                    qb["estimated_train_flops_observed_upper_bound"]
                if qb["estimated_train_flops_observed_upper_bound"] else None,
            )
        pairs.append(pair)
    complete = len(pairs) == len(out["protocol"]["seeds"])
    both = [p for p in pairs if p["attention_hit"] and p["memory_hit"]]
    load_rows = [r["memory_load"] for runs in out["runs"].values()
                 for arm in ARMS if arm in runs
                 for r in runs[arm]["checkpoints"]]
    high_load = any(
        row["load_avg_1_5_15"] is not None and row["cpu_count"] is not None
        and row["load_avg_1_5_15"][0] > 1.5 * row["cpu_count"]
        for row in load_rows)
    ans = dict(pairs=pairs, complete=complete, n_both_hit=len(both),
               n_attention_only=sum(p["attention_hit"] and not p["memory_hit"]
                                    for p in pairs),
               n_memory_only=sum(p["memory_hit"] and not p["attention_hit"]
                                 for p in pairs),
               n_neither=sum(not p["attention_hit"] and not p["memory_hit"]
                             for p in pairs),
               high_host_load_observed=high_load,
               host_load_rule="1-minute load average > 1.5 x logical CPUs",
               wall_clock_note="Synchronized train time includes the one-time "
                               "compiled first step; its cost is recorded per arm. "
                               "Host load is only a coarse contention indicator; "
                               "other GPU work may be invisible to it.")
    if both:
        t = [p["train_speedup_a_over_b"] for p in both]
        e = [p["estimated_train_flops_speedup_a_over_b"] for p in both]
        z = [p["total_speedup_a_over_b"] for p in both]
        ans.update(median_train_speedup_a_over_b=float(np.median(t)),
                   median_total_speedup_a_over_b=float(np.median(z)),
                   median_estimated_train_flops_speedup_a_over_b=
                       float(np.median(e)),
                   n_train_time_pair_wins=sum(v > 1 for v in t),
                   n_estimated_flops_pair_wins=sum(v > 1 for v in e))
    eligible = (out["protocol"]["primary"] and complete and
                len(both) == len(SEEDS) and not high_load and
                all(p["attention_final_bpc"] < out["bigram_bpc"]
                    for p in pairs))
    ans["primary_verdict_eligible"] = eligible
    ans["primary_efficiency_gate_passed"] = (
        eligible and ans["median_train_speedup_a_over_b"] >= MIN_SPEEDUP
        and ans["median_estimated_train_flops_speedup_a_over_b"] >= MIN_SPEEDUP
        and ans["n_train_time_pair_wins"] >= MIN_PAIR_WINS
        and ans["n_estimated_flops_pair_wins"] >= MIN_PAIR_WINS)
    if not eligible:
        ans["verdict_reason"] = (
            "diagnostic protocol, incomplete/censored pairs, failed attention "
            "bigram control, or high host load; inspect separate fields")
    elif ans["primary_efficiency_gate_passed"]:
        ans["verdict_reason"] = "predeclared tiny-corpus efficiency gate passed"
    else:
        ans["verdict_reason"] = "predeclared 2x tiny-corpus gate did not pass"
    return ans


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def main() -> None:
    protocol, out_path = parse_args()
    if (cg.CTX, cg.D, cg.N_LAYER, cg.N_HEAD, cg.MLP) != (
            CTX, D, LAYERS, HEADS, MLP):
        raise SystemExit("content_gate architecture env overrides conflict with "
                         "the fixed paired protocol")
    data, vocab = load_corpus(CORPUS)
    train, val = cg.make_splits(data, ctx=CTX)
    windows = cg.val_windows(val, ctx=CTX,
                             max_windows=protocol.max_val_windows)
    if not windows:
        raise SystemExit("the held-out split has no complete validation windows")
    bigram, unigram = floors(data)
    val_tokens = sum(int(y.size) for _, y in windows)
    if QUALITY_BPC >= bigram:
        raise SystemExit("the fixed target must be below the bigram floor")

    counts = {}
    flops = {}
    for arm in ARMS:
        mx.random.seed(0)
        model = cg.build(arm, vocab, d=D, ctx=CTX, n_layer=LAYERS)
        mx.eval(model.parameters())
        counts[arm] = cg.n_params(model)
        flops[arm] = cg.flops_per_token(arm, vocab, d=D, ctx=CTX,
                                        n_layer=LAYERS, mlp=MLP)
        if arm == "B_match" and any(block.ug is not None or block.p is None
                                    for block in model.blocks):
            raise SystemExit("B_match is not an input-only matched gate")
        del model
    spread = abs(counts["A_attn"] - counts["B_match"]) / max(counts.values())
    if spread > 0.02:
        raise SystemExit("attention and matched-memory params differ by >2%")
    scan_error = cg.verify_scan()
    if scan_error > 1e-5:
        raise SystemExit(f"parallel recurrence failed reference: {scan_error}")

    output = dict(
        status="running",
        source=dict(
            train_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            content_gate_sha256=hashlib.sha256(
                (ROOT / "experiments" / "content_gate.py").read_bytes()
            ).hexdigest(),
            llm_efficiency_sha256=hashlib.sha256(
                (ROOT / "experiments" / "llm_efficiency.py").read_bytes()
            ).hexdigest(),
            llm_tuned_sha256=hashlib.sha256(
                (ROOT / "experiments" / "llm_tuned.py").read_bytes()
            ).hexdigest(),
        ),
        protocol=dict(primary=protocol.primary, seeds=list(protocol.seeds),
                      steps=protocol.steps, eval_every=protocol.eval_every,
                      max_val_windows=protocol.max_val_windows,
                      compiled=protocol.compiled, quality_bpc=QUALITY_BPC,
                      ctx=CTX, d=D, layers=LAYERS, heads=HEADS, mlp=MLP,
                      batch=BATCH, lr=LR, warmup=WARMUP,
                      weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
                      min_speedup=MIN_SPEEDUP, min_pair_wins=MIN_PAIR_WINS,
                      arms=list(ARMS),
                      checkpoint_crossing="first full-validation bpc <= target; "
                      "observed upper bound with preceding failing checkpoint"),
        host=dict(platform=platform.platform(), machine=platform.machine(),
                  mlx_version=importlib.metadata.version("mlx"),
                  device=str(mx.default_device()),
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  memory_load_start=load_snapshot()),
        corpus=dict(path=str(Path(CORPUS).resolve()), chars=len(data),
                    vocab=vocab, train_chars=len(train), val_chars=len(val),
                    train_sha256=sha256_bytes(train),
                    val_sha256=sha256_bytes(val),
                    val_windows=len(windows), val_tokens=val_tokens),
        unigram_bpc=unigram, bigram_bpc=bigram,
        params=counts, parameter_spread=spread,
        estimated_train_flops_per_token=flops,
        scan_reference_max_abs_error=scan_error,
        limitations=[
            "One 1 MB character corpus and one tiny architecture; no claim for "
            "modern OSS model inference or training.",
            "A_attn and B_match are parameter-matched within 2%, but do not "
            "have equal per-token analytic FLOPs.",
            "The B_match extra trainable input projection is mathematically "
            "redundant; its runtime need not represent an optimized input-only gate.",
            "Full validation is checked only every 100 steps in the primary run; "
            "the first observed pass is an upper bound, not the exact crossing.",
            "FLOPs are analytic estimates from content_gate.py, not measured "
            "hardware counters or energy.",
            "Primary synchronized training seconds include the first-step "
            "JIT compilation cost; first_step_wall_s is reported per arm.",
        ],
        runs={}, summary=None,
    )
    atomic_json(out_path, output)
    print(f"quality target {QUALITY_BPC:.2f} bpc | bigram {bigram:.4f} | "
          f"validation {len(windows)} windows/{val_tokens:,} tokens", flush=True)
    print(f"params attention {counts['A_attn']:,}, memory {counts['B_match']:,} "
          f"({100*spread:.3f}% spread); paired seeds {protocol.seeds}", flush=True)
    print(f"output {out_path} | primary protocol: {protocol.primary}", flush=True)

    for i, seed in enumerate(protocol.seeds):
        schedule = paired_schedule(train, seed, protocol.steps)
        order = ARMS if i % 2 == 0 else tuple(reversed(ARMS))
        output["runs"][str(seed)] = {}
        for arm in order:
            print(f"\nseed {seed} {arm} begins; schedule {sha256_bytes(schedule)[:12]}",
                  flush=True)
            output["runs"][str(seed)][arm] = run_arm(
                arm, seed, train, windows, vocab, schedule, protocol)
            output["summary"] = summary(output)
            atomic_json(out_path, output)
    output["status"] = "complete"
    output["host"]["finished_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output["host"]["memory_load_end"] = load_snapshot()
    output["summary"] = summary(output)
    atomic_json(out_path, output)
    print("\n" + json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
