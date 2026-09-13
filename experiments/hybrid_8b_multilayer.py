"""Does the 1B transplant advantage reappear at 8B as MORE layers are replaced?

THE QUESTION, AND WHY IT IS THE ONE THAT MATTERS
------------------------------------------------
`hybrid_8b.py` replaced ONE attention sublayer of 32 at layer 16 and found a
plain NULL under the held-out protocol: transfer 8.7064 vs random 8.6875, with
3 of 3 control seeds beating the transplant. `verify_transplant.py` at 1B
(layer 8 of 16) found the opposite: transfer 22.7305 vs random 25.2227 with
0 of 6-8 control seeds beating it.

The obvious defence of the 8B null -- "the test is underpowered" -- is not
available in the naive form, because in NATS the 8B deletion gap is LARGER than
the 1B one. But there is a structural difference that the single-layer 8B run
cannot separate from a genuine scale limit:

    at 8B, deleting layer 16 of 32 leaves 31 attention layers intact, and the
    surviving layers can route around the damage.

That is a hypothesis about *how many* layers are replaced, and it is decidable
by a dose-response sweep. If the transplant advantage is real and merely masked
by redundancy, the transfer-minus-random gap should GROW with k and grow in step
with the deletion cost. If the gap stays flat or negative while the deletion
cost grows, the 1B effect does not scale, and that is a clean and valuable NULL.

Nothing is tuned post hoc: the decay is chosen on a SELECT window and the VAL
number at that decay is reported once. Every decay that was evaluated appears in
the JSON, selected or not.

WHAT EACH ARM MEANS
-------------------
  teacher        unmodified 8B-4bit. Reference point for the deletion cost.
  zero           the same k layers replaced by a carrier returning ZEROS, i.e.
                 attention deleted. Its ppl defines the size of the hole the
                 transplant is supposed to fill.
  transfer       carrier with W_v/W_o copied from EACH replaced layer's own
                 attention (dequantized -- this model is 4-bit affine).
  random_matched random W_v/W_o at the transplant's own per-layer weight std.
                 Same architecture, same output-rms normalisation, same decay.
                 The ONLY difference from `transfer` is whose weights they are.

Rms normalisation is applied per replaced layer: each carrier's output rms is
set to that layer's own attention output rms on real activations. The rule is
identical across arms, which is what makes this a comparison of weight CONTENT
rather than of output scale.

NO DOWNLOADS TO THE INTERNAL DISK
---------------------------------
The internal APFS volume is low on free space (~124 GB at the time of writing)
and the 8B weights live on an external ExFAT volume mounted at /Volumes/T9.
HF_HOME must point there and the script REFUSES to run otherwise, because the
default would silently pull a 4.5 GB checkpoint onto the internal disk.

Usage (run in screen -- it takes tens of minutes):

    screen -dmS b8ml /bin/zsh -c 'cd /Users/richie/Documents/github/human-brain && \
      HF_HOME=/Volumes/T9/hf8b ~/zbrain/venv/bin/python \
      experiments/hybrid_8b_multilayer.py > /tmp/zz_b8ml.log 2>&1'

Env knobs, all optional and all recorded in the output:
    B8ML_KS      default "1,2,4,8,16"     replaced-layer counts
    B8ML_DECAYS  default "0.5,0.7,0.8,0.9,0.99,1.0"
    B8ML_SEEDS   default "0,1,2"          random-control seeds
    B8ML_TEACHER default "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    B8ML_OUT     default experiments/results/hybrid_8b_multilayer.json

Results are written after EVERY completed evaluation, so a run that is killed
part-way still leaves a valid partial record.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import sys
import time

import numpy as np

# HF_HOME is set BEFORE mlx_lm is imported, and then CHECKED rather than
# trusted: a wrong HF_HOME silently downloads 4.5 GB to a nearly-full disk.
os.environ.setdefault("HF_HOME", "/Volumes/T9/hf8b")

import mlx.core as mx  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_hybrid import (  # noqa: E402
    attention_module, transplant, perplexity, n_params, GatedMemoryCarrier,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "data", "tinyshakespeare.txt")
TEACHER = os.environ.get("B8ML_TEACHER",
                         "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
KS = [int(x) for x in os.environ.get("B8ML_KS", "1,2,4,8,16").split(",")]
DECAYS = [float(x) for x in
          os.environ.get("B8ML_DECAYS", "0.5,0.9,0.99,1.0").split(",")]
SEEDS = [int(x) for x in os.environ.get("B8ML_SEEDS", "0,1,2").split(",")]
OUT = os.environ.get("B8ML_OUT",
                     os.path.join(HERE, "results", "hybrid_8b_multilayer.json"))
WINDOW = 3000
SELECT_FRAC, VAL_FRAC = 0.6, 0.9


def spread_layers(n_layers, k):
    """k layers spread evenly over n_layers, in increasing order.

    Even spacing is the point: contiguous layers would confound "how many
    replaced" with "how deep", since the same block would be hit at every k.
    The centres of the k equal-depth bins are used, which for k=1 reproduces the
    committed single-layer choice (layer 16 of 32) exactly -- so the k=1 row is
    a direct replication check on hybrid_8b.py rather than a new experiment.
    """
    if k < 1 or k > n_layers:
        raise ValueError(f"cannot replace {k} layers of {n_layers}")
    return [int((i + 0.5) * n_layers / k) for i in range(k)]


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


def sha1(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def probe_layers(model, layers, probe_ids):
    """Run the model once, capturing (input, attention output rms) per layer.

    The capture wrapper calls the ORIGINAL attention and records its input, so
    the probe is the activations the layer genuinely sees on real corpus
    tokens, not synthetic noise. Returns {layer_index: {h, rms_target, ...}}.
    """
    captured, handles = {}, []
    for li in layers:
        layer, attr, original = attention_module(model, li)
        handles.append((int(li), layer, attr, original))
        box = {}

        class Capture:
            def __init__(self, inner, box):
                self.inner = inner
                self.box = box

            def __call__(self, x, *a, **k):
                self.box["h"] = x
                return self.inner(x, *a, **k)

        setattr(layer, attr, Capture(original, box))
        captured[int(li)] = box

    model(mx.array(probe_ids[None, :].astype(np.int32)))
    mx.eval([b["h"] for b in captured.values()])

    out = {}
    for li, layer, attr, original in handles:
        h = captured[li]["h"]
        ref = original(h, None, None)
        mx.eval(ref)
        out[li] = dict(h=h, rms_target=rms(ref), rms_in=rms(h),
                       shape=[int(x) for x in h.shape])
    for _, layer, attr, original in handles:
        setattr(layer, attr, original)
    mx.eval(model.parameters())
    return out


def restore(model, handles):
    for _, layer, attr, original in handles:
        setattr(layer, attr, original)
    mx.eval(model.parameters())


def build_carrier(model, li, layer, attr, original, probes, *, decay, mode,
                  seed, legacy_control_seed=False):
    """Install one carrier at layer `li`; return (record).

    Every arm is normalised with the SAME rule -- out_gain set so the carrier's
    output rms on the layer's real probe activations equals that layer's
    attention output rms -- so the arms differ only in whose weights they hold.

    Determinism: `mx.random.seed(0)` is called before every transplant, exactly
    as `hybrid_8b.py` does. The gate's INPUT weights are randomly initialised
    (its bias is the swept decay), so without a fixed seed the arms would differ
    in gate init as well as in weight provenance -- a confound, and a violation
    of the reproducibility criterion.
    """
    probe = probes[li]
    mx.random.seed(0)
    # `decay=` AND `decays=` are both passed on purpose. `decays` sets the gate
    # bias; `decay` is what tells `transplant` this is an EXPLICIT override, so
    # it does not run the recency fit and then record `fitted_decay` as if the
    # gate had been fitted. Without `decay=`, the record claims a measured fit
    # while the gate actually holds the swept value -- the exact "fitted vs
    # defaulted" confusion this project has already been burned by once, and it
    # additionally costs a 32-head 1024x1024 attention profile per layer per arm.
    carrier, rec = transplant(model, li, "transfer", original=original,
                              probe_h=probe["h"], banks=1, decays=(decay,),
                              decay=decay)
    d = int(rec["d"])
    assert rec["decay_source"] == "explicit override", rec["decay_source"]
    assert abs(float(rec["fitted_decay"]) - float(decay)) < 1e-12

    if mode == "random_matched":
        if legacy_control_seed:
            # EXACTLY the committed single-layer draw: one rng(seed) for the
            # one replaced layer, no per-layer offset. Used only for the k=1
            # replication check, so that row can be compared number-for-number
            # with hybrid_8b_b.json.
            rng = np.random.default_rng(seed)
        else:
            # Seed is (control seed, layer) so every layer of the control gets
            # an INDEPENDENT draw. Reusing one seed across layers would make all
            # k layers share a draw, which is a different, weaker control.
            rng = np.random.default_rng(seed * 100003 + li)
        s_native = float(np.array(carrier.mem.v.weight).std())
        carrier.mem.v.weight = mx.array(
            rng.normal(0, s_native, (d, d)).astype(np.float32))
        carrier.mem.o.weight = mx.array(
            rng.normal(0, s_native, (d, d)).astype(np.float32))
        rec["control_seed"] = int(seed)
        rec["legacy_control_seed"] = bool(legacy_control_seed)
        rec["control_std"] = float(s_native)

    mx.eval(model.parameters())
    carrier.out_gain = 1.0
    o1 = carrier(probe["h"], None, None)
    mx.eval(o1)
    pre = rms(o1)
    carrier.out_gain = float(probe["rms_target"] / pre)
    o2 = carrier(probe["h"], None, None)
    mx.eval(o2)
    post = rms(o2)
    rec.update(mode=mode, decay=float(decay), rms_pre_norm=float(pre),
               rms_post_norm=float(post),
               rms_target=float(probe["rms_target"]),
               out_gain=float(carrier.out_gain),
               normalisation_reldiff=float(
                   abs(post / probe["rms_target"] - 1.0)))
    return rec


def evaluate_k(model, layers, probes, *, decay, mode, seed, select, val,
               legacy_control_seed=False):
    """Score one arm on SELECT and VAL, then restore every replaced layer.

    All k carriers are installed AT ONCE and the whole model is evaluated, so
    the arms interact through the network exactly as they would in the model
    under test. Reporting k independent single-layer deltas instead would be
    wrong, because those deltas are not additive in a deep residual network.

    Handles are recorded BEFORE each transplant runs, and the restore is in a
    `finally`, so an exception part-way through cannot leave the model with some
    layers replaced while the sweep continues -- which on a multi-hour run would
    silently corrupt every later number.
    """
    handles = [(int(li),) + attention_module(model, li) for li in layers]
    built = []
    try:
        if mode == "zero":
            for (li, layer, attr, original) in handles:
                setattr(layer, attr, GatedMemoryCarrier(
                    int(model.args.hidden_size), mode="zero"))
            mx.eval(model.parameters())
        else:
            for (li, layer, attr, original) in handles:
                built.append(build_carrier(
                    model, li, layer, attr, original, probes, decay=decay,
                    mode=mode, seed=seed,
                    legacy_control_seed=legacy_control_seed))
        t0 = time.perf_counter()
        p_sel = perplexity(model, select)["ppl"]
        p_val = perplexity(model, val)["ppl"]
        wall = time.perf_counter() - t0
    finally:
        restore(model, [(li, layer, attr, original)
                        for (li, layer, attr, original) in handles])
    return dict(mode=mode, decay=None if decay is None else float(decay),
                control_seed=None if seed is None else int(seed),
                select_ppl=float(p_sel), ppl=float(p_val), wall_s=float(wall),
                n_layers_replaced=len(layers), per_layer=built)


def recovery(base_ppl, zero_ppl, arm_ppl):
    """Fraction of the deletion gap an arm recovers; None when undefined."""
    if zero_ppl is None or arm_ppl is None or zero_ppl <= base_ppl:
        return None
    return float((zero_ppl - arm_ppl) / (zero_ppl - base_ppl))


def main():
    from mlx_lm import load

    hf_home = os.environ.get("HF_HOME", "")
    if "/Volumes/T9" not in hf_home:
        raise RuntimeError(
            f"HF_HOME={hf_home!r} does not point at the external volume. The "
            f"internal disk is nearly full and the 8B checkpoint must not be "
            f"downloaded to it. Set HF_HOME=/Volumes/T9/hf8b.")

    t_start = time.time()
    print(f"HF_HOME={hf_home}", flush=True)
    print(f"loading {TEACHER} ...", flush=True)
    t_load = time.perf_counter()
    model, tok = load(TEACHER)
    load_s = time.perf_counter() - t_load
    print(f"  loaded in {load_s:.1f}s", flush=True)

    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)
    s0, v0 = int(SELECT_FRAC * n), int(VAL_FRAC * n)
    select, val = ids[s0:s0 + WINDOW], ids[v0:v0 + WINDOW]
    assert not np.shares_memory(select, val), "SELECT and VAL overlap"
    probe_ids = ids[:1024]

    n_layers = len(model.model.layers)
    d = int(model.args.hidden_size)
    plan = {int(k): spread_layers(n_layers, int(k)) for k in KS}
    print(f"  {n_layers} layers, d={d}, corpus {n:,} tokens", flush=True)
    for k, ls in plan.items():
        print(f"    k={k:<3} layers {ls}", flush=True)

    out = dict(
        meta=dict(
            teacher=TEACHER,
            n_params=n_params(model),
            n_layers=n_layers, d=d,
            corpus=CORPUS, corpus_tokens=int(n),
            ks=[int(k) for k in KS],
            decays=DECAYS, seeds=SEEDS,
            layers_by_k={str(k): v for k, v in plan.items()},
            layer_selection=(
                "centres of k equal-depth bins; k=1 reproduces layer 16 of 32, "
                "the committed single-layer choice in hybrid_8b.py"),
            select_window=[int(s0), int(s0 + WINDOW)],
            val_window=[int(v0), int(v0 + WINDOW)],
            window_tokens=WINDOW,
            protocol=(
                "arms rms-matched per layer to the attention they replace; "
                "decay chosen on SELECT and reported once on VAL; every "
                "evaluated decay reported per arm; control-seed sweep at the "
                "SELECT-chosen decay"),
            select_sha1=sha1(select), val_sha1=sha1(val),
            driver="experiments/hybrid_8b_multilayer.py",
            host=platform.platform(), machine=platform.machine(),
            python=sys.version.split()[0],
            mlx_device=str(mx.default_device()),
            environment=(
                "Quantized 4-bit 8B (mlx-community/Meta-Llama-3.1-8B-Instruct"
                "-4bit). The internal APFS volume is LOW ON FREE SPACE and the "
                "model is cached on an external ExFAT volume mounted at "
                "/Volumes/T9 (HF_HOME=/Volumes/T9/hf8b); nothing is downloaded "
                "to the internal disk. Weights are read through "
                "llm_hybrid.dense_weight(), which dequantizes packed uint32 "
                "4-bit weights -- reading .weight directly is a "
                "silent-corruption bug on this model."),
            load_s=float(load_s),
            started_epoch=float(t_start),
            elapsed_s=None,
        ),
        teacher_ppl=None, ks={}, verdict=None)

    def dump():
        out["meta"]["elapsed_s"] = time.time() - t_start
        # memory is reported at every dump, so a partial record carries the
        # memory actually used rather than a figure from process start
        out["meta"]["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        out["meta"]["max_rss_gb"] = round(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, OUT)

    t0 = time.perf_counter()
    base_val = perplexity(model, val)["ppl"]
    base_sel = perplexity(model, select)["ppl"]
    out["teacher_ppl"] = float(base_val)
    out["teacher_select_ppl"] = float(base_sel)
    out["meta"]["teacher_eval_s"] = time.perf_counter() - t0
    dump()
    print(f"  teacher val ppl {base_val:.4f} select {base_sel:.4f}", flush=True)

    for k in KS:
        layers = plan[int(k)]
        key = str(int(k))
        print(f"\n=== k={k}  layers {layers} ===", flush=True)
        probes = probe_layers(model, layers, probe_ids)
        rec = dict(
            k=int(k), layers=[int(x) for x in layers],
            probe={str(li): dict(shape=p["shape"], rms_in=float(p["rms_in"]),
                                 rms_attn_out=float(p["rms_target"]))
                   for li, p in probes.items()},
            zero_ppl=None, paired=[], per_arm={},
            control_seed_sweep=None,
            control_seed_convention=(
                "committed hybrid_8b.py convention (np.random.default_rng(seed), "
                "no per-layer offset) -- k=1 is a direct replication of the "
                "single-layer run" if int(k) == 1 else
                "one independent draw per (seed, layer): "
                "np.random.default_rng(seed*100003 + layer), so the k layers do "
                "not share a single random draw"))

        t0 = time.perf_counter()
        z = evaluate_k(model, layers, probes, decay=None, mode="zero", seed=None,
                       select=select, val=val)
        rec["zero_ppl"] = z["ppl"]
        rec["zero_select_ppl"] = z["select_ppl"]
        rec["zero_wall_s"] = z["wall_s"]
        rec["deletion_cost_ppl"] = float(z["ppl"] - base_val)
        rec["deletion_cost_nats"] = float(math.log(z["ppl"]) - math.log(base_val))
        rec["zero_recovery_frac"] = recovery(base_val, z["ppl"], z["ppl"])
        out["ks"][key] = rec
        dump()
        print(f"  zero (attention deleted) val {z['ppl']:.4f}  "
              f"(cost {rec['deletion_cost_ppl']:+.4f} ppl, "
              f"{rec['deletion_cost_nats']:.4f} nats) [{z['wall_s']:.1f}s]",
              flush=True)

        # ------------------------------------------------ paired decay sweep
        for decay in DECAYS:
            row = dict(decay=float(decay))
            for arm, seed, mode in (("transfer", None, "transfer"),
                                    ("random_matched", SEEDS[0], "random_matched")):
                r = evaluate_k(model, layers, probes, decay=decay, mode=mode,
                               seed=seed, select=select, val=val,
                               legacy_control_seed=(int(k) == 1))
                r["recovery_frac"] = recovery(base_val, z["ppl"], r["ppl"])
                row[arm] = r
                print(f"    d={decay:<6} {arm:<15} sel {r['select_ppl']:>8.4f} "
                      f"val {r['ppl']:>8.4f} rec "
                      f"{'n/a' if r['recovery_frac'] is None else format(r['recovery_frac'], '+.3f')}"
                      f" [{r['wall_s']:.1f}s]", flush=True)
            row["ppl_gap_transfer_minus_control"] = float(
                row["transfer"]["ppl"] - row["random_matched"]["ppl"])
            row["gap_at_same_decay_transfer_minus_control_recovery"] = (
                None if row["transfer"]["recovery_frac"] is None
                else float(row["transfer"]["recovery_frac"]
                           - row["random_matched"]["recovery_frac"]))
            rec["paired"].append(row)
            rec["zero_ppl"] = z["ppl"]
            out["ks"][key] = rec
            dump()

        # ------------------------------------------ k=1 replication check
        # At k=1 the design is IDENTICAL to hybrid_8b.py (layer 16, same
        # protocol, same decay grid, same control seed convention). Reproducing
        # committed numbers here is what licenses the larger-k rows to be read
        # as an extension of that experiment rather than as a new one with a
        # different harness. A mismatch is reported, not smoothed over.
        if int(k) == 1 and int(n_layers) == 32:
            comm_path = os.path.join(HERE, "results", "hybrid_8b_b.json")
            repl = dict(committed_file=(comm_path if os.path.exists(comm_path)
                                        else None), reference={}, rows=[])
            try:
                comm = json.load(open(comm_path))
                cref = comm["layers"]["16"]
                repl["reference"] = dict(
                    teacher_ppl=comm["teacher_ppl"],
                    zero_ppl=cref["zero_ppl"],
                    val_sha1=comm["meta"].get("val_sha1"))
                my_val_sha = sha1(val)
                repl["val_sha1_matches"] = bool(
                    my_val_sha == comm["meta"].get("val_sha1"))
                repl["teacher_ppl_matches"] = bool(
                    abs(base_val - comm["teacher_ppl"]) < 1e-9)
                repl["zero_ppl_matches"] = bool(
                    abs(z["ppl"] - cref["zero_ppl"]) < 1e-9)
                for cdecay in DECAYS:
                    cmine = [r for r in rec["paired"]
                             if abs(r["decay"] - cdecay) < 1e-12]
                    crows = [r for r in cref["paired"]
                             if abs(r["decay"] - cdecay) < 1e-12]
                    if not cmine or not crows:
                        continue
                    c0, r0 = crows[0], cmine[0]
                    row = dict(decay=float(cdecay))
                    for arm in ("transfer", "random_matched"):
                        row[arm + "_mine"] = float(r0[arm]["ppl"])
                        row[arm + "_committed"] = float(c0[arm]["ppl"])
                        row[arm + "_absdiff"] = float(
                            abs(r0[arm]["ppl"] - c0[arm]["ppl"]))
                    repl["rows"].append(row)
                diffs = [r[a + "_absdiff"] for r in repl["rows"]
                         for a in ("transfer", "random_matched")]
                repl["max_abs_ppl_diff"] = float(max(diffs)) if diffs else None
                repl["agrees_with_committed"] = bool(diffs and max(diffs) < 1e-6)
            except Exception as e:
                repl["error"] = f"{type(e).__name__}: {e}"
                repl["agrees_with_committed"] = False
            rec["replication_check_k1"] = repl
            out["ks"][key] = rec
            dump()
            print(f"  k=1 replication vs hybrid_8b_b.json: "
                  f"max |ppl diff| = {repl.get('max_abs_ppl_diff')} "
                  f"(agrees={repl['agrees_with_committed']})", flush=True)

        # per-arm SELECT-chosen decay, with the VAL number reported once
        for arm in ("transfer", "random_matched"):
            rows = [dict(decay=r["decay"], ppl=r[arm]["ppl"],
                         select_ppl=r[arm]["select_ppl"],
                         recovery_frac=r[arm]["recovery_frac"],
                         wall_s=r[arm]["wall_s"]) for r in rec["paired"]]
            best_sel = min(rows, key=lambda x: x["select_ppl"])
            best_val = min(rows, key=lambda x: x["ppl"])
            rec["per_arm"][arm] = dict(
                chosen_on_select=best_sel, best_val=best_val,
                val_curve=[dict(decay=x["decay"], ppl=x["ppl"],
                                select_ppl=x["select_ppl"])
                           for x in rows],
                select_is_best_val=bool(best_sel["decay"] == best_val["decay"]))
            print(f"  -> {arm} SELECT picks d={best_sel['decay']} "
                  f"(sel {best_sel['select_ppl']:.4f}) -> VAL {best_sel['ppl']:.4f}"
                  f" | best VAL d={best_val['decay']} ({best_val['ppl']:.4f})"
                  f" | select_is_best_val="
                  f"{rec['per_arm'][arm]['select_is_best_val']}", flush=True)

        tb = rec["per_arm"]["transfer"]["chosen_on_select"]
        cb = rec["per_arm"]["random_matched"]["chosen_on_select"]
        rec["headline_selected"] = dict(
            transfer_decay=tb["decay"], transfer_val_ppl=tb["ppl"],
            transfer_select_ppl=tb["select_ppl"],
            transfer_recovery_frac=tb["recovery_frac"],
            control_decay=cb["decay"], control_val_ppl=cb["ppl"],
            control_select_ppl=cb["select_ppl"],
            control_recovery_frac=cb["recovery_frac"],
            margin_ppl=float(cb["ppl"] - tb["ppl"]),
            margin_recovery=(
                None if (tb["recovery_frac"] is None or cb["recovery_frac"] is None)
                else float(tb["recovery_frac"] - cb["recovery_frac"])),
            transfer_wins=bool(tb["ppl"] < cb["ppl"]))
        out["ks"][key] = rec
        dump()
        h = rec["headline_selected"]
        print(f"  SELECT-chosen: transfer {h['transfer_val_ppl']:.4f} @ "
              f"{h['transfer_decay']} vs control {h['control_val_ppl']:.4f} @ "
              f"{h['control_decay']} -> margin "
              f"{h['margin_ppl']:+.4f} ppl "
              f"({'TRANSFER WINS' if h['transfer_wins'] else 'CONTROL WINS'})",
              flush=True)

        # -------------------------------------------------- control-seed sweep
        # Same decay for BOTH arms: the SELECT-chosen decay of the transfer arm
        # is used, because that is the decay the protocol would ship. If the
        # control's own SELECT choice differs, that is reported separately and
        # the sweep is repeated there too when the two disagree.
        sweep_decays = []
        for cand in (tb["decay"], cb["decay"]):
            if cand not in sweep_decays:
                sweep_decays.append(cand)
        if len(sweep_decays) > 1:
            print(f"  NOTE: the two arms SELECT different decays "
                  f"(transfer {tb['decay']}, control {cb['decay']}); the seed "
                  f"sweep is run at BOTH and the transfer-consistent one is "
                  f"reported as primary", flush=True)
        sweeps = []
        for sweep_decay in sweep_decays:
            tr = next(r for r in rec["paired"] if r["decay"] == sweep_decay)["transfer"]
            rows = []
            for s in SEEDS:
                r = evaluate_k(model, layers, probes, decay=sweep_decay,
                               mode="random_matched", seed=s, select=select,
                               val=val, legacy_control_seed=(int(k) == 1))
                # evaluate_k returns raw scores only; the recovery fraction is
                # derived here because it needs the zero arm. Reading it off `r`
                # raised KeyError and killed the sweep after the first k.
                rows.append(dict(seed=int(s), ppl=float(r["ppl"]),
                                 select_ppl=float(r["select_ppl"]),
                                 recovery_frac=recovery(base_val, z["ppl"],
                                                        r["ppl"]),
                                 wall_s=r["wall_s"]))
                print(f"    seed {s} @ d={sweep_decay}: val {r['ppl']:.4f}",
                      flush=True)
            ppls = [r["ppl"] for r in rows]
            sweeps.append(dict(
                decay=float(sweep_decay),
                source=("transfer SELECT choice" if sweep_decay == tb["decay"]
                        else "control SELECT choice"),
                transfer_ppl=float(tr["ppl"]),
                transfer_select_ppl=float(tr["select_ppl"]),
                transfer_recovery_frac=tr["recovery_frac"],
                seeds=rows,
                mean_ppl=float(np.mean(ppls)),
                std_ppl=(float(np.std(ppls, ddof=1)) if len(ppls) > 1 else None),
                min_ppl=float(min(ppls)), max_ppl=float(max(ppls)),
                n_seeds_beating_transfer=int(sum(1 for p in ppls
                                                 if p < tr["ppl"])),
                n_seeds=len(rows),
                best_seed_recovery_frac=min(
                    (r["recovery_frac"] for r in rows
                     if r["recovery_frac"] is not None), default=None),
                min_paired_margin_ppl=float(min(p - tr["ppl"] for p in ppls))))
            # `control_seed_sweep` is ALWAYS the primary dict (the sweep at the
            # transfer arm's SELECT-chosen decay, i.e. the decay the protocol
            # would ship). Any additional sweep required because the control
            # arm's own SELECT choice differed goes in `control_seed_sweeps_extra`
            # -- a list-valued field here would force every reader to know which
            # shape it is looking at.
            rec["control_seed_sweep"] = sweeps[0]
            rec["control_seed_sweeps_extra"] = sweeps[1:]
            out["ks"][key] = rec
            dump()
        primary = sweeps[0]
        print(f"  control seeds @ d={primary['decay']}: "
              f"{[round(r['ppl'], 4) for r in primary['seeds']]} vs transfer "
              f"{primary['transfer_ppl']:.4f} -> "
              f"{primary['n_seeds_beating_transfer']}/{primary['n_seeds']} beat "
              f"transfer", flush=True)
        del probes  # release the per-layer activation caches before the next k

    # --------------------------------------------------------------- verdict
    rows = []
    for k in KS:
        rec = out["ks"].get(str(int(k)))
        if not rec or rec.get("headline_selected") is None:
            continue
        h = rec["headline_selected"]
        sweep = rec["control_seed_sweep"]
        gap_best = max((r["ppl_gap_transfer_minus_control"] for r in rec["paired"]),
                       default=None)
        n_grid_decays_transfer_wins = int(sum(
            1 for r in rec["paired"] if r["ppl_gap_transfer_minus_control"] < 0))
        rows.append(dict(
            k=int(k), layers=rec["layers"],
            teacher_ppl=float(out["teacher_ppl"]),
            zero_ppl=rec["zero_ppl"],
            deletion_cost_ppl=rec["deletion_cost_ppl"],
            deletion_cost_nats=rec["deletion_cost_nats"],
            transfer_val_ppl=h["transfer_val_ppl"],
            transfer_decay=h["transfer_decay"],
            control_val_ppl=h["control_val_ppl"],
            control_decay=h["control_decay"],
            margin_ppl=h["margin_ppl"],
            transfer_recovery_frac=h["transfer_recovery_frac"],
            control_recovery_frac=h["control_recovery_frac"],
            transfer_wins=h["transfer_wins"],
            n_grid_decays=len(rec["paired"]),
            n_grid_decays_transfer_wins=n_grid_decays_transfer_wins,
            best_grid_margin_ppl=gap_best,
            n_seeds_beating_transfer=(None if sweep is None
                                      else sweep["n_seeds_beating_transfer"]),
            n_seeds=(None if sweep is None else sweep["n_seeds"]),
            seed_sweep_decay=(None if sweep is None else sweep["decay"])))

    decreasing = None
    if len(rows) >= 2:
        # "gap grows with k" read as: margin_ppl (control minus transfer) is
        # increasing in k. Margin is signed so positive means transfer ahead.
        margins = [r["margin_ppl"] for r in rows]
        decreasing = bool(all(b > a for a, b in zip(margins, margins[1:])))

    corr = None
    if len(rows) >= 3:
        cost = np.array([r["deletion_cost_nats"] for r in rows], dtype=np.float64)
        marg = np.array([r["margin_ppl"] for r in rows], dtype=np.float64)
        if cost.std() > 0 and marg.std() > 0:
            corr = float(np.corrcoef(cost, marg)[0, 1])

    n_transfer_wins = int(sum(1 for r in rows if r["transfer_wins"]))
    n_seeded = int(sum(1 for r in rows if r["n_seeds_beating_transfer"] == 0))

    if not rows:
        verdict = dict(status="INCOMPLETE",
                       summary="no k completed; see per-k records")
    elif n_transfer_wins == len(rows) and n_seeded == len(rows):
        verdict = dict(
            status="TRANSFER ADVANTAGE REAPPEARS AT EVERY k",
            summary=("transfer beats the matched-random control at the "
                     "SELECT-chosen decay for every k, and no control seed "
                     "beats it at any k; the single-layer 8B null is "
                     "explained by the single-layer design"))
    elif n_transfer_wins == 0:
        verdict = dict(
            status="NULL HOLDS: NO TRANSPLANT ADVANTAGE AT ANY k",
            summary=("the matched-random control beats transfer at the "
                     "SELECT-chosen decay for every k tested, including the "
                     "largest; the 1B transplant effect does not reproduce at "
                     "8B even when many layers are replaced, so it is not a "
                     "power problem of the single-layer test"))
    else:
        verdict = dict(
            status="MIXED",
            summary=(f"transfer wins at {n_transfer_wins} of {len(rows)} k "
                     f"values; see per-k margins rather than a single claim"))

    verdict.update(
        per_k=rows,
        n_k=int(len(rows)),
        n_k_transfer_wins=n_transfer_wins,
        n_k_with_zero_seeds_beating_transfer=n_seeded,
        margin_monotone_increasing_in_k=decreasing,
        corr_deletion_cost_nats_vs_margin_ppl=corr,
        hypothesis_under_test=(
            "if the 8B null is caused by redundant surviving attention layers "
            "routing around the damage of ONE replaced layer, then replacing "
            "more layers should reduce that redundancy and the transplant-minus"
            "-random margin should GROW with k, tracking the deletion cost"),
        interpretation_rule=(
            "gap grows with k -> the single-layer 8B test was underpowered and "
            "the 1B effect is present at 8B once the intervention is large "
            "enough. gap flat or negative while the deletion cost grows -> the "
            "1B effect does not scale; the NULL is the finding"),
        caveat=(
            "a growing deletion cost is not by itself evidence for the "
            "transplant: the zero arm moving further from the teacher proves "
            "the intervention is larger, and only the transfer-minus-control "
            "margin speaks to whose weights are better"))
    out["verdict"] = verdict
    dump()

    print("\n=== VERDICT ===", flush=True)
    print(f"  teacher {out['teacher_ppl']:.4f}", flush=True)
    print(f"  {'k':>3} {'zero':>9} {'dCostNat':>9} {'transfer':>9} {'random':>9}"
          f" {'margin':>8} {'recT':>7} {'recC':>7} {'seedsBeat':>9}", flush=True)
    for r in rows:
        rec_t = r["transfer_recovery_frac"]
        rec_c = r["control_recovery_frac"]
        nbeat = (f"{r['n_seeds_beating_transfer']}/{r['n_seeds']}"
                 if r["n_seeds_beating_transfer"] is not None else "n/a")
        print(f"  {r['k']:>3} {r['zero_ppl']:>9.4f} "
              f"{r['deletion_cost_nats']:>9.4f} "
              f"{r['transfer_val_ppl']:>9.4f} {r['control_val_ppl']:>9.4f} "
              f"{r['margin_ppl']:>+8.4f} "
              f"{(rec_t if rec_t is not None else float('nan')):>7.3f} "
              f"{(rec_c if rec_c is not None else float('nan')):>7.3f} "
              f"{nbeat:>9}", flush=True)
    print(f"\n  {verdict['status']}", flush=True)
    print(f"  {verdict['summary']}", flush=True)
    print(f"\nwrote {OUT} in {out['meta']['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
