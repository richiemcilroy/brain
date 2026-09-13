"""ADVERSARIAL VERIFICATION of the corrected transplant headline.

WHAT IS UNDER TEST
------------------
`experiments/hybrid_decay.py` + `docs/HYBRID_DECAY.md` claim, for layer 8 of
`unsloth/Llama-3.2-1B` at the decay chosen per arm on a held-out SELECT split:

    transfer (attention's own W_V, W_O transplanted)  val ppl 22.8744  33.0% recovery
    random_matched (same shapes, random weights)      val ppl 25.2481   0.0% recovery
    zero (attention deleted)                          val ppl 24.1044
    teacher                                            val ppl 20.3756

and read the 2.37-ppl margin as "the transplanted weights encode real temporal
structure that random weights do not". This script tries to falsify that.

This script MUTATES NOTHING. It re-implements the committed protocol using the
same `llm_hybrid` primitives (`attention_module`, `transplant`, `perplexity`,
`GatedMemoryCarrier`) so the comparison is apples-to-apples, and writes only its
own result file. `llm_hybrid.py` and `hybrid_decay.py` are not imported-and-run
and not edited.

THE FIVE ATTACKS
----------------
A. PAIRED SAME-DECAY. The headline compares each arm at its own best decay
   (transfer 0.7/0.8, control 1.0). That is a best-of-8-grid selection per arm,
   so it conflates "whose weights are better" with "whose grid search got a
   better draw". Here both arms are evaluated at every grid point and the
   per-decay margin is reported, so the claim can be judged as a curve rather
   than as two tuned points.

B. SELECT/VAL DISJOINTNESS. The committed script asserts only
   `int(0.5n) <= int(0.6n) < int(0.9n)`, an assertion about INTERVALS that cannot
   fail for a positive corpus. Whether the 3000-token SELECT and VAL windows are
   actually disjoint is asserted here on token indices, and re-checked against
   the tokenizer the run actually uses.

C. RMS-NORMALISATION CONFOUND. Every arm is scaled to the replaced attention's
   output rms, so the comparison is supposed to be scale-free. This measures the
   post-normalisation rms of BOTH arms at every decay: if they are not equal, the
   arms are still being compared partly by scale, and the residual is quantified.

D. RANDOM-CONTROL SEED SENSITIVITY. The committed control is a single draw with
   `np.random.default_rng(0)`. Its std is copied from the transplant so the seed
   is the only thing standing between "the weights carry temporal structure" and
   "this particular random matrix happened to be bad". The control is re-drawn
   over >=5 seeds and each seed gets its own best-of-grid score, i.e. exactly the
   tuning freedom the treatment was given.

E. GENERALISATION TO LAYERS 4, 12, 15. The headline is a single layer. The
   corrected doc explicitly says the old layer sweep "was measured at the wrong
   decay throughout and needs re-running before it means anything". It is
   re-run here under the corrected protocol.

F. SAME-WEIGHTS SHUFFLE (extra, beyond the five requested). An iid gaussian
   control at matched std also has a matched spectrum, so it cannot separate
   "the learned mapping matters" from "the learned scale/spectrum matters". A
   permutation of the input and output spaces of W_o@W_v preserves the weight
   multiset, the singular values and the norm exactly, and destroys only which
   input feature maps to which output. This is the strongest available control
   for the "the weights carry temporal structure" claim, so it is run here.

Because the two arms are evaluated inside one loop per decay, the paired table is
free of cross-run drift: a slow-down or thermal change hits both arms of a pair
equally.

Usage (see docs/VERIFY_TRANSPLANT.md for the full command):

    HF_HOME=~/zbrain/hf ~/zbrain/venv/bin/python experiments/verify_transplant.py

Env knobs, all optional and all recorded in the output:
    VT_LAYERS  default "8,4,12,15"
    VT_DECAYS  default "0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    VT_SEEDS   default "0,1,2,3,4,5,6,7"   (random-control seeds, attack D)
    VT_GATES   default "0,1,2"             (gate-init seeds, see attack C note)
    VT_ARM_SEED default 0                  (seed for the shuffled control, attack F)
    VT_TEACHER default "unsloth/Llama-3.2-1B"
    VT_OUT     default experiments/results/verify_transplant.json
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import mlx.core as mx

os.environ.setdefault("HF_HOME", os.path.expanduser("~/zbrain/hf"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llm_hybrid import (  # noqa: E402
    attention_module, transplant, perplexity, n_params, GatedMemoryCarrier,
)

ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "data", "tinyshakespeare.txt")
TEACHER = os.environ.get("VT_TEACHER", "unsloth/Llama-3.2-1B")
LAYERS = [int(x) for x in os.environ.get("VT_LAYERS", "8,4,12,15").split(",")]
DECAYS = [float(x) for x in os.environ.get(
    "VT_DECAYS", "0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0").split(",")]
SEEDS = [int(x) for x in os.environ.get("VT_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
GATES = [int(x) for x in os.environ.get("VT_GATES", "0,1,2").split(",")]
OUT = os.environ.get("VT_OUT", os.path.join(HERE, "results", "verify_transplant.json"))
CONTROL_SEED = int(os.environ.get("VT_CONTROL_SEED", "0"))
# The gate INPUT weights are random-initialised inside `transplant` and are not
# controlled by the committed protocol. Pinning the global MLX RNG here makes
# the main table reproducible; attack C measures how much that init can move the
# result by varying it deliberately.
GATE_SEED_DEFAULT = int(os.environ.get("VT_GATE_SEED_DEFAULT", "0"))

# Fraction of the corpus used for each window, identical to hybrid_decay.py.
SELECT_FRAC, VAL_FRAC, WINDOW = 0.6, 0.9, 3000


def rms(a):
    return float(mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)))


def recovery(zero, base, ppl):
    """Recovered fraction of the attention-deletion gap. Same formula as the
    committed verdict, guarded for zero<=base."""
    if zero <= base:
        return None
    return float((zero - ppl) / (zero - base))


def control_weights(carrier, seed, d):
    """Exactly the committed random_matched construction.

    The committed code copies `s_native = std(carrier.mem.v.weight)` straight
    after `transplant`, i.e. the std of the TRANSPLANTED pretrained weights --
    it depends only on which layer's W_V was copied, never on the decay or on
    the carrier. It is layer-dependent, so it is recomputed here per layer.
    """
    rng = np.random.default_rng(seed)
    s_native = float(np.array(carrier.mem.v.weight).std())
    carrier.mem.v.weight = mx.array(
        rng.normal(0, s_native, (d, d)).astype(np.float32))
    carrier.mem.o.weight = mx.array(
        rng.normal(0, s_native, (d, d)).astype(np.float32))
    return s_native


def shuffle_weights(carrier, seed, d):
    """ATTACK F: same weights, scrambled directions.

    The composite of the transplanted arms is `W_o @ W_v` (the gate factorises
    the temporal mixing out). Permuting the input and output spaces of that
    composite via P_rho (W_o W_v) P_sigma^T leaves the MULTISET of weights and
    the singular values identical, but destroys which input feature maps to
    which output.

    Any hidden-channel permutation cancels in (P_tau W_o)(P_tau W_v), so only
    the outer two are applied. This is a strictly stronger control than iid
    gaussian at the same std: it asks whether the LEARNED DIRECTIONS matter
    rather than whether the weight SCALE and spectrum do. If the transplant
    only ties this arm, the effect is not about the learned mapping.
    """
    rng = np.random.default_rng(seed)
    sigma = rng.permutation(d)
    rho = rng.permutation(d)
    v = np.array(carrier.mem.v.weight)[rho][:, sigma]
    o = np.array(carrier.mem.o.weight)[:, rho]
    carrier.mem.v.weight = mx.array(v.astype(np.float32))
    carrier.mem.o.weight = mx.array(o.astype(np.float32))
    return dict(perm_seed=int(seed), sigma_sha=hashlib.sha1(sigma.tobytes()).hexdigest()[:12],
                rho_sha=hashlib.sha1(rho.tobytes()).hexdigest()[:12])


def evaluate(model, li, layer, attr, original, probe_h, *, decay, seed,
             gate_seed, rms_target, select, val, mode="random_matched"):
    """Install an arm, normalise it to `rms_target`, score SELECT and VAL.

    `li` is the layer INDEX (transplant/attention_module address layers by
    index); `layer`/`attr` are used only to restore the original module.
    """
    if gate_seed is not None:
        mx.random.seed(int(gate_seed))
    carrier, rec = transplant(model, li, "transfer", original=original,
                              probe_h=probe_h, banks=1, decays=(decay,))
    d = int(rec["d"])
    if seed is not None and mode == "random_matched":
        rec["control_std"] = control_weights(carrier, seed, d)
        rec["control_seed"] = int(seed)
    elif seed is not None and mode == "shuffled":
        rec["shuffle"] = shuffle_weights(carrier, seed, d)
    mx.eval(model.parameters())
    carrier.out_gain = 1.0
    o1 = carrier(probe_h, None, None)
    mx.eval(o1)
    pre_rms = rms(o1)
    carrier.out_gain = float(rms_target / pre_rms)
    # confirm the normalisation actually landed, rather than trusting the algebra
    o2 = carrier(probe_h, None, None)
    mx.eval(o2)
    post_rms = rms(o2)
    p_sel = perplexity(model, select)["ppl"]
    p_val = perplexity(model, val)["ppl"]
    setattr(layer, attr, original)
    return dict(decay=float(decay), ppl=float(p_val), select_ppl=float(p_sel),
                rms_pre_norm=float(pre_rms), rms_post_norm=float(post_rms),
                rms_target=float(rms_target), out_gain=float(carrier.out_gain),
                control_std=rec.get("control_std"),
                control_seed=rec.get("control_seed"),
                shuffle=rec.get("shuffle"))


def capture_probe(model, ids, layer_idx):
    layer, attr, original = attention_module(model, layer_idx)
    cap = {}

    class Capture:
        def __call__(self, x, *a, **k):
            cap["h"] = x
            return original(x, *a, **k)

    setattr(layer, attr, Capture())
    model(mx.array(ids[:1024][None, :].astype(np.int32)))
    probe_h = cap["h"]
    ref_out = original(probe_h, None, None)
    mx.eval(probe_h, ref_out)
    setattr(layer, attr, original)
    return layer, attr, original, probe_h, rms(ref_out)


def best_of(rows, key="select_ppl"):
    return min(rows, key=lambda r: r[key]) if rows else None


def main():
    from mlx_lm import load

    t_start = time.time()
    print(f"loading {TEACHER} ...", flush=True)
    model, tok = load(TEACHER)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int32)
    n = len(ids)

    # ---------------------------------------------------------------- ATTACK B
    # Intervals, asserted on the TOKEN INDICES the run actually uses.
    s0, s1 = int(SELECT_FRAC * n), int(SELECT_FRAC * n) + WINDOW
    v0, v1 = int(VAL_FRAC * n), int(VAL_FRAC * n) + WINDOW
    assert 0 <= s0 < s1 <= n, (s0, s1, n)
    assert 0 <= v0 < v1 <= n, (v0, v1, n)
    # the real disjointness property -- overlaps of both ranges, not of the
    # fractional boundaries the committed script asserts
    overlap = max(0, min(s1, v1) - max(s0, v0))
    select = ids[s0:s1]
    val = ids[v0:v1]
    shared_tokens = len(set(select.tolist()) & set(val.tolist()))
    committed_assert = int(0.5 * n) <= int(0.6 * n) < int(0.9 * n)
    print(f"corpus {n:,} tokens | SELECT [{s0},{s1}) VAL [{v0},{v1}) | "
          f"index overlap {overlap} | shared token IDs {shared_tokens} | "
          f"committed assert holds: {committed_assert}", flush=True)
    assert overlap == 0, f"SELECT/VAL overlap by {overlap} token indices"
    assert s1 <= v0, "SELECT must end before VAL begins"

    out = dict(
        meta=dict(
            teacher=TEACHER, n_params=n_params(model),
            n_layers=len(model.model.layers), corpus_tokens=int(n),
            decays=DECAYS, layers=LAYERS, control_seeds=SEEDS,
            gate_seeds=GATES, control_seed_default=CONTROL_SEED,
            gate_seed_default=GATE_SEED_DEFAULT,
            host=platform.platform(), mlx_machine=platform.machine(),
            driver="experiments/verify_transplant.py",
            reimplements="experiments/hybrid_decay.py (same primitives, same protocol)",
            edits_to_others="none",
            protocol=dict(
                selection="decay chosen on best-of-grid SELECT ppl, reported on VAL",
                eval="frozen, fixed 3000-token window, no sampling",
                norm="every arm scaled so output rms equals replaced attention's",
                paired="both arms evaluated inside one loop at each decay",
                determinism=("gate init pinned to VT_GATE_SEED_DEFAULT in the "
                             "main table and the control-seed sweep; its "
                             "sensitivity is reported under attack C")),
            elapsed_s=None),
        splits=dict(n=int(n), select=[s0, s1], val=[v0, v1],
                    index_overlap=int(overlap),
                    committed_interval_assert_holds=bool(committed_assert),
                    committed_assert_implies_disjoint=(
                        "no: int(0.5n)<=int(0.6n)<int(0.9n) is about fractional "
                        "boundaries, not about the 3000-token windows"),
                    shared_token_ids=int(shared_tokens),
                    select_sha1=hashlib.sha1(select.tobytes()).hexdigest(),
                    val_sha1=hashlib.sha1(val.tobytes()).hexdigest()),
        teacher_ppl=None, layers={}, attacks={})

    base = perplexity(model, val)["ppl"]
    out["teacher_ppl"] = float(base)
    print(f"  teacher ppl {base:.4f}", flush=True)

    for li in LAYERS:
        print(f"\n=== LAYER {li} ===", flush=True)
        layer, attr, original, probe_h, rms_target = capture_probe(model, ids, li)
        setattr(layer, attr, GatedMemoryCarrier(int(probe_h.shape[-1]), mode="zero"))
        mx.eval(model.parameters())
        zero_ppl = perplexity(model, val)["ppl"]
        setattr(layer, attr, original)
        print(f"  probe rms_in {rms(probe_h):.4f} attn out rms {rms_target:.5f} | "
              f"zero ppl {zero_ppl:.4f}", flush=True)

        arm_names = {"transfer", "random_matched", "zero"}
        rec = dict(probe=dict(rms_in=rms(probe_h), rms_attn_out=rms_target,
                              shape=[int(x) for x in probe_h.shape]),
                   zero_ppl=float(zero_ppl),
                   paired=[], per_arm={}, random_seed_sensitivity=None,
                   gate_init_sensitivity=None)

        # ------------------------------------------------ ATTACK A + C (+ D seed 0)
        arms = [("transfer", None, "transfer"),
                ("random_matched", CONTROL_SEED, "random_matched")]
        if li == LAYERS[0]:
            arm_names.add("shuffled")
            arms.append(("shuffled", CONTROL_SEED, "shuffled"))
        for decay in DECAYS:
            row = dict(decay=float(decay))
            for arm, seed, mode in arms:
                t0 = time.time()
                r = evaluate(model, li, layer, attr, original, probe_h, decay=decay,
                             seed=seed, gate_seed=GATE_SEED_DEFAULT,
                             rms_target=rms_target, select=select, val=val,
                             mode=mode)
                r["wall_s"] = time.time() - t0
                row[arm] = r
                print(f"    d={decay:<6} {arm:<15} sel {r['select_ppl']:>8.4f} "
                      f"val {r['ppl']:>8.4f} rms_post {r['rms_post_norm']:.6f}", flush=True)
            row["ppl_gap_transfer_minus_control"] = float(
                row["transfer"]["ppl"] - row["random_matched"]["ppl"])
            row["rms_post_reldiff"] = float(
                abs(row["transfer"]["rms_post_norm"] /
                    row["random_matched"]["rms_post_norm"] - 1.0))
            row["rms_target_reldiff"] = float(
                abs(row["transfer"]["rms_post_norm"] / rms_target - 1.0))
            rec["paired"].append(row)
            out["layers"][str(li)] = rec
            dump(out)

        rec["per_arm"] = {}
        for arm in sorted(arm_names):
            if arm == "zero":
                rows = [dict(decay=None, ppl=float(zero_ppl), select_ppl=None)]
                rec["per_arm"]["zero"] = dict(
                    at_best=None, best_val_of_grid=rows[0],
                    recovery_frac=recovery(zero_ppl, base, zero_ppl))
                continue
            rows = [{k: r[arm][k] for k in ("decay", "ppl", "select_ppl")}
                    for r in rec["paired"]]
            b = best_of(rows)
            rec["per_arm"][arm] = dict(
                at_best=b, recovery_frac=recovery(zero_ppl, base, b["ppl"]),
                best_val_of_grid=min(rows, key=lambda x: x["ppl"]),
                val_curve=[dict(decay=r["decay"], ppl=r["ppl"]) for r in rows])
        tb = rec["per_arm"]["transfer"]["at_best"]
        cb = rec["per_arm"]["random_matched"]["at_best"]
        rec["headline_reproduced"] = dict(
            transfer_best=tb, control_best=cb,
            transfer_beats_control_at_own_best=bool(tb["ppl"] < cb["ppl"]),
            margin_ppl=float(cb["ppl"] - tb["ppl"]),
            transfer_recovery_frac=rec["per_arm"]["transfer"]["recovery_frac"],
            control_recovery_frac=rec["per_arm"]["random_matched"]["recovery_frac"])
        out["layers"][str(li)] = rec
        dump(out)
        print(f"  -> transfer {tb['ppl']:.4f} @ {tb['decay']} | control "
              f"{cb['ppl']:.4f} @ {cb['decay']} | margin {cb['ppl']-tb['ppl']:.4f}",
              flush=True)

        # ------------------------------------------------------------- ATTACK D
        if li == LAYERS[0]:
            sens = []
            for seed in SEEDS:
                rows = []
                for decay in DECAYS:
                    r = evaluate(model, li, layer, attr, original, probe_h, decay=decay,
                                 seed=seed, gate_seed=GATE_SEED_DEFAULT,
                                 rms_target=rms_target, select=select, val=val)
                    rows.append({k: r[k] for k in ("decay", "ppl", "select_ppl")})
                b = best_of(rows)
                sens.append(dict(seed=int(seed), at_best=b,
                                 recovery_frac=recovery(zero_ppl, base, b["ppl"]),
                                 val_curve=rows))
                print(f"    control seed {seed}: best-of-grid select {b['select_ppl']:.4f} "
                      f"-> val {b['ppl']:.4f} (decay {b['decay']})", flush=True)
                rec["random_seed_sensitivity"] = sens
                out["layers"][str(li)] = rec
                dump(out)
            bests = [s["at_best"]["ppl"] for s in sens]
            tb_ppl = rec["per_arm"]["transfer"]["at_best"]["ppl"]
            rec["random_seed_sensitivity_summary"] = dict(
                n_seeds=len(SEEDS), best_ppl=float(min(bests)),
                worst_ppl=float(max(bests)), mean_ppl=float(np.mean(bests)),
                std_ppl=float(np.std(bests, ddof=1)) if len(bests) > 1 else None,
                best_of_best=float(min(bests)), n_seeds_beating_transfer=int(
                    sum(1 for b in bests if b < tb_ppl)),
                transfer_ppl=float(tb_ppl),
                transfer_beats_every_seed=bool(all(b > tb_ppl for b in bests)),
                margin_vs_best_seed=float(min(bests) - tb_ppl))
            out["layers"][str(li)] = rec
            dump(out)

            # ------------------------------------------- ATTACK C (gate-init) + A
            gsens = []
            for gseed in GATES:
                rows = []
                for decay in DECAYS:
                    r = evaluate(model, li, layer, attr, original, probe_h, decay=decay,
                                 seed=None, gate_seed=gseed, rms_target=rms_target,
                                 select=select, val=val)
                    rows.append({k: r[k] for k in ("decay", "ppl", "select_ppl",
                                                   "rms_post_norm")})
                b = best_of(rows)
                gsens.append(dict(gate_seed=int(gseed), at_best=b,
                                  recovery_frac=recovery(zero_ppl, base, b["ppl"]),
                                  val_curve=rows))
                print(f"    gate-init seed {gseed}: best-of-grid select "
                      f"{b['select_ppl']:.4f} -> val {b['ppl']:.4f} (decay {b['decay']})",
                      flush=True)
                rec["gate_init_sensitivity"] = gsens
                out["layers"][str(li)] = rec
                dump(out)
            gbests = [s["at_best"]["ppl"] for s in gsens]
            rec["gate_init_sensitivity_summary"] = dict(
                n_seeds=len(GATES), best_ppl=float(min(gbests)),
                worst_ppl=float(max(gbests)), spread_ppl=float(max(gbests) - min(gbests)),
                ppl_by_seed={int(s["gate_seed"]): float(s["at_best"]["ppl"])
                             for s in gsens},
                note=("gate input weights are random-initialised inside the "
                      "transplant and are NOT controlled for by the committed "
                      "protocol; this measures how much of the headline margin "
                      "they can account for"))
            out["layers"][str(li)] = rec
            dump(out)

    # --------------------------------------------------------------- verdicts
    l8 = out["layers"][str(LAYERS[0])]
    rss = l8.get("random_seed_sensitivity_summary") or {}
    gss = l8.get("gate_init_sensitivity_summary") or {}
    margins = [dict(layer=int(k), margin=float(v["headline_reproduced"]["margin_ppl"]),
                    transfer=v["headline_reproduced"]["transfer_best"]["ppl"],
                    control=v["headline_reproduced"]["control_best"]["ppl"],
                    transfer_recovery=v["headline_reproduced"]["transfer_recovery_frac"],
                    control_recovery=v["headline_reproduced"]["control_recovery_frac"])
               for k, v in sorted(out["layers"].items(), key=lambda kv: int(kv[0]))]
    sh = l8.get("per_arm", {}).get("shuffled")
    out["attacks"] = dict(
        A_paired_same_decay=dict(
            done=True,
            per_layer_min_margin=min(abs(m["margin"]) for m in margins),
            positive_margin_every_layer=bool(all(m["margin"] > 0 for m in margins)),
            layers=margins),
        B_disjointness=dict(
            done=True, index_overlap=int(overlap), disjoint=bool(overlap == 0),
            committed_assert=("int(0.5n)<=int(0.6n)<int(0.9n) -- holds, but it is "
                              "not a disjointness test"),
            shared_token_ids=int(shared_tokens)),
        C_rms_confound=dict(
            done=True,
            max_rms_reldiff=float(max(r["rms_post_reldiff"] for r in l8["paired"])),
            note=("every arm is scaled to exactly the attention output rms, so "
                  "post-norm rms is matched by construction; the confound that "
                  "survives is the random gate-input init, reported separately"),
            gate_init_spread_ppl=gss.get("spread_ppl")),
        D_control_seed_sensitivity=dict(done=True, **rss),
        E_layer_generalisation=dict(done=True, layers=LAYERS, per_layer=margins),
        F_shuffled_weight_control=dict(
            done=bool(sh and sh.get("at_best")),
            at_best=(sh or {}).get("at_best"),
            recovery_frac=(sh or {}).get("recovery_frac"),
            transfer_margin_ppl=(float(sh["at_best"]["ppl"] - l8["per_arm"]["transfer"]["at_best"]["ppl"])
                                 if sh and sh.get("at_best") else None),
            note=("same weight multiset and singular values as the transplant, "
                  "input/output spaces permuted. Tests whether the LEARNED "
                  "DIRECTIONS matter, not just scale/spectrum. Extra attack, "
                  "beyond the five requested, added because it is the strongest "
                  "available control for the 'weights carry structure' claim")))
    out["meta"]["elapsed_s"] = time.time() - t_start
    dump(out)
    print(f"\nwrote {OUT} in {out['meta']['elapsed_s']:.1f}s")
    return out


def dump(out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(tmp, OUT)


if __name__ == "__main__":
    main()
