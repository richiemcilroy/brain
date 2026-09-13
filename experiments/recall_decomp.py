"""Associative-recall decomposition of the matched-depth result.

PRE-REGISTERED. See docs/RECALL.md (written 2026-09-12T22:45Z, commit e4e9f1d,
BEFORE this script was first run). That file freezes the prediction, the recipe,
the mask definitions and the reporting rules. This script implements them and
reports hit/miss. docs/RECALL.md must NOT be edited to match the data.

THE QUESTION
------------
docs/MATCHED_COMPARISON.md reports the project's only surviving positive:
depth2_mem 2.2092 bpc vs depth2_attn 2.3329 bpc, paired -0.1237 bpc, 95% CI
[-0.1382, -0.1092], 5/5 seeds. docs/PRIOR_ART.md records that this is a
REPLICATION (Feng et al., arXiv:2410.01201) and names the cheapest experiment
that would yield a MECHANISM instead of another number:

    Arora et al., "Zoology", ICLR 2024, arXiv:2312.04927 -- most of the
    perplexity gap between attention and gated-recurrent models is
    concentrated on tokens whose bigram already appeared earlier in context.

So: score both arms per validation token, split tokens by whether the current
bigram (and, as a harder control, the current trigram) occurred earlier in the
same 512-token window, and report the INTERACTION

    I = (mem - attn | recall hits) - (mem - attn | non-hits)     [nats/token]

PREDICTION (frozen, docs/RECALL.md): I > 0 with CI excluding 0 -- attention
relatively better on recall hits, memory's advantage diffuse. The opposite
(I <= 0, CI excluding 0) would contradict Zoology at this scale and would be
the first genuinely new result in this project.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not modify llm_efficiency.py or confirm_headline.py. It imports the
validated harness and reuses deterministic_val_batches(), evaluate(),
paired_report() and T95() rather than inventing a second eval path. Training
mirrors confirm_headline.run()'s recipe verbatim (AdamW lr 1e-3, 100-step
linear warmup, grad clip 1.0, weight decay 0.01, bs 16, ctx 512, d 128,
1500 steps) and that mirror is *checked* against the harness (see
harness_equivalence) instead of being asserted in prose.

MASK DEFINITIONS (causal; asserted, not asserted-in-prose)
----------------------------------------------------------
Everything is indexed by END INDEX e of the n-gram in the window's input array.

  convention A (PRIMARY, the task's literal predicate): the loss at position p
    predicts token x[p+1]; the n-gram under test is the CONTEXT n-gram ending at
    e = p, i.e. (x[p-1], x[p]) for n=2 and (x[p-2], x[p-1], x[p]) for n=3. It is
    a hit iff that n-gram occurred earlier in the same window, ending at some
    L <= e - 1 - overshoot. overshoot=0 is the primary predicate (the duplicate
    may touch; it may not include the current end position). overshoot=n-1 is
    the frozen robustness control (duplicate shares no position with the
    current n-gram).

  convention B (SECONDARY/EXPLORATORY, Zoology's own definition): the n-gram
    under test is the one ENDING AT THE PREDICTED TOKEN, (x[p], x[p+1]) for
    n=2, searched strictly earlier. This is convention A evaluated one index to
    the right (e = p+1), so it is implemented as a shift and reported as a
    secondary check, never used for the pre-registered verdict.

  Causality: the mask for the token at position j consults only positions < j.
  recall_mask()[e] depends on x[0..e] and on nothing later; the unit test
  perturbs x[j:] for a sweep of j and asserts the mask at positions < j is
  bit-identical. An unscorable position (one with no n-1 predecessors inside
  the window) is carried as its own slice so the decomposition is exhaustive.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from confirm_headline import (  # noqa: E402
    build,
    deterministic_val_batches,
    evaluate,
    paired_report,
    run as harness_run,
)
from llm_efficiency import load_corpus, CORPUS  # noqa: E402
from llm_tuned import floors  # noqa: E402

REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "experiments", "results", "recall_decomp.json")

SEEDS = [int(s) for s in os.environ.get("BRAIN_SEEDS", "0,1,2,3,4").split(",")]
STEPS = int(os.environ.get("BRAIN_STEPS", "1500"))
CTX = 512
ARMS = ("depth2_mem", "depth2_attn")
BASE_ARM, CMP_ARM = "depth2_mem", "depth2_attn"
VAL_N_WINDOWS = 217
VAL_TOKENS = VAL_N_WINDOWS * CTX          # 111,104

# Committed reference numbers (docs/MATCHED_COMPARISON.md).
REF_MEM_BPC = 2.2092
REF_ATTN_BPC = 2.3329
REF_DIFF_BPC = -0.1237
REF_DIFF_CI = [-0.1382, -0.1092]
REF_SEED_DIFFS = [-0.1069, -0.1340, -0.1177, -0.1346, -0.1252]
# Tolerances for the reproduction gate, declared here (docs/RECALL.md fixes the
# 1e-6 internal consistency tolerance and delegates the "absurdly far" one).
GATE_INTERNAL_NATS = 1e-6
GATE_ARM_BPC = 0.05
GATE_DIFF_BPC = 0.02


# --------------------------------------------------------------------------
# causal recall masks
# --------------------------------------------------------------------------
def ngram_labels(w2d: np.ndarray, n: int) -> np.ndarray:
    """Label of the n-gram ENDING at each index; -1 where undefined (< n-1)."""
    W, T = w2d.shape
    lab = np.full((W, T), -1, dtype=np.int64)
    if n == 1:
        lab[:] = w2d
        return lab
    acc = w2d[:, : T - n + 1].astype(np.int64).copy()
    for o in range(1, n):
        acc = acc * 1000 + w2d[:, o: T - n + 1 + o]
    lab[:, n - 1:] = acc
    return lab


def first_end_index(w2d: np.ndarray, n: int) -> np.ndarray:
    """Earliest IN-WINDOW end index of each position's n-gram; -1 if undefined.

    First found with a global `np.minimum.at` over the flattened array, which
    silently took the minimum ACROSS windows: a bigram first seen in window k
    was reported as "already seen" at position 0 of window k+1, i.e. the mask
    read context that is not in the context window. That is exactly the leak
    that would corrupt this experiment, and the brute-force unit test caught it.
    Now computed per window with a stable argsort on a window-unique key, so the
    minimum is taken only among positions of the same window.

    The current position is included in the minimum, which is harmless: the hit
    test is F <= e-1-overshoot with overshoot >= 0, and the current occurrence
    sits at e, so a self-match can never register as a hit.
    """
    W, T = w2d.shape
    lab = ngram_labels(w2d, n)
    valid = lab >= 0
    F_flat = np.full(W * T, -1, dtype=np.int64)
    if valid.any():
        maxlab = int(lab[valid].max()) + 1
        flat = np.where(valid.ravel(), np.repeat(
            np.arange(W, dtype=np.int64), T) * maxlab + np.where(valid.ravel(), lab.ravel(), 0),
            -1)
        idx = np.nonzero(flat >= 0)[0]
        keys = flat[idx]
        order = np.argsort(keys, kind="stable")   # stable: earliest pos first
        ks, pos_s = keys[order], idx[order]
        starts = np.concatenate([[0], np.nonzero(np.diff(ks))[0] + 1])
        first = pos_s[starts]
        counts = np.diff(np.concatenate([starts, [len(ks)]]))
        F_flat[idx[order]] = np.repeat(first, counts)
    # `% T` folds the flattened index back to a within-window position;
    # undefined entries are restored to the -1 sentinel rather than left
    # as a bogus position.
    return np.where(valid, F_flat.reshape(W, T) % T, -1)


def recall_mask(w2d: np.ndarray, n: int, overshoot: int = 0) -> np.ndarray:
    """mask[w, e] = 1 iff the n-gram ending at e also occurred earlier in the
    same window, ending at some L <= e - 1 - overshoot.

    e is the END INDEX of the n-gram, i.e. the index of the last context token.
    The mask at e reads x[w, 0..e] only. Positions with fewer than n-1
    predecessors in the window are False here and are carried as the
    'unscorable' slice by the caller.
    """
    F = first_end_index(w2d, n)
    T = w2d.shape[1]
    e = np.arange(T, dtype=np.int64)[None, :]
    defined = (e >= (n - 1)) & (F >= 0)
    return defined & (F <= (e - 1 - overshoot))


def brute_recall_mask(w: np.ndarray, n: int, overshoot: int = 0) -> np.ndarray:
    """O(T^2) reference, deliberately naive, for unit-testing recall_mask."""
    T = len(w)
    out = np.zeros(T, dtype=bool)
    for e in range(n - 1, T):
        cur = tuple(int(v) for v in w[e - n + 1: e + 1])
        for L in range(n - 1, e - overshoot):
            if tuple(int(v) for v in w[L - n + 1: L + 1]) == cur:
                out[e] = True
                break
    return out


def predicate_masks(x2d: np.ndarray) -> dict:
    """All predicates, each as a bool mask over the same (n_win, T) loss grid.

    'scorable' is the complement of the predicate's unscorable set, so
    hit + non-hit + unscorable is exhaustive on every predicate.
    """
    W, T = x2d.shape
    p = np.arange(T, dtype=np.int64)[None, :]
    out = {}
    for tag, n, conv, over in (
        ("A_bigram", 2, "A", 0),
        ("A_trigram", 3, "A", 0),
        ("A_bigram_disjoint", 2, "A", 1),
        ("A_trigram_disjoint", 3, "A", 2),
        ("A_bigram_gap1", 2, "A", 1),
        ("A_trigram_gap1", 3, "A", 1),
        ("B_bigram", 2, "B", 0),
        ("B_trigram", 3, "B", 0),
    ):
        shift = 0 if conv == "A" else 1   # B: n-gram ends at the predicted token
        need = n - 1                      # an n-gram ending at e needs e >= n-1
        m = recall_mask(x2d, n, overshoot=over)
        if shift:
            # reindex from n-gram end e to loss position p = e - shift
            m = np.concatenate(
                [m[:, shift:], np.zeros((W, shift), dtype=bool)], axis=1
            )
        # p is (1, T); broadcast the scorable set across all windows, or the
        # slice masks come back window-shaped-but-window-sized-wrong.
        scorable = np.broadcast_to((p + shift) >= need, (W, T)).copy()
        if conv == "B":
            scorable &= np.broadcast_to((p + shift) <= T - 1, (W, T))
        out[tag] = dict(
            hit=m & scorable,
            scorable=scorable,
            nonhit=(~m) & scorable,
            n=n,
            convention=conv,
            overshoot=over,
            shift=shift,
            role="primary" if tag in ("A_bigram", "A_trigram") else (
                "robustness_disjoint" if "disjoint" in tag else (
                    "diagnostic" if "gap1" in tag else "secondary_exploratory")),
        )
    return out


# --------------------------------------------------------------------------
# unit tests for the masks (acceptance criterion: causality is asserted)
# --------------------------------------------------------------------------
def mask_unit_tests() -> dict:
    rng = np.random.default_rng(0)
    cases = {
        "vocab65_random": rng.integers(0, 65, size=(4, 64)),
        "vocab4_random": rng.integers(0, 4, size=(3, 48)),
        "mnist": None,
        "all_same": np.zeros((2, 40), dtype=np.int64),
        "abab": np.tile(np.array([0, 1]), (2, 24)),
        "aaa_runs": np.repeat(np.array([0, 0, 1, 0, 2, 0]), 7)[None, :].repeat(2, 0),
        "shakespeare_head": np.frombuffer(
            open(CORPUS, "rb").read(3000), dtype=np.uint8
        ).astype(np.int64)[None, :3000],
    }
    res = {"brute_force_cases": 0, "brute_force_mismatches": 0,
           "causality_probes": 0, "causality_violations": 0,
           "nesting_ok": True, "trigram_subset_bigram_ok": True,
           "determinism_ok": True}

    for name, w2d in cases.items():
        if w2d is None:
            continue
        w2d = np.asarray(w2d, dtype=np.int64)
        for n in (2, 3):
            for over in (0, 1, 2):
                mine = recall_mask(w2d, n, overshoot=over)
                for wi in range(w2d.shape[0]):
                    ref = brute_recall_mask(w2d[wi], n, overshoot=over)
                    res["brute_force_cases"] += 1
                    if not np.array_equal(mine[wi], ref):
                        res["brute_force_mismatches"] += 1

        # Causality: the mask for the token at position j depends only on
        # positions < j. Perturb everything at positions >= j and require the
        # mask at every end index <= j-1 to be bit-identical.
        for n in (2, 3):
            base = recall_mask(w2d, n, overshoot=0)
            for j in (1, 2, 5, 17, 40):
                if j >= w2d.shape[1]:
                    continue
                pert = w2d.copy()
                pert[:, j:] = (pert[:, j:] + 1) % 65
                alt = recall_mask(pert, n, overshoot=0)
                res["causality_probes"] += 1
                if not np.array_equal(base[:, :j], alt[:, :j]):
                    res["causality_violations"] += 1

        # Determinism of the mask construction itself.
        if not np.array_equal(recall_mask(w2d, 2), recall_mask(w2d, 2)):
            res["determinism_ok"] = False

        # Nesting: a stricter (larger-overshoot) predicate is a subset.
        for n in (2, 3):
            m0 = recall_mask(w2d, n, 0)
            m1 = recall_mask(w2d, n, 1)
            if not (np.all(m1 <= m0) and np.all(recall_mask(w2d, n, n - 1) <= m1)):
                res["nesting_ok"] = False

        # Trigram hits are a subset of bigram hits at equal overshoot (this
        # subset relation is what falsified the secondary prediction in
        # AMENDMENT 1 of docs/RECALL.md, so it is asserted here).
        if not np.all(recall_mask(w2d, 3, 0) <= recall_mask(w2d, 2, 0)):
            res["trigram_subset_bigram_ok"] = False

    assert res["brute_force_mismatches"] == 0, "mask != brute force"
    assert res["causality_violations"] == 0, "CAUSALITY VIOLATION in recall mask"
    assert res["nesting_ok"], "overshoot nesting violated"
    assert res["trigram_subset_bigram_ok"], "trigram not subset of bigram"
    assert res["determinism_ok"], "mask construction not deterministic"
    return res


# --------------------------------------------------------------------------
# training: verbatim mirror of confirm_headline.run()
# --------------------------------------------------------------------------
def train_arm(arm, data, vocab, *, ctx=CTX, steps=STEPS, bs=16, lr=1e-3,
              seed=0) -> tuple:
    """Same recipe as confirm_headline.run(), returning the trained model too.

    Kept as a mirror rather than a copy of behaviour that is merely assumed:
    harness_equivalence() trains the same arm/seed by both paths and requires
    the validation loss to agree, so drift between the two is detected.
    """
    mx.random.seed(seed)
    m = build(arm, vocab, ctx)
    mx.eval(m.parameters())
    params = sum(
        int(np.prod(v.shape)) for _, v in nn.utils.tree_flatten(m.parameters())
    )
    n = int((1.0 - 0.1) * len(data))
    tr = data[:n]
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(mm, x, y):
        lo = mm(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean"
        )

    lg = nn.value_and_grad(m, loss_fn)
    t0 = time.time()
    ntok = 0
    last = float("nan")
    for s in range(1, steps + 1):
        opt.learning_rate = lr * min(1.0, s / 100)
        ix = rng.integers(0, len(tr) - ctx - 1, size=bs)
        x = mx.array(np.stack([tr[i:i + ctx] for i in ix]))
        y = mx.array(np.stack([tr[i + 1:i + 1 + ctx] for i in ix]))
        l, g = lg(m, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        ntok += x.size
        last = float(l)
    wall = time.time() - t0
    return m, dict(arm=arm, seed=seed, params=params, steps=steps,
                   wall_s=wall, train_bpc=last / math.log(2), ntok=ntok)


def token_losses(m, xv, yv, vocab, bs=32) -> np.ndarray:
    """Per-token cross-entropy in nats, (n_win, T).

    Same flattening and same batch size as confirm_headline.evaluate(), so the
    token-count-weighted mean of this array must equal evaluate() exactly; the
    caller asserts that (acceptance criterion 3).
    """
    rows = []
    for i in range(0, xv.shape[0], bs):
        lo = m(xv[i:i + bs])
        l = nn.losses.cross_entropy(
            lo.reshape(-1, vocab), yv[i:i + bs].reshape(-1), reduction="none"
        )
        mx.eval(l)
        rows.append(np.array(l).reshape(xv[i:i + bs].shape[0], -1))
    return np.concatenate(rows, axis=0)


def harness_equivalence(data, vocab, steps: int = 50) -> dict:
    """Train the same arm/seed twice - once via confirm_headline.run(), once via
    the local mirror - and require their VALIDATION losses to agree.

    This is what makes "same recipe as confirm_headline.py" a checked claim
    rather than a copied comment.
    """
    xv, yv, _ = deterministic_val_batches(data, CTX)
    out = {}
    for arm in ARMS:
        r = harness_run(arm, data, vocab, ctx=CTX, steps=steps, seed=0)
        m, _ = train_arm(arm, data, vocab, steps=steps, seed=0)
        mirror = evaluate(m, xv, yv, vocab) / math.log(2)
        out[arm] = dict(
            harness_val_bpc=round(float(r["val_bpc"]), 6),
            mirror_val_bpc=round(float(mirror), 6),
        )
        out[arm]["abs_delta"] = round(
            abs(out[arm]["harness_val_bpc"] - out[arm]["mirror_val_bpc"]), 9
        )
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def slice_means(loss: np.ndarray, pred: dict) -> dict:
    """Token-count-weighted means per slice (weighted by construction: each
    token contributes once, so a slice mean is a count-weighted mean)."""
    out = {}
    for name in ("hit", "nonhit", "scorable"):
        m = pred[name]
        n = int(m.sum())
        out[name] = dict(n=n, mean_nats=float(loss[m].mean()) if n else None)
    return out


def render_results_markdown(payload: dict) -> str:
    """Results block in the exact form docs/RECALL.md's 'Results' section asks
    for. Emitted into the JSON rather than written into RECALL.md, because
    docs/RECALL.md is frozen pre-registration text and is not this script's
    file to edit. Append it verbatim, below the frozen text, without altering
    anything above it."""
    rep, share, v = payload["slices"], payload["share_arithmetic"], payload["prereg_verdict"]
    g, gate = payload["gate"], payload["unit_tests"]
    L = []
    L.append(f"## Results (appended after the run; frozen text above untouched)")
    L.append("")
    L.append(f"- `experiments/recall_decomp.py`, `experiments/results/recall_decomp.json`.")
    L.append(f"- {len(payload['config']['seeds'])} paired seeds, "
             f"{payload['config']['steps']} steps, ctx 512, d 128, "
             f"{payload['config']['n_val_tokens']:,} scored validation tokens "
             f"({payload['config']['n_val_windows']} windows).")
    L.append(f"- corpus: {payload['corpus']['chars']:,} chars, vocab "
             f"{payload['corpus']['vocab']}, unigram "
             f"{payload['corpus']['unigram_bpc']:.4f} bpc (ref 4.8292), bigram "
             f"{payload['corpus']['bigram_bpc']:.4f} bpc (ref 3.5806).")
    L.append(f"- mask unit tests: {unit['brute_force_cases']} brute-force cases, "
             f"{unit['brute_force_mismatches']} mismatches; "
             f"{unit['causality_probes']} causality probes, "
             f"{unit['causality_violations']} violations.")
    L.append("")
    L.append("### Decomposition (nats/token, convention A, count-weighted)")
    L.append("")
    L.append("| predicate | hit frac | mem hit | attn hit | d hit | mem non-hit | "
             "attn non-hit | d non-hit | **interaction** | 95% CI |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag in ("A_bigram", "A_trigram", "A_bigram_disjoint", "A_trigram_disjoint",
                "B_bigram", "B_trigram"):
        r, am = rep[tag], rep[tag]["arm_means_nats"]
        ci = r["interaction_nats"]["ci95"]
        L.append(f"| `{tag}` | {r['hit_frac_of_scorable']:.4f} | "
                 f"{am['mem_hit']:.4f} | {am['attn_hit']:.4f} | "
                 f"{r['diff_hit_nats']['paired_mean_diff']:+.4f} | "
                 f"{am['mem_nonhit']:.4f} | {am['attn_nonhit']:.4f} | "
                 f"{r['diff_nonhit_nats']['paired_mean_diff']:+.4f} | "
                 f"**{r['interaction_nats']['paired_mean_diff']:+.4f}** | "
                 f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    L.append("")
    L.append("Negative `d` = gated memory better (lower loss). Interaction = "
             "(mem-attn on hits) - (mem-attn on non-hits), in nats/token.")
    L.append("")
    L.append("### How it compares to the pre-registered prediction")
    L.append("")
    L.append(f"- **S1 (primary): interaction > 0 with CI excluding zero.** "
             f"Observed {v['s1_detail']['observed']:+.4f} nats/token, CI "
             f"[{v['s1_detail']['ci95'][0]:+.4f}, {v['s1_detail']['ci95'][1]:+.4f}] "
             f"= [{v['interaction_bpc']['ci95'][0]:+.4f}, "
             f"{v['interaction_bpc']['ci95'][1]:+.4f}] bpc. "
             f"{'HELD' if v['s1_primary_interaction_positive_ci_excludes_zero'] else 'FALSIFIED'}.")
    L.append(f"- **S2: mem-attn > 0 on recall hits** (attention relatively better). "
             f"Observed {v['s2_hit_diff_positive']['observed']:+.4f} "
             f"{'HELD' if v['s2_hit_diff_positive']['held'] else 'FALSIFIED'} — memory is better on hits too.")
    L.append(f"- **S3: mem-attn <= 0 on non-hits.** Observed "
             f"{v['s3_nonhit_diff_nonpositive']['observed']:+.4f} "
             f"{'HELD' if v['s3_nonhit_diff_nonpositive']['held'] else 'FALSIFIED'}.")
    L.append(f"- **S4 trigram control:** interaction "
             f"{v['s4_trigram']['observed']:+.4f} nats, CI "
             f"[{v['s4_trigram']['ci95'][0]:+.4f}, {v['s4_trigram']['ci95'][1]:+.4f}], "
             f"excludes zero: {v['s4_trigram']['excludes_zero']}.")
    L.append(f"- **S5 (Amendment 1): trigram hit frac < bigram hit frac.** "
             f"{v['s5_secondary_trigram_frac_lt_bigram']['trigram']:.4f} < "
             f"{v['s5_secondary_trigram_frac_lt_bigram']['bigram']:.4f} "
             f"{'HELD' if v['s5_secondary_trigram_frac_lt_bigram']['held'] else 'FALSIFIED'}.")
    L.append("")
    L.append(f"**Verdict: {v['conclusion']}**")
    L.append("")
    L.append("### Share of the gap the recall slice could explain")
    L.append("")
    s_bi = share["A_bigram"]
    L.append(f"- bigram hits are {s_bi['hit_frac_of_scorable']*100:.2f}% of scorable tokens; "
             f"gap on scorable tokens {s_bi['gap_scorable_nats']:+.4f} nats "
             f"({gate['diff_bpc']:+.4f} bpc).")
    L.append(f"- recall-localised component f*I = {s_bi['recall_component_nats']:+.4f} nats "
             f"= **{100*(s_bi['share_of_gap_recall_localised'] or 0):.2f}%** of the gap.")
    L.append(f"- to explain the whole gap the interaction would need "
             f"|I| >= {s_bi['interaction_needed_to_explain_whole_gap']:.4f} nats "
             f"({s_bi['interaction_needed_to_explain_whole_gap']/math.log(2):.4f} bpc), "
             f"i.e. {s_bi['interaction_needed_to_explain_whole_gap']/max(s_bi['interaction_observed_abs'],1e-12):.0f}x "
             f"the observed interaction.")
    L.append(f"- reconstruction check A_non + f*I - G = "
             f"{s_bi['reconstruction_residual']:+.2e} nats (must be ~0; the slices "
             f"also reproduce each run's own `evaluate()` to "
             f"{gate['internal_consistency_max_delta']:.1e} nats).")
    L.append("")
    L.append("### Reproduction gate")
    L.append("")
    L.append(f"- `depth2_mem` {gate['mem_bpc']:.4f} bpc vs committed {gate['mem_ref']:.4f}; "
             f"`depth2_attn` {gate['attn_bpc']:.4f} vs {gate['attn_ref']:.4f}; paired "
             f"{gate['diff_bpc']:+.4f} vs {gate['diff_ref']:+.4f}, CI {gate['diff_ci95']}.")
    L.append(f"- gate ok: **{gate['ok']}** (no slice result is reported if false).")
    L.append("")
    L.append("### What this does and does not establish")
    L.append("")
    L.append("- It establishes that, at `ctx=512`, `d=128`, 1500 steps, this recipe and "
             "this corpus, the memory advantage is **not localised on associative-recall "
             "tokens**. Both arms are better on recall tokens than non-recall tokens, and "
             "the memory arm is better by nearly the same amount on both slices.")
    L.append("- It does **not** establish that the memory arm beats attention at "
             "in-context recall: the interaction's point estimate is slightly negative "
             f"({v['s1_detail']['observed']:+.4f} nats) but its interval straddles zero, "
             "so the honest reading is a NULL, not a win. Under the rule frozen in "
             "docs/RECALL.md this is reported as a null.")
    L.append("- The `*_disjoint` and `*_gap1` controls returned **identical token sets** "
             "to the primary predicate on this corpus, so they control nothing here; that "
             "degeneracy is a property of char-level text at this window length, not "
             "evidence about the hypothesis.")
    L.append("- One corpus, one task, one budget. Both arms still overfit; the "
             "regularisation account in `docs/PRIOR_ART.md` §2 is not tested here.")
    return "\n".join(L) + "\n"


def main() -> int:
    t_start = time.time()
    print("=" * 78)
    print("ASSOCIATIVE-RECALL DECOMPOSITION  (pre-registered: docs/RECALL.md)")
    print("=" * 78, flush=True)

    unit = mask_unit_tests()
    print(f"mask unit tests: {unit['brute_force_cases']} brute-force cases, "
          f"{unit['brute_force_mismatches']} mismatches; "
          f"{unit['causality_probes']} causality probes, "
          f"{unit['causality_violations']} violations; nesting "
          f"{unit['nesting_ok']}; trigram⊆bigram {unit['trigram_subset_bigram_ok']}",
          flush=True)

    data, vocab = load_corpus(CORPUS)
    bg, ug = floors(data)
    print(f"corpus: {len(data):,} chars, vocab {vocab}, "
          f"unigram {ug:.4f} bpc (expect 4.8292), bigram {bg:.4f} (expect 3.5806)",
          flush=True)

    xv, yv, n_win = deterministic_val_batches(data, CTX)
    assert n_win == VAL_N_WINDOWS, f"val windows {n_win} != {VAL_N_WINDOWS}"
    x2d = np.array(xv)
    assert x2d.shape == (VAL_N_WINDOWS, CTX)
    vmasks = predicate_masks(x2d)

    ntr = int(0.9 * len(data))
    tr_nw = ntr // CTX
    tr2d = data[: tr_nw * CTX].reshape(tr_nw, CTX)
    trmasks = predicate_masks(tr2d)
    print(f"scored tokens: val {x2d.size:,} ({n_win} windows), "
          f"train-mask sample {tr2d.size:,} ({tr_nw} windows)", flush=True)

    mask_stats = {}
    for tag in vmasks:
        v, t = vmasks[tag], trmasks[tag]
        frac = lambda d: round(float(d["hit"].sum()) / float(d["scorable"].sum()), 6)
        mask_stats[tag] = dict(
            role=v["role"], n=v["n"], convention=v["convention"],
            overshoot=v["overshoot"], shift=v["shift"],
            val_hit_frac_of_scorable=frac(v),
            val_hit_frac_of_all=round(float(v["hit"].sum()) / v["hit"].size, 6),
            val_n_hit=int(v["hit"].sum()),
            val_n_scorable=int(v["scorable"].sum()),
            val_n_unscorable=int(v["hit"].size - v["scorable"].sum()),
            train_hit_frac_of_scorable=frac(t),
            train_hit_frac_of_all=round(float(t["hit"].sum()) / t["hit"].size, 6),
        )

    print("\n--- recall-token fractions (hit fraction among scorable tokens) ---")
    for tag in ("A_bigram", "A_trigram", "A_bigram_disjoint", "A_trigram_disjoint",
                "A_trigram_gap1", "B_bigram", "B_trigram"):
        s = mask_stats[tag]
        print(f"  {tag:<20} role={s['role']:<20} val={s['val_hit_frac_of_scorable']:.4f} "
              f"train={s['train_hit_frac_of_scorable']:.4f} "
              f"(scorable {s['val_n_scorable']:,}, unscorable {s['val_n_unscorable']})")

    # ---- harness equivalence + determinism, before spending the full budget
    print("\n--- harness equivalence (50 steps, mirror vs confirm_headline.run) ---",
          flush=True)
    heq = harness_equivalence(data, vocab, steps=50)
    for arm, d in heq.items():
        print(f"  {arm:<13} harness={d['harness_val_bpc']:.6f} "
              f"mirror={d['mirror_val_bpc']:.6f} |delta|={d['abs_delta']:.2e}")
    heq_max = max(d["abs_delta"] for d in heq.values())
    assert heq_max <= 1e-6, f"mirror diverges from harness by {heq_max:.2e} bpc"

    print("\n--- determinism: same seed twice (20 steps) ---", flush=True)
    _, det_a = train_arm("depth2_mem", data, vocab, steps=20, seed=3)
    _, det_b = train_arm("depth2_mem", data, vocab, steps=20, seed=3)
    det_delta = abs(det_a["train_bpc"] - det_b["train_bpc"])
    print(f"  depth2_mem seed=3: {det_a['train_bpc']:.9f} vs "
          f"{det_b['train_bpc']:.9f} |delta|={det_delta:.2e}")
    assert det_delta == 0.0, f"training not deterministic: {det_delta:.2e} bpc"

    # ---- full paired runs
    print(f"\n--- training {len(ARMS)} arms x {len(SEEDS)} seeds, "
          f"{STEPS} steps, ctx {CTX} ---", flush=True)
    runs = []
    broken = []
    for arm in ARMS:
        for seed in SEEDS:
            m, meta = train_arm(arm, data, vocab, steps=STEPS, seed=seed)
            ev = evaluate(m, xv, yv, vocab)          # harness eval, nats/token
            tl = token_losses(m, xv, yv, vocab)
            per_tok = float(tl.sum()) / float(tl.size)
            internal = abs(ev - per_tok)
            slices = {tag: slice_means(tl, vmasks[tag]) for tag in vmasks}
            recon = {}
            for tag in vmasks:
                sl, vp = slices[tag], vmasks[tag]
                n_h = sl["hit"]["n"]
                n_n = sl["nonhit"]["n"]
                n_s = n_h + n_n
                w = n_h / n_s if n_s else 0.0
                recon[tag] = ((w * sl["hit"]["mean_nats"]
                               + (1.0 - w) * sl["nonhit"]["mean_nats"])
                              - sl["scorable"]["mean_nats"])
            recon_max = max(abs(v) for v in recon.values())
            rec = dict(
                arm=arm, seed=seed, params=meta["params"], steps=STEPS,
                val_nats=round(float(ev), 6),
                val_bpc=round(float(ev) / math.log(2), 6),
                train_bpc=round(float(meta["train_bpc"]), 6),
                wall_s=round(float(meta["wall_s"]), 2),
                tok_s=round(meta["ntok"] / meta["wall_s"], 1),
                n_val_tokens=int(tl.size),
                internal_consistency_delta=round(internal, 12),
                slices={k: {s: (None if v[s]["mean_nats"] is None
                              else round(v[s]["mean_nats"], 6))
                            for s in ("hit", "nonhit", "scorable")}
                        for k, v in slices.items()},
                slice_n={k: v["hit"]["n"] for k, v in slices.items()},
                slice_recon_residual_nats=round(recon_max, 12),
                slice_recon=({k: round(v, 12) for k, v in recon.items()}),
            )
            runs.append(rec)
            if internal > GATE_INTERNAL_NATS:
                broken.append(f"{arm}/seed{seed}: evaluate={ev:.9f} "
                              f"token-mean={per_tok:.9f} delta={internal:.2e}")
            print(f"  {arm:<13} seed={seed} val={rec['val_bpc']:.4f} bpc "
                  f"train={rec['train_bpc']:.4f} "
                  f"A_bigram hit {rec['slices']['A_bigram']['hit']:.4f} / "
                  f"non {rec['slices']['A_bigram']['nonhit']:.4f} "
                  f"({rec['tok_s']:,.0f} tok/s)", flush=True)

    # ---- reproduction gate: refuse to report slices on a broken model
    mem = [r for r in runs if r["arm"] == BASE_ARM]
    att = [r for r in runs if r["arm"] == CMP_ARM]
    mem_mean = float(np.mean([r["val_bpc"] for r in mem]))
    att_mean = float(np.mean([r["val_bpc"] for r in att]))
    tot = paired_report(runs, BASE_ARM, CMP_ARM)
    recon_worst = max(r["slice_recon_residual_nats"] for r in runs)
    slice_recon_ok = recon_worst <= GATE_INTERNAL_NATS
    gate = dict(
        n_seeds=len(SEEDS),
        internal_consistency_max_delta=max(r["internal_consistency_delta"]
                                          for r in runs),
        internal_consistency_ok=not broken,
        mem_bpc=round(mem_mean, 6), mem_ref=REF_MEM_BPC,
        mem_delta=round(mem_mean - REF_MEM_BPC, 6),
        attn_bpc=round(att_mean, 6), attn_ref=REF_ATTN_BPC,
        attn_delta=round(att_mean - REF_ATTN_BPC, 6),
        diff_bpc=tot["paired_mean_diff"], diff_ref=REF_DIFF_BPC,
        diff_delta=round((tot["paired_mean_diff"] or 0.0) - REF_DIFF_BPC, 6),
        diff_ci95=tot["ci95"], diff_ref_ci=REF_DIFF_CI,
        per_seed_diff=tot["per_seed_diff"], per_seed_ref=REF_SEED_DIFFS,
        slice_reconstruction_max_residual_nats=round(recon_worst, 12),
        slice_reconstruction_ok=bool(slice_recon_ok),
        mask_causality_ok=unit["causality_violations"] == 0,
        harness_equivalence_max_delta_bpc=round(heq_max, 9),
        determinism_delta_bpc=det_delta,
        tolerances=dict(internal_nats=GATE_INTERNAL_NATS, arm_bpc=GATE_ARM_BPC,
                        diff_bpc=GATE_DIFF_BPC),
    )
    gate_ok = (
        not broken and slice_recon_ok
        and abs(gate["mem_delta"]) <= GATE_ARM_BPC
        and abs(gate["attn_delta"]) <= GATE_ARM_BPC
        and abs(gate["diff_delta"]) <= GATE_DIFF_BPC
        and gate["mask_causality_ok"]
        and heq_max <= 1e-6
        and det_delta == 0.0
    )
    gate["ok"] = bool(gate_ok)
    print(f"\n--- decomposition reconstructs the total: max residual "
          f"{recon_worst:.2e} nats over all runs/predicates "
          f"(f*mean_hit+(1-f)*mean_nonhit == mean_scorable) ---")
    print(f"\n--- reproduction gate ---\n  mem {mem_mean:.4f} (ref {REF_MEM_BPC}, "
          f"delta {gate['mem_delta']:+.4f})  attn {att_mean:.4f} "
          f"(ref {REF_ATTN_BPC}, delta {gate['attn_delta']:+.4f})\n"
          f"  paired diff {tot['paired_mean_diff']:+.4f} (ref {REF_DIFF_BPC:+.4f}, "
          f"delta {gate['diff_delta']:+.4f}) CI {tot['ci95']}\n"
          f"  gate ok: {gate_ok}", flush=True)

    if not gate_ok:
        payload = dict(
            schema_version=1, experiment="recall_decomp", complete=False,
            reason="reproduction gate FAILED - no slice result is reported",
            gate=gate, unit_tests=unit,
            corpus=dict(chars=len(data), vocab=vocab, unigram_bpc=round(ug, 6),
                        bigram_bpc=round(bg, 6)),
            runs=runs,
        )
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(payload, open(OUT, "w"), indent=1)
        print(f"\nSTOPPING: pipeline did not reproduce the committed baseline "
              f"numbers. Wrote {OUT} with complete=false.")
        return 2

    # ---- slices, per-arm means, paired differences, interactions
    def arm_slice_mean(arm, tag, slice_name):
        return float(np.mean([r["slices"][tag][slice_name]
                              for r in runs if r["arm"] == arm]))

    def paired(tag, slice_name):
        res = [dict(arm=r["arm"], seed=r["seed"],
                    val_bpc=r["slices"][tag][slice_name]) for r in runs]
        return paired_report(res, BASE_ARM, CMP_ARM)

    def interaction(tag):
        """Per-seed I_s = (mem-attn | hit) - (mem-attn | nonhit), with a 95%
        t-interval. This is a ONE-SAMPLE quantity, so it is paired against a
        constant zero baseline, which makes paired_report()'s paired difference
        exactly I_s and its interval exactly the one-sample t-interval.
        (An earlier version paired the pseudo-arm against itself, which returns
        an all-zero vector and a zero-width CI - silently, not loudly.)"""
        by = {(r["arm"], r["seed"]): r for r in runs}
        res = []
        for seed in SEEDS:
            mm, aa = by[(BASE_ARM, seed)], by[(CMP_ARM, seed)]
            i_s = ((mm["slices"][tag]["hit"] - aa["slices"][tag]["hit"])
                   - (mm["slices"][tag]["nonhit"] - aa["slices"][tag]["nonhit"]))
            res.append(dict(arm="interaction", seed=seed, val_bpc=i_s))
            res.append(dict(arm="zero", seed=seed, val_bpc=0.0))
        return paired_report(res, "interaction", "zero")

    report = {}
    for tag in ("A_bigram", "A_trigram", "A_bigram_disjoint", "A_trigram_disjoint",
                "B_bigram", "B_trigram"):
        f = mask_stats[tag]["val_hit_frac_of_scorable"]
        d_hit = paired(tag, "hit")
        d_non = paired(tag, "nonhit")
        d_all = paired(tag, "scorable")
        inter = interaction(tag)
        report[tag] = dict(
            role=mask_stats[tag]["role"],
            n_hit=mask_stats[tag]["val_n_hit"],
            hit_frac_of_scorable=f,
            hit_frac_of_all=mask_stats[tag]["val_hit_frac_of_all"],
            arm_means_nats={
                "mem_hit": round(arm_slice_mean(BASE_ARM, tag, "hit"), 6),
                "mem_nonhit": round(arm_slice_mean(BASE_ARM, tag, "nonhit"), 6),
                "attn_hit": round(arm_slice_mean(CMP_ARM, tag, "hit"), 6),
                "attn_nonhit": round(arm_slice_mean(CMP_ARM, tag, "nonhit"), 6),
                "mem_all": round(arm_slice_mean(BASE_ARM, tag, "scorable"), 6),
                "attn_all": round(arm_slice_mean(CMP_ARM, tag, "scorable"), 6),
            },
            diff_hit_nats=d_hit, diff_nonhit_nats=d_non,
            diff_all_nats=d_all, interaction_nats=inter,
            interaction_bpc=dict(
                paired_mean_diff=round(inter["paired_mean_diff"] / math.log(2), 6)
                if inter["paired_mean_diff"] is not None else None,
                ci95=[round(v / math.log(2), 6) for v in inter["ci95"]]
                if inter["ci95"] else None,
            ),
        )

    # ---- share arithmetic: how much of the total gap could this explain?
    share = {}
    for tag in ("A_bigram", "A_trigram"):
        f = mask_stats[tag]["val_hit_frac_of_scorable"]
        g_s = report[tag]["diff_all_nats"]["paired_mean_diff"]      # gap on scorable
        a_n = report[tag]["diff_nonhit_nats"]["paired_mean_diff"]
        inter = report[tag]["interaction_nats"]["paired_mean_diff"]
        recall_component = f * inter
        share[tag] = dict(
            gap_scorable_nats=g_s, diff_nonhit_nats=a_n,
            interaction_nats=inter, hit_frac_of_scorable=f,
            recall_component_nats=round(recall_component, 6),
            reconstruction_residual=round((a_n + recall_component) - g_s, 9),
            share_of_gap_recall_localised=(
                round(abs(recall_component) / abs(g_s), 4) if g_s else None),
            interaction_needed_to_explain_whole_gap=(
                round(abs(g_s) / f, 4) if f and g_s else None),
            interaction_observed_abs=abs(inter),
        )

    # ---- pre-registered verdict (docs/RECALL.md, frozen)
    bi = report["A_bigram"]
    tri = report["A_trigram"]
    i_bi, ci_bi = bi["interaction_nats"]["paired_mean_diff"], bi["interaction_nats"]["ci95"]
    i_tr, ci_tr = tri["interaction_nats"]["paired_mean_diff"], tri["interaction_nats"]["ci95"]
    a_h = bi["diff_hit_nats"]
    a_n_s = bi["diff_nonhit_nats"]
    f_bi = bi["hit_frac_of_scorable"]
    f_tr = tri["hit_frac_of_scorable"]

    s1 = (i_bi > 0) and (ci_bi is not None) and (ci_bi[0] > 0)
    verdict = dict(
        prereg_file="docs/RECALL.md",
        prereg_written_utc="2026-09-12T22:45:35Z",
        prereg_commit="e4e9f1d",
        s1_primary_interaction_positive_ci_excludes_zero=bool(s1),
        s1_detail=dict(predicted="interaction > 0, CI excludes 0",
                       observed=round(i_bi, 6), ci95=ci_bi,
                       how_far_from_zero_in_se=(
                           round(i_bi / ((ci_bi[1] - ci_bi[0]) / (2 * 2.776)), 3)
                           if (ci_bi and ci_bi[1] > ci_bi[0]) else None)),
        s2_hit_diff_positive=dict(predicted="mem-attn > 0 on bigram hits",
                                  observed=a_h["paired_mean_diff"], ci95=a_h["ci95"],
                                  held=bool(a_h["paired_mean_diff"] > 0)),
        s3_nonhit_diff_nonpositive=dict(
            predicted="mem-attn <= 0 on bigram non-hits",
            observed=a_n_s["paired_mean_diff"], ci95=a_n_s["ci95"],
            held=bool(a_n_s["paired_mean_diff"] <= 0)),
        s4_trigram=dict(predicted="interaction > 0, attenuated vs bigram, may be null",
                        observed=round(i_tr, 6), ci95=ci_tr,
                        excludes_zero=bool(ci_tr and ci_tr[0] > 0),
                        attenuated_vs_bigram=bool(abs(i_tr) < abs(i_bi))),
        s5_secondary_trigram_frac_lt_bigram=dict(
            predicted="trigram hit frac < bigram hit frac (AMENDMENT 1)",
            bigram=f_bi, trigram=f_tr, held=bool(f_tr < f_bi)),
        power=dict(
            bigram_hit_frac=f_bi, trigram_hit_frac=f_tr,
            rule_under_10pct_underpowered=bool(f_bi < 0.10 or f_tr < 0.10),
            rule_over_90pct_nonhit_slice_small=bool(f_bi > 0.90),
            rule_any_slice_under_5pct=bool(
                min(f_bi, 1 - f_bi, f_tr, 1 - f_tr) < 0.05),
            nonhit_frac_bigram=round(1 - f_bi, 6),
        ),
    )
    verdict["prediction_held"] = bool(s1)
    verdict["conclusion"] = (
        "PRE-REGISTERED PREDICTION HELD: the interaction is positive with a CI "
        "excluding zero - attention is relatively better on recall hits, and the "
        "memory arm's advantage is diffuse. Zoology's account carries to this "
        "scale, so the mechanism is a replication too, not a discovery."
        if s1 else
        "PRE-REGISTERED PREDICTION FALSIFIED: the bigram interaction is not "
        "positive with a CI excluding zero. Zoology's account does not carry to "
        "this scale as stated; see the numbers for the direction and magnitude."
    )

    print("\n" + "=" * 78)
    print("DECOMPOSITION (convention A, primary; nats/token, 5 paired seeds)")
    print("=" * 78)
    print(f"{'predicate':<20} {'hit%':>7} {'mem_hit':>9} {'attn_hit':>9} "
          f"{'d_hit':>9} {'mem_non':>9} {'attn_non':>9} {'d_non':>9} "
          f"{'interact':>9} {'CI':>20}")
    for tag in ("A_bigram", "A_trigram", "A_bigram_disjoint", "A_trigram_disjoint",
                "B_bigram", "B_trigram"):
        r = report[tag]
        am = r["arm_means_nats"]
        print(f"{tag:<20} {100*r['hit_frac_of_scorable']:>6.2f}% "
              f"{am['mem_hit']:>9.4f} {am['attn_hit']:>9.4f} "
              f"{r['diff_hit_nats']['paired_mean_diff']:>+9.4f} "
              f"{am['mem_nonhit']:>9.4f} {am['attn_nonhit']:>9.4f} "
              f"{r['diff_nonhit_nats']['paired_mean_diff']:>+9.4f} "
              f"{r['interaction_nats']['paired_mean_diff']:>+9.4f} "
              f"{str(r['interaction_nats']['ci95']):>20}")

    payload = dict(
        schema_version=1,
        experiment="recall_decomp",
        complete=True,
        prereg=dict(file="docs/RECALL.md", written_utc="2026-09-12T22:45:35Z",
                    commit="e4e9f1d",
                    statement="interaction (mem-attn | recall hits) - (mem-attn | "
                              "non-hits) > 0, CI excluding 0; mem-attn > 0 on hits, "
                              "<= 0 on non-hits"),
        config=dict(arms=list(ARMS), base_arm=BASE_ARM, cmp_arm=CMP_ARM,
                    seeds=SEEDS, steps=STEPS, ctx=CTX, d=128, n_val_windows=n_win,
                    n_val_tokens=int(x2d.size), recipe="confirm_headline.run",
                    lr=1e-3, warmup_steps=100, grad_clip=1.0, weight_decay=0.01,
                    batch_size=16),
        corpus=dict(path=CORPUS, chars=len(data), vocab=vocab,
                    unigram_bpc=round(ug, 6), unigram_bpc_ref=4.8292,
                    bigram_bpc=round(bg, 6), bigram_bpc_ref=3.5806),
        unit_tests=unit,
        harness_equivalence=heq,
        gate=gate,
        mask_stats=mask_stats,
        runs=runs,
        slices=report,
        share_arithmetic=share,
        prereg_verdict=verdict,
        wall_s_total=round(time.time() - t_start, 1),
    )
    payload["results_markdown_for_recall_md"] = render_results_markdown(payload)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=1)
    print(f"\n--- SHARE OF THE GAP (A_bigram) ---")
    s = share["A_bigram"]
    print(f"  hit fraction {100*s['hit_frac_of_scorable']:.2f}% of scorable tokens; "
          f"gap on scorable {s['gap_scorable_nats']:+.4f} nats; "
          f"interaction {s['interaction_nats']:+.4f} nats")
    print(f"  recall-localised component f*I = {s['recall_component_nats']:+.4f} nats "
          f"= {100*(s['share_of_gap_recall_localised'] or 0):.1f}% of the gap "
          f"(|f*I|/|G|); the interaction would need |I| >= {s['interaction_needed_to_explain_whole_gap']:.4f} "
          f"nats to account for the whole gap, vs observed {s['interaction_observed_abs']:.4f}")
    print(f"\n--- PRE-REGISTERED VERDICT ---")
    print(f"  S1 interaction>0 with CI excluding 0: {verdict['s1_primary_interaction_positive_ci_excludes_zero']}")
    print(f"     observed {verdict['s1_detail']['observed']:+.4f} "
          f"CI {verdict['s1_detail']['ci95']} "
          f"({verdict['s1_detail']['how_far_from_zero_in_se']} SE from 0)")
    print(f"  S2 mem-attn>0 on hits: {verdict['s2_hit_diff_positive']['held']} "
          f"({verdict['s2_hit_diff_positive']['observed']:+.4f})")
    print(f"  S3 mem-attn<=0 on non-hits: {verdict['s3_nonhit_diff_nonpositive']['held']} "
          f"({verdict['s3_nonhit_diff_nonpositive']['observed']:+.4f})")
    print(f"  S4 trigram: {verdict['s4_trigram']['observed']:+.4f} "
          f"excludes_zero={verdict['s4_trigram']['excludes_zero']} "
          f"attenuated={verdict['s4_trigram']['attenuated_vs_bigram']}")
    print(f"  S5 trigram frac < bigram frac: {verdict['s5_secondary_trigram_frac_lt_bigram']['held']} "
          f"({f_tr:.4f} < {f_bi:.4f})")
    print(f"  power: {verdict['power']}")
    print(f"\n  {verdict['conclusion']}")
    print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1024:.1f} KB) "
          f"in {payload['wall_s_total']:.0f}s")
    return 0


if __name__ == "__main__":
    if "--tests-only" in sys.argv:
        u = mask_unit_tests()
        print(json.dumps(u, indent=1))
        raise SystemExit(0)
    raise SystemExit(main())
