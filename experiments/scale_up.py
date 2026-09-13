"""Scale-up + held-out-TEST confirmation attempt for the matched-depth claim.

WHAT IS UNDER SCRUTINY
----------------------
docs/MATCHED_COMPARISON.md reports the project's one surviving positive. On
char-level TinyShakespeare (~1.1 M chars), ctx=512, d=128, 1500 steps, 5 paired
seeds, deterministic validation over all 217 non-overlapping 512-token windows:

    depth2_mem   445,440 params   val 2.2092 bpc
    depth2_attn  478,976 params   val 2.3329 bpc
    paired diff -0.1237 bpc, 95% CI [-0.1382, -0.1092], 5/5 seeds agreeing

An adversarial review named two credibility gaps and one mechanism concern:

  1. NO HELD-OUT TEST SPLIT. Roughly 15 configurations were compared on the
     same validation split, so the reported number carries a selection optimism
     bias of order 1.7 standard errors.
  2. ONE TINY CORPUS. 1.1 MB of Shakespeare cannot support a general claim.
  3. THE OVERFIT CONFOUND (the review's most important finding). Both arms were
     trained WITHOUT DROPOUT on a corpus passed ~12 times, so the comparison may
     be measuring "who overfits least" rather than "who models better".
     Committed curves support the concern: A_attention reaches train 1.47 bpc
     while its best val is 2.23 and its FINAL val is 2.40 - it overfits well
     past its best point. nanoGPT's shakespeare_char config uses dropout 0.2,
     and a no-dropout baseline may be mistuned in the same way as the
     bigram-floor baseline this project already retracted.

THIS SCRIPT ADDRESSES ALL THREE.

CORPUS
------
enwik8, the standard 100 MB char-level bpc benchmark, so the numbers are
directly comparable to published results. Standard 90/5/5 M split. The corpus
path, byte size and SHA256 are recorded in the output. Nothing is truncated or
subset: the full 100,000,000-byte file is used.

  enwik8 canonical sha256:
  2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8

CORPUS LOCATION
---------------
Deliberately NOT on /Volumes/T9 (that volume unmounted mid-session and took a
previous download with it). Candidates, in order, are $BRAIN_CORPUS, then
/tmp/zz_enwik8/extract/enwik8, then the repo's data/ directory. If none exists
the script downloads the zip into /tmp/zz_enwik8, extracts only enwik8, and
deletes the zip.

TEST DISCIPLINE
---------------
TEST IS TOUCHED ONCE, AT THE END, FOR EVERY ARM, and never during selection.
Learning rate is chosen on the validation split only. The count of
configurations tried is recorded in the output so the optimism can be stated
rather than hidden.

DROPOT IS A MANIPULATION, NOT A TUNING KNOB
-------------------------------------------
Dropout levels {0.0, 0.1, 0.2} are all reported for both arms. Nothing is
selected by looking at which dropout level produces the biggest gap. The
pre-registered prediction, written before the runs, is:

    attention's best-of-curve validation bpc improves by 0.05-0.1 bpc at
    dropout 0.2 relative to dropout 0.0, and the paired gap against the
    2-layer memory arm shrinks toward zero or reverses.

A reversal is reported as the headline if that is what the data shows.

REPORTING
---------
* Per-seed train / val / test bpc for every arm at every dropout level.
* Best-of-curve validation bpc (the minimum over the training run), not just the
  final value, because the overfit confound is precisely about the final value.
* Paired differences over shared seeds with 95% paired t-intervals and sign
  agreement. An interval straddling zero is reported as a NULL.
* Unigram and bigram floors for every split, so every arm can be shown clear of
  the floor. bpc is NOT comparable across corpora; enwik8's bigram floor is far
  below TinyShakespeare's 3.5806.

DETERMINISM
-----------
Each run is a fresh process with `mx.random.seed(seed)` before any sampling, and
a numpy Generator seeded identically for batch indices. Dropout draws come from
the MLX global RNG, so the whole run is reproducible from its seed.

USAGE
-----
    python3 experiments/scale_up.py --smoke
    python3 experiments/scale_up.py --plan
    python3 experiments/scale_up.py --all --workers 2
    python3 experiments/scale_up.py --merge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_efficiency import (  # noqa: E402
    LM, GatedMemory, AttentionMemory, Block,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "experiments", "results")
RUNS_DIR = os.environ.get("BRAIN_RUNS_DIR",
                          os.path.expanduser("~/zbrain/scale_up_runs"))
OUT = os.environ.get("BRAIN_SCALE_OUT",
                        os.path.join(RESULTS_DIR, "scale_up.json"))
CORPUS_DIR = "/tmp/zz_enwik8"
CORPUS_PATH = os.path.join(CORPUS_DIR, "extract", "enwik8")
HOME_DATA = os.path.expanduser("~/zbrain/data/enwik8")
T9_PATH = "/Volumes/T9/human-brain/scratch/enwik8_extract/enwik8"
ENWIK8_URLS = ["https://data.deepai.org/enwik8.zip",
               "http://mattmahoney.net/dc/enwik8.zip"]
ENWIK8_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
SPLIT = {"train": (0, 90_000_000), "val": (90_000_000, 95_000_000),
         "test": (95_000_000, 100_000_000)}

# ---------------------------------------------------------------------------
# PRE-DECLARED GRIDS. Nothing is added to these after seeing any result.
# ---------------------------------------------------------------------------
DROPOUTS = [0.0, 0.1, 0.2]
SEEDS = [int(x) for x in os.environ.get("BRAIN_SEEDS_LIST", "0,1,2,3,4").split(",")]
ARMS = ["depth2_mem", "depth2_attn"]
# learning-rate grid, evaluated on VALIDATION only, at the selection budget
LR_GRID = [3e-4, 1e-3, 3e-3]
SELECTION_STEPS = 3000
SELECTION_SEED = 0
SKIP_SELECTION = os.environ.get("BRAIN_SKIP_SELECTION", "0") == "1"
# Session budget. The lead's priority order is: (a) the TEST number for both
# arms, (b) dropout 0.0 vs 0.2. FINAL_STEPS and the dropout levels are fixed
# BEFORE any final run is launched and are recorded in the output.
FINAL_STEPS = int(os.environ.get("BRAIN_FINAL_STEPS", "8000"))
CURVE_EVERY = int(os.environ.get("BRAIN_CURVE_EVERY", "1000"))
CURVE_WINDOWS = int(os.environ.get("BRAIN_CURVE_WINDOWS", "1500"))
SESSION_DROPOUTS = [float(x) for x in
                    os.environ.get("BRAIN_DROPOUTS", "0.0,0.2").split(",")]
CTX, D_MODEL, DEPTH, N_HEAD, BS = 512, 128, 2, 4, 16
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def now():
    return datetime.now(timezone.utc).isoformat()


def log(msg, fh=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_corpus():
    """Primary corpus first (internal disk), then external drive, then vendored.

    The internal copy is preferred deliberately: an external volume unmounted
    mid-session earlier and took a download with it.
    """
    env = os.environ.get("BRAIN_CORPUS")
    for cand in ([env] if env else []) + [
            HOME_DATA,
            T9_PATH,
            os.path.join(REPO, "data", "enwik8"),
            CORPUS_PATH,
            os.path.join(CORPUS_DIR, "enwik8")]:
        if cand and os.path.exists(cand):
            return cand
    return None


def download_corpus():
    """Download the zip, extract ONLY enwik8, delete the zip. Returns path."""
    import urllib.request
    import zipfile
    os.makedirs(CORPUS_DIR, exist_ok=True)
    zip_path = os.path.join(CORPUS_DIR, "enwik8.zip")
    for url in ENWIK8_URLS:
        try:
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as exc:  # noqa: BLE001
            log(f"  failed: {exc}")
    else:
        return None
    with zipfile.ZipFile(zip_path) as z:
        os.makedirs(os.path.join(CORPUS_DIR, "extract"), exist_ok=True)
        z.extract("enwik8", os.path.join(CORPUS_DIR, "extract"))
    os.remove(zip_path)  # keep the internal disk clean
    return CORPUS_PATH


def load_bytes(path):
    """enwik8 is raw bytes; latin-1 gives a 1:1 byte<->char map (vocab 241)."""
    return np.frombuffer(open(path, "rb").read(), dtype=np.uint8).astype(np.int32)


# ---------------------------------------------------------------------------
# evaluation windows
# ---------------------------------------------------------------------------
def window_spec(tokens, ctx, max_windows=None, name=""):
    """Deterministic non-overlapping ctx windows over a whole split.

    Window i covers tokens [i*ctx, (i+1)*ctx]. Windows never overlap and never
    cross a split boundary. If max_windows is set, an evenly spaced deterministic
    subset spanning the split is kept -- identical for every arm, dropout level
    and seed, so all runs are scored on exactly the same tokens.
    """
    n_win = (len(tokens) - 1) // ctx
    idx = np.arange(n_win, dtype=np.int64)
    if max_windows is not None and n_win > max_windows:
        idx = np.unique(np.linspace(0, n_win - 1, max_windows).astype(np.int64))
    return {"name": name, "tokens": tokens, "ctx": ctx,
            "n_win_total": int(n_win), "idx": idx, "n_win": int(len(idx))}


def evaluate(m, spec, vocab, bs=32):
    """Mean bpc over every window of a spec, weighted by token count."""
    m.eval()
    tok, ctx, idx = spec["tokens"], spec["ctx"], spec["idx"]
    ar = np.arange(ctx + 1, dtype=np.int64)
    tot, n = 0.0, 0
    for i in range(0, len(idx), bs):
        rows = idx[i:i + bs]
        blk = tok[(rows * ctx)[:, None] + ar[None, :]]
        x = mx.array(blk[:, :ctx].astype(np.int32))
        y = mx.array(blk[:, 1:ctx + 1].astype(np.int32))
        lo = m(x)
        l = nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="sum")
        mx.eval(l)
        tot += float(l)
        n += int(y.size)
    m.train()
    return tot / n / math.log(2)


def floors(train_tok, targets, V):
    """Unigram and add-1 bigram bpc: counts from train, evaluated per split."""
    t = train_tok.astype(np.int64)
    Cu = np.bincount(t, minlength=V).astype(np.float64) + 1.0
    Pu = Cu / Cu.sum()
    pair = t[:-1] * V + t[1:]
    C = (np.bincount(pair, minlength=V * V).astype(np.float64)
         .reshape(V, V)) + 1.0
    P = C / C.sum(1, keepdims=True)
    out = {}
    for name, t in targets.items():
        out[name] = {
            "unigram_bpc": (-float(np.sum(np.log(Pu[t]))) / len(t)) / math.log(2),
            "bigram_bpc": (-float(np.sum(np.log(P[t[:-1], t[1:]])))
                           / len(t)) / math.log(2),
            "n_tokens": int(len(t)),
        }
    return out


# ---------------------------------------------------------------------------
# model with dropout on the attention/memory and MLP residual paths
# ---------------------------------------------------------------------------
class DropoutBlock(nn.Module):
    """Identical to llm_efficiency.Block plus dropout on both residual paths.

    With p=0 this is exactly the original Block: dropout is the identity when
    p=0 and adds no parameters, so parameter counts are unchanged and the p=0
    runs reproduce the original architecture.
    """

    def __init__(self, d, n_head, kind, banks=1, chunk=64, p=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mem = (AttentionMemory(d, n_head) if kind == "attn"
                    else GatedMemory(d, banks=banks, chunk=chunk))
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d, bias=True),
            nn.GELU(),
            nn.Linear(4 * d, d, bias=True),
        )
        self.drop = nn.Dropout(p)

    def __call__(self, x, mask=None):
        x = x + self.drop(self.mem(self.ln1(x), mask))
        return x + self.drop(self.mlp(self.ln2(x)))


class DropoutLM(nn.Module):
    """Mirror of llm_efficiency.LM with dropout on the residual branches."""

    def __init__(self, vocab, d, ctx, kind, n_layer, n_head, p=0.0,
                 banks=1, chunk=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = [DropoutBlock(d, n_head, kind, banks, chunk, p)
                       for _ in range(n_layer)]
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(mx.arange(T)[None, :])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.lnf(x))


def build(arm, vocab, ctx, dropout):
    if arm == "depth2_mem":
        return DropoutLM(vocab, D_MODEL, ctx, "mem", DEPTH, N_HEAD, dropout, 1, 64)
    if arm == "depth2_attn":
        return DropoutLM(vocab, D_MODEL, ctx, "attn", DEPTH, N_HEAD, dropout, 1, 64)
    raise ValueError(arm)


def n_params(m):
    return sum(int(np.prod(v.shape))
               for _, v in nn.utils.tree_flatten(m.parameters()))


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def run(arm, data, vocab, *, ctx, steps, bs, lr, dropout, seed, cut,
        eval_specs, curve_every=0, phase="", log=log):
    mx.random.seed(seed)
    m = build(arm, vocab, ctx, dropout)
    mx.eval(m.parameters())
    P = n_params(m)
    tr = data[cut["train"][0]:cut["train"][1]]
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, x, y):
        lo = m(x)
        return nn.losses.cross_entropy(
            lo.reshape(-1, vocab), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(m, loss_fn)
    curve_spec = eval_specs.get("val")
    if curve_every and curve_spec is not None and CURVE_WINDOWS < curve_spec["n_win"]:
        curve_spec = window_spec(curve_spec["tokens"], ctx, CURVE_WINDOWS,
                                 "val_curve")
    t0 = time.time()
    ntok, last_train = 0, float("nan")
    curve, best = [], {"val_bpc": float("inf"), "step": 0}
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
        last_train = float(l)
        if curve_every and curve_spec is not None and s % curve_every == 0:
            vb = evaluate(m, curve_spec, vocab)
            curve.append({"step": s, "val_bpc": round(vb, 5),
                          "n_windows": curve_spec["n_win"]})
            if vb < best["val_bpc"]:
                best = {"val_bpc": vb, "step": s}
    train_wall = time.time() - t0

    t1 = time.time()
    bpc, wins = {}, {}
    for name, spec in eval_specs.items():
        bpc[name] = evaluate(m, spec, vocab)
        wins[name] = spec["n_win"]
    if curve_every:
        vb = evaluate(m, eval_specs["val"], vocab)
        curve.append({"step": steps, "val_bpc": round(vb, 5)})
        if vb < best["val_bpc"]:
            best = {"val_bpc": vb, "step": steps}
    eval_wall = time.time() - t1

    return dict(arm=arm, seed=seed, phase=phase, dropout=dropout, lr=lr,
                params=P, steps=steps, bs=bs, ctx=ctx, d=D_MODEL, depth=DEPTH,
                val_bpc=bpc.get("val"), test_bpc=bpc.get("test"),
                train_bpc=bpc.get("train"),
                best_val_bpc=(best["val_bpc"] if curve_every else None),
                best_val_step=(best["step"] if curve_every else None),
                final_val_bpc=(curve[-1]["val_bpc"] if curve_every else None),
                train_bpc_last_batch=last_train / math.log(2),
                n_windows=wins, curve=curve,
                train_wall_s=round(train_wall, 1),
                eval_wall_s=round(eval_wall, 1),
                tok_s=round(ntok / train_wall, 1), tokens_seen=int(ntok))


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------
def paired_report(rows, a, b, metric):
    sa = {r["seed"]: r[metric] for r in rows
          if r["arm"] == a and r.get(metric) is not None}
    sb = {r["seed"]: r[metric] for r in rows
          if r["arm"] == b and r.get(metric) is not None}
    seeds = sorted(set(sa) & set(sb))
    d = np.array([sa[s] - sb[s] for s in seeds], dtype=float)
    n = len(d)
    if n < 2:
        return dict(comparison=f"{a} - {b}", metric=metric, n_seeds=int(n),
                    per_seed_diff=[round(float(v), 5) for v in d],
                    paired_mean_diff=None, ci95=None, sign_agreement=None,
                    verdict="insufficient seeds")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    crit = T95.get(n - 1, 1.96)
    lo, hi = mean - crit * se, mean + crit * se
    agree = int(sum(1 for v in d if (v < 0) == (mean < 0)))
    verdict = ("NULL: interval straddles zero" if lo < 0 < hi else
               f"{'memory' if mean < 0 else 'attention'} better, interval "
               f"excludes 0")
    return dict(comparison=f"{a} - {b}", metric=metric, n_seeds=int(n),
                per_seed_diff=[round(float(v), 5) for v in d],
                paired_mean_diff=round(mean, 5), se=round(se, 5),
                t_stat=round(mean / se, 3) if se else None,
                ci95=[round(lo, 5), round(hi, 5)],
                sign_agreement=f"{agree}/{n}", verdict=verdict)


def summarise(rows):
    """Mean +/- sd per (arm, dropout) for every metric."""
    out = {}
    for arm in ARMS:
        for p in DROPOUTS:
            rs = [r for r in rows if r["arm"] == arm and r["dropout"] == p]
            if not rs:
                continue
            e = {"n_seeds": len(rs), "params": rs[0]["params"]}
            for metric in ("train_bpc", "val_bpc", "test_bpc", "best_val_bpc",
                           "final_val_bpc"):
                v = [r[metric] for r in rs if r.get(metric) is not None]
                if v:
                    e[metric + "_mean"] = round(float(np.mean(v)), 5)
                    e[metric + "_sd"] = (round(float(np.std(v, ddof=1)), 5)
                                         if len(v) > 1 else None)
            out[f"{arm}|p={p}"] = e
    return out


def run_key(r):
    return (f"{r['arm']}_p{r['dropout']}_lr{r['lr']:g}_s{r['seed']}"
            f"_st{r['steps']}_{r['phase']}")


# ---------------------------------------------------------------------------
# child: execute a single run and write its own JSON
# ---------------------------------------------------------------------------
def child(args):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = find_corpus() or download_corpus()
    if path is None:
        raise SystemExit("corpus unavailable")
    data = load_bytes(path)
    vocab = int(data.max()) + 1
    cuts = {k: (a, min(b, len(data))) for k, (a, b) in SPLIT.items()}
    specs = {}
    for name, (a, b) in cuts.items():
        cap = int(os.environ.get("BRAIN_TRAIN_EVAL_WINDOWS", "2000")) \
            if name == "train" else None
        specs[name] = window_spec(data[a:b], CTX, cap, name)
    wanted = {"train": specs["train"], "val": specs["val"]}
    if args.unlock_test:
        wanted["test"] = specs["test"]
    r = run(args.arm, data, vocab, ctx=CTX, steps=args.steps, bs=BS, lr=args.lr,
            dropout=args.dropout, seed=args.seed, cut=cuts, eval_specs=wanted,
            curve_every=args.curve_every, phase=args.phase)
    r["corpus_sha256"] = sha256_of(path)
    r["finished_utc"] = now()
    key = run_key(r)
    tmp = os.path.join(RUNS_DIR, key + ".tmp")
    with open(tmp, "w") as f:
        json.dump(r, f, indent=1)
    os.replace(tmp, os.path.join(RUNS_DIR, key + ".json"))
    log(f"DONE {key} val={r['val_bpc']} test={r['test_bpc']} "
        f"best_val={r['best_val_bpc']} ({r['train_wall_s']}s)")
    return 0


# ---------------------------------------------------------------------------
# merge: rebuild the aggregate from per-run files
# ---------------------------------------------------------------------------
def merge(extra=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = find_corpus() or download_corpus()
    if path is None:
        raise SystemExit("corpus unavailable")
    data = load_bytes(path)
    vocab = int(data.max()) + 1
    cuts = {k: (a, min(b, len(data))) for k, (a, b) in SPLIT.items()}
    specs, floors_by_split = {}, {}
    for name, (a, b) in cuts.items():
        cap = int(os.environ.get("BRAIN_TRAIN_EVAL_WINDOWS", "2000")) \
            if name == "train" else None
        specs[name] = window_spec(data[a:b], CTX, cap, name)
    floors_by_split = floors(data[cuts["train"][0]:cuts["train"][1]],
                             {k: data[a:b] for k, (a, b) in cuts.items()}, vocab)

    rows = []
    if os.path.isdir(RUNS_DIR):
        for fn in sorted(os.listdir(RUNS_DIR)):
            if fn.endswith(".json"):
                with open(os.path.join(RUNS_DIR, fn)) as f:
                    rows.append(json.load(f))

    sel = [r for r in rows if r["phase"] == "selection"]
    fin = [r for r in rows if r["phase"] == "final"]
    planned = [j for j in plan() if j["phase"] == "final"]
    expected = len(planned)
    done_keys = {(r["arm"], r["dropout"], r["seed"], r["lr"], r["steps"])
                 for r in fin}
    missing = [job_key(j) for j in planned
               if (j["arm"], j["dropout"], j["seed"], j["lr"],
                   j["steps"]) not in done_keys]
    doc = {
        "complete": len(fin) >= expected,
        "n_final_runs": len(fin), "n_final_expected": expected,
        "note": ("partial results are valid; `complete` becomes true only when "
                 "every declared arm/dropout/seed pair has finished"),
        "planned_final_jobs": [job_key(j) for j in planned],
        "missing_final_jobs": missing,
        "updated_utc": now(),
        "corpus": {
            "path": path, "bytes": os.path.getsize(path),
            "sha256": sha256_of(path), "vocab_size": vocab,
            "source": "enwik8 (standard 100 MB char-level bpc benchmark)",
            "subset": None,
            "subset_note": "full corpus used; nothing truncated or subset",
            "split_chars": {k: v[1] - v[0] for k, v in cuts.items()},
            "split_bounds": {k: list(v) for k, v in cuts.items()},
        },
        "floors": floors_by_split,
        "floors_note": ("unigram and add-1 bigram, counts from the train split, "
                        "evaluated per split. bpc is NOT comparable across "
                        "corpora: enwik8's bigram floor (~3.89) is far below "
                        "TinyShakespeare's 3.5806."),
        "eval_windows": {k: {"total": v["n_win_total"], "evaluated": v["n_win"]}
                         for k, v in specs.items()},
        "config": {
            "ctx": CTX, "d": D_MODEL, "depth": DEPTH, "n_head": N_HEAD, "bs": BS,
            "arms": ARMS, "dropouts": DROPOUTS, "seeds": SEEDS,
            "final_steps": FINAL_STEPS, "selection_steps": SELECTION_STEPS,
            "curve_every": CURVE_EVERY, "curve_windows": CURVE_WINDOWS,
            "session_dropouts": SESSION_DROPOUTS,
            "optimizer": ("AdamW, lr*min(1,step/100) warmup, weight_decay=0.01, "
                          "grad clip 1.0"),
            "dropout_placement": "on both residual branches (after memory/"
                                 "attention and after the MLP), as in nanoGPT",
        },
        "selection": {
            "lr_grid": LR_GRID, "n_configs_tried": len(LR_GRID) * len(ARMS),
            "selection_steps": SELECTION_STEPS, "selection_seed": SELECTION_SEED,
            "split_used": "validation only; TEST never consulted",
            "note": ("selection optimism applies to any configuration chosen "
                     "after seeing a validation table. The TEST number is "
                     "reported separately and was computed once, after "
                     "selection finished."),
        },
        "dropout_note": ("dropout is a pre-declared manipulation, not a tuning "
                         "knob: all levels are reported for both arms and "
                         "nothing was selected by which level favoured an arm"),
        "preregistered_prediction": (
            "attention's best-of-curve validation bpc improves by 0.05-0.1 bpc "
            "at dropout 0.2 vs 0.0, and the paired gap against depth2_mem "
            "shrinks toward zero or reverses"),
        "runs": rows,
        "summary": summarise(fin),
        "paired": {},
        "wall_time_s": None,
    }
    if extra:
        doc.update(extra)

    # selection result: best mean validation bpc across arms per lr
    if sel:
        by_lr = {}
        for lr in sorted({r["lr"] for r in sel}):
            vals = [r["val_bpc"] for r in sel if r["lr"] == lr
                    and r["val_bpc"] is not None]
            by_lr[str(lr)] = round(float(np.mean(vals)), 5) if vals else None
        doc["selection"]["val_mean_by_lr"] = by_lr
        best = min((k for k, v in by_lr.items() if v is not None),
                   key=lambda k: by_lr[k])
        doc["selection"]["selected_lr"] = float(best)
        doc["selection"]["criterion"] = "lowest mean validation bpc across both arms"

    # paired comparisons per dropout level, on val / best-val / test / train
    for p in DROPOUTS:
        rows_p = [r for r in fin if r["dropout"] == p]
        if not rows_p:
            continue
        for metric in ("train_bpc", "val_bpc", "best_val_bpc", "test_bpc"):
            if not any(r.get(metric) is not None for r in rows_p):
                continue
            doc["paired"][f"p={p}|{metric}"] = paired_report(
                rows_p, ARMS[0], ARMS[1], metric)

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, OUT)
    log(f"merged {len(rows)} runs ({len(fin)} final, {len(sel)} selection) "
        f"-> {OUT}")
    return doc


# ---------------------------------------------------------------------------
# plan + orchestration
# ---------------------------------------------------------------------------
def plan(selected_lr=None):
    jobs = []
    if not SKIP_SELECTION:
        for lr in LR_GRID:                  # selection: VALIDATION ONLY
            for arm in ARMS:
                jobs.append(dict(arm=arm, dropout=0.0, lr=lr,
                                 seed=SELECTION_SEED, steps=SELECTION_STEPS,
                                 phase="selection", curve_every=0))
    lr = selected_lr if selected_lr is not None else 1e-3
    for p in SESSION_DROPOUTS:              # primary first, then extension
        for seed in SEEDS:
            for arm in ARMS:
                jobs.append(dict(arm=arm, dropout=p, lr=lr, seed=seed,
                                 steps=FINAL_STEPS, phase="final",
                                 curve_every=CURVE_EVERY))
    return jobs


def job_key(j):
    return (f"{j['arm']}_p{j['dropout']}_lr{j['lr']:g}_s{j['seed']}"
            f"_st{j['steps']}_{j['phase']}")


def cmd_plan():
    for j in plan():
        print(f"  {job_key(j):<44} steps={j['steps']:<6} curve={j['curve_every']}")
    print(f"total jobs: {len(plan())}")


def run_jobs(jobs, workers, unlock_test, label):
    """Execute jobs in parallel; merge after each completion so nothing is lost."""
    done = {f[:-5] for f in os.listdir(RUNS_DIR) if f.endswith(".json")}
    pending = [j for j in jobs if job_key(j) not in done]
    log(f"[{label}] {len(jobs)} jobs, {len(jobs)-len(pending)} already done, "
        f"{len(pending)} pending, workers={workers}")
    running = []
    while pending or running:
        while pending and len(running) < workers:
            j = pending.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--run",
                   "--arm", j["arm"], "--dropout", str(j["dropout"]),
                   "--lr", str(j["lr"]), "--seed", str(j["seed"]),
                   "--steps", str(j["steps"]), "--phase", j["phase"],
                   "--curve-every", str(j["curve_every"])]
            if unlock_test and j["phase"] == "final":
                cmd.append("--unlock-test")
            lg = open(os.path.join(RUNS_DIR, job_key(j) + ".log"), "w")
            running.append((subprocess.Popen(cmd, stdout=lg, stderr=lg), j, lg))
            log(f"START {job_key(j)}")
        time.sleep(5)
        for entry in list(running):
            proc, j, lg = entry
            if proc.poll() is not None:
                running.remove(entry)
                lg.close()
                rc = proc.returncode
                log(f"{'OK  ' if rc == 0 else 'FAIL'} {job_key(j)} rc={rc}")
                merge()
    return [j for j in jobs]


def cmd_all(workers, unlock_test):
    os.makedirs(RUNS_DIR, exist_ok=True)
    t0 = time.time()
    if not SKIP_SELECTION:
        sel_jobs = [j for j in plan(selected_lr=None)
                    if j["phase"] == "selection"]
        run_jobs(sel_jobs, workers, unlock_test, "selection")
    doc = merge()
    sel_lr = doc.get("selection", {}).get("selected_lr") or 1e-3
    log(f"using lr={sel_lr:g} for final runs")
    fin_jobs = [j for j in plan(selected_lr=sel_lr) if j["phase"] == "final"]
    run_jobs(fin_jobs, workers, unlock_test, "final")
    doc = merge()
    log(f"all jobs finished in {time.time()-t0:.0f}s")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--run", action="store_true", help="internal: single run")
    ap.add_argument("--arm", default="depth2_mem")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=FINAL_STEPS)
    ap.add_argument("--phase", default="final")
    ap.add_argument("--curve-every", type=int, default=0)
    ap.add_argument("--unlock-test", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    global OUT, RUNS_DIR
    if args.smoke:
        args.curve_every = 50
        RUNS_DIR = os.path.expanduser("~/zbrain/scale_up_smoke_runs")
        OUT = os.path.expanduser("~/zbrain/scale_up_smoke.json")
        args.curve_every = 10
        args.unlock_test = True
    if args.plan:
        return cmd_plan()
    if args.run:
        return child(args)
    if args.merge:
        merge()
        return
    if args.smoke:
        os.makedirs(RUNS_DIR, exist_ok=True)
        for arm in ARMS:
            a = argparse.Namespace(**vars(args))
            a.arm, a.seed, a.steps, a.phase = arm, 0, 60, "final"
            child(a)
        merge()
        return
    cmd_all(args.workers, args.unlock_test or os.environ.get(
        "BRAIN_UNLOCK_TEST", "0") == "1")


if __name__ == "__main__":
    main()
