"""Read-only audit of the LoRA and content-gate control artifacts.

This script uses only the Python standard library. It does not import MLX, load
a model, train, or modify the result files. Differences are paired by seed and
learning-rate banks are never pooled.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
T975 = {
    1: 12.7062047364,
    2: 4.30265272991,
    3: 3.18244630528,
    4: 2.77644510520,
    5: 2.57058183564,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.30600413520,
    9: 2.26215716285,
    10: 2.22813885196,
}


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation, equivalent to NumPy's default percentile."""
    position = (len(sorted_values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * (position - lower)


def paired_stats(values: list[float]) -> dict:
    """Student t interval and exhaustive paired-bootstrap percentile interval."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "per_seed": [], "t95": None,
                "bootstrap95": None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    t95 = None
    if sd is not None and n - 1 in T975:
        half_width = T975[n - 1] * sd / math.sqrt(n)
        t95 = [mean - half_width, mean + half_width]
    bootstrap95 = None
    if 2 <= n <= 6:
        # Resample paired seed differences, rather than individual arm rows.
        means = sorted(
            statistics.mean(values[index] for index in sample)
            for sample in itertools.product(range(n), repeat=n)
        )
        bootstrap95 = [percentile(means, 0.025), percentile(means, 0.975)]
    return {
        "n": n, "mean": mean, "sd": sd, "per_seed": values,
        "t95": t95, "bootstrap95": bootstrap95,
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
    }


def indexed_rows(rows: list[dict], key_fields: tuple[str, ...]) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError(f"duplicate result row for {key_fields}: {key}")
        result[key] = row
    return result


def summarize_lora(data: dict) -> dict:
    expected_seeds = sorted(set(data["seeds"]))
    expected_arms = list(data["arms"])
    rows = indexed_rows(data["runs"], ("lr", "seed", "arm"))
    observed_lrs = sorted({key[0] for key in rows})
    declared_lrs = sorted(data.get("lrs", [data.get("lr")]))
    banks = []
    for lr in observed_lrs:
        missing = [
            {"seed": seed, "arm": arm}
            for seed in expected_seeds for arm in expected_arms
            if (lr, seed, arm) not in rows
        ]
        bank = {
            "lr": lr, "expected_rows": len(expected_seeds) * len(expected_arms),
            "observed_rows": sum(key[0] == lr for key in rows),
            "complete": not missing, "missing": missing,
            "arm_means": {}, "comparisons": {},
        }
        for metric, baseline in (
            ("after_val_ppl", data["teacher_val_ppl"]),
            ("after_val_ppl_hp", data["teacher_val_ppl_hp"]),
        ):
            means = {}
            for arm in expected_arms:
                values = [rows[(lr, seed, arm)][metric]
                          for seed in expected_seeds
                          if (lr, seed, arm) in rows]
                if values:
                    means[arm] = {
                        "n": len(values), "after": statistics.mean(values),
                        "gain_vs_teacher": baseline - statistics.mean(values),
                    }
            bank["arm_means"][metric] = means
            for a, b in (("T", "L1"), ("T", "L2"), ("T", "R")):
                paired_seeds = [seed for seed in expected_seeds
                                if (lr, seed, a) in rows
                                and (lr, seed, b) in rows]
                values = [rows[(lr, seed, b)][metric]
                          - rows[(lr, seed, a)][metric]
                          for seed in paired_seeds]
                stats = paired_stats(values)
                stats["seeds"] = paired_seeds
                stats["definition"] = f"{b} - {a}; positive means {a} is better"
                bank["comparisons"].setdefault(f"{a}_vs_{b}", {})[metric] = stats
        banks.append(bank)
    return {
        "teacher": data["teacher"], "expected_seeds": expected_seeds,
        "expected_arms": expected_arms, "declared_lrs": declared_lrs,
        "observed_lrs": observed_lrs,
        "metadata_lrs_match": declared_lrs == observed_lrs,
        "banks": banks,
    }


def summarize_gate(data: dict) -> dict:
    expected_seeds = sorted(set(data["meta"]["seeds"]))
    expected_arms = list(data["param_table"])
    rows = indexed_rows(
        [row for arm_rows in data["arms"].values() for row in arm_rows],
        ("arm", "seed"),
    )
    missing = {
        arm: [seed for seed in expected_seeds if (arm, seed) not in rows]
        for arm in expected_arms
    }
    arm_means = {
        arm: statistics.mean(rows[(arm, seed)]["bpc"]
                             for seed in expected_seeds
                             if (arm, seed) in rows)
        for arm in expected_arms if any((arm, seed) in rows
                                        for seed in expected_seeds)
    }
    comparisons = {}
    for a, b in (("C_statedep", "B_match"),
                 ("B_match", "B_plain")):
        paired_seeds = [seed for seed in expected_seeds
                        if (a, seed) in rows and (b, seed) in rows]
        stats = paired_stats([
            rows[(a, seed)]["bpc"] - rows[(b, seed)]["bpc"]
            for seed in paired_seeds
        ])
        stats["seeds"] = paired_seeds
        stats["definition"] = f"{a} - {b}; negative means {a} is better"
        comparisons[f"{a}_vs_{b}"] = stats
    floor = data["meta"]["bigram_bpc"]
    attention = arm_means.get("A_attn")
    matched = data.get("matched_set", {})
    return {
        "expected_seeds": expected_seeds, "expected_arms": expected_arms,
        "missing_seeds": missing, "complete": all(not x for x in missing.values()),
        "arm_means_bpc": arm_means, "comparisons": comparisons,
        "bigram_floor_bpc": floor, "attention_clears_floor":
            attention is not None and attention < floor,
        "matched_memory_params": bool(matched.get("exact_params")),
        "matched_memory_macs": bool(matched.get("exact_macs")),
        "attention_macs_ratio_vs_C": matched.get("attention_macs_ratio_vs_C"),
        "timing_recorded": bool(data.get("timing")),
        "recency_recorded": bool(data.get("recency")),
        "load_avg_at_start": data["meta"].get("load_avg_at_start"),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cross = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cross / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None


def summarize_recency(data: dict) -> dict:
    rows = data["rows"]
    # Prefer the trained gate's measured fit when that field exists. The current
    # artifact predates it, so its correlations describe gate-bias initialization.
    has_trained_fit = all(
        all("cdf_l1_trained" in row["arms"][arm]
            for arm in ("B_plain", "C_statedep"))
        for row in rows
    )
    basis = "trained_gate" if has_trained_fit else "initial_decay"
    correlations = {}
    recorded = {}
    for arm in ("B_plain", "C_statedep"):
        fits = [row["arms"][arm]["cdf_l1_trained"]
                if has_trained_fit else row["cdf_l1"] for row in rows]
        bpcs = [row["arms"][arm]["bpc_mean"] for row in rows]
        correlations[arm] = pearson(fits, bpcs)
        key = (f"corr_cdf_l1_trained_vs_bpc_{arm}" if has_trained_fit
               else f"corr_cdf_l1_vs_bpc_{arm}")
        recorded[arm] = data.get(key)
    # Both quantities are errors: lower fit error and lower bpc are better.
    # Positive correlation => better fit helps; negative => better fit hurts.
    rb, rc = correlations["B_plain"], correlations["C_statedep"]
    anti_b = rb is not None and rb < -0.3
    anti_c = rc is not None and rc < -0.3
    reported = data.get("verdict", {})
    floor = data.get("meta", {}).get("bigram_bpc")
    attention = data.get("attn_bpc")
    smoke = (data["steps"] <= 1 or len(data["seeds"]) < 2
             or len(data["decays"]) < 3)
    return {
        "steps": data["steps"], "seeds": data["seeds"],
        "decays": data["decays"], "fit_basis": basis,
        "correlations_recomputed": correlations,
        "correlations_recorded": recorded,
        "correlation_sign": "positive: better fit helps; negative: better fit hurts",
        "correct_anti_input_only": anti_b,
        "correct_anti_state_dependent": anti_c,
        "correct_explains_anti_correlation": anti_b and not anti_c,
        "reported_verdict": reported,
        "reported_verdict_present": bool(reported),
        "reported_sign_is_wrong": bool(reported) and (
            reported.get("anti_correlation_present_input_only") != anti_b
            or reported.get("anti_correlation_present_state_dep") != anti_c
            or reported.get("explains_anti_correlation") != (anti_b and not anti_c)
        ),
        "attention_bpc": attention, "bigram_floor_bpc": floor,
        "attention_clears_floor": (attention is not None and floor is not None
                                    and attention < floor),
        "smoke_run": smoke,
        "correlation_degenerate_two_decays": len(data["decays"]) == 2,
    }


def format_interval(interval: list[float] | None) -> str:
    return "unavailable" if interval is None else (
        f"[{interval[0]:+.6f}, {interval[1]:+.6f}]"
    )


def format_number(value: float | None, *, signed: bool = True) -> str:
    if value is None:
        return "unavailable"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def print_report(report: dict) -> None:
    lora = report["lora"]
    print("LoRA control: positive paired difference means T beats the comparator")
    print(f"  declared LRs {lora['declared_lrs']} | observed LRs "
          f"{lora['observed_lrs']} | metadata match "
          f"{lora['metadata_lrs_match']}")
    for bank in lora["banks"]:
        print(f"  lr={bank['lr']:g}: {bank['observed_rows']}/"
              f"{bank['expected_rows']} rows, complete={bank['complete']}")
        if bank["missing"]:
            print(f"    missing {bank['missing']}")
        means = bank["arm_means"]["after_val_ppl_hp"]
        print("    float32 gains vs teacher: " + ", ".join(
            f"{arm}={means[arm]['gain_vs_teacher']:+.6f}"
            for arm in lora["expected_arms"] if arm in means
        ))
        for name in ("T_vs_L1", "T_vs_L2", "T_vs_R"):
            stats = bank["comparisons"][name]["after_val_ppl_hp"]
            print(f"    {stats['definition']}: n={stats['n']} "
                  f"mean={format_number(stats['mean'])} "
                  f"t95={format_interval(stats['t95'])} "
                  f"bootstrap95={format_interval(stats['bootstrap95'])}")
            print(f"      seeds {stats['seeds']} | differences "
                  f"{[round(x, 6) for x in stats['per_seed']]}")
        repo = bank["comparisons"]["T_vs_L1"]["after_val_ppl"]
        print(f"    repo metric L1 - T: {format_number(repo['mean'])}, "
              f"t95={format_interval(repo['t95'])}, "
              f"bootstrap95={format_interval(repo['bootstrap95'])}")

    gate = report["gate"]
    print("Gate control: negative C - B_match means state dependence helps")
    print(f"  complete={gate['complete']} | missing seeds "
          f"{gate['missing_seeds']}")
    print(f"  attention {format_number(gate['arm_means_bpc'].get('A_attn'), signed=False)} bpc "
          f"vs bigram floor {gate['bigram_floor_bpc']:.6f}; "
          f"clears={gate['attention_clears_floor']}")
    for name in ("C_statedep_vs_B_match", "B_match_vs_B_plain"):
        stats = gate["comparisons"][name]
        print(f"  {stats['definition']}: n={stats['n']} "
              f"mean={format_number(stats['mean'])} "
              f"t95={format_interval(stats['t95'])} "
              f"bootstrap95={format_interval(stats['bootstrap95'])}")
        print(f"    seeds {stats['seeds']} | differences "
              f"{[round(x, 6) for x in stats['per_seed']]}")
    print(f"  memory params/MACs matched="
          f"{gate['matched_memory_params']}/{gate['matched_memory_macs']}; "
          f"attention MAC ratio="
          f"{format_number(gate['attention_macs_ratio_vs_C'], signed=False)}")
    print(f"  dedicated timing/recency present="
          f"{gate['timing_recorded']}/{gate['recency_recorded']}; "
          f"start load average={gate['load_avg_at_start']}")

    recency = report["recency"]
    print("Recency check: positive corr(fit error, bpc) means better fit helps")
    print(f"  {recency['steps']} step(s), {len(recency['seeds'])} seed(s), "
          f"{len(recency['decays'])} decay(s); smoke={recency['smoke_run']}; "
          f"fit basis={recency['fit_basis']}")
    print(f"  attention {format_number(recency['attention_bpc'], signed=False)} "
          f"bpc vs bigram floor "
          f"{format_number(recency['bigram_floor_bpc'], signed=False)}; "
          f"clears={recency['attention_clears_floor']}")
    print(f"  recomputed r: "
          f"B={format_number(recency['correlations_recomputed']['B_plain'])}, "
          f"C={format_number(recency['correlations_recomputed']['C_statedep'])}; "
          f"two-decay degeneracy={recency['correlation_degenerate_two_decays']}")
    print(f"  correct anti-correlation: B="
          f"{recency['correct_anti_input_only']}, "
          f"C={recency['correct_anti_state_dependent']}; "
          f"reported verdict sign wrong={recency['reported_sign_is_wrong']}")
    print("  claim status: incomplete gate run; recency verdict invalid; "
          "no compute-reduction claim established by these controls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--json", action="store_true",
                        help="print the audit as JSON instead of readable text")
    args = parser.parse_args()
    report = {
        "lora": summarize_lora(read_json(args.results_dir
                                    / "hybrid_lora_control.json")),
        "gate": summarize_gate(read_json(args.results_dir
                                    / "content_gate.json")),
        "recency": summarize_recency(read_json(args.results_dir
                                          / "content_gate_recency.json")),
    }
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
