"""A fit-error sign reversal must never produce a supported gate verdict."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from content_gate import recency_verdict  # noqa: E402


def result(input_corr, state_corr, *, steps=800, attention_bpc=2.3):
    return {
        "corr_cdf_l1_trained_vs_bpc_B_plain": input_corr,
        "corr_cdf_l1_trained_vs_bpc_C_statedep": state_corr,
        "steps": steps,
        "seeds": [0, 1],
        "decays": [0.5, 0.7, 0.9],
        "attn_bpc": attention_bpc,
        "meta": {"bigram_bpc": 3.5806},
    }


def test_negative_fit_error_correlation_is_the_anti_correlation():
    verdict = recency_verdict(result(-0.8, 0.2))
    assert verdict["valid_run"]
    assert verdict["anti_correlation_present_input_only"]
    assert not verdict["anti_correlation_present_state_dep"]
    assert verdict["explains_anti_correlation"]


def test_positive_fit_error_correlation_means_fit_helps():
    verdict = recency_verdict(result(0.8, -0.5))
    assert verdict["valid_run"]
    assert not verdict["explains_anti_correlation"]
    assert "REVERSED" in verdict["statement"]


def test_smoke_or_floor_failing_run_cannot_support_hypothesis():
    for values in (result(-1, 1, steps=1),
                   result(-1, 1, attention_bpc=6.09)):
        verdict = recency_verdict(values)
        assert not verdict["valid_run"]
        assert not verdict["explains_anti_correlation"]
        assert "INVALID" in verdict["statement"]
