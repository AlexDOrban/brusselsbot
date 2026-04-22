import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_decision import GATES, gate_pass


def test_exact_tie_passes_all_gates():
    agg = {"hit_at_1": 0.889, "hit_at_3": 1.000, "mrr": 0.944}
    for key, threshold in GATES.items():
        assert gate_pass(agg, key, threshold), (
            f"exact tie on {key} ({agg[key]} vs {threshold}) must pass"
        )


def test_one_thousandth_below_each_threshold_fails():
    for key, threshold in GATES.items():
        agg = {"hit_at_1": 1.0, "hit_at_3": 1.0, "mrr": 1.0}
        agg[key] = threshold - 0.001
        assert not gate_pass(agg, key, threshold), (
            f"{key}={agg[key]:.4f} is 0.001 below threshold {threshold} and must fail"
        )


def test_eight_ninths_counts_as_tie_with_0_889():
    """Baseline measures 8/9 in-scope hits. Displayed as 0.889; must be treated
    as tying a 0.889 threshold, not failing it by floating-point precision."""
    agg = {"hit_at_1": 8 / 9, "hit_at_3": 1.000, "mrr": 0.944}
    assert gate_pass(agg, "hit_at_1", 0.889)
