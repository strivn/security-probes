"""Tests for paired win-rate statistics (Wilson CI + exact-binomial sign test)."""

from __future__ import annotations

import math

import pytest
from scipy.stats import binomtest

from agentic_sec_probe.paired_stats import (
    mcnemar_paired,
    paired_win_rate,
    wilson_interval,
)


class TestWilsonInterval:
    def test_zero_n_returns_zero_width(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_bounds_within_unit_interval(self) -> None:
        lo, hi = wilson_interval(0, 10)
        assert lo == 0.0  # lower bound clamped at 0
        assert 0.0 < hi < 1.0  # Wilson does NOT collapse to zero width at k=0
        lo, hi = wilson_interval(10, 10)
        assert hi == 1.0
        assert 0.0 < lo < 1.0

    def test_center_brackets_point_estimate(self) -> None:
        # For a non-degenerate proportion the interval brackets k/n.
        lo, hi = wilson_interval(150, 235)
        assert lo < 150 / 235 < hi

    def test_known_value_95pct(self) -> None:
        # Reference: Wilson 95% CI for 150/235 = (0.5751, 0.6971), verified
        # independently against an independent from-scratch closed form.
        lo, hi = wilson_interval(150, 235, confidence=0.95)
        assert lo == pytest.approx(0.5751, abs=1e-3)
        assert hi == pytest.approx(0.6971, abs=1e-3)

    def test_wider_at_higher_confidence(self) -> None:
        lo95, hi95 = wilson_interval(60, 100, confidence=0.95)
        lo99, hi99 = wilson_interval(60, 100, confidence=0.99)
        assert (hi99 - lo99) > (hi95 - lo95)


class TestPairedWinRate:
    def test_counts_wins_losses_ties(self) -> None:
        gaps = [0.3, -0.1, 0.0, 0.5, -0.2, 0.7, 0.0]  # wins: .3,.5,.7  losses: -.1,-.2
        r = paired_win_rate(gaps)
        assert r.wins == 3
        assert r.losses == 2
        assert r.ties == 2
        assert r.n_pairs == 7
        assert r.n_effective == 5  # ties dropped

    def test_ties_dropped_from_rate(self) -> None:
        # 3 wins, 1 loss, 6 ties -> rate is 3/4, NOT 3/10.
        gaps = [0.1, 0.1, 0.1, -0.1] + [0.0] * 6
        r = paired_win_rate(gaps)
        assert r.win_rate == pytest.approx(0.75)
        assert r.ties == 6

    def test_tie_epsilon(self) -> None:
        gaps = [1e-13, -1e-13, 0.5, -0.5]
        # With tie_eps below the magnitudes, the tiny gaps count as win/loss.
        r0 = paired_win_rate(gaps, tie_eps=0.0)
        assert r0.ties == 0
        # With a larger epsilon, the tiny gaps become ties.
        r1 = paired_win_rate(gaps, tie_eps=1e-12)
        assert r1.ties == 2
        assert r1.n_effective == 2

    def test_empty_input(self) -> None:
        r = paired_win_rate([])
        assert r.n_pairs == 0
        assert r.n_effective == 0
        assert r.win_rate == 0.0
        assert r.ci_lower == 0.0
        assert r.ci_upper == 0.0
        assert r.p_value == 1.0

    def test_all_wins(self) -> None:
        r = paired_win_rate([0.5] * 20)
        assert r.win_rate == 1.0
        assert r.ci_upper == 1.0
        assert r.ci_lower > 0.0  # Wilson lower bound is not 0 at k=n
        assert r.p_value < 1e-4  # 20/20 wins is significant vs 0.5

    def test_p_value_matches_scipy_binomtest(self) -> None:
        # 150 wins, 85 losses (235 effective). Sign test == exact binomial.
        gaps = [0.1] * 150 + [-0.1] * 85
        r = paired_win_rate(gaps)
        expected = float(binomtest(150, 235, p=0.5, alternative="two-sided").pvalue)
        assert r.p_value == pytest.approx(expected)
        assert r.win_rate == pytest.approx(150 / 235)

    def test_chance_level_not_significant(self) -> None:
        # Exactly balanced -> p-value == 1.0 (cannot reject 0.5).
        r = paired_win_rate([0.1] * 50 + [-0.1] * 50)
        assert r.win_rate == pytest.approx(0.5)
        assert r.p_value == pytest.approx(1.0)

    def test_as_dict_rounds_and_serializes(self) -> None:
        r = paired_win_rate([0.1] * 150 + [-0.1] * 85)
        d = r.as_dict()
        assert d["wins"] == 150
        assert d["losses"] == 85
        assert isinstance(d["win_rate"], float)
        # Rounded to 4 dp.
        assert d["win_rate"] == round(150 / 235, 4)
        assert not math.isnan(d["ci_lower"])


class TestMcNemarPaired:
    def test_contingency_cells(self) -> None:
        # A: T T T F F ; B: T F F T F
        # a(both T)=1, b(A only)=2, c(B only)=1, d(both F)=1
        a = [True, True, True, False, False]
        b = [True, False, False, True, False]
        r = mcnemar_paired(a, b)
        assert r.n == 5
        assert r.n_both_correct == 1
        assert r.n_a_only == 2
        assert r.n_b_only == 1
        assert r.n_both_wrong == 1
        assert r.n_discordant == 3

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            mcnemar_paired([True, False], [True])

    def test_exact_used_below_threshold(self) -> None:
        # b+c = 10 < 25 -> exact binomial.
        a = [True] * 8 + [False] * 2 + [True] * 5
        b = [False] * 8 + [True] * 2 + [True] * 5
        r = mcnemar_paired(a, b)
        assert r.exact is True
        assert r.n_a_only == 8
        assert r.n_b_only == 2
        # Exact statistic = min(b, c) = 2.
        assert r.statistic == 2.0
        # p == two-sided binomial on 8 of 10, p=0.5 (independent recompute).
        expected = float(binomtest(8, 10, p=0.5, alternative="two-sided").pvalue)
        assert r.p_value == pytest.approx(expected)

    def test_asymptotic_used_above_threshold(self) -> None:
        # b+c = 30 >= 25 -> chi-square with continuity correction.
        a = [True] * 20 + [False] * 10 + [True] * 5
        b = [False] * 20 + [True] * 10 + [True] * 5
        r = mcnemar_paired(a, b)
        assert r.exact is False
        # statistic = (|20-10|-1)^2 / 30 = 81/30 = 2.7.
        assert r.statistic == pytest.approx(2.7)

    def test_no_discordant_pairs(self) -> None:
        # The two arms agree on every item -> undefined direction, p=1.0.
        a = [True, True, False, False]
        b = [True, True, False, False]
        r = mcnemar_paired(a, b)
        assert r.n_discordant == 0
        assert r.p_value == 1.0
        assert math.isnan(r.odds_ratio)

    def test_accuracy_and_effect_size(self) -> None:
        # a=3, b=4, c=1, d=2 -> n=10; acc_a=(3+4)/10=0.7, acc_b=(3+1)/10=0.4.
        a = [True] * 3 + [True] * 4 + [False] * 1 + [False] * 2
        b = [True] * 3 + [False] * 4 + [True] * 1 + [False] * 2
        r = mcnemar_paired(a, b)
        assert r.accuracy_a == pytest.approx(0.7)
        assert r.accuracy_b == pytest.approx(0.4)
        assert r.accuracy_diff == pytest.approx((4 - 1) / 10)
        assert r.odds_ratio == pytest.approx(4.0)

    def test_odds_ratio_inf_when_c_zero(self) -> None:
        # b>0, c=0 -> OR is +inf (arm A strictly dominates the discordant cells).
        a = [True] * 5 + [True] * 3
        b = [True] * 5 + [False] * 3
        r = mcnemar_paired(a, b)
        assert r.n_b_only == 0
        assert r.odds_ratio == float("inf")

    def test_as_dict_serializes(self) -> None:
        a = [True] * 8 + [False] * 2
        b = [False] * 8 + [True] * 2
        d = mcnemar_paired(a, b).as_dict()
        assert d["n_a_only"] == 8
        assert isinstance(d["p_value"], float)
        assert d["exact"] is True
