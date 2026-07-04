"""Paired win-rate statistics for the probe-vs-fix ranking metric.

The headline metric is: given a (vulnerable, fixed) pair, how often does the scorer
assign a higher risk score to the vulnerable file than to its fix? Each pair yields a
risk gap = risk(vulnerable) - risk(fixed); the pair is a "win" if the gap is positive,
a "loss" if negative, a "tie" if exactly zero.

We report this as a fraction-correct (paired win-rate) over the non-tie pairs, with:
  - a Wilson score confidence interval on that proportion
    (Brown, Cai & DasGupta 2001 — most accurate/robust small-n interval for k/n), and
  - an exact-binomial (sign) test of the proportion against the 0.5 chance baseline
    (scipy.stats.binomtest; the sign test IS the exact-binomial test on wins vs losses).

Ties are DROPPED, not counted as losses: the test is run on m = wins + losses. Counting
ties as losses would deflate a baseline that frequently produces exactly-equal scores
(e.g. a binary YES/NO prompted scorer). We report the tie count so the drop is visible.

References:
  Wilson 1927; Brown, Cai & DasGupta 2001, "Interval Estimation for a Binomial Proportion".
  McKenzie et al. 2025 (arXiv:2506.10805) §2.2 use the same paired-ranking framing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from scipy.stats import binomtest, norm


@dataclass(frozen=True)
class PairedWinRate:
    """Result of a paired win-rate analysis over a set of risk gaps."""

    n_pairs: int  # total pairs considered (wins + losses + ties)
    wins: int  # risk gap > 0  (scorer ranked vulnerable above its fix)
    losses: int  # risk gap < 0
    ties: int  # risk gap == 0  (dropped from the test)
    n_effective: int  # wins + losses (the denominator for the rate + test)
    win_rate: float  # wins / n_effective  (NaN-safe: 0.0 if n_effective == 0)
    ci_lower: float  # Wilson score interval lower bound on win_rate
    ci_upper: float  # Wilson score interval upper bound
    ci_level: float  # confidence level (e.g. 0.95)
    p_value: float  # exact-binomial (sign) test, two-sided, H0: win_rate == 0.5

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (rounds floats to 4 dp for stable artifacts)."""
        d = asdict(self)
        for k in ("win_rate", "ci_lower", "ci_upper"):
            d[k] = round(d[k], 4)
        return d


def wilson_interval(k: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n.

    Closed form (no statsmodels dependency). Returns (0.0, 0.0) for n == 0.

    The Wilson interval is bounded in [0, 1], does not collapse to a zero-width
    interval at k == 0 or k == n, and is accurate at small n — unlike the Wald
    (normal-approximation) interval. See Brown, Cai & DasGupta 2001.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))) / denom
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    # At the extremes the Wilson interval is one-sided (lower=0 at k=0, upper=1 at
    # k=n); snap to the exact boundary so floating-point dust doesn't leak through.
    if k == 0:
        lower = 0.0
    if k == n:
        upper = 1.0
    return (lower, upper)


def paired_win_rate(
    risk_gaps: list[float],
    *,
    confidence: float = 0.95,
    tie_eps: float = 0.0,
) -> PairedWinRate:
    """Compute paired win-rate, Wilson CI, and exact-binomial sign test.

    Args:
        risk_gaps: per-pair risk(vulnerable) - risk(fixed). Positive = win.
        confidence: CI confidence level (default 0.95).
        tie_eps: gaps with |gap| <= tie_eps are treated as ties and dropped.
            Default 0.0 means only exactly-equal scores are ties. Use a small
            epsilon (e.g. 1e-12) if floating-point scores never compare exactly.

    Returns:
        PairedWinRate with ties dropped from the rate and the test.
    """
    wins = sum(1 for g in risk_gaps if g > tie_eps)
    losses = sum(1 for g in risk_gaps if g < -tie_eps)
    ties = len(risk_gaps) - wins - losses
    n_eff = wins + losses

    win_rate = wins / n_eff if n_eff > 0 else 0.0
    ci_lower, ci_upper = wilson_interval(wins, n_eff, confidence=confidence)
    # Exact-binomial (sign) test on wins vs losses; ties already excluded.
    p_value = (
        float(binomtest(wins, n_eff, p=0.5, alternative="two-sided").pvalue) if n_eff > 0 else 1.0
    )

    return PairedWinRate(
        n_pairs=len(risk_gaps),
        wins=wins,
        losses=losses,
        ties=ties,
        n_effective=n_eff,
        win_rate=win_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=confidence,
        p_value=p_value,
    )


@dataclass(frozen=True)
class McNemarResult:
    """Between-arm paired comparison (probe vs prompted) on the SAME items.

    Tests H0: the two scorers have equal accuracy on the same CVEs (marginal
    homogeneity). Uses ONLY the discordant cells; invariant to concordant ones.
    This is an ADDITION to the per-arm win-rates, not a replacement: the per-arm
    PairedWinRate answers "is each arm above chance?", McNemar answers "is arm A
    above arm B on the same items?". Canonical ML reference: Dietterich 1998
    (Neural Computation 10(7)), which shows two independent one-sample proportion
    tests have high Type-I error and McNemar has low Type-I error.
    """

    n: int  # number of paired items (CVEs) — the unit of analysis
    n_both_correct: int  # a: both arms correct
    n_a_only: int  # b: arm A (probe) correct, arm B (prompted) wrong
    n_b_only: int  # c: arm A wrong, arm B correct
    n_both_wrong: int  # d
    n_discordant: int  # b + c
    statistic: float  # exact: min(b, c); asymptotic: chi-square
    p_value: float  # two-sided
    exact: bool  # True if the exact binomial test was used (b + c < 25)
    accuracy_a: float  # arm A accuracy = (a + b) / n
    accuracy_b: float  # arm B accuracy = (a + c) / n
    accuracy_diff: float  # (b - c) / n  (positive => A more accurate)
    odds_ratio: float  # b / c  (inf if c == 0 and b > 0; nan if both 0)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (rounds floats to 4 dp; OR/inf left as-is)."""
        d = asdict(self)
        for k in ("statistic", "p_value", "accuracy_a", "accuracy_b", "accuracy_diff"):
            d[k] = round(d[k], 4)
        return d


def mcnemar_paired(
    correct_a: list[bool],
    correct_b: list[bool],
    *,
    exact_threshold: int = 25,
) -> McNemarResult:
    """Exact (or continuity-corrected chi-square) McNemar test on paired correctness.

    Args:
        correct_a: per-item correctness for arm A (e.g. the probe). One bool per CVE.
        correct_b: per-item correctness for arm B (e.g. prompted). Same items, same order.
        exact_threshold: use the exact binomial test when b + c < this (default 25,
            the standard guidance — the chi-square approximation is unreliable below it).

    The exact test is a two-sided binomial test on b successes out of b+c trials, p=0.5
    (scipy.stats.binomtest — the same primitive the sign test uses). The asymptotic test
    is chi-square with Edwards continuity correction: (|b-c|-1)^2 / (b+c).

    Each item must be ONE independent observation (the CVE), not multiple function-pairs
    from the same CVE — within-cluster correlation inflates the test (Obuchowski 1998).
    """
    if len(correct_a) != len(correct_b):
        msg = f"correct_a ({len(correct_a)}) and correct_b ({len(correct_b)}) length mismatch"
        raise ValueError(msg)
    n = len(correct_a)

    a = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and y)
    b = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if not x and y)
    d = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if not x and not y)
    n_disc = b + c

    if n_disc == 0:
        # No discordant pairs: the two arms agree everywhere — undefined direction.
        statistic, p_value, exact = 0.0, 1.0, True
    elif n_disc < exact_threshold:
        # Exact: two-sided binomial on min vs the discordant total, p=0.5.
        statistic = float(min(b, c))  # statsmodels-compatible exact statistic
        p_value = float(binomtest(b, n_disc, p=0.5, alternative="two-sided").pvalue)
        exact = True
    else:
        # Asymptotic chi-square with Edwards continuity correction.
        statistic = (abs(b - c) - 1.0) ** 2 / n_disc if n_disc > 0 else 0.0
        # Two-sided p from the chi-square(1) survival function via the normal tail.
        z = math.sqrt(statistic)
        p_value = float(2.0 * (1.0 - norm.cdf(z)))
        exact = False

    acc_a = (a + b) / n if n > 0 else 0.0
    acc_b = (a + c) / n if n > 0 else 0.0
    if c == 0:  # noqa: SIM108 — nested ternary is less readable than the if/else here
        odds_ratio = float("inf") if b > 0 else float("nan")
    else:
        odds_ratio = b / c

    return McNemarResult(
        n=n,
        n_both_correct=a,
        n_a_only=b,
        n_b_only=c,
        n_both_wrong=d,
        n_discordant=n_disc,
        statistic=statistic,
        p_value=p_value,
        exact=exact,
        accuracy_a=acc_a,
        accuracy_b=acc_b,
        accuracy_diff=(b - c) / n if n > 0 else 0.0,
        odds_ratio=odds_ratio,
    )
