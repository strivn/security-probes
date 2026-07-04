"""Random baseline: expected AUC under permuted labels.

Establishes the significance floor for each scope. With small eval sets
(e.g., CWE-190 has only 9 eval pairs = 18 samples), the random AUC
distribution can be wide — a probe must exceed the 95% CI upper bound
to be considered meaningful.

Uses the split JSON to compute baselines on the actual eval set sizes.
No GPU needed — runs locally.

Output: outputs/phase1/random_baseline.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.data import C_CPP_CWES, PYTHON_CWES

SEED = 42
N_PERMUTATIONS = 10_000


def random_auc_distribution(
    n_positive: int,
    n_negative: int,
    n_perms: int,
    rng: np.random.RandomState,
) -> npt.NDArray[np.float64]:
    """Compute AUC distribution under random predictions.

    Creates true labels (n_positive 1s, n_negative 0s), then for each
    permutation, generates random scores and computes AUC.
    """
    y_true = np.array([1] * n_positive + [0] * n_negative)
    aucs = np.empty(n_perms)
    for i in range(n_perms):
        scores = rng.random(len(y_true))
        aucs[i] = _fast_auc(y_true, scores)
    return aucs


def _fast_auc(
    y_true: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
) -> float:
    """Compute AUC via the Mann-Whitney U statistic (faster than sklearn for this)."""
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    u_stat = stats.mannwhitneyu(pos_scores, neg_scores, alternative="two-sided").statistic
    return float(u_stat / (len(pos_scores) * len(neg_scores)))


def compute_scope_baseline(
    samples: list[dict[str, object]],
    cwe_filter: set[str] | None,
    scope_name: str,
    rng: np.random.RandomState,
) -> dict[str, object]:
    """Compute random baseline for a given scope (universal/python/c_cpp)."""
    eval_samples = [s for s in samples if s["split"] == "eval"]
    if cwe_filter is not None:
        eval_samples = [s for s in eval_samples if s["cwe"] in cwe_filter]

    n_pos = sum(1 for s in eval_samples if s["label"] == 1)
    n_neg = sum(1 for s in eval_samples if s["label"] == 0)

    print(f"  {scope_name}: {n_pos} pos + {n_neg} neg = {len(eval_samples)} eval samples")

    aucs = random_auc_distribution(n_pos, n_neg, N_PERMUTATIONS, rng)

    return {
        "scope": scope_name,
        "n_eval": len(eval_samples),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "ci_95_lower": float(np.percentile(aucs, 2.5)),
        "ci_95_upper": float(np.percentile(aucs, 97.5)),
        "ci_99_upper": float(np.percentile(aucs, 99.5)),
        "median_auc": float(np.median(aucs)),
    }


def compute_per_cwe_baseline(
    samples: list[dict[str, object]],
    rng: np.random.RandomState,
) -> dict[str, dict[str, object]]:
    """Compute random baseline per individual CWE."""
    eval_samples = [s for s in samples if s["split"] == "eval"]
    cwes = sorted({str(s["cwe"]) for s in eval_samples})
    per_cwe: dict[str, dict[str, object]] = {}

    for cwe in cwes:
        cwe_samples = [s for s in eval_samples if s["cwe"] == cwe]
        n_pos = sum(1 for s in cwe_samples if s["label"] == 1)
        n_neg = sum(1 for s in cwe_samples if s["label"] == 0)

        aucs = random_auc_distribution(n_pos, n_neg, N_PERMUTATIONS, rng)
        per_cwe[cwe] = {
            "n_eval": len(cwe_samples),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "mean_auc": float(np.mean(aucs)),
            "ci_95_upper": float(np.percentile(aucs, 97.5)),
            "ci_99_upper": float(np.percentile(aucs, 99.5)),
        }
        print(f"    {cwe}: n={len(cwe_samples)}, 95% CI upper={per_cwe[cwe]['ci_95_upper']:.3f}")

    return per_cwe


def main() -> None:
    split_path = PROJECT / "data" / "splits" / "phase1_split.json"
    with open(split_path) as f:
        split = json.load(f)

    samples: list[dict[str, object]] = split["samples"]
    rng = np.random.RandomState(SEED)

    print(f"Random baseline: {N_PERMUTATIONS} permutations, seed={SEED}")
    print(f"Eval set: {split['n_eval_samples']} samples ({split['n_eval_pairs']} pairs)\n")

    results: dict[str, object] = {
        "seed": SEED,
        "n_permutations": N_PERMUTATIONS,
        "scopes": {},
        "per_cwe": {},
    }

    # Scope-level baselines
    scopes: list[tuple[str, set[str] | None]] = [
        ("universal", None),
        ("python", PYTHON_CWES),
        ("c_cpp", C_CPP_CWES),
    ]
    print("Scope baselines:")
    scope_results = {}
    for scope_name, cwe_filter in scopes:
        scope_results[scope_name] = compute_scope_baseline(
            samples,
            cwe_filter,
            scope_name,
            rng,
        )
    results["scopes"] = scope_results

    # Per-CWE baselines
    print("\nPer-CWE baselines:")
    results["per_cwe"] = compute_per_cwe_baseline(samples, rng)

    out_path = PROJECT / "outputs" / "phase1" / "random_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\nSignificance thresholds (probe AUC must exceed to be meaningful):")
    for name, r in scope_results.items():
        print(f"  {name}: 95% CI upper = {r['ci_95_upper']:.3f}, mean = {r['mean_auc']:.3f}")


if __name__ == "__main__":
    main()
