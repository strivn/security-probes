"""Semgrep baseline on PatchEval CVEs.

For each CVE with Python vul/fix function pairs:
  1. Run semgrep on vul_func snippet → n_findings_vul
  2. Run semgrep on fix_func snippet → n_findings_fix
  3. delta = n_findings_vul - n_findings_fix
  4. Correct if delta > 0 (more findings on vul than fix)

Computes overall accuracy, per-CWE breakdown, statistical tests.

Input:  data/patcheval/python_cves.json
Output: outputs/phase1/patcheval_semgrep.json

Runs on CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

# Shared seen/unseen split with the probe OOD eval (07) so semgrep reports on the
# identical 235-CVE unseen slice (full-label-set rule). See patcheval.py.
from agentic_sec_probe.patcheval import (
    assert_expected_split,
    classify_cve,
    cve_primary_cwe,
)


def run_semgrep(code: str) -> dict[str, Any]:
    """Run semgrep on a Python code snippet. Returns findings dict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [
                "semgrep",
                "--config",
                "r/python.lang.security",
                "--json",
                "--quiet",
                "--no-git-ignore",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = json.loads(result.stdout) if result.stdout.strip() else {"results": []}
        findings = output.get("results", [])
        return {
            "n_findings": len(findings),
            "rule_ids": [f["check_id"] for f in findings],
        }
    except subprocess.TimeoutExpired:
        return {"n_findings": 0, "rule_ids": [], "error": "timeout"}
    except Exception as exc:
        return {"n_findings": 0, "rule_ids": [], "error": str(exc)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    cve_path = PROJECT / "data" / "patcheval" / "python_cves.json"
    with open(cve_path) as f:
        cves: list[dict[str, Any]] = json.load(f)

    valid = [c for c in cves if c.get("vul_func") and c.get("fix_func")]
    print(f"PatchEval CVEs: {len(valid)}/{len(cves)} with both vul and fix", flush=True)
    assert_expected_split(valid)  # repro guard: 404 valid / 169 seen / 235 unseen

    per_cve: list[dict[str, Any]] = []

    for i, cve in enumerate(valid):
        cve_id = cve["cve_id"]
        cwe = cve_primary_cwe(cve)
        split = classify_cve(cve)  # "seen" / "unseen" (full-label-set rule)

        vul_code = cve["vul_func"][0].get("snippet", "")
        fix_code = cve["fix_func"][0].get("snippet", "")

        if not vul_code or not fix_code:
            per_cve.append(
                {
                    "cve_id": cve_id,
                    "cwe": cwe,
                    "split": split,
                    "error": "empty_code",
                }
            )
            continue

        vul_result = run_semgrep(vul_code)
        fix_result = run_semgrep(fix_code)

        n_vul = vul_result["n_findings"]
        n_fix = fix_result["n_findings"]
        delta = n_vul - n_fix

        per_cve.append(
            {
                "cve_id": cve_id,
                "cwe": cwe,
                "split": split,
                "n_findings_vul": n_vul,
                "n_findings_fix": n_fix,
                "delta": delta,
                "vul_rules": vul_result["rule_ids"],
                "fix_rules": fix_result["rule_ids"],
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(valid)}]", flush=True)

    # Filter to valid results
    results_with_delta = [r for r in per_cve if "delta" in r]
    deltas = np.array([r["delta"] for r in results_with_delta], dtype=np.float64)
    n = len(deltas)

    # Overall metrics
    n_correct = int(np.sum(deltas > 0))
    n_wrong = int(np.sum(deltas < 0))
    n_tied = int(np.sum(deltas == 0))
    pct_positive = n_correct / n if n > 0 else 0

    # How many CVEs had ANY findings at all
    n_any_vul = sum(1 for r in results_with_delta if r["n_findings_vul"] > 0)
    n_any_fix = sum(1 for r in results_with_delta if r["n_findings_fix"] > 0)

    # Wilcoxon on non-zero deltas
    nonzero = deltas[deltas != 0]
    if len(nonzero) >= 10:
        stat, p_value = stats.wilcoxon(nonzero)
        wilcoxon = {"stat": float(stat), "p_value": float(p_value), "n_nonzero": len(nonzero)}
    else:
        wilcoxon = {"stat": None, "p_value": None, "n_nonzero": len(nonzero)}

    # Per-CWE breakdown
    cwe_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results_with_delta:
        cwe_groups[r["cwe"]].append(r)

    per_cwe_summary: dict[str, dict[str, Any]] = {}
    for cwe in sorted(cwe_groups, key=lambda c: len(cwe_groups[c]), reverse=True):
        items = cwe_groups[cwe]
        cwe_deltas = np.array([r["delta"] for r in items], dtype=np.float64)
        cwe_n = len(items)
        per_cwe_summary[cwe] = {
            "n": cwe_n,
            "n_correct": int(np.sum(cwe_deltas > 0)),
            "n_wrong": int(np.sum(cwe_deltas < 0)),
            "n_tied": int(np.sum(cwe_deltas == 0)),
            "accuracy": round(float(np.sum(cwe_deltas > 0)) / cwe_n, 4) if cwe_n > 0 else 0,
            "n_any_findings": sum(
                1 for r in items if r["n_findings_vul"] > 0 or r["n_findings_fix"] > 0
            ),
        }

    # Seen vs unseen bug types (per-CVE full-label-set rule, shared with 07).
    id_deltas = np.array([r["delta"] for r in results_with_delta if r["split"] == "seen"])
    ood_deltas = np.array([r["delta"] for r in results_with_delta if r["split"] == "unseen"])

    output = {
        "n_total": n,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_tied": n_tied,
        "accuracy": round(pct_positive, 4),
        "n_any_findings_vul": n_any_vul,
        "n_any_findings_fix": n_any_fix,
        "wilcoxon": wilcoxon,
        "in_distribution": {
            "n": len(id_deltas),
            "accuracy": round(float(np.sum(id_deltas > 0)) / len(id_deltas), 4)
            if len(id_deltas) > 0
            else 0,
            "n_tied": int(np.sum(id_deltas == 0)),
        },
        "out_of_distribution": {
            "n": len(ood_deltas),
            "accuracy": round(float(np.sum(ood_deltas > 0)) / len(ood_deltas), 4)
            if len(ood_deltas) > 0
            else 0,
            "n_tied": int(np.sum(ood_deltas == 0)),
        },
        "per_cwe": per_cwe_summary,
        "per_cve": per_cve,
    }

    out_path = PROJECT / "outputs" / "phase1" / "patcheval_semgrep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}", flush=True)
    print(f"\nOverall (n={n}):", flush=True)
    print(f"  Correct (delta > 0): {n_correct} ({pct_positive:.1%})", flush=True)
    print(f"  Wrong (delta < 0):   {n_wrong}", flush=True)
    print(f"  Tied (delta == 0):   {n_tied}", flush=True)
    print(f"  CVEs with ANY semgrep findings: vul={n_any_vul}, fix={n_any_fix}", flush=True)
    if wilcoxon["p_value"] is not None:
        print(f"  Wilcoxon p={wilcoxon['p_value']:.2e}", flush=True)


if __name__ == "__main__":
    main()
