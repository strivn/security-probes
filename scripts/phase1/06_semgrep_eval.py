"""Semgrep baseline: static analysis on eval Python samples.

For each Python eval sample:
1. Write code to a temp .py file
2. Run semgrep with Python security rules
3. has_findings → predicted vulnerable

Computes AUC, accuracy, F1 per scope (universal filters to Python-only
since semgrep rules are language-specific).

Output: outputs/phase1/semgrep_eval.json

Runs on CPU. No GPU needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.data import PYTHON_CWES

SEMGREP_CONFIGS = [
    "r/python.lang.security",
]


def run_semgrep_on_code(code: str, cwe: str) -> dict[str, Any]:
    """Run semgrep on a code sample. Returns result dict."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
    ) as f:
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
            "has_findings": len(findings) > 0,
            "n_findings": len(findings),
            "rule_ids": [f["check_id"] for f in findings],
        }
    except subprocess.TimeoutExpired:
        return {"has_findings": False, "n_findings": 0, "rule_ids": [], "error": "timeout"}
    except Exception as exc:
        return {"has_findings": False, "n_findings": 0, "rule_ids": [], "error": str(exc)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    split_path = PROJECT / "data" / "splits" / "phase1_split.json"
    with open(split_path) as f:
        split_data = json.load(f)

    # Load actual code from SVEN pairs
    from agentic_sec_probe.data import load_sven_pairs, pairs_to_samples

    data_root = PROJECT / "data" / "raw"
    train_pairs = load_sven_pairs(data_root / "sven_train")
    val_pairs = load_sven_pairs(data_root / "sven_val")
    all_pairs = train_pairs + val_pairs
    samples = pairs_to_samples(all_pairs)

    # Filter to Python eval samples only
    meta_list: list[dict[str, Any]] = split_data["samples"]
    eval_indices = [
        i for i, m in enumerate(meta_list) if m["split"] == "eval" and m["cwe"] in PYTHON_CWES
    ]

    print(f"Python eval samples: {len(eval_indices)}")

    results_per_sample = []
    y_true = []
    y_pred = []
    y_score = []  # binary: 1.0 if findings, 0.0 if not (for AUC)

    for idx_num, idx in enumerate(eval_indices):
        sample = samples[idx]
        meta = meta_list[idx]

        semgrep_result = run_semgrep_on_code(sample.code, sample.cwe)

        predicted_vul = 1 if semgrep_result["has_findings"] else 0
        y_true.append(meta["label"])
        y_pred.append(predicted_vul)
        y_score.append(1.0 if semgrep_result["has_findings"] else 0.0)

        results_per_sample.append(
            {
                "sample_idx": idx,
                "pair_id": meta["pair_id"],
                "cwe": meta["cwe"],
                "label": meta["label"],
                "predicted": predicted_vul,
                "n_findings": semgrep_result["n_findings"],
                "rule_ids": semgrep_result["rule_ids"],
                "error": semgrep_result.get("error"),
            }
        )

        if (idx_num + 1) % 20 == 0:
            print(f"  [{idx_num + 1}/{len(eval_indices)}]")

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_score_arr = np.array(y_score, dtype=np.float64)

    # Compute metrics
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0.0))

    # AUC needs variation in scores — semgrep is binary, so AUC may be degenerate
    auc = float(roc_auc_score(y_true_arr, y_score_arr)) if len(np.unique(y_score_arr)) > 1 else 0.5

    # Per-CWE breakdown
    per_cwe: dict[str, dict[str, Any]] = {}
    for cwe in sorted(PYTHON_CWES):
        cwe_mask = np.array([m["cwe"] == cwe for m in [meta_list[i] for i in eval_indices]])
        if cwe_mask.sum() == 0:
            continue
        cwe_true = y_true_arr[cwe_mask]
        cwe_pred = y_pred_arr[cwe_mask]
        cwe_score = y_score_arr[cwe_mask]

        cwe_acc = float(accuracy_score(cwe_true, cwe_pred))
        cwe_f1 = float(f1_score(cwe_true, cwe_pred, zero_division=0.0))
        if len(np.unique(cwe_score)) > 1:
            cwe_auc = float(roc_auc_score(cwe_true, cwe_score))
        else:
            cwe_auc = 0.5

        # Detection rates
        vul_mask = cwe_true == 1
        sec_mask = cwe_true == 0
        tp = int(cwe_pred[vul_mask].sum()) if vul_mask.sum() > 0 else 0
        fp = int(cwe_pred[sec_mask].sum()) if sec_mask.sum() > 0 else 0

        per_cwe[cwe] = {
            "n_eval": int(cwe_mask.sum()),
            "accuracy": round(cwe_acc, 4),
            "f1": round(cwe_f1, 4),
            "auc": round(cwe_auc, 4),
            "true_positive_rate": round(tp / vul_mask.sum(), 4) if vul_mask.sum() > 0 else 0.0,
            "false_positive_rate": round(fp / sec_mask.sum(), 4) if sec_mask.sum() > 0 else 0.0,
        }

    output = {
        "n_eval_python": len(eval_indices),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "per_cwe": per_cwe,
        "per_sample": results_per_sample,
    }

    out_path = PROJECT / "outputs" / "phase1" / "semgrep_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"\nResults (Python eval, n={len(eval_indices)}):")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1:       {f1:.3f}")
    print("\nPer-CWE:")
    for cwe, r in per_cwe.items():
        tpr = r["true_positive_rate"]
        fpr = r["false_positive_rate"]
        print(f"  {cwe}: AUC={r['auc']:.3f}, TPR={tpr:.3f}, FPR={fpr:.3f}")


if __name__ == "__main__":
    main()
