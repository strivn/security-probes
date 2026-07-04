"""Tests for the elicitation driver's pure orchestration (no GPU/model).

Loads 13_*.py by file path (its filename starts with a digit, so it can't be a normal
import). Tests build_items / build_fewshot_prefix / summarize.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT / "scripts" / "phase1" / "13_patcheval_prompted_elicit.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("elicit_driver", DRIVER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cve(cve_id: str, labels: list[str]) -> dict:
    return {
        "cve_id": cve_id,
        "cwe_info": {label: {} for label in labels},
        "vul_func": [{"snippet": "def f(x):\n    return eval(x)"}],
        "fix_func": [{"snippet": "def f(x):\n    return int(x)"}],
    }


def test_modes_constant(driver) -> None:
    assert driver.MODES == (
        "noshot-logprob",
        "noshot-token",
        "fewshot-logprob",
        "fewshot-token",
        "cot-logprob",
        "cot-token",
    )


def test_split_mode(driver) -> None:
    assert driver.split_mode("cot-logprob") == ("cot", "logprob")
    assert driver.split_mode("noshot-token") == ("noshot", "token")
    assert driver.split_mode("fewshot-token") == ("fewshot", "token")
    with pytest.raises(ValueError, match="unknown mode"):
        driver.split_mode("greedy")


def test_binary_failure_modes_are_token_readouts(driver) -> None:
    # Only the token (generate-and-parse) readouts can NaN; logprob readouts read a logit.
    assert driver.BINARY_FAILURE_MODES == ("noshot-token", "fewshot-token", "cot-token")


def test_build_items_interleaves_vul_fix(driver) -> None:
    cves = [_cve("CVE-1", ["CWE-601"]), _cve("CVE-2", ["CWE-22"])]
    items, meta = driver.build_items(cves, "noshot-logprob")
    assert len(items) == 4  # 2 CVEs x (vul, fix)
    assert len(meta) == 2
    assert meta[0]["split"] == "unseen"  # CWE-601 not trained
    assert meta[1]["split"] == "seen"  # CWE-22 trained
    assert all(it["cot"] is False for it in items)


def test_cot_logprob_flags_items(driver) -> None:
    items, _ = driver.build_items([_cve("CVE-1", ["CWE-601"])], "cot-logprob")
    assert all(it["cot"] is True for it in items)
    assert all(it["cot_greedy"] is False for it in items)
    assert "VERDICT" in items[0]["text"]


def test_cot_token_flags_items(driver) -> None:
    # cot-token uses the CoT prompt but the token (parse the emitted VERDICT) readout.
    items, _ = driver.build_items([_cve("CVE-1", ["CWE-601"])], "cot-token")
    assert all(it["cot"] is False for it in items)
    assert all(it["cot_greedy"] is True for it in items)
    assert "VERDICT" in items[0]["text"]


def test_fewshot_token_flags_items(driver) -> None:
    # fewshot-token = fewshot prompt + token readout: greedy flag set, prefix present.
    items, _ = driver.build_items([_cve("CVE-1", ["CWE-601"])], "fewshot-token")
    assert all(it["greedy"] is True for it in items)
    assert "Example 1" in items[0]["text"]


def test_build_items_carry_transcript_identity(driver) -> None:
    # Each scoring item must carry (cve_id, pair_idx, which) so the per-response transcript
    # can label every row, and they must follow the [vul, fix] interleave per pair.
    items, _ = driver.build_items([_cve("CVE-1", ["CWE-601"])], "noshot-logprob")
    assert len(items) == 2  # one pair -> vul, fix
    assert (
        items[0]["cve_id"] == "CVE-1" and items[0]["pair_idx"] == 0 and items[0]["which"] == "vul"
    )
    assert (
        items[1]["cve_id"] == "CVE-1" and items[1]["pair_idx"] == 0 and items[1]["which"] == "fix"
    )


def test_fewshot_prefix_prepended(driver) -> None:
    items, _ = driver.build_items([_cve("CVE-1", ["CWE-601"])], "fewshot-logprob")
    assert "Example 1" in items[0]["text"]
    assert "Example 2" in items[0]["text"]


def test_build_items_skips_empty_code(driver) -> None:
    cve = _cve("CVE-X", ["CWE-601"])
    cve["vul_func"][0]["snippet"] = ""
    items, meta = driver.build_items([cve], "noshot-logprob")
    assert items == []
    assert meta == []


def test_summarize_win_rate_and_auc(driver) -> None:
    # 3 unseen CVEs, probe-like scores: vul > fix on 2 of 3.
    per_cve = [
        {"split": "unseen", "vul_score": 0.9, "fix_score": 0.2, "risk_gap": 0.7},
        {"split": "unseen", "vul_score": 0.8, "fix_score": 0.3, "risk_gap": 0.5},
        {"split": "unseen", "vul_score": 0.4, "fix_score": 0.6, "risk_gap": -0.2},
        {"split": "seen", "vul_score": 0.9, "fix_score": 0.1, "risk_gap": 0.8},
    ]
    out = driver.summarize(per_cve)
    assert out["unseen"]["wins"] == 2
    assert out["unseen"]["losses"] == 1
    assert out["unseen"]["n_effective"] == 3
    assert out["unseen"]["win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert 0.0 <= out["unseen"]["auc"] <= 1.0
    assert out["seen"]["wins"] == 1
    assert out["overall"]["n_pairs"] == 4
