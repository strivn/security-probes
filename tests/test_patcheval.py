"""Tests for the shared PatchEval seen/unseen classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_sec_probe.patcheval import (
    EXPECTED_PAIRING,
    EXPECTED_REPO_OVERLAP,
    EXPECTED_SPLIT,
    TRAINED_CWES,
    assert_expected_split,
    classify_cve,
    cve_in_trained_repo,
    cve_label_set,
    cve_primary_cwe,
    cve_repo,
    cve_stratum,
    pair_cve_snippets,
    primary_def,
)

CVE_PATH = Path("data/patcheval/python_cves.json")


def test_trained_cwes() -> None:
    assert {"CWE-22", "CWE-78", "CWE-79", "CWE-89"} == TRAINED_CWES


def test_classify_seen_on_any_trained_label() -> None:
    # Primary untrained (CWE-94) but secondary trained (CWE-78) -> seen.
    cve = {"cwe_info": {"CWE-94": {}, "CWE-77": {}, "CWE-78": {}}}
    assert classify_cve(cve) == "seen"
    # Primary trained -> seen.
    assert classify_cve({"cwe_info": {"CWE-22": {}}}) == "seen"
    # No trained label anywhere -> unseen.
    assert classify_cve({"cwe_info": {"CWE-601": {}}}) == "unseen"


def test_primary_cwe_is_first_key() -> None:
    assert cve_primary_cwe({"cwe_info": {"CWE-94": {}, "CWE-78": {}}}) == "CWE-94"
    assert cve_primary_cwe({"cwe_info": {}}) == "unknown"


def test_label_set() -> None:
    assert cve_label_set({"cwe_info": {"CWE-22": {}, "CWE-79": {}}}) == {"CWE-22", "CWE-79"}


@pytest.fixture()
def valid_cves() -> list[dict]:
    if not CVE_PATH.exists():
        pytest.skip("PatchEval data not present")
    cves = json.loads(CVE_PATH.read_text())
    return [c for c in cves if c.get("vul_func") and c.get("fix_func")]


def test_real_data_matches_expected_split(valid_cves: list[dict]) -> None:
    n_seen = sum(1 for c in valid_cves if classify_cve(c) == "seen")
    actual = {"valid": len(valid_cves), "seen": n_seen, "unseen": len(valid_cves) - n_seen}
    assert actual == EXPECTED_SPLIT


def test_assert_expected_split_passes_on_real_data(valid_cves: list[dict]) -> None:
    assert_expected_split(valid_cves)  # must not raise


def test_assert_expected_split_raises_on_wrong_count() -> None:
    with pytest.raises(ValueError, match="split mismatch"):
        assert_expected_split([{"cwe_info": {"CWE-601": {}}}])


# ── snippet pairing ───────────────────────────────────────────────────────


def test_primary_def() -> None:
    assert primary_def("def foo(x):\n    pass") == "foo"
    assert primary_def("@deco\nasync def bar(y):\n    pass") == "bar"
    assert primary_def("x = 1\ny = 2") is None  # module-level, no def


def _snip(path: str, code: str) -> dict:
    return {"file_path": path, "snippet": code}


def test_pair_single_direct() -> None:
    # 1 vul, 1 fix -> direct pair regardless of names.
    v = [_snip("a.py", "def foo(): pass")]
    f = [_snip("a.py", "def renamed(): return 1")]
    pairs, unpaired = pair_cve_snippets(v, f)
    assert pairs == [(0, 0)]
    assert unpaired == []
    assert cve_stratum(v, f) == "A_single"


def test_pair_multi_exact_name() -> None:
    v = [_snip("a.py", "def foo(): pass"), _snip("a.py", "def bar(): pass")]
    f = [_snip("a.py", "def bar(): return 2"), _snip("a.py", "def foo(): return 1")]
    pairs, unpaired = pair_cve_snippets(v, f)
    assert sorted(pairs) == [(0, 1), (1, 0)]  # foo->foo, bar->bar (order-independent)
    assert unpaired == []
    assert cve_stratum(v, f) == "B_multi_full"


def test_pair_containment_fallback() -> None:
    # rename delete -> delete_all matched by containment.
    v = [_snip("a.py", "def delete(): pass"), _snip("a.py", "def keep(): pass")]
    f = [_snip("a.py", "def delete_all(): return 1"), _snip("a.py", "def keep(): return 2")]
    pairs, unpaired = pair_cve_snippets(v, f)
    assert unpaired == []
    assert cve_stratum(v, f) == "B_multi_full"


def test_pair_partial_drop_unmatched() -> None:
    # vul has a func with no fix counterpart -> partial.
    v = [_snip("a.py", "def foo(): pass"), _snip("a.py", "def gone(): pass")]
    f = [_snip("a.py", "def foo(): return 1")]
    pairs, unpaired = pair_cve_snippets(v, f)
    assert pairs == [(0, 0)]
    assert unpaired == [1]
    assert cve_stratum(v, f) == "C_partial"


def test_pair_different_file_not_matched() -> None:
    # same name but different file -> no match.
    v = [_snip("a.py", "def foo(): pass"), _snip("b.py", "def bar(): pass")]
    f = [_snip("a.py", "def foo(): return 1"), _snip("c.py", "def bar(): return 2")]
    pairs, unpaired = pair_cve_snippets(v, f)
    assert pairs == [(0, 0)]  # foo in a.py matches; bar in b.py vs c.py does not
    assert unpaired == [1]


def test_real_data_matches_expected_pairing(valid_cves: list[dict]) -> None:
    from collections import Counter

    cat: Counter[str] = Counter()
    pairs = 0
    for c in valid_cves:
        cat[cve_stratum(c["vul_func"], c["fix_func"])] += 1
        p, _ = pair_cve_snippets(c["vul_func"], c["fix_func"])
        pairs += len(p)
    exp = EXPECTED_PAIRING["all"]
    assert cat["A_single"] == exp["A_single"]
    assert cat["B_multi_full"] == exp["B_multi_full"]
    assert cat["C_partial"] == exp["C_partial"]
    assert cat["D_unpairable"] == exp["D_unpairable"]
    assert pairs == exp["pairs"]


def test_real_data_unseen_pairing(valid_cves: list[dict]) -> None:
    from collections import Counter

    unseen = [c for c in valid_cves if classify_cve(c) == "unseen"]
    cat: Counter[str] = Counter()
    pairs = 0
    for c in unseen:
        cat[cve_stratum(c["vul_func"], c["fix_func"])] += 1
        p, _ = pair_cve_snippets(c["vul_func"], c["fix_func"])
        pairs += len(p)
    exp = EXPECTED_PAIRING["unseen"]
    assert cat["A_single"] == exp["A_single"]
    assert cat["B_multi_full"] == exp["B_multi_full"]
    assert pairs == exp["pairs"]


# ── repo overlap ──────────────────────────────────────────────────────────


def test_cve_repo_parses_github_url() -> None:
    assert cve_repo({"repo": "https://github.com/aio-libs/aiohttp"}) == "aio-libs/aiohttp"
    assert cve_repo({"repo": "https://github.com/Foo/Bar/"}) == "foo/bar"
    assert cve_repo({}) == ""


def test_cve_in_trained_repo() -> None:
    assert cve_in_trained_repo({"repo": "https://github.com/tensorflow/tensorflow"}) is True
    assert cve_in_trained_repo({"repo": "https://github.com/aio-libs/aiohttp"}) is False


def test_real_data_repo_overlap(valid_cves: list[dict]) -> None:
    n = sum(1 for c in valid_cves if cve_in_trained_repo(c))
    n_unseen = sum(1 for c in valid_cves if cve_in_trained_repo(c) and classify_cve(c) == "unseen")
    assert n == EXPECTED_REPO_OVERLAP["cves"]
    assert n_unseen == EXPECTED_REPO_OVERLAP["unseen_cves"]
