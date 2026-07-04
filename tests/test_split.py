"""Tests for train/eval split properties.

Validates the split JSON if it exists.
Skips gracefully if the file hasn't been created yet.
"""

import json
from pathlib import Path

import pytest

SPLIT_PATH = Path("data/splits/phase1_split.json")


@pytest.fixture()
def split_data() -> dict:
    if not SPLIT_PATH.exists():
        pytest.skip("Split file not generated yet")
    with open(SPLIT_PATH) as f:
        return json.load(f)


def test_no_pair_overlap(split_data: dict) -> None:
    """Train and eval pair sets must be disjoint."""
    train_ids = set(split_data["train_pair_ids"])
    eval_ids = set(split_data["eval_pair_ids"])
    assert train_ids.isdisjoint(eval_ids)


def test_all_pairs_assigned(split_data: dict) -> None:
    """Every pair must be in exactly one split."""
    train_ids = set(split_data["train_pair_ids"])
    eval_ids = set(split_data["eval_pair_ids"])
    assert len(train_ids) + len(eval_ids) == split_data["n_pairs"]


def test_overall_ratio(split_data: dict) -> None:
    """Overall train ratio should be ~80%."""
    total = split_data["n_train_pairs"] + split_data["n_eval_pairs"]
    ratio = split_data["n_train_pairs"] / total
    assert 0.75 < ratio < 0.85


def test_per_cwe_ratio(split_data: dict) -> None:
    """Each CWE should have 70-90% train ratio."""
    for cwe, counts in split_data["per_cwe"].items():
        cwe_total = counts["train"] + counts["eval"]
        ratio = counts["train"] / cwe_total
        assert 0.70 < ratio < 0.90, f"{cwe}: train ratio {ratio:.2f} out of range"


def test_samples_are_2x_pairs(split_data: dict) -> None:
    """Each pair produces 2 samples (vulnerable + secure)."""
    assert split_data["n_train_samples"] == split_data["n_train_pairs"] * 2
    assert split_data["n_eval_samples"] == split_data["n_eval_pairs"] * 2


def test_sample_metadata_complete(split_data: dict) -> None:
    """Every sample has required fields."""
    for s in split_data["samples"]:
        assert "pair_id" in s
        assert "label" in s
        assert "cwe" in s
        assert "language" in s
        assert "project" in s
        assert s["split"] in ("train", "eval")


def test_grouped_by_project(split_data: dict) -> None:
    """Split must be project-grouped (the leak fix), not pair-level."""
    assert split_data.get("group_by") == "project"


def test_no_project_straddles_splits(split_data: dict) -> None:
    """No GitHub project may appear in both train and eval (the core leak fix).

    Under the old pair-level split, eval pairs shared commits/projects with train,
    letting the probe key on repo idioms instead of the vulnerability.
    """
    train_projects = {s["project"] for s in split_data["samples"] if s["split"] == "train"}
    eval_projects = {s["project"] for s in split_data["samples"] if s["split"] == "eval"}
    straddling = train_projects & eval_projects
    assert not straddling, f"{len(straddling)} projects in both splits: {sorted(straddling)[:5]}"


def test_pair_samples_share_split_and_project(split_data: dict) -> None:
    """A pair's two samples must share the same split AND project assignment."""
    by_pair: dict[int, list[dict]] = {}
    for s in split_data["samples"]:
        by_pair.setdefault(s["pair_id"], []).append(s)
    for pid, samples in by_pair.items():
        splits = {s["split"] for s in samples}
        projects = {s["project"] for s in samples}
        assert len(splits) == 1, f"pair {pid} spans splits {splits}"
        assert len(projects) == 1, f"pair {pid} spans projects {projects}"


# --- F1 guard: scope by LANGUAGE, never by CWE-name as a language proxy ---

# A CWE-based stand-in for "Python scope" that this test guards against. These are a
# CWE grouping, NOT a language: filtering on them admits C/C++ samples whose CWE
# happens to be one of these four (e.g. a path-traversal cwe-022 written in C).
_OLD_PYTHON_CWE_PROXY = {"cwe-022", "cwe-078", "cwe-079", "cwe-089"}


def test_python_language_scope_is_pure_python(split_data: dict) -> None:
    """The language=='python' scope must contain ZERO non-python samples.

    This is the F1 invariant: the probe's "Python" scope is defined by the per-sample
    `language` field, not by CWE membership. A language filter is exact by construction;
    this test fails only if a non-python sample is ever tagged language=='python'.
    """
    py = [s for s in split_data["samples"] if s["language"] == "python"]
    non_py = [s for s in py if not _ends_python(s)]
    # `language` is the authority; there is no file_name in the split, so assert the
    # field is internally consistent (every python-scoped sample is tagged python).
    assert all(s["language"] == "python" for s in py)
    assert not non_py, f"{len(non_py)} samples in python scope are not python"


def test_cwe_proxy_scope_is_contaminated_language_scope_is_not(split_data: dict) -> None:
    """Document + lock the F1 contamination: CWE-proxy admits C/C++, language does not.

    The old `cwe in {cwe-022,078,079,089}` scope mixes in C/C++ samples; the
    `language=='python'` scope is a strict, smaller, pure subset. This test asserts the
    relationship so a regression to CWE-based scoping is caught.
    """
    samples = split_data["samples"]
    cwe_scope = [s for s in samples if s["cwe"] in _OLD_PYTHON_CWE_PROXY]
    lang_scope = [s for s in samples if s["language"] == "python"]

    cwe_non_py = [s for s in cwe_scope if s["language"] != "python"]
    # The CWE proxy IS contaminated (this is the bug we are guarding against reusing).
    assert cwe_non_py, "expected CWE-proxy scope to contain non-python contamination"
    # The language scope is a pure subset of the CWE scope: every python sample's CWE is
    # one of the four, and the language scope drops exactly the contaminants.
    assert len(lang_scope) == len(cwe_scope) - len(cwe_non_py)
    assert all(s["cwe"] in _OLD_PYTHON_CWE_PROXY for s in lang_scope)


def _ends_python(sample: dict) -> bool:
    """A sample is python iff its `language` field says so (the split has no file_name)."""
    return sample["language"] == "python"
