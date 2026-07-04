"""Shared PatchEval CVE classification — the seen/unseen-bug-types split.

Single source of truth for how a PatchEval CVE is classified relative to the four
SVEN-Python CWEs the probe is trained on. Used by the probe OOD eval (07) and the
semgrep baseline (11) so both report on the IDENTICAL slice.

A CVE is "unseen" (out-of-distribution) only if NONE of the trained CWEs appears
anywhere in its FULL cwe_info label set — not just its primary label.
Splitting on the primary label alone let 119 CVEs whose primary CWE was untrained but
a SECONDARY CWE was trained leak into the "OOD" set; the full-label-set rule yields
235 clean unseen CVEs from the 404 valid (169 seen), verified against
data/patcheval/python_cves.json.
"""

from __future__ import annotations

import re
from typing import Any

# SVEN lowercase "cwe-022" ↔ PatchEval "CWE-22".
SVEN_TO_PATCHEVAL_CWE = {
    "cwe-022": "CWE-22",
    "cwe-078": "CWE-78",
    "cwe-079": "CWE-79",
    "cwe-089": "CWE-89",
}
TRAINED_CWES = set(SVEN_TO_PATCHEVAL_CWE.values())

# Expected split on the canonical data/patcheval/python_cves.json (404 valid CVEs).
# Asserted by callers so a silent data change or rule regression is caught loudly.
EXPECTED_SPLIT = {"valid": 404, "seen": 169, "unseen": 235}


def cve_label_set(cve: dict[str, Any]) -> set[str]:
    """Full set of CWE labels PatchEval assigns to a CVE (e.g. {'CWE-94','CWE-77'})."""
    return set(cve.get("cwe_info", {}).keys())


def cve_primary_cwe(cve: dict[str, Any]) -> str:
    """Primary CWE for the per-CWE breakdown label ONLY (first key; deterministic).

    Not used for the seen/unseen split — that uses the full label set via classify_cve.
    """
    keys = list(cve.get("cwe_info", {}).keys())
    return keys[0] if keys else "unknown"


def classify_cve(cve: dict[str, Any]) -> str:
    """Return "seen" if any trained CWE is in the full label set, else "unseen"."""
    return "seen" if (cve_label_set(cve) & TRAINED_CWES) else "unseen"


def assert_expected_split(valid_cves: list[dict[str, Any]]) -> None:
    """Raise if the seen/unseen split of `valid_cves` != EXPECTED_SPLIT (repro guard)."""
    n_seen = sum(1 for c in valid_cves if classify_cve(c) == "seen")
    actual = {"valid": len(valid_cves), "seen": n_seen, "unseen": len(valid_cves) - n_seen}
    if actual != EXPECTED_SPLIT:
        msg = (
            f"PatchEval split mismatch: got {actual}, expected {EXPECTED_SPLIT}. "
            "Data file or classify_cve rule changed — refusing to proceed so the "
            "published headline n cannot silently drift."
        )
        raise ValueError(msg)


# ── snippet pairing ──────────────────────────────────────────────────────
# PatchEval stores LISTS of function snippets per CVE. Scoring only vul_func[0] vs
# fix_func[0] would silently compare different functions on 38% of CVEs.
# This rule pairs vul↔fix snippets within a CVE by (file_path, primary def-name): exact
# name equality first, containment fallback, injective assignment. Research-verified:
# PatchEval defines no pairing; name-first
# matching is phase 1 of RefactoringMiner/CodeShovel; CVEfixes pairs by method name in
# the changed file. Replicated on data/patcheval/python_cves.json.

# Re-baselined on the canonical data with THIS rule. Asserted so a data or
# rule change is caught loudly. all = full 404-valid; unseen = the headline slice.
EXPECTED_PAIRING = {
    "all": {"A_single": 251, "B_multi_full": 135, "C_partial": 17, "D_unpairable": 1, "pairs": 526},
    "unseen": {
        "A_single": 147,
        "B_multi_full": 77,
        "C_partial": 10,
        "D_unpairable": 1,
        "pairs": 304,
    },
}

# "primary def" = first line-anchored def/async def in the snippet. Decorators sit on
# their own lines above the def, so the def line is the anchor; module-level snippets
# (no def) return None and cannot produce a ranking pair.
_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)


def primary_def(snippet: str) -> str | None:
    """First def/async def name in a snippet, or None if module-level (no def)."""
    m = _DEF_RE.search(snippet)
    return m.group(1) if m else None


def pair_cve_snippets(
    vul: list[dict[str, Any]],
    fix: list[dict[str, Any]],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Pair vul↔fix snippet indices for one CVE.

    Returns (pairs, unpaired_vul_indices). 1v1f pairs directly. Otherwise injective
    matching by (file_path, primary def-name): exact equality first, containment
    fallback. Each fix snippet is used at most once. Unmatched vul snippets are dropped
    (returned in `unpaired`) — fix-by-deletion/rename/add-helper cannot yield a pair.
    """
    if len(vul) == 1 and len(fix) == 1:
        return [(0, 0)], []

    used_fix: set[int] = set()
    pairs: list[tuple[int, int]] = []
    vul_meta = [
        (i, v.get("file_path"), primary_def(v.get("snippet", ""))) for i, v in enumerate(vul)
    ]
    fix_meta = [
        (j, f.get("file_path"), primary_def(f.get("snippet", ""))) for j, f in enumerate(fix)
    ]

    # Pass 1: exact (file_path, def-name) equality.
    for vi, vpath, vname in vul_meta:
        if vname is None:
            continue
        match = next(
            (j for j, fp, fn in fix_meta if j not in used_fix and fp == vpath and fn == vname),
            None,
        )
        if match is not None:
            used_fix.add(match)
            pairs.append((vi, match))

    # Pass 2: containment fallback (renames like delete -> delete_all), same file_path.
    paired = {vi for vi, _ in pairs}
    for vi, vpath, vname in vul_meta:
        if vi in paired or vname is None:
            continue
        match = next(
            (
                j
                for j, fp, fn in fix_meta
                if j not in used_fix
                and fp == vpath
                and fn is not None
                and (vname in fn or fn in vname)
            ),
            None,
        )
        if match is not None:
            used_fix.add(match)
            pairs.append((vi, match))
            paired.add(vi)

    unpaired = [vi for vi, _, _ in vul_meta if vi not in paired]
    return pairs, unpaired


def cve_stratum(vul: list[dict[str, Any]], fix: list[dict[str, Any]]) -> str:
    """Classify a CVE into a pairing stratum: 'A_single', 'B_multi_full', 'C_partial',
    or 'D_unpairable'. A = the headline single-snippet stratum."""
    if len(vul) == 1 and len(fix) == 1:
        return "A_single"
    pairs, unpaired = pair_cve_snippets(vul, fix)
    if not pairs:
        return "D_unpairable"
    return "B_multi_full" if not unpaired else "C_partial"


# ── repo-level leakage into the "OOD" test ───────────────────────────────
# 8 repos appear in BOTH the SVEN training set and PatchEval (exact owner/repo match,
# computed from data/splits/phase1_split.json ∩ data/patcheval/python_cves.json).
# The CWE-based seen/unseen split does NOT control codebase overlap, so a
# CVE from one of these repos may let the probe key on codebase idioms it saw in training
# (the exact threat the project-level SVEN split was built to prevent). Cascade reporting:
# the repo-disjoint subset is PRIMARY, the full set is shown alongside.
SVEN_PATCHEVAL_OVERLAP_REPOS = {
    "bit-team/backintime",
    "devsnd/cherrymusic",
    "ipython/ipython",
    "lepture/mistune",
    "python-pillow/pillow",
    "saltstack/salt",
    "tensorflow/tensorflow",
    "wagtail/wagtail",
}
# On the canonical data this flags 13 CVEs (7 unseen). Asserted by callers.
EXPECTED_REPO_OVERLAP = {"cves": 13, "unseen_cves": 7}


def cve_repo(cve: dict[str, Any]) -> str:
    """Return the CVE's GitHub 'owner/repo' (lowercased), or '' if unparseable."""
    url = str(cve.get("repo", "")).rstrip("/")
    if "github.com/" in url:
        return url.split("github.com/")[-1].lower()
    return url.lower()


def cve_in_trained_repo(cve: dict[str, Any]) -> bool:
    """True if the CVE comes from a repo that also appears in SVEN training (F8 leak)."""
    return cve_repo(cve) in SVEN_PATCHEVAL_OVERLAP_REPOS
