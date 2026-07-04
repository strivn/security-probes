"""CPU tests for the cot_greedy verdict-parse semantics.

The GPU half (model.generate + token-space VERDICT-marker search) runs only on the
box. The PARSE rule -- given the decoded text AFTER the last VERDICT marker, return
YES=1.0 / NO=0.0 / NaN on no standalone verdict word -- is pure string logic and is
pinned here. This mirrors the worker's cot_greedy_score tail exactly (word-boundary
regex, first-standalone-wins). Keeping it in sync with the worker is the same
discipline as test_worker_parity for the logit path.
"""

from __future__ import annotations

import math
import re

# Mirror of the worker's word-boundary verdict regexes (script 13, _YES_RE/_NO_RE).
_YES_RE = re.compile(r"\byes\b")
_NO_RE = re.compile(r"\bno\b")


def parse_verdict_after_marker(text_after_marker: str) -> float:
    """Replicates cot_greedy_score's tail: first standalone yes/no after VERDICT wins."""
    after = text_after_marker.strip().lower()
    y = _YES_RE.search(after)
    n = _NO_RE.search(after)
    if y and not n:
        return 1.0
    if n and not y:
        return 0.0
    if y and n:
        return 1.0 if y.start() < n.start() else 0.0
    return float("nan")


def test_plain_yes() -> None:
    assert parse_verdict_after_marker(" YES") == 1.0


def test_plain_no() -> None:
    assert parse_verdict_after_marker(" NO") == 0.0


def test_yes_with_trailing_prose() -> None:
    assert parse_verdict_after_marker(" YES, this function is vulnerable.") == 1.0


def test_first_standalone_wins() -> None:
    # "YES ... no" -> YES (earlier position). The worker takes the first verdict word.
    assert parse_verdict_after_marker(" YES. There is no other issue.") == 1.0


def test_no_before_yes() -> None:
    assert parse_verdict_after_marker(" NO. It is not a yes-case.") == 0.0


def test_not_does_not_match_no() -> None:
    # Word boundary: "not"/"cannot" must NOT match the standalone "no".
    assert parse_verdict_after_marker(" cannot determine, but YES") == 1.0


def test_unparseable_is_nan() -> None:
    assert math.isnan(parse_verdict_after_marker(" the function looks fine overall"))


def test_empty_is_nan() -> None:
    assert math.isnan(parse_verdict_after_marker(""))


# ── cut_after_verdict (vLLM two-pass: truncate reasoning at the model's own VERDICT) ──
_VERDICT_RE = re.compile(r"\**\s*verdict\s*\**\s*:", re.IGNORECASE)
_MARKER = " VERDICT:"


def cut_after_verdict(reasoning: str) -> str:
    """Mirror of task0_vllm_probe.cut_after_verdict (kept in sync; pinned here)."""
    matches = list(_VERDICT_RE.finditer(reasoning))
    if matches:
        return reasoning[: matches[-1].end()]
    return reasoning + _MARKER


def test_cut_plain_verdict() -> None:
    # Ends right after the colon, so the next token would be the model's YES/NO.
    assert cut_after_verdict("...analysis. VERDICT: YES").endswith("VERDICT:")


def test_cut_markdown_bold_verdict() -> None:
    # qwen's actual form: "**VERDICT: YES**". Cut after the colon, dropping " YES**".
    out = cut_after_verdict("...risk.  **VERDICT: NO**")
    assert out.endswith(":")
    assert "NO" not in out.split("VERDICT")[-1].replace(":", "")


def test_cut_uses_last_verdict() -> None:
    # If the word appears twice (e.g. in reasoning + final), cut at the LAST one.
    out = cut_after_verdict("I will give a VERDICT: below. VERDICT: YES")
    assert out.count("VERDICT") == 2
    assert out.endswith("VERDICT:")


def test_cut_appends_marker_when_absent() -> None:
    # Model rambled without a verdict -> append our own marker so there's a slot to score.
    assert cut_after_verdict("The function seems okay.") == "The function seems okay." + _MARKER
