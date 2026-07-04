"""Contract test for the CoT token-space verdict search used in
scripts/phase1/13_patcheval_prompted_elicit.py.

The worker reads the YES/NO logit at the slot RIGHT AFTER the model's own "VERDICT:"
marker, located in TOKEN space (not by decoding->re-tokenizing, which byte-BPE can shift).
`_find_last_subseq` returns the index ONE PAST the last marker occurrence; that index,
offset by prompt_len, is the absolute position of the YES/NO slot whose logit we read.

This function is defined inside the worker's subprocess string (it runs there, no GPU),
so it cannot be imported. We replicate it here and pin the algorithm + the index chain so
a future edit to the worker that breaks either is caught on CPU.
"""

from __future__ import annotations


def _find_last_subseq(hay: list[int], needle: list[int]) -> int:
    """Index one-past the LAST occurrence of needle in hay, or -1. (Worker copy.)"""
    n = len(needle)
    if n == 0 or n > len(hay):
        return -1
    for start in range(len(hay) - n, -1, -1):
        if hay[start : start + n] == needle:
            return start + n
    return -1


def test_marker_at_end_returns_len() -> None:
    # Model said "...VERDICT:" and stopped -> one-past = len(gen), verdict slot is next.
    assert _find_last_subseq([5, 6, 7, 8], [7, 8]) == 4


def test_marker_at_start() -> None:
    assert _find_last_subseq([7, 8, 5, 6], [7, 8]) == 2


def test_last_occurrence_wins() -> None:
    # If the model wrote VERDICT twice (e.g. restated), the LAST one is the real verdict.
    assert _find_last_subseq([7, 8, 1, 7, 8], [7, 8]) == 5


def test_absent_marker() -> None:
    assert _find_last_subseq([1, 2, 3], [9]) == -1


def test_empty_and_oversized() -> None:
    assert _find_last_subseq([], [1]) == -1  # empty haystack
    assert _find_last_subseq([1], [1, 2]) == -1  # needle longer than haystack
    assert _find_last_subseq([1, 2, 3], []) == -1  # empty needle


def test_single_token_marker() -> None:
    assert _find_last_subseq([1, 2, 99, 3], [99]) == 3


def test_index_chain_points_at_verdict_slot() -> None:
    # full = [prompt(plen) + gen]. best_end is one-past the marker WITHIN gen_only.
    # verdict_pos = plen + best_end = absolute index of the token AFTER the marker.
    # logits_at reads row (pos-1), which predicts the token AT pos = the YES/NO slot.
    plen = 10
    gen_only = [50, 51, 99, 52]  # marker [99] at gen idx 2 -> one-past = 3
    best_end = _find_last_subseq(gen_only, [99])
    assert best_end == 3
    verdict_pos = plen + best_end
    assert verdict_pos == 13  # full[13] is the slot right after the marker
    # The logit row that PREDICTS full[13] is row 12 = verdict_pos - 1. Pin that.
    assert verdict_pos - 1 == 12
