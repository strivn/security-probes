"""Tests for the logprob YES/NO scoring primitive (pure, no model)."""

from __future__ import annotations

import math

import pytest

from agentic_sec_probe.logprob_scoring import (
    AnswerTokenIds,
    resolve_answer_token_ids,
    yes_score_from_logits,
)


class FakeTokenizer:
    """Maps known strings to fixed id lists; everything else splits to 2 tokens."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.mapping.get(text, [999, 998])  # default: multi-token


class TestResolveAnswerTokenIds:
    def test_single_token_variants_collected(self) -> None:
        # Explicit variants so no unmapped string falls through to the multi-token default.
        tok = FakeTokenizer({"YES": [10], "Yes": [11], " Yes": [12], "NO": [20], "No": [21]})
        ids = resolve_answer_token_ids(
            tok, yes_variants=("YES", "Yes", " Yes"), no_variants=("NO", "No")
        )
        assert ids.yes_ids == frozenset({10, 11, 12})
        assert ids.no_ids == frozenset({20, 21})
        assert ids.usable

    def test_multi_token_variant_contributes_first_subtoken(self) -> None:
        # " YES" splits to [5, 6]: its FIRST sub-token (5) is added (discriminates from NO),
        # and the variant is recorded in multi_token_variants. (DeepSeek's real case.)
        tok = FakeTokenizer({"YES": [10], " YES": [5, 6], "NO": [20]})
        ids = resolve_answer_token_ids(tok, yes_variants=("YES", " YES"), no_variants=("NO",))
        assert ids.yes_ids == frozenset({10, 5})
        assert " YES" in ids.multi_token_variants
        assert ids.usable

    def test_ambiguous_leading_subtoken_dropped_from_both_sides(self) -> None:
        # If YES and NO share a leading sub-token (id 7), it cannot discriminate -> dropped.
        tok = FakeTokenizer({"YES": [7, 1], "NO": [7, 2]})
        ids = resolve_answer_token_ids(tok, yes_variants=("YES",), no_variants=("NO",))
        assert 7 not in ids.yes_ids
        assert 7 not in ids.no_ids
        assert not ids.usable  # both sides emptied by the drop

    def test_unusable_when_a_side_empty(self) -> None:
        # NO encodes to nothing -> no_ids empty -> unusable. (A side with any token, even a
        # first sub-token, is now usable; only a truly empty side is unusable.)
        tok = FakeTokenizer({"YES": [10], "NO": []})
        ids = resolve_answer_token_ids(tok, yes_variants=("YES",), no_variants=("NO",))
        assert not ids.usable


def _logits(vocab: int, settings: dict[int, float]) -> list[float]:
    v = [-10.0] * vocab
    for i, val in settings.items():
        v[i] = val
    return v


class TestYesScoreFromLogits:
    def test_unsure_is_half(self) -> None:
        ids = AnswerTokenIds(frozenset({10}), frozenset({20}), ())
        logits = _logits(50, {10: 2.0, 20: 2.0})  # equal mass
        assert yes_score_from_logits(logits, ids) == pytest.approx(0.5)

    def test_confident_yes(self) -> None:
        ids = AnswerTokenIds(frozenset({10}), frozenset({20}), ())
        logits = _logits(50, {10: 8.0, 20: 0.0})  # YES >> NO
        score = yes_score_from_logits(logits, ids)
        assert score == pytest.approx(1.0 / (1.0 + math.exp(-8.0)))
        assert score > 0.99

    def test_confident_no(self) -> None:
        ids = AnswerTokenIds(frozenset({10}), frozenset({20}), ())
        logits = _logits(50, {10: 0.0, 20: 8.0})
        assert yes_score_from_logits(logits, ids) < 0.01

    def test_mass_aggregated_over_variants(self) -> None:
        # Two YES ids vs one NO id; aggregated YES mass = logsumexp.
        ids = AnswerTokenIds(frozenset({10, 11}), frozenset({20}), ())
        logits = _logits(50, {10: 1.0, 11: 1.0, 20: 1.0})
        log_yes = math.log(math.exp(1.0) + math.exp(1.0))
        log_no = 1.0
        expected = 1.0 / (1.0 + math.exp(-(log_yes - log_no)))
        assert yes_score_from_logits(logits, ids) == pytest.approx(expected)
        # With double the YES mass, score must exceed 0.5.
        assert yes_score_from_logits(logits, ids) > 0.5

    def test_score_in_unit_interval(self) -> None:
        ids = AnswerTokenIds(frozenset({10}), frozenset({20}), ())
        for yv, nv in [(-50.0, 50.0), (50.0, -50.0), (0.0, 0.0), (3.3, -1.1)]:
            s = yes_score_from_logits(_logits(50, {10: yv, 20: nv}), ids)
            assert 0.0 <= s <= 1.0

    def test_raises_on_unusable_ids(self) -> None:
        ids = AnswerTokenIds(frozenset({10}), frozenset(), ())
        with pytest.raises(ValueError, match="empty YES or NO"):
            yes_score_from_logits(_logits(50, {10: 1.0}), ids)

    def test_raises_on_out_of_vocab_id(self) -> None:
        ids = AnswerTokenIds(frozenset({100}), frozenset({20}), ())
        with pytest.raises(ValueError, match="out of vocab"):
            yes_score_from_logits(_logits(50, {20: 1.0}), ids)
