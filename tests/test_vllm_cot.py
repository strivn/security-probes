"""CPU parity tests for the vLLM logprob -> YES/NO adapter.

The load-bearing guarantee: scoring a vLLM-shaped sparse {token_id: logprob} dict
must equal scoring the dense logit vector with the HF primitive, so the two
backends agree on the math. (Cross-backend *kernel* numerics still differ; that is
measured separately by the E3 logprob anchor. These tests pin the SCORING math.)
"""

from __future__ import annotations

import math

import pytest

from agentic_sec_probe.logprob_scoring import AnswerTokenIds, yes_score_from_logits
from agentic_sec_probe.vllm_cot import (
    MISSING_LOGPROB_FLOOR,
    find_answer_slot,
    find_verdict_slot,
    score_from_prompt_logprobs,
    score_generated_slot,
    score_generated_verdict_slot,
)

# A toy vocab with two YES ids and two NO ids (mirrors surface-variant aggregation).
YES_IDS = frozenset({10, 11})
NO_IDS = frozenset({20, 21})
ANSWER = AnswerTokenIds(yes_ids=YES_IDS, no_ids=NO_IDS, multi_token_variants=())


def _log_softmax(logits: list[float]) -> list[float]:
    """Dense logits -> per-id logprobs (what vLLM returns, computed from the same vector)."""
    m = max(logits)
    denom = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - denom for x in logits]


@pytest.mark.parametrize(
    "logits",
    [
        [0.0] * 22 + [3.0],  # padding ids, YES/NO mid-range
        [1.0, -2.0, 0.5] + [0.1] * 19 + [5.0],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -3.0,
            0.0,
        ],  # YES strongly favored
        [0.0] * 10 + [-2.0, -1.0] + [0.0] * 8 + [4.0, 3.0, 0.0],  # NO strongly favored
    ],
)
def test_sparse_dict_matches_dense_when_all_ids_present(logits: list[float]) -> None:
    """Full top-N (all YES/NO ids present) -> identical score to the dense primitive."""
    dense_score = yes_score_from_logits(logits, ANSWER)

    # vLLM returns logprobs (log-softmax of logits), as a sparse dict. Build the dict
    # with EVERY YES/NO id present so no floor is needed -> must match dense exactly.
    logprobs = _log_softmax(logits)
    position = {i: logprobs[i] for i in (*YES_IDS, *NO_IDS)}
    sparse_score = score_from_prompt_logprobs(position, ANSWER)

    assert sparse_score == pytest.approx(dense_score, abs=1e-9)


def test_missing_no_side_saturates_toward_yes() -> None:
    """If NO ids are absent from the top-N, the floor makes the score saturate to ~1."""
    # Only YES ids present, both high logprob; NO ids fall back to the floor.
    position = {10: -0.1, 11: -0.2}
    score = score_from_prompt_logprobs(position, ANSWER)
    assert score > 0.99


def test_missing_yes_side_saturates_toward_no() -> None:
    """If YES ids are absent, the score saturates toward 0."""
    position = {20: -0.1, 21: -0.2}
    score = score_from_prompt_logprobs(position, ANSWER)
    assert score < 0.01


def test_floor_value_is_used_for_absent_ids() -> None:
    """An absent id contributes exactly `floor`, matching a dense vector with that floor."""
    # YES present at -0.5; NO absent -> floor. Compare to the dense equivalent where
    # the NO ids sit at the floor and YES ids at -0.5.
    position = {10: -0.5, 11: -0.5}
    sparse_score = score_from_prompt_logprobs(position, ANSWER)

    log_yes = math.log(2) + (-0.5)  # logsumexp of two equal -0.5 logprobs
    log_no = math.log(2) + MISSING_LOGPROB_FLOOR
    diff = log_yes - log_no
    expected = 1.0 / (1.0 + math.exp(-diff))

    assert sparse_score == pytest.approx(expected, abs=1e-9)


def test_empty_answer_ids_raises() -> None:
    """An empty YES or NO side is a tokenizer-resolution failure, not a silent 0.5."""
    bad = AnswerTokenIds(yes_ids=frozenset(), no_ids=NO_IDS, multi_token_variants=())
    with pytest.raises(ValueError, match="empty YES or NO side"):
        score_from_prompt_logprobs({20: -0.1}, bad)


# ── find_verdict_slot: the generated-stream slot locator (the sound cot-logprob path) ──
#
# A fake tokenizer maps ids -> surface strings so find_verdict_slot's running-decode reconstructs
# realistic text. The vocab mirrors the real models: a single-token " YES"/" NO" (Qwen/Llama/
# Devstral) AND the DeepSeek split " Y","ES" so both branches are covered in one suite.

# id -> string. Verdict-relevant ids chosen to overlap nothing by accident.
_FAKE_VOCAB = {
    1: "The",
    2: " code",
    3: " is",
    4: " unsafe",
    5: ".",
    6: " yes",  # an in-REASONING 'yes' (lower) that must NOT be picked as the verdict
    7: "\n\n",
    8: "VERDICT",
    9: ":",
    10: " ",
    11: "**",
    12: " It",  # hedge token (neither YES nor NO)
    13: " depends",
    14: "ES",  # DeepSeek second sub-token after " Y"
    15: "\n",
    # answer tokens:
    100: " YES",  # single-token YES (Qwen/Llama/Devstral style)
    101: " NO",  # single-token NO
    102: " Y",  # DeepSeek first sub-token of " YES"
}


def _fake_decode(ids: list[int]) -> str:
    return "".join(_FAKE_VOCAB[i] for i in ids)


# YES side includes the single-token YES (100) and DeepSeek's first sub-token " Y" (102);
# NO side is the single-token " NO" (101). Mirrors resolve_answer_token_ids' output.
SLOT_YES = frozenset({100, 102})
SLOT_NO = frozenset({101})
SLOT_ANSWER = AnswerTokenIds(yes_ids=SLOT_YES, no_ids=SLOT_NO, multi_token_variants=(" YES",))


@pytest.mark.parametrize(
    ("gen_ids", "want_slot", "want_label", "name"),
    [
        # 1. Clean single-token YES: "The code is unsafe.\n\nVERDICT: YES"
        ([1, 2, 3, 4, 5, 7, 8, 9, 100], 8, "YES", "single_yes"),
        # 2. Clean single-token NO: "...VERDICT: NO"
        ([1, 2, 3, 4, 5, 7, 8, 9, 101], 8, "NO", "single_no"),
        # 3. Markdown bold: "**VERDICT:** YES" -> marker spans 11,8,9,11; word at the next id.
        ([1, 5, 11, 8, 9, 11, 100], 6, "YES", "markdown_yes"),
        # 4. DeepSeek split "VERDICT: Y" then "ES": slot is the " Y" sub-token, label YES.
        ([1, 2, 3, 7, 8, 9, 102, 14], 6, "YES", "deepseek_subtoken_yes"),
        # 5. Hedge "VERDICT: It depends" -> no YES/NO id in window -> (None, None) -> NaN.
        ([1, 2, 7, 8, 9, 12, 13], None, None, "hedge_none"),
        # 6. In-reasoning ' yes' (id 6) BEFORE the marker must be ignored; final verdict is NO.
        ([1, 6, 3, 4, 7, 8, 9, 101], 7, "NO", "ignore_inreasoning_yes"),
        # 7. No VERDICT marker at all -> (None, None).
        ([1, 2, 3, 4, 5], None, None, "no_marker"),
        # 8. Two markers: "VERDICT: NO ... VERDICT: YES" -> LAST one wins (YES).
        ([8, 9, 101, 5, 8, 9, 100], 6, "YES", "last_marker_wins"),
    ],
)
def test_find_verdict_slot(
    gen_ids: list[int],
    want_slot: int | None,
    want_label: str | None,
    name: str,
) -> None:
    slot, label = find_verdict_slot(gen_ids, _fake_decode, SLOT_ANSWER)
    assert (slot, label) == (want_slot, want_label), name
    # When found, the token AT the slot must actually be a resolved answer id (self-consistency).
    if slot is not None:
        assert gen_ids[slot] in (SLOT_YES | SLOT_NO), name


def test_find_verdict_slot_respects_window() -> None:
    """A YES/NO id far past the marker (beyond `window`) is NOT picked (it's post-verdict prose)."""
    # marker at index 1 ("VERDICT:"), then 9 filler tokens, THEN a YES id at index 11.
    gen_ids = [8, 9] + [5] * 9 + [100]
    slot, label = find_verdict_slot(gen_ids, _fake_decode, SLOT_ANSWER, window=8)
    assert (slot, label) == (None, None)
    # With a wide enough window it IS found.
    slot2, label2 = find_verdict_slot(gen_ids, _fake_decode, SLOT_ANSWER, window=20)
    assert (slot2, label2) == (11, "YES")


def test_score_generated_verdict_slot_reads_the_right_dict() -> None:
    """End-to-end: the scorer reads logprobs AT the located slot and softmaxes YES vs NO there."""
    # "VERDICT: YES" with the YES token (id 100) at index 2.
    gen_ids = [8, 9, 100]
    # Per-token logprob dicts; only the slot's dict matters. YES(100) high, NO(101) low.
    logprobs: list[dict[int, float]] = [
        {8: -0.1},
        {9: -0.1},
        {100: -0.05, 101: -3.0},  # the verdict slot
    ]
    score, label, ly, ln = score_generated_verdict_slot(
        gen_ids, logprobs, _fake_decode, SLOT_ANSWER
    )
    assert label == "YES"
    assert score > 0.9  # YES strongly favored at the slot
    assert ly == pytest.approx(-0.05)
    assert ln == pytest.approx(-3.0)


def test_score_generated_verdict_slot_deepseek_subtoken() -> None:
    """DeepSeek " Y"(102) vs " NO"(101) at the generated slot scores a graded P(YES)."""
    gen_ids = [8, 9, 102, 14]  # "VERDICT: Y" + "ES"
    logprobs = [
        {8: -0.1},
        {9: -0.1},
        {102: -0.7, 101: -1.5},  # the " Y" verdict slot: both sides present
        {14: -0.2},
    ]
    score, label, ly, ln = score_generated_verdict_slot(
        gen_ids, logprobs, _fake_decode, SLOT_ANSWER
    )
    assert label == "YES"
    assert 0.5 < score < 1.0  # graded, not floored
    assert ly == pytest.approx(-0.7)
    assert ln == pytest.approx(-1.5)


def test_score_generated_verdict_slot_hedge_is_nan() -> None:
    """A hedge (no YES/NO id after the marker) -> NaN, no label, no logmass."""
    gen_ids = [8, 9, 12, 13]  # "VERDICT: It depends"
    logprobs = [{8: -0.1}, {9: -0.1}, {12: -0.2}, {13: -0.2}]
    score, label, ly, ln = score_generated_verdict_slot(
        gen_ids, logprobs, _fake_decode, SLOT_ANSWER
    )
    assert math.isnan(score)
    assert label is None
    assert ly is None and ln is None


# ── find_answer_slot: the marker-less slot locator (noshot/fewshot, answer at token ~0) ──


@pytest.mark.parametrize(
    ("gen_ids", "want_slot", "want_label", "name"),
    [
        # 1. Bare answer at token 0 (the verified common case): " YES" / " NO"
        ([100], 0, "YES", "token0_yes"),
        ([101], 0, "NO", "token0_no"),
        # 2. DeepSeek bare " Y" at token 0 (first sub-token), then "ES"
        ([102, 14], 0, "YES", "token0_deepseek_y"),
        # 3. Leading non-answer token (space/markdown) then the answer within the window
        ([10, 100], 1, "YES", "leading_space_then_yes"),  # 10=" "
        ([11, 101], 1, "NO", "leading_markdown_then_no"),  # 11="**"
        # 4. A short preamble "The answer is YES" (3 filler then YES) within window=4
        ([1, 3, 10, 100], 3, "YES", "preamble_within_window"),
        # 5. No YES/NO id at all -> NaN signal
        ([1, 2, 5], None, None, "no_answer"),
        # 6. Answer beyond the window -> NOT found (avoids latching onto later prose)
        ([1, 2, 5, 7, 100], None, None, "beyond_window"),
    ],
)
def test_find_answer_slot(
    gen_ids: list[int],
    want_slot: int | None,
    want_label: str | None,
    name: str,
) -> None:
    slot, label = find_answer_slot(gen_ids, _fake_decode, SLOT_ANSWER, window=4)
    assert (slot, label) == (want_slot, want_label), name
    if slot is not None:
        assert gen_ids[slot] in (SLOT_YES | SLOT_NO), name


# ── score_generated_slot: one scorer for either finder (CoT verdict OR noshot/fewshot answer) ──


def test_score_generated_slot_answer_finder_token0() -> None:
    """noshot/fewshot path: bare ' YES' at token 0, scored via find_answer_slot."""
    gen_ids = [100]  # " YES"
    logprobs = [{100: -0.05, 101: -3.0}]
    score, label, ly, ln = score_generated_slot(
        gen_ids, logprobs, _fake_decode, SLOT_ANSWER, finder=find_answer_slot
    )
    assert label == "YES"
    assert score > 0.9
    assert ly == pytest.approx(-0.05)
    assert ln == pytest.approx(-3.0)


def test_score_generated_slot_verdict_parity() -> None:
    """Verdict finder via the shared scorer matches the existing score_generated_verdict_slot."""
    gen_ids = [8, 9, 100]  # "VERDICT: YES"
    logprobs = [{8: -0.1}, {9: -0.1}, {100: -0.05, 101: -3.0}]
    shared = score_generated_slot(
        gen_ids, logprobs, _fake_decode, SLOT_ANSWER, finder=find_verdict_slot
    )
    legacy = score_generated_verdict_slot(gen_ids, logprobs, _fake_decode, SLOT_ANSWER)
    assert shared == legacy
