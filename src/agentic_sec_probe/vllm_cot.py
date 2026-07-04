"""vLLM logprob -> YES/NO score adapter (Phase 2 CoT, vLLM backend).

The HF prompted worker reads a DENSE next-token logit vector and scores it with
`logprob_scoring.yes_score_from_logits`. vLLM does not expose a dense vector: a
`SamplingParams(logprobs=N)` / `prompt_logprobs=N` request returns only the top-N
tokens at each position as a sparse `dict[int, Logprob]` (verified against vLLM
v0.23.0 `vllm/logprobs.py`: `LogprobsOnePosition = dict[int, Logprob]`, with
`Logprob.logprob: float`).

This module adapts that sparse dict onto the SAME scoring math as the HF path so
there is exactly one copy of the softmax. The only new concern is the top-N cap:
a YES or NO surface-variant id may be absent from the returned top-N (the model
was confident about the other side). We assign such a missing id a floor logprob
so the score degrades gracefully (saturates toward the present side) instead of
raising on an empty side.

Pure + model-free + vLLM-free: unit-tested on CPU against `yes_score_from_logits`.
The worker unwraps each `Logprob.logprob` into a plain `dict[int, float]` before
calling here, so this file never imports vllm.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from agentic_sec_probe.logprob_scoring import AnswerTokenIds, _logsumexp

if TYPE_CHECKING:
    from collections.abc import Callable

# Floor for a YES/NO id absent from vLLM's returned top-N. vLLM's default
# `max_logprobs` is 20; a binary YES/NO slot almost always has both sides in the
# top-20, but if one is missing it is far down the distribution, so a low floor is
# the faithful stand-in. -100.0 (in nats) is ~exp(-100) probability: negligible.
MISSING_LOGPROB_FLOOR = -100.0

# Cut a CoT reasoning to end right BEFORE the model's own yes/no verdict word, so a second
# prompt ending there puts the verdict word in the FINAL slot (where prompt_logprobs[-1] reads
# the YES-vs-NO logprob). The match consumes "VERDICT:" + the whitespace/markdown that precedes
# the verdict word, so the prompt ends at "...VERDICT: " and the next token IS the yes/no.
# Cutting only after the colon (leaving the prompt to end ON the colon token) lands the slot on
# a formatting/colon continuation, not the verdict -- so we include the trailing separators.
# Tolerates markdown bold + spacing ("**VERDICT:**", "VERDICT :"). vllm-free + unit-tested.
_VERDICT_COLON_RE = re.compile(r"\**\s*verdict\s*\**\s*:[\s*]*", re.IGNORECASE)


def cut_after_verdict(reasoning: str) -> tuple[str, bool]:
    """Return (reasoning truncated to end just before the last verdict word, found?).

    The cut keeps everything up to and including 'VERDICT:' plus the trailing whitespace/markdown,
    so the score prompt ends right where the model's yes/no token comes next (the slot we read).
    If the model never volunteered a verdict, returns (reasoning, False); the caller scores NaN
    and never appends a fabricated marker (appending one floored slots to 0.5 in task0 v1).
    """
    matches = list(_VERDICT_COLON_RE.finditer(reasoning))
    if not matches:
        return reasoning, False
    return reasoning[: matches[-1].end()], True


def score_from_prompt_logprobs(
    position_logprobs: dict[int, float],
    answer_ids: AnswerTokenIds,
    *,
    floor: float = MISSING_LOGPROB_FLOOR,
) -> float:
    """P(YES) from a single position's sparse {token_id: logprob} dict (vLLM top-N).

    Mirrors `logprob_scoring.yes_score_from_logits` exactly (log-sum-exp the YES
    ids, log-sum-exp the NO ids, 2-way softmax == sigmoid of the difference), but
    reads logprobs from a sparse dict instead of a dense vector. Any YES/NO id not
    present in `position_logprobs` contributes `floor` rather than being dropped,
    so a confident one-sided distribution saturates instead of raising.

    `position_logprobs` is one entry of vLLM's prompt_logprobs/logprobs list with
    each `Logprob` already unwrapped to its `.logprob` float (token_id -> logprob).
    """
    if not answer_ids.usable:
        msg = "answer_ids has an empty YES or NO side; fall back to sequence scoring"
        raise ValueError(msg)

    # Gather each side's log-mass, flooring ids vLLM did not return in the top-N.
    yes_logprobs = [position_logprobs.get(i, floor) for i in answer_ids.yes_ids]
    no_logprobs = [position_logprobs.get(i, floor) for i in answer_ids.no_ids]

    log_yes = _logsumexp(yes_logprobs)
    log_no = _logsumexp(no_logprobs)

    # 2-way softmax over the aggregated log-masses == sigmoid(log_yes - log_no).
    diff = log_yes - log_no
    if diff >= 0:
        return 1.0 / (1.0 + math.exp(-diff))
    e = math.exp(diff)
    return e / (1.0 + e)


# ── Generated-slot readout (the sound cot-logprob path; replaces cut+re-feed) ─────────────────
#
# Instead of cutting the reasoning and re-feeding it (which reads a prefill conditional that, for
# DeepSeek, is NOT the token the model decodes), we read
# the YES/NO logprob at the slot the model ACTUALLY generated its verdict. The worker generates the
# CoT once with vLLM `logprobs=20`, so `out.outputs[0].logprobs[k]` is the decode distribution at
# generated token k. We locate the verdict token IN that generated stream and score its dict.
#
# The marker regex matches "VERDICT:" (bold/spacing tolerant) but NOT the verdict word -- the word
# is a separate generated token we find by id, so the marker only scopes the region.
_VERDICT_MARKER_RE = re.compile(r"\**\s*verdict\s*\**\s*:", re.IGNORECASE)

# How many tokens after the marker to scan for a YES/NO id before giving up (hedge -> NaN). The
# verdict word is normally the next 1-3 tokens ("VERDICT:" then maybe a space/markdown token then
# the word); 8 is generous without reaching a later stray yes/no in the post-verdict prose.
VERDICT_SLOT_WINDOW = 8


def find_verdict_slot(
    gen_ids: list[int],
    decode: Callable[[list[int]], str],
    answer_ids: AnswerTokenIds,
    *,
    window: int = VERDICT_SLOT_WINDOW,
) -> tuple[int | None, str | None]:
    """Locate the YES/NO verdict token in a generated token stream (token space, no char offsets).

    Two-stage scan over `gen_ids`, the per-token ids vLLM generated (greedy):
      1. Find the token index by which the model has finished writing its LAST 'VERDICT:' marker
         (running-decode contains the marker). This scopes us to the model's intended answer and
         ignores any 'yes'/'no' that appeared earlier in the reasoning body.
      2. Scan forward up to `window` tokens; return the FIRST token whose id is a resolved YES or
         NO id. Its index is the slot to read logprobs at; its side gives the label.

    Returns (slot_index, label) where label is 'YES' or 'NO'; (None, None) if there is no marker,
    or no YES/NO id within the window (a hedge like 'VERDICT: It depends' -> caller scores NaN).

    Pure: `decode` is injected (the tokenizer's `decode`), so this is vLLM- and model-free and
    unit-tested on CPU with a fake tokenizer.
    """
    yes_ids, no_ids = answer_ids.yes_ids, answer_ids.no_ids

    # Stage 1: largest k whose running decode newly contains the marker. We walk forward keeping a
    # running string so a marker split across tokens ("VER","DICT",":") is still detected, and we
    # keep the LAST marker (the model may mention 'verdict:' mid-reasoning before its final one).
    last_marker_k: int | None = None
    running = ""
    for k, tid in enumerate(gen_ids):
        running += decode([tid])
        if _VERDICT_MARKER_RE.search(running):
            last_marker_k = k
            running = ""  # reset so the NEXT marker (if any) is found independently -> keeps last
    if last_marker_k is None:
        return None, None

    # Stage 2: first YES/NO id within `window` tokens after the marker token.
    end = min(last_marker_k + 1 + window, len(gen_ids))
    for k in range(last_marker_k + 1, end):
        tid = gen_ids[k]
        if tid in yes_ids:
            return k, "YES"
        if tid in no_ids:
            return k, "NO"
    return None, None


# noshot/fewshot have NO 'VERDICT:' marker: the prompt asks for one word, so the model emits a
# BARE YES/NO at generation token 0 (verified from the HF fewshot-token transcripts). The window
# absorbs an occasional leading space/markdown token or a short "The answer is " preamble; small
# so a later stray yes/no in trailing prose is never picked.
ANSWER_SLOT_WINDOW = 4


def find_answer_slot(
    gen_ids: list[int],
    decode: Callable[[list[int]], str],
    answer_ids: AnswerTokenIds,
    *,
    window: int = ANSWER_SLOT_WINDOW,
) -> tuple[int | None, str | None]:
    """Locate the YES/NO answer token for a marker-less prompt (noshot/fewshot).

    The answer is at/near generation token 0 (the prompt asks for exactly one word). Scan the
    FIRST `window` generated tokens; return the FIRST whose id is a resolved YES or NO id. No
    marker scan (unlike find_verdict_slot) -- generation start IS the anchor. Returns
    (slot_index, label) or (None, None) if no YES/NO id appears in the window (parse-fail/refusal
    -> caller scores NaN). Pure: `decode` is unused here but kept in the signature for a uniform
    finder interface with find_verdict_slot (so score_generated_slot can take either).
    """
    yes_ids, no_ids = answer_ids.yes_ids, answer_ids.no_ids
    end = min(window, len(gen_ids))
    for k in range(end):
        tid = gen_ids[k]
        if tid in yes_ids:
            return k, "YES"
        if tid in no_ids:
            return k, "NO"
    return None, None


def score_generated_slot(
    gen_ids: list[int],
    logprobs: list[dict[int, float]],
    decode: Callable[[list[int]], str],
    answer_ids: AnswerTokenIds,
    finder: Callable[..., tuple[int | None, str | None]],
    *,
    window: int = VERDICT_SLOT_WINDOW,
) -> tuple[float, str | None, float | None, float | None]:
    """Continuous P(YES) at a generated YES/NO slot located by `finder` + the decoded label.

    `finder` is find_verdict_slot (CoT) or find_answer_slot (noshot/fewshot): both return
    (slot_index, label). One copy of the softmax/logmass logic for every prompt style. Returns
    (score, label, yes_logmass, no_logmass); NaN/None if the finder found no clean slot. `gen_ids`
    and `logprobs` are aligned per-generated-token (vLLM out.outputs[0].token_ids / .logprobs, each
    Logprob already unwrapped to its .logprob float).
    """
    slot, label = finder(gen_ids, decode, answer_ids, window=window)
    if slot is None or slot >= len(logprobs):
        return float("nan"), None, None, None
    position = logprobs[slot]
    score = score_from_prompt_logprobs(position, answer_ids)
    ly = [position[i] for i in answer_ids.yes_ids if i in position]
    ln = [position[i] for i in answer_ids.no_ids if i in position]
    return score, label, (max(ly) if ly else None), (max(ln) if ln else None)


def score_generated_verdict_slot(
    gen_ids: list[int],
    logprobs: list[dict[int, float]],
    decode: Callable[[list[int]], str],
    answer_ids: AnswerTokenIds,
    *,
    window: int = VERDICT_SLOT_WINDOW,
) -> tuple[float, str | None, float | None, float | None]:
    """CoT readout: P(YES) at the model's own VERDICT slot. Thin wrapper over score_generated_slot
    with the verdict finder; public signature + outputs unchanged (parity-tested)."""
    return score_generated_slot(
        gen_ids, logprobs, decode, answer_ids, find_verdict_slot, window=window
    )
