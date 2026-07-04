"""Logprob-based YES/NO scoring — the load-bearing primitive for the stronger
prompted baseline (Phase 2 elicitation suite).

Instead of generating text and parsing it, we read the model's probability of
answering YES vs NO directly from the next-token logits and convert it to a
continuous classification score:

    score = softmax([logp(YES), logp(NO)])[YES]

This is McKenzie et al. 2025 (arXiv:2506.10805) §2.2's method: "A softmax is applied
to the log-likelihood of the two continuations to create a classification probability."
It gives prompting a real ranking score (so it has an AUC + Wilson CI, comparable to
the probe) and removes two failure modes of text parsing at the source: the binary
YES/NO tie (every score is now continuous) and the byte-BPE marker bug (no decode/parse).

Two implementation rules from the literature cross-check (both verified):
  1. Use RAW logits, not processed `scores`. A single forward pass `model(**inputs)
     .logits[:, -1, :]` is rawest and avoids generation warpers entirely. (If using
     generate(), read `output_logits`, NOT `scores` — HF generation-utils docs.)
  2. Do NOT hardcode `tokenizer.encode("Yes")[0]`. Resolve the answer-token ids under
     the EXACT post-chat-template continuation and aggregate probability mass over
     surface variants {" Yes","Yes"," yes","yes",...}. A model may tokenize " Yes" and
     "Yes" to different ids; missing a variant silently deflates the score.

The pure functions here (id resolution, mass aggregation, softmax score) are unit-tested
on CPU with no model. The driver (scripts/phase1/13_*) wraps them in the GPU worker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

# Surface variants whose probability mass counts toward each answer. Lowercased
# comparison is NOT used — these are exact continuation strings a chat model emits
# right after the generation prompt; case matters to the tokenizer.
YES_VARIANTS = ("YES", "Yes", "yes", " YES", " Yes", " yes")
NO_VARIANTS = ("NO", "No", "no", " NO", " No", " no")


class _Tokenizer(Protocol):
    """Minimal tokenizer interface needed for id resolution (duck-typed)."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


@dataclass(frozen=True)
class AnswerTokenIds:
    """Resolved single-token ids for the YES and NO answer variants.

    yes_ids / no_ids are the vocab ids of every surface variant that tokenizes to a
    SINGLE token. multi_token_variants lists variants that did not (informational —
    if a whole side is empty, the caller must fall back to sequence scoring).
    """

    yes_ids: frozenset[int]
    no_ids: frozenset[int]
    multi_token_variants: tuple[str, ...]

    @property
    def usable(self) -> bool:
        """True if both YES and NO have at least one single-token id."""
        return bool(self.yes_ids) and bool(self.no_ids)


def resolve_answer_token_ids(
    tokenizer: _Tokenizer,
    yes_variants: tuple[str, ...] = YES_VARIANTS,
    no_variants: tuple[str, ...] = NO_VARIANTS,
) -> AnswerTokenIds:
    """Map YES/NO surface variants to vocab ids we read at the answer slot.

    A single-token variant contributes its id directly. A MULTI-token variant (e.g.
    DeepSeek splits " YES" into [" Y", "ES"]) contributes its FIRST sub-token: at the
    answer slot the model predicts that first token, and it discriminates YES from NO as
    long as the two sides' first sub-tokens differ. After collecting both sides we DROP
    any id that ended up on both (an ambiguous leading sub-token is not discriminative).

    Why first-sub-token, not drop: a model steered toward an ALL-CAPS verdict (the CoT
    "VERDICT: YES" prompt) emits exactly the multi-token form; dropping it floored DeepSeek
    to 0. Single-token tokenizers (Qwen/Llama) never hit the multi-token branch, so their
    resolved ids — and all committed results that used them — are unchanged.
    """
    yes_ids: set[int] = set()
    no_ids: set[int] = set()
    multi: list[str] = []

    def collect(variants: tuple[str, ...], dst: set[int]) -> None:
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            if len(ids) > 1:
                multi.append(variant)
            dst.add(ids[0])  # single id, or the discriminating FIRST sub-token

    collect(yes_variants, yes_ids)
    collect(no_variants, no_ids)

    # An id on BOTH sides cannot discriminate (e.g. a shared leading sub-token) -> drop it.
    ambiguous = yes_ids & no_ids
    yes_ids -= ambiguous
    no_ids -= ambiguous

    return AnswerTokenIds(
        yes_ids=frozenset(yes_ids),
        no_ids=frozenset(no_ids),
        multi_token_variants=tuple(multi),
    )


def _logsumexp(values: list[float]) -> float:
    """Numerically stable log-sum-exp over a list of log-probabilities/logits."""
    if not values:
        return -math.inf
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


def yes_score_from_logits(
    next_token_logits: list[float],
    answer_ids: AnswerTokenIds,
) -> float:
    """P(YES) = softmax([logp(YES mass), logp(NO mass)])[YES] from raw next-token logits.

    Aggregates probability mass over each side's surface-variant ids (sum in prob
    space = log-sum-exp of their logits), then applies a 2-way softmax over the two
    aggregated log-masses. Returns a value in (0, 1); 0.5 means YES and NO are
    equally likely (the model is unsure — this subsumes abstention).

    `next_token_logits` is the raw logit vector over the full vocab at the answer
    position (a single forward pass: model(**inputs).logits[0, -1, :].tolist()).
    """
    if not answer_ids.usable:
        msg = "answer_ids has an empty YES or NO side; fall back to sequence scoring"
        raise ValueError(msg)
    vocab = len(next_token_logits)
    yes_logits = [next_token_logits[i] for i in answer_ids.yes_ids if i < vocab]
    no_logits = [next_token_logits[i] for i in answer_ids.no_ids if i < vocab]
    if not yes_logits or not no_logits:
        msg = "resolved answer id out of vocab range; tokenizer/model mismatch"
        raise ValueError(msg)
    log_yes = _logsumexp(yes_logits)
    log_no = _logsumexp(no_logits)
    # 2-way softmax over the aggregated log-masses == sigmoid(log_yes - log_no).
    diff = log_yes - log_no
    if diff >= 0:
        return 1.0 / (1.0 + math.exp(-diff))
    e = math.exp(diff)
    return e / (1.0 + e)
