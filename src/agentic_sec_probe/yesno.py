"""Shared YES/NO verdict parser for prompted-baseline scoring.

Used by the SVEN prompted eval (05) and PatchEval prompted eval (12) so the parse
logic lives in one place (was duplicated and diverged-prone).

Two bugs this fixes vs the old per-script copies:

1. Byte-level BPE markers. Some tokenizers (DeepSeek-Coder, GPT-2
   family) decode raw token strings with the byte-level BPE convention where a
   leading space is rendered as 'Ġ' (U+0120) and a newline as 'Ċ' (U+010A). A
   confident "Yes" arrives as "ĠYes"; an old `startswith("YES")` / `\b...\b` regex
   then nulled it, deflating the prompted baseline (38/39 DeepSeek confident-YES OOD
   responses were dropped). We strip these markers before parsing.

2. Leading-prose false verdicts. `startswith("NO")` mislabels
   free-text like "No obvious issue, but YES it is vulnerable" as NO. We extract the
   LAST explicit verdict token instead, which is correct for reason-then-answer
   outputs (and identical to startswith for clean one-word answers).

Note: Phase 2's logprob scoring makes this parser unnecessary for the HEADLINE
(it reads YES/NO logits directly). This parser is for the legacy/free-text paths.
"""

from __future__ import annotations

import re

# Byte-level BPE glyphs: 'Ġ' = encoded space, 'Ċ' = encoded newline.
# Map them back to their real characters before parsing.
_BYTE_BPE_TRANSLATION = str.maketrans({"Ġ": " ", "Ċ": "\n", "Ā": " "})

_VERDICT_RE = re.compile(r"\b(YES|NO)\b")


def normalize_byte_bpe(text: str) -> str:
    """Replace byte-level BPE markers (Ġ/Ċ) with the characters they encode."""
    return text.translate(_BYTE_BPE_TRANSLATION)


def parse_yesno(text: str) -> str | None:
    """Extract a YES/NO verdict from a model response.

    Strips byte-BPE markers, then returns the LAST explicit YES or NO token (so a
    reason-then-verdict response is read by its final answer, not a leading prose
    word). Returns None if neither appears.
    """
    cleaned = normalize_byte_bpe(text).strip().upper()
    if not cleaned:
        return None
    matches = _VERDICT_RE.findall(cleaned)
    return matches[-1] if matches else None
