"""Tests for the shared YES/NO verdict parser."""

from __future__ import annotations

from agentic_sec_probe.yesno import normalize_byte_bpe, parse_yesno

# The actual byte-level BPE glyphs (verified codepoints).
G = "Ġ"  # 'Ġ' = encoded leading space
C = "Ċ"  # 'Ċ' = encoded newline


class TestNormalizeByteBpe:
    def test_glyph_codepoints(self) -> None:
        assert G == "Ġ"
        assert C == "Ċ"

    def test_space_marker(self) -> None:
        assert normalize_byte_bpe(f"{G}Yes") == " Yes"

    def test_newline_marker(self) -> None:
        assert normalize_byte_bpe(f"Yes{C}") == "Yes\n"

    def test_plain_text_unchanged(self) -> None:
        assert normalize_byte_bpe("Yes, vulnerable") == "Yes, vulnerable"


class TestParseYesno:
    def test_clean_yes(self) -> None:
        assert parse_yesno("YES") == "YES"
        assert parse_yesno("yes") == "YES"
        assert parse_yesno("Yes, it is vulnerable.") == "YES"

    def test_clean_no(self) -> None:
        assert parse_yesno("NO") == "NO"
        assert parse_yesno("No, looks safe.") == "NO"

    def test_byte_bpe_yes_recovered(self) -> None:
        # The DeepSeek bug: 'ĠYes' was nulled by the old parser.
        assert parse_yesno(f"{G}Yes") == "YES"
        assert parse_yesno(f"{G}YES{C}") == "YES"

    def test_byte_bpe_no_recovered(self) -> None:
        assert parse_yesno(f"{G}No") == "NO"

    def test_last_verdict_wins_over_leading_prose(self) -> None:
        # prompt-4: old startswith("NO") mislabeled this as NO.
        assert parse_yesno("No obvious issue at first, but YES it is vulnerable") == "YES"

    def test_reason_then_verdict(self) -> None:
        assert parse_yesno("The query uses string formatting.\nVERDICT: YES") == "YES"
        assert parse_yesno("This is parameterized.\nVERDICT: NO") == "NO"

    def test_no_verdict_returns_none(self) -> None:
        assert parse_yesno("I am not sure about this code") is None
        assert parse_yesno("") is None
        assert parse_yesno(f"{G}{C}  ") is None

    def test_substring_not_matched(self) -> None:
        # 'NOTE' / 'YESTERDAY' must not be read as verdicts (word-boundary).
        assert parse_yesno("NOTE: nothing here") is None
        assert parse_yesno("YESTERDAY we shipped") is None
