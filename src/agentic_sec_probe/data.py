"""SVEN dataset loader for probe training.

Each SVEN entry is a (vulnerable, secure) function pair from a security-fix commit.
The pairs are natural contrast pairs: same function, minimal edit, security-specific change.

Data structure per entry:
    func_name:       function name
    func_src_before: vulnerable version (pre-fix)
    func_src_after:  secure version (post-fix)
    commit_link:     GitHub commit URL
    file_name:       source file name (determines language: .py vs .c/.cpp)
    vul_type:        CWE type (from filename, e.g. "cwe-089")
    line_changes:    dict with 'deleted' and 'added' lists of changed lines
    char_changes:    dict with 'deleted' and 'added' lists of changed character spans
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CWE_DESCRIPTIONS: dict[str, str] = {
    "cwe-022": "Path Traversal",
    "cwe-078": "OS Command Injection",
    "cwe-079": "Cross-site Scripting (XSS)",
    "cwe-089": "SQL Injection",
    "cwe-125": "Out-of-bounds Read",
    "cwe-190": "Integer Overflow",
    "cwe-416": "Use After Free",
    "cwe-476": "NULL Pointer Dereference",
    "cwe-787": "Out-of-bounds Write",
}

PYTHON_CWES = {"cwe-022", "cwe-078", "cwe-079", "cwe-089"}
C_CPP_CWES = {"cwe-125", "cwe-190", "cwe-416", "cwe-476", "cwe-787"}


@dataclass(frozen=True)
class SVENPair:
    """A single vulnerable/secure function pair from SVEN."""

    func_name: str
    func_vulnerable: str
    func_secure: str
    cwe: str
    language: Literal["python", "c_cpp"]
    file_name: str
    commit_link: str
    line_changes: dict[str, Any]
    char_changes: dict[str, Any]

    @property
    def n_lines_changed(self) -> int:
        """Total number of lines deleted + added in the fix."""
        return len(self.line_changes.get("deleted", [])) + len(self.line_changes.get("added", []))

    @property
    def n_chars_changed(self) -> int:
        """Total number of character spans deleted + added in the fix."""
        deleted = sum(len(c.get("chars", "")) for c in self.char_changes.get("deleted", []))
        added = sum(len(c.get("chars", "")) for c in self.char_changes.get("added", []))
        return deleted + added


@dataclass(frozen=True)
class SVENSample:
    """A single labeled code sample (one side of a pair)."""

    code: str
    label: Literal[0, 1]  # 0 = secure, 1 = vulnerable
    cwe: str
    language: Literal["python", "c_cpp"]
    pair_id: int  # links back to the pair


def _detect_language(file_name: str) -> Literal["python", "c_cpp"]:
    if file_name.endswith(".py"):
        return "python"
    return "c_cpp"


def load_sven_pairs(
    data_dir: str | Path,
    *,
    language: Literal["python", "c_cpp", "all"] = "all",
    cwes: set[str] | None = None,
) -> list[SVENPair]:
    """Load SVEN pairs from a directory of per-CWE JSONL files.

    Args:
        data_dir: path containing cwe-*.jsonl files
        language: filter by language ("python", "c_cpp", or "all")
        cwes: filter by specific CWE types (e.g. {"cwe-089", "cwe-078"})

    Returns:
        list of SVENPair objects
    """
    data_dir = Path(data_dir)
    pairs: list[SVENPair] = []

    for jsonl_path in sorted(data_dir.glob("cwe-*.jsonl")):
        cwe = jsonl_path.stem  # e.g. "cwe-089"

        if cwes is not None and cwe not in cwes:
            continue

        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                lang = _detect_language(entry["file_name"])

                if language != "all" and lang != language:
                    continue

                pairs.append(
                    SVENPair(
                        func_name=entry["func_name"],
                        func_vulnerable=entry["func_src_before"],
                        func_secure=entry["func_src_after"],
                        cwe=cwe,
                        language=lang,
                        file_name=entry["file_name"],
                        commit_link=entry["commit_link"],
                        line_changes=entry["line_changes"],
                        char_changes=entry["char_changes"],
                    )
                )

    return pairs


def pairs_to_samples(pairs: list[SVENPair]) -> list[SVENSample]:
    """Flatten pairs into individual labeled samples.

    Each pair produces two samples: one vulnerable (label=1), one secure (label=0).
    The pair_id links them back together.
    """
    samples: list[SVENSample] = []
    for i, pair in enumerate(pairs):
        samples.append(
            SVENSample(
                code=pair.func_vulnerable,
                label=1,
                cwe=pair.cwe,
                language=pair.language,
                pair_id=i,
            )
        )
        samples.append(
            SVENSample(
                code=pair.func_secure,
                label=0,
                cwe=pair.cwe,
                language=pair.language,
                pair_id=i,
            )
        )
    return samples
