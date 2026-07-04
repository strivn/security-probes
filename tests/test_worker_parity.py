"""Parity test: the inline scoring in the GPU worker (a self-contained temp script
string in 13_*.py) must match the unit-tested module logprob_scoring.

The worker re-implements yes_score_from_logits inline because it runs as a detached
subprocess temp file. This test extracts that inline function and asserts it returns
identical scores to the module across many random logit vectors, so the two copies
can never silently drift.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from agentic_sec_probe.logprob_scoring import AnswerTokenIds, yes_score_from_logits

PROJECT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT / "scripts" / "phase1" / "13_patcheval_prompted_elicit.py"


def _load_worker_score():
    """Build the worker's inline yes_score_from_logits with fixed YES/NO id sets.

    Reads WORKER_SCRIPT as TEXT (does not import 13_*.py, which pulls torch/dotenv).
    """
    source = DRIVER.read_text()
    # Extract the WORKER_SCRIPT = r\"\"\"...\"\"\" block.
    m = re.search(r'WORKER_SCRIPT = r"""(.*?)"""', source, re.DOTALL)
    assert m, "WORKER_SCRIPT block not found"
    worker_src = m.group(1)

    # Execute only the pure helpers (logsumexp + yes_score_from_logits) with fixed
    # YES_IDS/NO_IDS, avoiding the torch/model parts of the worker string.
    ns: dict[str, object] = {"math": math, "YES_IDS": {10, 11}, "NO_IDS": {20}}
    lines = worker_src.splitlines()

    def grab(fn_start: str) -> str:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(fn_start))
        end = start + 1
        while end < len(lines) and (
            lines[end].startswith(("    ", "\t")) or not lines[end].strip()
        ):
            end += 1
        return "\n".join(lines[start:end])

    # yes_score_from_logits composes logsumexp + yes_no_logmass + score_from_logmass; exec
    # all of them so the parity guard tests the REAL worker composition (not a stale copy).
    exec(grab("def logsumexp"), ns)  # noqa: S102 - trusted in-repo worker string
    exec(grab("def yes_no_logmass"), ns)  # noqa: S102
    exec(grab("def score_from_logmass"), ns)  # noqa: S102
    exec(grab("def yes_score_from_logits"), ns)  # noqa: S102
    return ns["yes_score_from_logits"]


def test_worker_score_matches_module() -> None:
    worker_score = _load_worker_score()
    ids = AnswerTokenIds(frozenset({10, 11}), frozenset({20}), ())
    vocab = 50
    # Deterministic spread of logit configs (no RNG: vary by index).
    for k in range(40):
        row = [-10.0] * vocab
        row[10] = (k % 7) - 3.0
        row[11] = (k % 5) - 2.0
        row[20] = (k % 9) - 4.0
        expected = yes_score_from_logits(row, ids)
        got = worker_score(row)
        assert abs(got - expected) < 1e-9, f"drift at k={k}: worker={got} module={expected}"
