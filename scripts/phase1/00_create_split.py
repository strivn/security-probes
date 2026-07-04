"""Create stratified 80:20 train/eval split at PROJECT level.

Merges SVEN train + val into all available pairs. Splits per-CWE with seed=42 so
every CWE has ~80% train representation.

The split unit is the GitHub PROJECT (owner/repo parsed from commit_link), NOT the
individual pair. Every pair from one project lands in the same split. This is
strictly stronger than pair-level grouping:

  - pair-level (old) kept a function's own (vulnerable, secure) versions together,
    but let two DIFFERENT functions fixed by the same commit / same project land on
    opposite sides. A high-capacity probe can then key on project-specific idioms
    (helper names, variable names, the author's fix convention) and "pass" an eval
    pair by recognizing the codebase, not the vulnerability. Measured cross-split
    same-commit overlap under the old split: CWE-089 98%, CWE-078 77%, CWE-022 58%,
    CWE-079 50% of eval pairs (C/C++ CWEs 0%). See Allamanis 2019 (arXiv:1812.06469)
    on duplication inflating code-ML metrics.
  - project-level (this) guarantees no codebase appears in both train and eval, so
    the eval set is a genuine held-out generalization test.

The (vulnerable, secure) pair of a single function is still never split (both
samples share a pair_id and inherit the pair's project assignment).

Note: CWE-078 (command injection) is dominated by a single project (~60% of pairs).
Its split is still leakage-free, but its result reflects transfer from one codebase
rather than broad within-CWE generalization — disclose this in the paper.

Output: data/splits/phase1_split.json

Determinism: seeded project shuffle + size-descending greedy fill. Same input -> same
split. Asserts no project straddles train/eval (leakage guard).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.data import load_sven_pairs, pairs_to_samples

SEED = 42
TRAIN_RATIO = 0.8

_PROJECT_RE = re.compile(r"github\.com/([^/]+/[^/]+)/commit/", re.IGNORECASE)


def project_of(commit_link: str) -> str:
    """Parse 'owner/repo' from a GitHub commit_link.

    Falls back to the full commit_link if the pattern does not match, so an
    unexpected URL becomes its own singleton group rather than silently merging
    with another project. All 803 current SVEN links match the pattern (verified).
    """
    m = _PROJECT_RE.search(commit_link)
    return m.group(1) if m else commit_link


def assign_projects_globally(
    all_pairs_project: list[str],
    all_pairs_cwe: list[str],
    rng: np.random.RandomState,
) -> dict[str, str]:
    """Assign each project ENTIRELY to 'train' or 'eval' (global, cross-CWE).

    A project is one GitHub repo; 37/270 SVEN projects span multiple CWEs (e.g.
    ImageMagick appears in 6 CWEs). A per-CWE split would put such a project in
    train for one CWE and eval for another, leaking repo idioms across CWEs. So we
    assign the WHOLE project to one side regardless of CWE.

    Greedy, deficit-driven: process projects largest-first (seeded shuffle breaks
    ties), and place each project on the side whose per-CWE train/eval quota is
    currently furthest from being met. This holds every CWE near TRAIN_RATIO even
    though assignment is global. Deterministic for a fixed seed.
    """
    cwe_total: dict[str, int] = defaultdict(int)
    for cwe in all_pairs_cwe:
        cwe_total[cwe] += 1

    proj_cwe_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    proj_size: dict[str, int] = defaultdict(int)
    for proj, cwe in zip(all_pairs_project, all_pairs_cwe, strict=True):
        proj_cwe_counts[proj][cwe] += 1
        proj_size[proj] += 1

    proj_keys = list(proj_size.keys())
    rng.shuffle(proj_keys)
    proj_keys.sort(key=lambda k: -proj_size[k])  # largest project first

    train_cwe: dict[str, int] = defaultdict(int)
    eval_cwe: dict[str, int] = defaultdict(int)
    assignment: dict[str, str] = {}
    for proj in proj_keys:
        # How badly does each side still need this project's CWEs?
        train_need = sum(
            max(0.0, TRAIN_RATIO * cwe_total[cwe] - train_cwe[cwe]) for cwe in proj_cwe_counts[proj]
        )
        eval_need = sum(
            max(0.0, (1.0 - TRAIN_RATIO) * cwe_total[cwe] - eval_cwe[cwe])
            for cwe in proj_cwe_counts[proj]
        )
        side = "train" if train_need >= eval_need else "eval"
        assignment[proj] = side
        target = train_cwe if side == "train" else eval_cwe
        for cwe, n in proj_cwe_counts[proj].items():
            target[cwe] += n
    return assignment


def main() -> None:
    data_root = PROJECT / "data" / "raw"

    train_pairs = load_sven_pairs(data_root / "sven_train")
    val_pairs = load_sven_pairs(data_root / "sven_val")
    all_pairs = train_pairs + val_pairs
    print(f"Merged: {len(all_pairs)} pairs ({len(train_pairs)} train + {len(val_pairs)} val)")

    samples = pairs_to_samples(all_pairs)
    pair_project = [project_of(p.commit_link) for p in all_pairs]
    pair_cwe = [p.cwe for p in all_pairs]

    # Assign every project entirely to one split (global, cross-CWE), then read off
    # each pair's split from its project. Pair integrity is preserved for free: a
    # pair's two samples share a pair_id and thus the same project assignment.
    rng = np.random.RandomState(SEED)
    proj_assignment = assign_projects_globally(pair_project, pair_cwe, rng)

    train_ids: list[int] = []
    eval_ids: list[int] = []
    for pid in range(len(all_pairs)):
        (train_ids if proj_assignment[pair_project[pid]] == "train" else eval_ids).append(pid)

    # Report per-CWE outcome (stratification is an emergent property of the
    # deficit-greedy global assignment, not enforced per CWE).
    per_cwe: dict[str, dict[str, int]] = {}
    for cwe in sorted(set(pair_cwe)):
        ids = [i for i in range(len(all_pairs)) if pair_cwe[i] == cwe]
        tr = [i for i in ids if i in set(train_ids)]
        ev = [i for i in ids if i in set(eval_ids)]
        n_eval_proj = len({pair_project[i] for i in ev})
        per_cwe[cwe] = {
            "train": len(tr),
            "eval": len(ev),
            "total": len(ids),
            "eval_projects": n_eval_proj,
        }
        tr_pct = len(tr) / len(ids) * 100 if ids else 0.0
        print(
            f"  {cwe}: {len(tr)} train, {len(ev)} eval (total {len(ids)}, "
            f"{tr_pct:.0f}% train, {n_eval_proj} eval projects)"
        )

    # Leakage guard: assert no project appears in BOTH splits.
    train_id_set = set(train_ids)
    eval_id_set = set(eval_ids)
    train_projects = {pair_project[i] for i in train_ids}
    eval_projects = {pair_project[i] for i in eval_ids}
    straddling = train_projects & eval_projects
    if straddling:
        msg = (
            f"Project-level leakage: {len(straddling)} projects span both splits: "
            f"{sorted(straddling)[:5]}"
        )
        raise AssertionError(msg)
    print(
        f"\nLeakage guard OK: {len(train_projects)} train / "
        f"{len(eval_projects)} eval projects, 0 shared."
    )

    # Build per-sample metadata with split + project assignment.
    sample_meta = [
        {
            "pair_id": s.pair_id,
            "label": s.label,
            "cwe": s.cwe,
            "language": s.language,
            "project": pair_project[s.pair_id],
            "split": "train" if s.pair_id in train_id_set else "eval",
        }
        for s in samples
    ]

    output = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "group_by": "project",  # split unit (was "pair" previously)
        "n_pairs": len(all_pairs),
        "n_train_pairs": len(train_ids),
        "n_eval_pairs": len(eval_ids),
        "n_train_samples": len(train_ids) * 2,
        "n_eval_samples": len(eval_ids) * 2,
        "n_train_projects": len(train_projects),
        "n_eval_projects": len(eval_projects),
        "train_pair_ids": sorted(train_ids),
        "eval_pair_ids": sorted(eval_ids),
        "per_cwe": per_cwe,
        "samples": sample_meta,
    }

    out_path = PROJECT / "data" / "splits" / "phase1_split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")
    n_tr, n_ev = len(train_ids), len(eval_ids)
    print(f"Train: {n_tr} pairs ({n_tr * 2} samples) across {len(train_projects)} projects")
    print(f"Eval:  {n_ev} pairs ({n_ev * 2} samples) across {len(eval_projects)} projects")
    assert eval_id_set.isdisjoint(train_id_set), "pair-id overlap between splits"


if __name__ == "__main__":
    main()
