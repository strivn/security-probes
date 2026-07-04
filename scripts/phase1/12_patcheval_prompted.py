"""Prompted LLM baseline on PatchEval CVEs.

For each model's instruct variant, on each PatchEval CVE:
  1. Prompt "Is this function vulnerable?" on vul_func → parse YES/NO
  2. Prompt same question on fix_func → parse YES/NO
  3. delta = score(vul) - score(fix), where YES=1, NO=0
  4. Correct if delta > 0

Each model runs in a subprocess for full GPU memory isolation.

Input:  data/patcheval/python_cves.json
Output: outputs/phase1/patcheval_prompted_{slug}.json (per model)

Runs on GPU. ~30-60 min per model.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

load_dotenv(PROJECT / ".env")

from agentic_sec_probe.models import MODEL_REGISTRY
from agentic_sec_probe.yesno import parse_yesno

SVEN_TO_PATCHEVAL_CWE = {
    "cwe-022": "CWE-22",
    "cwe-078": "CWE-78",
    "cwe-079": "CWE-79",
    "cwe-089": "CWE-89",
}
IN_DIST_CWES = set(SVEN_TO_PATCHEVAL_CWE.values())

TASK_B_PROMPT = (
    "You are a security code reviewer."
    " Analyze the following function for security vulnerabilities.\n\n"
    "```\n{code}\n```\n\n"
    "Does this function contain a security vulnerability?"
    " Answer with exactly one word: YES or NO"
)

MAX_INPUT_TOKENS = 8192  # Raised from 3072 — all models support ≥8K context
CHECKPOINT_EVERY = 25  # Atomically write checkpoint every N prompts

WORKER_SCRIPT = """
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id     = sys.argv[1]
prompts_file = sys.argv[2]
output_file  = sys.argv[3]  # doubles as checkpoint path (deterministic)
quant        = sys.argv[4]
max_tokens   = int(sys.argv[5])
ckpt_every   = int(sys.argv[6])

with open(prompts_file) as f:
    prompts = json.load(f)

# Resume from existing checkpoint if present
responses = []
truncated = []
if os.path.exists(output_file):
    try:
        with open(output_file) as f:
            data = json.load(f)
        responses = list(data.get("responses", []))
        truncated = list(data.get("truncated", []))
        if len(responses) != len(truncated) or len(responses) > len(prompts):
            responses, truncated = [], []
    except (json.JSONDecodeError, OSError):
        responses, truncated = [], []

start_idx = len(responses)
if start_idx >= len(prompts):
    print(f"  All {len(prompts)} prompts already done", flush=True)
    sys.exit(0)
if start_idx > 0:
    print(f"  Resuming from prompt {start_idx}/{len(prompts)}", flush=True)

def save_ckpt():
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"responses": responses, "truncated": truncated}, f)
    os.replace(tmp, output_file)

tokenizer = AutoTokenizer.from_pretrained(model_id)

load_kwargs = {"device_map": "auto"}
if quant == "8bit":
    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
elif quant == "4bit":
    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
else:
    load_kwargs["torch_dtype"] = torch.float16

model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

input_device = next(model.parameters()).device

for i in range(start_idx, len(prompts)):
    prompt = prompts[i]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_tokens,
    ).to(input_device)
    was_truncated = len(tokenizer.encode(text)) > max_tokens
    truncated.append(was_truncated)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=16, do_sample=False,
            temperature=None, top_p=None,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    if (i + 1) % ckpt_every == 0:
        save_ckpt()
        n_trunc = sum(truncated)
        print(f"  [{i+1}/{len(prompts)}] (truncated: {n_trunc}) ckpt", flush=True)

save_ckpt()
"""


def run_model_subprocess(
    model_id: str,
    prompts: list[str],
    quantization: str | None,
    checkpoint_path: Path,
) -> tuple[list[str], list[bool]]:
    """Run model inference in subprocess for GPU memory isolation.

    Worker writes a checkpoint at ``checkpoint_path`` every
    ``CHECKPOINT_EVERY`` prompts so a killed run can resume.
    """
    print(f"\n  Loading {model_id} (subprocess)...", flush=True)
    print(f"  Checkpoint: {checkpoint_path}", flush=True)
    t0 = time.time()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(prompts, pf)
        prompts_path = pf.name

    script_path = tempfile.mktemp(suffix=".py")
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    quant_arg = quantization if quantization else "none"
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            model_id,
            prompts_path,
            str(checkpoint_path),
            quant_arg,
            str(MAX_INPUT_TOKENS),
            str(CHECKPOINT_EVERY),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=7200,
    )

    elapsed = time.time() - t0

    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)

    if result.returncode != 0:
        err_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        print(f"  STDERR: {err_msg}", flush=True)
        msg = f"Subprocess exit {result.returncode}"
        raise RuntimeError(msg)

    with open(checkpoint_path) as f:
        data: dict[str, Any] = json.load(f)
    responses: list[str] = data["responses"]
    truncated: list[bool] = data.get("truncated", [False] * len(responses))
    n_trunc = sum(truncated)
    print(f"  Done in {elapsed:.0f}s ({len(responses)} responses, {n_trunc} truncated)", flush=True)

    for p in [prompts_path, script_path]:
        with contextlib.suppress(OSError):
            os.unlink(p)

    return responses, truncated


def process_model(
    slug: str,
    cves: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Run prompted classification on PatchEval vul/fix pairs for one model."""
    spec = MODEL_REGISTRY[slug]

    # Build prompts: interleaved [vul_0, fix_0, vul_1, fix_1, ...]
    prompts: list[str] = []
    cve_meta: list[dict[str, Any]] = []
    for cve in cves:
        cve_id = cve["cve_id"]
        cwe_info = cve.get("cwe_info", {})
        cwe = list(cwe_info.keys())[0] if cwe_info else "unknown"

        vul_code = cve["vul_func"][0].get("snippet", "")
        fix_code = cve["fix_func"][0].get("snippet", "")

        if not vul_code or not fix_code:
            continue

        prompts.append(TASK_B_PROMPT.format(code=vul_code))
        prompts.append(TASK_B_PROMPT.format(code=fix_code))
        cve_meta.append({"cve_id": cve_id, "cwe": cwe})

    print(f"  {len(cve_meta)} CVEs, {len(prompts)} prompts", flush=True)

    # Deterministic checkpoint path so a killed run can resume
    checkpoint_path = out_dir / f"_ckpt_patcheval_prompted_{slug}.json"

    # Run inference
    responses, truncated_flags = run_model_subprocess(
        spec.hf_instruct_id,
        prompts,
        spec.quantization,
        checkpoint_path,
    )

    # Parse responses into per-CVE results
    per_cve: list[dict[str, Any]] = []
    deltas: list[float] = []

    for i, meta in enumerate(cve_meta):
        vul_raw = responses[2 * i]
        fix_raw = responses[2 * i + 1]

        vul_parsed = parse_yesno(vul_raw)
        fix_parsed = parse_yesno(fix_raw)

        vul_score = 1.0 if vul_parsed == "YES" else 0.0
        fix_score = 1.0 if fix_parsed == "YES" else 0.0
        delta = vul_score - fix_score

        deltas.append(delta)
        per_cve.append(
            {
                "cve_id": meta["cve_id"],
                "cwe": meta["cwe"],
                "vul_response": vul_raw[:200],
                "fix_response": fix_raw[:200],
                "vul_parsed": vul_parsed,
                "fix_parsed": fix_parsed,
                "vul_score": vul_score,
                "fix_score": fix_score,
                "delta": delta,
                "vul_truncated": truncated_flags[2 * i],
                "fix_truncated": truncated_flags[2 * i + 1],
            }
        )

    deltas_arr = np.array(deltas)
    n = len(deltas_arr)
    n_correct = int(np.sum(deltas_arr > 0))
    n_wrong = int(np.sum(deltas_arr < 0))
    n_tied = int(np.sum(deltas_arr == 0))

    # Wilcoxon on non-zero deltas
    nonzero = deltas_arr[deltas_arr != 0]
    if len(nonzero) >= 10:
        stat, p_value = stats.wilcoxon(nonzero)
        wilcoxon = {"stat": float(stat), "p_value": float(p_value), "n_nonzero": len(nonzero)}
    else:
        wilcoxon = {"stat": None, "p_value": None, "n_nonzero": len(nonzero)}

    # In-dist vs OOD
    id_deltas = np.array([r["delta"] for r in per_cve if r["cwe"] in IN_DIST_CWES])
    ood_deltas = np.array([r["delta"] for r in per_cve if r["cwe"] not in IN_DIST_CWES])

    # Per-CWE
    from collections import defaultdict

    cwe_groups: dict[str, list[float]] = defaultdict(list)
    for r in per_cve:
        cwe_groups[r["cwe"]].append(r["delta"])

    per_cwe_summary: dict[str, dict[str, Any]] = {}
    for cwe in sorted(cwe_groups, key=lambda c: len(cwe_groups[c]), reverse=True):
        cwe_d = np.array(cwe_groups[cwe])
        cwe_n = len(cwe_d)
        per_cwe_summary[cwe] = {
            "n": cwe_n,
            "n_correct": int(np.sum(cwe_d > 0)),
            "n_wrong": int(np.sum(cwe_d < 0)),
            "n_tied": int(np.sum(cwe_d == 0)),
            "accuracy": round(float(np.sum(cwe_d > 0)) / cwe_n, 4) if cwe_n > 0 else 0,
        }

    n_unparseable_vul = sum(1 for r in per_cve if r["vul_parsed"] is None)
    n_unparseable_fix = sum(1 for r in per_cve if r["fix_parsed"] is None)

    output = {
        "slug": slug,
        "model_id": spec.hf_instruct_id,
        "n_cves": n,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_tied": n_tied,
        "accuracy": round(n_correct / n, 4) if n > 0 else 0,
        "n_unparseable_vul": n_unparseable_vul,
        "n_unparseable_fix": n_unparseable_fix,
        "wilcoxon": wilcoxon,
        "in_distribution": {
            "n": len(id_deltas),
            "accuracy": round(float(np.sum(id_deltas > 0)) / len(id_deltas), 4)
            if len(id_deltas) > 0
            else 0,
        },
        "out_of_distribution": {
            "n": len(ood_deltas),
            "accuracy": round(float(np.sum(ood_deltas > 0)) / len(ood_deltas), 4)
            if len(ood_deltas) > 0
            else 0,
        },
        "per_cwe": per_cwe_summary,
        "per_cve": per_cve,
    }

    out_path = out_dir / f"patcheval_prompted_{slug}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Final output persisted; drop the checkpoint
    with contextlib.suppress(OSError):
        checkpoint_path.unlink()

    print(f"\n  Saved: {out_path}", flush=True)
    print(f"  Correct: {n_correct}/{n} ({n_correct / n:.1%})", flush=True)
    print(f"  Tied: {n_tied}, Wrong: {n_wrong}", flush=True)
    if wilcoxon["p_value"] is not None:
        print(f"  Wilcoxon p={wilcoxon['p_value']:.2e}", flush=True)


def main() -> None:
    cve_path = PROJECT / "data" / "patcheval" / "python_cves.json"
    with open(cve_path) as f:
        cves: list[dict[str, Any]] = json.load(f)

    valid = [c for c in cves if c.get("vul_func") and c.get("fix_func")]
    print(f"PatchEval CVEs: {len(valid)}/{len(cves)} with both vul and fix", flush=True)

    out_dir = PROJECT / "outputs" / "phase1"

    # Order largest-first for clean GPU
    ordered_slugs = sorted(
        MODEL_REGISTRY.keys(),
        key=lambda s: MODEL_REGISTRY[s].min_vram_gb,
        reverse=True,
    )

    for slug in ordered_slugs:
        out_path = out_dir / f"patcheval_prompted_{slug}.json"
        if out_path.exists():
            print(f"\nSKIP {slug}: results already exist", flush=True)
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"PatchEval prompted: {slug}", flush=True)
        print(f"{'=' * 60}", flush=True)

        try:
            process_model(slug, valid, out_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
