"""Prompted LLM baseline: single-function vulnerability classification.

For each model's instruct variant, on ALL eval samples:
  Prompt: "Does this function contain a security vulnerability? YES or NO"
  Parse yes/no → predicted label

Each model runs in a subprocess for full GPU memory isolation.
Computes AUC, accuracy, F1 per scope (universal, python, c_cpp) and per CWE.

Input:  data/splits/phase1_split.json
        data/raw/sven_train/, data/raw/sven_val/
Output: outputs/phase1/prompted_eval_{slug}.json (per model)

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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

load_dotenv(PROJECT / ".env")

from agentic_sec_probe.data import C_CPP_CWES, PYTHON_CWES, load_sven_pairs, pairs_to_samples
from agentic_sec_probe.models import MODEL_REGISTRY
from agentic_sec_probe.yesno import parse_yesno

TASK_B_PROMPT = (
    "You are a security code reviewer."
    " Analyze the following function for security vulnerabilities.\n\n"
    "```\n{code}\n```\n\n"
    "Does this function contain a security vulnerability?"
    " Answer with exactly one word: YES or NO"
)

SCOPES: dict[str, set[str] | None] = {
    "universal": None,
    "python": PYTHON_CWES,
    "c_cpp": C_CPP_CWES,
}

# Worker script runs in subprocess — full GPU memory released on exit
MAX_INPUT_TOKENS = 8192  # Raised from 3072 — all models support ≥8K context
CHECKPOINT_EVERY = 25  # Atomically write checkpoint every N prompts

WORKER_SCRIPT = """
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id     = sys.argv[1]
prompts_file = sys.argv[2]
output_file  = sys.argv[3]  # doubles as checkpoint path (deterministic)
quant        = sys.argv[4]  # "8bit", "4bit", or "none"
max_tokens   = int(sys.argv[5])  # max input tokens (truncation limit)
ckpt_every   = int(sys.argv[6])  # save every N prompts

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

# fp16 for ALL models: the bitsandbytes int8/4-bit path is a quant confound
# vs the fp16 probe extraction and triggered the ops.cu:388 crash. The `quant` arg is kept
# for the call signature but forced to full precision here.
load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

input_device = next(model.parameters()).device

# Backend split (matches script 13): Devstral/Mistral has NO Jinja chat template; HF
# apply_chat_template byte-BPE-mangles its [INST]/<s> markers into literal-text ids
# (control-token soup). Route it through mistral-common encode_chat_completion. Models with a
# proper Jinja template (Qwen/DeepSeek/Llama) use apply_chat_template(tokenize=False) and are
# re-tokenized with add_special_tokens=False to avoid a DOUBLE BOS (the template already adds it).
USE_MISTRAL_COMMON = tokenizer.chat_template is None
if USE_MISTRAL_COMMON:
    from huggingface_hub import hf_hub_download
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    _mt = MistralTokenizer.from_file(hf_hub_download(repo_id=model_id, filename="tekken.json"))

def chat_input_ids(user_prompt):
    if USE_MISTRAL_COMMON:
        toks = _mt.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=user_prompt)])
        ).tokens
        was_trunc = len(toks) > max_tokens
        return torch.tensor([toks[:max_tokens]], device=input_device), was_trunc
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    full = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    was_trunc = full["input_ids"].shape[1] > max_tokens
    enc = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=max_tokens, return_tensors="pt")
    return enc["input_ids"].to(input_device), was_trunc

for i in range(start_idx, len(prompts)):
    prompt = prompts[i]
    input_ids, was_truncated = chat_input_ids(prompt)
    truncated.append(was_truncated)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=16, do_sample=False,
            temperature=None, top_p=None,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
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
    """Run model inference in a subprocess for GPU memory isolation.

    The worker writes a checkpoint file at ``checkpoint_path`` every
    ``CHECKPOINT_EVERY`` prompts, so a killed run can resume by re-invoking
    with the same path.

    Returns (responses, truncated_flags).
    """
    print(f"\n  Loading {model_id} (subprocess)...")
    print(f"  Checkpoint: {checkpoint_path}")
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
        timeout=3600,
    )

    elapsed = time.time() - t0

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.returncode != 0:
        err_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        print(f"  STDERR: {err_msg}")
        msg = f"Subprocess exit {result.returncode}"
        raise RuntimeError(msg)

    with open(checkpoint_path) as f:
        data: dict[str, Any] = json.load(f)
    responses: list[str] = data["responses"]
    truncated: list[bool] = data.get("truncated", [False] * len(responses))
    n_trunc = sum(truncated)
    print(f"  Done in {elapsed:.0f}s ({len(responses)} responses, {n_trunc} truncated)")

    for p in [prompts_path, script_path]:
        with contextlib.suppress(OSError):
            os.unlink(p)

    return responses, truncated


def compute_metrics(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> dict[str, float]:
    """Compute accuracy, F1, precision, recall for binary YES/NO predictions.

    AUC is intentionally NOT reported: prompted outputs are binary labels
    (YES/NO) so roc_auc_score is degenerate — it collapses to a linear
    transform of accuracy at a single decision threshold and inflates the
    apparent signal. Accuracy/F1/precision/recall are the honest metrics
    for binary classifier outputs.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def process_model(
    slug: str,
    samples_with_meta: list[tuple[Any, dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Run prompted classification for one model on all eval samples."""
    spec = MODEL_REGISTRY[slug]

    # Build prompts
    prompts = [TASK_B_PROMPT.format(code=sample.code) for sample, _meta in samples_with_meta]
    print(f"  {len(prompts)} prompts")

    # Deterministic checkpoint path so a killed run can resume
    checkpoint_path = out_dir / f"_ckpt_prompted_eval_{slug}.json"

    # Run inference
    responses, truncated_flags = run_model_subprocess(
        spec.hf_instruct_id,
        prompts,
        spec.quantization,
        checkpoint_path,
    )

    # Parse responses
    per_sample: list[dict[str, Any]] = []
    y_true_list: list[int] = []
    y_pred_list: list[int] = []

    for i, (_sample, meta) in enumerate(samples_with_meta):
        raw = responses[i]
        was_truncated = truncated_flags[i]
        parsed = parse_yesno(raw)
        predicted = 1 if parsed == "YES" else 0

        y_true_list.append(meta["label"])
        y_pred_list.append(predicted)

        per_sample.append(
            {
                "sample_idx": meta["idx"],
                "pair_id": meta["pair_id"],
                "cwe": meta["cwe"],
                "label": meta["label"],
                "predicted": predicted,
                "parsed": parsed,
                "truncated": was_truncated,
                "raw_response": raw[:200],
            }
        )

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Per-scope metrics
    scope_metrics: dict[str, Any] = {}
    for scope_name, cwe_filter in SCOPES.items():
        if cwe_filter is not None:
            mask = np.array([m["cwe"] in cwe_filter for _s, m in samples_with_meta])
        else:
            mask = np.ones(len(samples_with_meta), dtype=bool)

        if mask.sum() < 4:
            continue
        scope_metrics[scope_name] = {
            "n": int(mask.sum()),
            **compute_metrics(y_true[mask], y_pred[mask]),
        }

    # Per-CWE metrics
    per_cwe: dict[str, Any] = {}
    all_cwes = sorted({m["cwe"] for _s, m in samples_with_meta})
    for cwe in all_cwes:
        cwe_mask = np.array([m["cwe"] == cwe for _s, m in samples_with_meta])
        if cwe_mask.sum() < 4:
            continue

        cwe_true = y_true[cwe_mask]
        cwe_pred = y_pred[cwe_mask]

        vul_mask = cwe_true == 1
        sec_mask = cwe_true == 0
        tp = int(cwe_pred[vul_mask].sum()) if vul_mask.sum() > 0 else 0
        fp = int(cwe_pred[sec_mask].sum()) if sec_mask.sum() > 0 else 0

        per_cwe[cwe] = {
            "n": int(cwe_mask.sum()),
            **compute_metrics(cwe_true, cwe_pred),
            "true_positive_rate": round(tp / vul_mask.sum(), 4) if vul_mask.sum() > 0 else 0.0,
            "false_positive_rate": round(fp / sec_mask.sum(), 4) if sec_mask.sum() > 0 else 0.0,
        }

    # Unparseable and truncated counts
    n_unparseable = sum(1 for ps in per_sample if ps["parsed"] is None)
    n_truncated = sum(1 for ps in per_sample if ps["truncated"])

    output = {
        "slug": slug,
        "model_id": spec.hf_instruct_id,
        "n_samples": len(samples_with_meta),
        "n_unparseable": n_unparseable,
        "n_truncated": n_truncated,
        "scope_metrics": scope_metrics,
        "per_cwe": per_cwe,
        "per_sample": per_sample,
    }

    out_path = out_dir / f"prompted_eval_{slug}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Final output persisted; drop the checkpoint
    with contextlib.suppress(OSError):
        checkpoint_path.unlink()

    # Summary
    for scope_name, metrics in scope_metrics.items():
        print(
            f"  {scope_name}: acc={metrics['accuracy']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"P={metrics['precision']:.3f}, "
            f"R={metrics['recall']:.3f}"
        )
    if n_unparseable > 0:
        print(f"  Unparseable: {n_unparseable}/{len(per_sample)}")
    if n_truncated > 0:
        print(f"  Truncated: {n_truncated}/{len(per_sample)}")


def main() -> None:
    # Load split and samples
    split_path = PROJECT / "data" / "splits" / "phase1_split.json"
    with open(split_path) as f:
        split_data = json.load(f)

    data_root = PROJECT / "data" / "raw"
    train_pairs = load_sven_pairs(data_root / "sven_train")
    val_pairs = load_sven_pairs(data_root / "sven_val")
    all_pairs = train_pairs + val_pairs
    samples = pairs_to_samples(all_pairs)

    meta_list: list[dict[str, Any]] = split_data["samples"]

    # Filter to eval samples only
    samples_with_meta: list[tuple[Any, dict[str, Any]]] = []
    for i, (sample, meta) in enumerate(zip(samples, meta_list)):
        if meta["split"] == "eval":
            samples_with_meta.append((sample, {**meta, "idx": i}))

    print(f"Total eval samples: {len(samples_with_meta)}")

    out_dir = PROJECT / "outputs" / "phase1"

    # In-scope reviewer models. The full registry has more; the rest are out of
    # scope for the reported results (missing prompted data / OOD transfer / hardware).
    IN_SCOPE = [
        "qwen2.5-coder-7b",
        "qwen2.5-coder-14b",
        "deepseek-coder-33b",
        "devstral-small",
        "llama-3.1-8b",
    ]
    # Order largest-first so they get a clean GPU
    ordered_slugs = sorted(
        IN_SCOPE,
        key=lambda s: MODEL_REGISTRY[s].min_vram_gb,
        reverse=True,
    )

    for slug in ordered_slugs:
        out_path = out_dir / f"prompted_eval_{slug}.json"
        if out_path.exists():
            print(f"\nSKIP {slug}: prompted results already exist")
            continue

        print(f"\n{'=' * 60}")
        print(f"Prompted baseline: {slug}")
        print(f"{'=' * 60}")

        try:
            process_model(slug, samples_with_meta, out_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")


if __name__ == "__main__":
    main()
