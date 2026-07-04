"""Task 0 — vLLM viability probe (qwen2.5-coder-7b, ~6 real CVEs).

GATE for the whole vLLM CoT plan. Proves the MECHANISM (ignore speed):
  1. qwen-7b loads under vLLM in fp16 (matching the probe extraction dtype);
  2. batched LLM.generate produces CoT reasoning for a batch of prompts;
  3. a second batched call with prompt_logprobs reads the YES/NO logprob at the
     verdict slot (the two-pass design);
  4. score_from_prompt_logprobs turns that into a P(YES) in (0,1).

If this prints finite YES/NO logprobs + scores for every CVE snippet, vLLM is viable
and scaling to bigger models is a speed question, not a feasibility one. If we cannot
get clean verdict-slot logprobs here, vLLM is out -> fall back to the HF path.

This is a DIAGNOSTIC: it observes (loads a model, runs real inference, reads logits)
and prints what it observed. Run on the GPU box only:
    uv run python scripts/phase1/task0_vllm_probe.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agentic_sec_probe.logprob_scoring import resolve_answer_token_ids
from agentic_sec_probe.models import MODEL_REGISTRY
from agentic_sec_probe.patcheval import pair_cve_snippets
from agentic_sec_probe.vllm_cot import score_from_prompt_logprobs

N_CVES = 6
# Model slug from argv[1] (default qwen-7b). Lets one script probe each model in its own
# subprocess so the GPU is fully released between models (only one ~80GB model fits at a time).
MODEL_SLUG = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder-7b"
COT_PROMPT = (
    "You are a security code reviewer."
    " Analyze the following function for security vulnerabilities.\n\n"
    "```\n{code}\n```\n\n"
    "Think step by step about whether this function contains a security"
    " vulnerability, then end your answer with 'VERDICT: YES' or 'VERDICT: NO'."
)
VERDICT_MARKER = " VERDICT:"  # appended only if the model volunteered no verdict of its own

# Match the model's OWN verdict marker anywhere in the reasoning, tolerating markdown bold
# / spacing (e.g. "**VERDICT:**", "VERDICT :"). We cut the text right AFTER the colon so the
# next-token slot is the model's yes/no.
_VERDICT_RE = re.compile(r"\**\s*verdict\s*\**\s*:", re.IGNORECASE)


def cut_after_verdict(reasoning: str) -> str:
    """Return reasoning truncated to just after the model's own 'VERDICT:' (so the next-token
    slot is its yes/no). If the model never volunteered one, append our VERDICT_MARKER."""
    matches = list(_VERDICT_RE.finditer(reasoning))
    if matches:
        return reasoning[: matches[-1].end()]  # keep up to and including the last "VERDICT:"
    return reasoning + VERDICT_MARKER


def load_probe_snippets() -> list[tuple[str, str, str]]:
    """Return [(cve_id, which, code)] for the first few aligned vul/fix snippets."""
    cves = json.loads((PROJECT / "data/patcheval/python_cves.json").read_text())
    valid = [c for c in cves if c.get("vul_func") and c.get("fix_func")][:N_CVES]
    out: list[tuple[str, str, str]] = []
    for cve in valid:
        pairs, _ = pair_cve_snippets(cve["vul_func"], cve["fix_func"])
        if not pairs:
            continue
        vi, fi = pairs[0]
        vul = cve["vul_func"][vi].get("snippet", "")
        fix = cve["fix_func"][fi].get("snippet", "")
        if vul and fix:
            out.append((cve["cve_id"], "vul", vul))
            out.append((cve["cve_id"], "fix", fix))
    return out


def main() -> None:
    spec = MODEL_REGISTRY[MODEL_SLUG]
    model_id = spec.probe_model_id  # instruct checkpoint, same as the prompt arm
    print(f"Task 0 vLLM probe: {model_id} (fp16), {N_CVES} CVEs\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    answer_ids = resolve_answer_token_ids(tokenizer)
    print(f"YES ids={sorted(answer_ids.yes_ids)} NO ids={sorted(answer_ids.no_ids)}", flush=True)
    if not answer_ids.usable:
        print("FATAL: empty YES/NO id set", flush=True)
        sys.exit(2)

    snippets = load_probe_snippets()
    print(f"Loaded {len(snippets)} snippets ({len(snippets) // 2} CVEs)\n", flush=True)

    # fp16 to match the probe extraction numerics (models.py: quantization=None for in-scope).
    # max_model_len capped at 8192: our prompts are a code snippet + <=512-token reasoning, far
    # under the models' native 32-64k context. The cap shrinks the KV-cache reservation so the
    # 33B model fits on one 80GB card (default 64k needs 15.5GiB KV; only ~6GiB is left after
    # 66GB of fp16 weights). Harmless for the small models (they never hit 8192 anyway).
    t_load = time.time()
    llm = LLM(
        model=model_id,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )
    print(f"  [load] {time.time() - t_load:.1f}s", flush=True)

    # Build chat-templated prompts (same template path as the HF worker for qwen).
    def chat_prompt(code: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": COT_PROMPT.format(code=code)}],
            tokenize=False,
            add_generation_prompt=True,
        )

    prompts = [chat_prompt(code) for _, _, code in snippets]

    # ── Pass A: batched CoT generation (greedy = temperature 0) ──
    t_gen = time.time()
    gen = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
    reasonings = [o.outputs[0].text for o in gen]
    print(f"  [genA] {time.time() - t_gen:.1f}s for {len(prompts)} snippets", flush=True)

    # ── Build the verdict-slot scoring prompt PER ITEM ──
    # The model usually VOLUNTEERS its own "VERDICT:" (often markdown-bold, e.g.
    # "**VERDICT: YES**"). If we blindly append " VERDICT:" after a reasoning that already
    # ends in a verdict, the final slot is NOT a yes/no position (Task 0 v1 bug: those slots
    # floored to 0.5). Instead: cut the reasoning right AFTER the model's own "VERDICT:" so
    # the next-token slot is the model's own yes/no. Only if it never volunteered one do we
    # append our marker. (Mirrors HF cot_then_verdict_score, but in text space for vLLM.)
    score_prompts = [p + cut_after_verdict(r) for p, r in zip(prompts, reasonings, strict=True)]
    scored = llm.generate(
        score_prompts,
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20),
    )

    print("\n=== RESULTS ===", flush=True)
    all_finite = True
    for (cve_id, which, _code), reasoning, out in zip(snippets, reasonings, scored, strict=True):
        # prompt_logprobs: list (one per prompt token); final entry = the YES/NO slot.
        plps = out.prompt_logprobs
        last = plps[-1] if plps else None
        if not last:
            print(f"{cve_id} {which}: NO prompt_logprobs at final slot", flush=True)
            all_finite = False
            continue
        # Unwrap vLLM Logprob objects -> {token_id: logprob float}.
        position = {tid: lp.logprob for tid, lp in last.items()}
        score = score_from_prompt_logprobs(position, answer_ids)
        yes_present = any(i in position for i in answer_ids.yes_ids)
        no_present = any(i in position for i in answer_ids.no_ids)
        verdict_tail = reasoning.strip()[-40:].replace("\n", " ")
        print(
            f"{cve_id} {which}: P(YES)={score:.4f} "
            f"[yes_in_top20={yes_present} no_in_top20={no_present}] "
            f"...{verdict_tail!r}",
            flush=True,
        )

    print(f"\nVIABLE: {all_finite}", flush=True)
    if not all_finite:
        sys.exit(1)


if __name__ == "__main__":
    main()
