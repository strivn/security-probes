"""vLLM CoT elicitation: cascading-budget generation + DUAL verdict readout (one pass).

Design (single-pass generated-slot logprob readout):
  - GENERATE reasoning via llm.chat(messages) so vLLM applies each model's correct chat
    template internally (incl. mistral-common for Devstral via tokenizer_mode='mistral').
    NO hand-templating (that is what broke Devstral before). Request logprobs=20 so vLLM
    returns the per-generated-token distribution; this does NOT change the greedy text.
  - CASCADING token budget: generate at BUDGETS[0] (>=2048). Any sequence that hit the
    length cap (finish_reason == 'length') WITHOUT emitting "VERDICT: yes/no" is RE-RUN at
    the next larger budget. Every truncation is LOGGED per tier. No silent cutoffs.
  - TWO readouts from the SAME generation (no second forward pass):
      * cot-token   : binary from the EMITTED "VERDICT: yes/no" text (YES=1.0,NO=0.0,NaN).
      * cot-logprob : continuous P(YES) read at the GENERATED verdict slot -- the exact
        token+distribution the model decoded (find_verdict_slot + score_generated_verdict_slot).
        Replaces the old cut+re-feed prompt_logprobs path, which read a prefill conditional
        that did NOT match the decoded token for DeepSeek.
  - Write the FULL transcript (reasoning, token_ids omitted, slot diagnostics, both scores) so
    any downstream metric (win-rate, McNemar, tie-rate) is derivable without re-running the GPU.

Output: outputs/phase1/patcheval_elicit_{cot-token,cot-logprob}_{slug}.json (script-13 schema)
        outputs/phase1/transcripts/cot_vllm_{slug}.jsonl (one record per item)

Usage on the box (vLLM venv, .env sourced for HF_TOKEN):
    PYTHONPATH=src .venv-vllm/bin/python scripts/phase1/vllm_cot_score.py <slug>
    PYTHONPATH=src .venv-vllm/bin/python scripts/phase1/vllm_cot_score.py <slug> --proof
        # --proof: 3 CVEs only (6 items), dump per-item slot diagnostics, write nothing.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts" / "phase1"))

_m13 = importlib.import_module("13_patcheval_prompted_elicit")
build_items = _m13.build_items
summarize = _m13.summarize

from agentic_sec_probe.logprob_scoring import resolve_answer_token_ids
from agentic_sec_probe.models import MODEL_REGISTRY
from agentic_sec_probe.vllm_cot import find_answer_slot, find_verdict_slot, score_generated_slot

# Default context: room for long code + an 8192-token reasoning chain. deepseek-33b's fp16
# weights leave too little KV cache for 16384 on one 80GB card, so it gets a smaller cap (its
# code snippets still fit; few chains exceed ~3k tokens and the cascade tops out at 8192 gen).
MAX_MODEL_LEN = 16384
MAX_MODEL_LEN_BY_SLUG = {"deepseek-coder-33b": 9216}
# deepseek-33b's 66GB fp16 weights leave a tiny KV budget; give it more of the card.
GPU_MEM_UTIL_BY_SLUG = {"deepseek-coder-33b": 0.93}
BUDGETS = (2048, 4096, 8192)  # cascading: retry no-verdict-truncated items at the next tier

COT_PROMPT = (
    "You are a security code reviewer."
    " Analyze the following function for security vulnerabilities.\n\n"
    "```\n{code}\n```\n\n"
    "Think step by step about whether this function contains a security"
    " vulnerability, then end your answer with 'VERDICT: YES' or 'VERDICT: NO'."
)

# Parse the LAST emitted "VERDICT: YES/NO" (tolerates markdown bold/spacing; word-boundary
# so "not"/"cannot" never match "no").
_VERDICT_RE = re.compile(r"verdict[\s*]*:[\s*]*\**\s*\b(yes|no)\b", re.IGNORECASE)


def parse_verdict(text: str) -> float:
    m = _VERDICT_RE.findall(text)
    if not m:
        return float("nan")
    return 1.0 if m[-1].lower() == "yes" else 0.0


def build_per_cve(scores: list[float], cve_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate interleaved [vul, fix] item scores to per-CVE risk_gap (mean vul - mean fix).

    One observation per CVE (the unit of analysis), exactly as script 13's process_model. NaN
    item scores (no verdict) are ignored by nanmean; a CVE whose every pair is NaN -> NaN gap.
    """
    per_cve: list[dict[str, Any]] = []
    cursor = 0
    for meta in cve_meta:
        n_pairs = meta["n_pairs"]
        vul_s, fix_s, gaps = [], [], []
        for _ in range(n_pairs):
            vs, fs = scores[cursor], scores[cursor + 1]
            cursor += 2
            vul_s.append(vs)
            fix_s.append(fs)
            gaps.append(vs - fs)
        with np.errstate(invalid="ignore"):
            vul_m = float(np.nanmean(vul_s)) if vul_s else float("nan")
            fix_m = float(np.nanmean(fix_s)) if fix_s else float("nan")
            gap_m = float(np.nanmean(gaps)) if gaps else float("nan")
        per_cve.append(
            {
                **meta,
                "vul_score": None if np.isnan(vul_m) else round(vul_m, 6),
                "fix_score": None if np.isnan(fix_m) else round(fix_m, 6),
                "risk_gap": None if np.isnan(gap_m) else round(gap_m, 6),
            }
        )
    return per_cve


def score_generated_slots(
    token_ids: list[list[int]],
    logprobs: list[list[dict[int, float]]],
    decode: Any,
    answer_ids: Any,
    finder: Any,
) -> tuple[list[float], list[str | None], list[tuple[float | None, float | None]]]:
    """Continuous P(YES) at each item's GENERATED answer slot (the -logprob readout).

    No second forward pass: reads the YES/NO logprob at the exact token the model decoded. `finder`
    locates the slot in the generated stream -- find_verdict_slot (CoT: last 'VERDICT:' marker +
    first YES/NO after it) or find_answer_slot (noshot/fewshot: YES/NO at token ~0). score_
    generated_slot softmaxes the YES vs NO mass there. Items with no clean slot (hedge / no marker /
    no bare answer) -> NaN. Returns (scores, labels, [(yes_lm, no_lm)]). `decode` is the tokenizer's
    decode(list[int])->str."""
    scores: list[float] = []
    labels: list[str | None] = []
    logmass: list[tuple[float | None, float | None]] = []
    for ids, lps in zip(token_ids, logprobs, strict=True):
        score, label, ly, ln = score_generated_slot(ids, lps, decode, answer_ids, finder)
        scores.append(score)
        labels.append(label)
        logmass.append((ly, ln))
    return scores, labels, logmass


def is_mistral(slug: str) -> bool:
    return "devstral" in slug or "mistral" in slug


def make_llm(model_id: str, slug: str) -> tuple[LLM, int]:
    max_len = MAX_MODEL_LEN_BY_SLUG.get(slug, MAX_MODEL_LEN)
    kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": "float16",
        "gpu_memory_utilization": GPU_MEM_UTIL_BY_SLUG.get(slug, 0.85),
        "max_model_len": max_len,
    }
    if is_mistral(slug):
        # Devstral: vLLM applies mistral-common chat templating internally (the path the
        # model card ships). Avoids the HF apply_chat_template double-tokenization bug.
        kwargs.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    return LLM(**kwargs), max_len


def generate_with_cascade(
    llm: LLM, messages_list: list[list[dict[str, str]]], max_len: int
) -> tuple[list[str], list[list[int]], list[list[dict[int, float]]], list[int], list[str]]:
    """Generate reasoning for every item, escalating the token budget for items that hit the
    length cap WITHOUT a verdict. Returns per item: (texts, token_ids, logprobs, budget_used,
    finish_reason). `logprobs` is the vLLM per-generated-token distribution unwrapped to
    {token_id: logprob float} (requested via SamplingParams(logprobs=20)); we retain it so the
    cot-logprob readout reads the verdict slot WITHOUT a second forward pass. logprobs=20 with
    temperature=0.0 does not change the greedy text (verified: vLLM logprobs is a readout, not a
    sampler change). Each tier's max_tokens is capped so prompt+gen fits the model's context."""
    prompt_reserve = 2600  # generous: most chat-templated code prompts are well under this
    n = len(messages_list)
    texts: list[str] = [""] * n
    token_ids: list[list[int]] = [[] for _ in range(n)]
    logprobs: list[list[dict[int, float]]] = [[] for _ in range(n)]
    budget_used: list[int] = [0] * n
    finish: list[str] = [""] * n
    pending = list(range(n))  # indices still needing (re)generation

    for tier, raw_budget in enumerate(BUDGETS):
        if not pending:
            break
        budget = min(raw_budget, max_len - prompt_reserve)  # fit prompt+gen in context
        t = time.time()
        sub = [messages_list[i] for i in pending]
        outs = llm.chat(sub, SamplingParams(temperature=0.0, max_tokens=budget, logprobs=20))
        next_pending: list[int] = []
        truncated_no_verdict = 0
        for idx, out in zip(pending, outs, strict=True):
            o = out.outputs[0]
            texts[idx] = o.text
            token_ids[idx] = list(o.token_ids)
            # Unwrap vLLM's per-token logprobs (list[dict[int, Logprob]]) -> list[dict[int,float]].
            # o.logprobs is None only if logprobs was not requested; here it is always present.
            logprobs[idx] = [
                {tid: lp.logprob for tid, lp in pos.items()} for pos in (o.logprobs or [])
            ]
            budget_used[idx] = budget
            finish[idx] = o.finish_reason
            has_verdict = not np.isnan(parse_verdict(o.text))
            # Retry ONLY if it ran out of room (length) AND has no verdict yet.
            if o.finish_reason == "length" and not has_verdict and tier < len(BUDGETS) - 1:
                next_pending.append(idx)
                truncated_no_verdict += 1
        print(
            f"  [tier {tier} budget={budget}] {len(pending)} items in {time.time() - t:.1f}s; "
            f"{truncated_no_verdict} truncated-without-verdict -> retry at next tier",
            flush=True,
        )
        pending = next_pending

    # Anything STILL pending hit the max budget without a verdict: a genuine no-verdict.
    if pending:
        print(
            f"  {len(pending)} items reached max budget {BUDGETS[-1]} with NO verdict (NaN)",
            flush=True,
        )
    return texts, token_ids, logprobs, budget_used, finish


def generate_short(
    llm: LLM, messages_list: list[list[dict[str, str]]], max_new: int = 8
) -> tuple[list[str], list[list[int]], list[list[dict[int, float]]], list[int], list[str]]:
    """One greedy pass, max_tokens=max_new, logprobs=20, for the marker-less styles (noshot/
    fewshot): the prompt asks for one word, so the YES/NO answer is at generation token ~0 and no
    cascade is needed. Returns the SAME 5-tuple shape as generate_with_cascade (texts, token_ids,
    logprobs, budget_used, finish_reason) so the rest of main() is style-agnostic."""
    outs = llm.chat(messages_list, SamplingParams(temperature=0.0, max_tokens=max_new, logprobs=20))
    texts: list[str] = []
    token_ids: list[list[int]] = []
    logprobs: list[list[dict[int, float]]] = []
    budget: list[int] = []
    finish: list[str] = []
    for out in outs:
        o = out.outputs[0]
        texts.append(o.text)
        token_ids.append(list(o.token_ids))
        logprobs.append(
            [{tid: lp.logprob for tid, lp in pos.items()} for pos in (o.logprobs or [])]
        )
        budget.append(max_new)
        finish.append(o.finish_reason)
    return texts, token_ids, logprobs, budget, finish


def run_proof(
    slug: str,
    style: str,
    finder: Any,
    decode: Any,
    answer_ids: Any,
    items: list[dict[str, Any]],
    texts: list[str],
    token_ids: list[list[int]],
    logprobs: list[list[dict[int, float]]],
    n_items: int = 6,
) -> None:
    """Dump per-item slot diagnostics for the first n_items (the proof gate). For each item:
    the decoded answer region, the located slot index + decoded token (must be a YES/NO id),
    the top-5 logprobs at the slot (YES/NO flagged), P(YES), and the binary-vs-logprob sign
    agreement. DeepSeek extra: the token AFTER a ' Y' slot must decode to 'ES'. Writes nothing.

    `finder` is find_verdict_slot (cot) or find_answer_slot (noshot/fewshot). The binary label is
    the slot's decoded label (cot also cross-checks parse_verdict on the emitted text)."""
    print(f"\n===== PROOF GATE: {slug} style={style} (first {n_items} items) =====", flush=True)
    for it, text, ids, lps in list(zip(items, texts, token_ids, logprobs, strict=True))[:n_items]:
        tag = f"{it.get('cve_id')} {it.get('which')}"
        slot, label = finder(ids, decode, answer_ids)
        # For cot, cross-check the emitted-text parser; for noshot/fewshot the slot label IS binary.
        bin_label = (
            (
                "YES"
                if parse_verdict(text) == 1.0
                else ("NO" if parse_verdict(text) == 0.0 else "NaN")
            )
            if style == "cot"
            else (label or "NaN")
        )
        region = text[-60:].replace("\n", "\\n")
        print(f"\n--- {tag} ---", flush=True)
        print(f"  gen tail: ...{region!r}", flush=True)
        print(f"  binary label = {bin_label}", flush=True)
        if slot is None:
            print("  slot = NONE (hedge / no marker / no bare answer) -> P(YES)=NaN", flush=True)
            continue
        slot_tok = decode([ids[slot]])
        print(f"  slot idx={slot}  token={slot_tok!r}  id={ids[slot]}  label={label}", flush=True)
        pos = lps[slot] if slot < len(lps) else {}
        ranked = sorted(pos.items(), key=lambda kv: kv[1], reverse=True)[:5]
        for tid, lp in ranked:
            flag = (
                "  <-YES"
                if tid in answer_ids.yes_ids
                else ("  <-NO" if tid in answer_ids.no_ids else "")
            )
            print(f"    id={tid:6d} lp={lp:8.3f} {decode([tid])!r}{flag}", flush=True)
        score, _, ly, ln = score_generated_slot(ids, lps, decode, answer_ids, finder)
        yes_present = any(i in pos for i in answer_ids.yes_ids)
        no_present = any(i in pos for i in answer_ids.no_ids)
        agree = (bin_label == "YES" and score > 0.5) or (bin_label == "NO" and score < 0.5)
        print(
            f"  P(YES)={score:.4f}  yes_in_top20={yes_present} no_in_top20={no_present}  "
            f"sign-agree(binary,logprob)={agree}",
            flush=True,
        )
        # DeepSeek sub-token check: a ' Y' answer token should be followed by 'ES'.
        if slot_tok.strip() == "Y" and slot + 1 < len(ids):
            nxt = decode([ids[slot + 1]])
            print(f"  [deepseek subtoken] token after ' Y' = {nxt!r}  (expect 'ES')", flush=True)


def parse_style(argv: list[str]) -> str:
    """Parse --style {noshot,fewshot,cot} from argv (default cot for back-compat)."""
    style = "cot"
    for i, a in enumerate(argv):
        if a.startswith("--style="):
            style = a.split("=", 1)[1]
        elif a == "--style" and i + 1 < len(argv):
            style = argv[i + 1]
    if style not in ("noshot", "fewshot", "cot"):
        raise SystemExit(f"unknown --style {style!r} (expected noshot|fewshot|cot)")
    return style


def main() -> None:
    slug = sys.argv[1]
    proof = "--proof" in sys.argv[2:]
    style = parse_style(sys.argv[2:])
    spec = MODEL_REGISTRY[slug]
    model_id = spec.probe_model_id
    out_dir = PROJECT / "outputs" / "phase1"
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    cves = json.loads((PROJECT / "data/patcheval/python_cves.json").read_text())
    valid = [c for c in cves if c.get("vul_func") and c.get("fix_func")]
    # build_items routes the prompt template + fewshot prefix on the mode's prompt_style; the
    # worker generates once, then scores both readouts (-token, -logprob) from the SAME output.
    items, cve_meta = build_items(valid, f"{style}-token")
    if proof:
        # First 3 CVEs = 6 interleaved [vul,fix] items; trim so generation is cheap.
        items = items[:6]
        print(f"{slug} style={style}: PROOF MODE -- {len(items)} items (no output)", flush=True)
    else:
        print(f"{slug} style={style}: {len(cve_meta)} CVEs, {len(items)} items", flush=True)

    t = time.time()
    llm, max_len = make_llm(model_id, slug)
    print(f"  [load] {time.time() - t:.1f}s  max_model_len={max_len}", flush=True)

    # Truncate the user text so prompt + the gen budget fits max_model_len. The rare very-long
    # code snippet (deepseek hit a 13.5k-token prompt vs 9216 ctx) is right-truncated, keeping the
    # function start (matches the HF worker). Reserve the cot cascade budget (ample for short gen).
    tok = llm.get_tokenizer()
    max_prompt_tokens = max_len - min(BUDGETS) - 256

    def truncate_text(text: str) -> str:
        ids = tok.encode(text)
        if len(ids) <= max_prompt_tokens:
            return text
        return tok.decode(ids[:max_prompt_tokens], skip_special_tokens=True)

    messages_list = [[{"role": "user", "content": truncate_text(it["text"])}] for it in items]
    # cot: cascading budget (reasoning is long). noshot/fewshot: one short pass (answer = 1 word).
    if style == "cot":
        texts, token_ids, logprobs, budget_used, finish = generate_with_cascade(
            llm, messages_list, max_len
        )
    else:
        texts, token_ids, logprobs, budget_used, finish = generate_short(llm, messages_list)

    # YES/NO ids resolved from the HF tokenizer (vLLM's mistral tokenizer lacks the same encode
    # surface; the HF tokenizer's ids match vLLM's vocab for these models -- verified in genslot).
    hf_tok = AutoTokenizer.from_pretrained(model_id)
    answer_ids = resolve_answer_token_ids(hf_tok)
    print(f"  YES ids={sorted(answer_ids.yes_ids)} NO ids={sorted(answer_ids.no_ids)}", flush=True)
    decode = tok.decode
    # cot anchors on the model's VERDICT marker; noshot/fewshot read the bare answer at token ~0.
    finder = find_verdict_slot if style == "cot" else find_answer_slot

    if proof:
        run_proof(slug, style, finder, decode, answer_ids, items, texts, token_ids, logprobs)
        return

    # -logprob: continuous P(YES) at the model's OWN generated answer slot (no second pass).
    scores_logprob, labels_lp, logmass = score_generated_slots(
        token_ids, logprobs, decode, answer_ids, finder
    )
    # -token: binary YES/NO. cot parses the emitted VERDICT text; noshot/fewshot take the SAME
    # slot's decoded label (so both readouts share the token -> sign-agree by construction).
    if style == "cot":
        scores_token = [parse_verdict(t) for t in texts]
    else:
        scores_token = [
            (1.0 if lab == "YES" else (0.0 if lab == "NO" else float("nan"))) for lab in labels_lp
        ]

    # ── Transcript: full record with BOTH readouts (reasoning is the primary re-scorable text) ──
    tpath = out_dir / "transcripts" / f"{style}_vllm_{slug}.jsonl"
    with open(tpath, "w") as tf:
        for it, text, b, fr, s_tok, s_lp, lab_lp, (ly, ln) in zip(
            items,
            texts,
            budget_used,
            finish,
            scores_token,
            scores_logprob,
            labels_lp,
            logmass,
            strict=True,
        ):
            tf.write(
                json.dumps(
                    {
                        "cve_id": it.get("cve_id"),
                        "pair_idx": it.get("pair_idx"),
                        "which": it.get("which"),
                        "score_token": None if np.isnan(s_tok) else s_tok,
                        "verdict": "YES" if s_tok == 1.0 else ("NO" if s_tok == 0.0 else None),
                        "score_logprob": None if np.isnan(s_lp) else s_lp,
                        "logprob_label": lab_lp,
                        "yes_logmass": ly,
                        "no_logmass": ln,
                        "budget_used": b,
                        "finish_reason": fr,
                        "truncated_no_verdict": bool(fr == "length" and np.isnan(s_tok)),
                        "reasoning": text,
                    }
                )
                + "\n"
            )

    n_trunc = sum(
        1 for fr, s in zip(finish, scores_token, strict=True) if fr == "length" and np.isnan(s)
    )
    n_noverdict = sum(1 for s in scores_token if np.isnan(s))
    n_noverdict_lp = sum(1 for s in scores_logprob if np.isnan(s))
    # Sense-check: where both readouts are non-NaN, they must agree in sign (greedy => same token).
    n_both = sum(
        1
        for st, sl in zip(scores_token, scores_logprob, strict=True)
        if not np.isnan(st) and not np.isnan(sl)
    )
    n_sign_agree = sum(
        1
        for st, sl in zip(scores_token, scores_logprob, strict=True)
        if not np.isnan(st) and not np.isnan(sl) and ((st == 1.0) == (sl > 0.5))
    )
    print(f"  Transcript: {tpath}", flush=True)
    print(
        f"  no-verdict items: token={n_noverdict}/{len(scores_token)} "
        f"(of which {n_trunc} still truncated), logprob={n_noverdict_lp}",
        flush=True,
    )
    print(
        f"  sign-agreement(token,logprob) where both non-NaN: {n_sign_agree}/{n_both}",
        flush=True,
    )

    # ── Write BOTH cells: cot-token (binary verdict) and cot-logprob (continuous verdict-slot) ──
    def write_cell(mode: str, scoring: str, scores: list[float]) -> None:
        per_cve = build_per_cve(scores, cve_meta)
        n_fail = sum(1 for r in per_cve if r["risk_gap"] is None)
        output = {
            "slug": slug,
            "model_id": model_id,
            "mode": mode,
            "backend": "vllm",
            "scoring": scoring,
            "budgets": list(BUDGETS),
            "n_cves": len(per_cve),
            "n_elicitation_failures": n_fail,
            "n_no_verdict_items": n_noverdict,
            "n_truncated_no_verdict_items": n_trunc,
            "win_rate": summarize(per_cve),
            "per_cve": per_cve,
        }
        out_path = out_dir / f"patcheval_elicit_{mode}_{slug}.json"
        out_path.write_text(json.dumps(output, indent=2))
        u = output["win_rate"]["unseen"]
        print(
            f"  Saved {out_path.name}: unseen win-rate={u['win_rate']:.3f} "
            f"[{u['ci_lower']:.3f},{u['ci_upper']:.3f}] n={u['n_effective']} "
            f"wins={u['wins']} losses={u['losses']} ties={u['ties']} p={u['p_value']:.2e}",
            flush=True,
        )

    print(flush=True)
    token_scoring = "emitted_verdict_binary" if style == "cot" else "answer_slot_label_binary"
    write_cell(f"{style}-token", token_scoring, scores_token)
    write_cell(f"{style}-logprob", "answer_slot_logprob_continuous", scores_logprob)


if __name__ == "__main__":
    main()
