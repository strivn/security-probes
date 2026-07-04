"""Stronger prompted baseline: 3-mode elicitation suite on PatchEval (Phase 2).

The reviewers asked for a stronger prompted baseline than forced one-word YES/NO.
This script gives prompting a CONTINUOUS score via the YES-vs-NO logprob primitive
(src/agentic_sec_probe/logprob_scoring.py), so the prompted side gets a real ranking
score with an AUC + Wilson CI, comparable to the probe, and the binary-tie + byte-BPE
parse failures disappear.

Three modes, all unifying on the same logprob scoring primitive:

  logprob (default, load-bearing): one forward pass on the prompt + chat generation
      prompt; score = softmax([logp(YES), logp(NO)])[YES] from RAW next-token logits.
      = McKenzie 2025 (arXiv:2506.10805) §2.2.

  cot: generate reasoning until the model emits its OWN "VERDICT:" (budget escalates on
      retry; no hard cap, no force-append), then read the next-token YES/NO logprob at that
      slot in token space. If the model never volunteers a verdict, the item scores NaN (a
      genuine failure, dropped). Reading a logit instead of regex-parsing free text avoids
      the misjudgement xFinder 2405.11874 finds in ~1/5 free-text verdicts.

  fewshot: prepend a fixed set of labeled TRAIN exemplars (excluding any train item
      sharing pair_id OR commit/project with an eval item), then logprob-score.

Per-CVE: risk_gap = score(vul) - score(fix). Headline = paired win-rate (Wilson 95%
CI + exact-binomial sign test, ties dropped) on the unseen-bug-types slice, shared
with the probe (patcheval.classify_cve). A continuous-score AUC is also reported.

Input:  data/patcheval/python_cves.json
Output: outputs/phase1/patcheval_elicit_{mode}_{slug}.json (per model x mode)

Runs each model in a subprocess for GPU memory isolation, checkpointing every N items
(resume-safe). The scoring math is unit-tested on CPU (tests/test_logprob_scoring.py);
this driver wraps it in the GPU worker.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

load_dotenv(PROJECT / ".env")

from agentic_sec_probe.models import MODEL_REGISTRY
from agentic_sec_probe.paired_stats import paired_win_rate
from agentic_sec_probe.patcheval import (
    assert_expected_split,
    classify_cve,
    cve_in_trained_repo,
    cve_primary_cwe,
    cve_stratum,
    pair_cve_snippets,
)

# The elicitation suite is a 2x3 grid: prompt style {noshot, fewshot, cot} x readout
# {logprob, token}. A mode name is "{prompt}-{readout}". `logprob` reads the YES-vs-NO logit
# at the answer slot (no parse failure); `token` greedily generates and parses the emitted
# verdict (can NaN on a parse/no-verdict failure). "greedy" was a misnomer (every mode decodes
# greedily) -- the real axis is logit-readout vs token-readout.
MODES = (
    "noshot-logprob",
    "noshot-token",
    "fewshot-logprob",
    "fewshot-token",
    "cot-logprob",
    "cot-token",
)


def split_mode(mode: str) -> tuple[str, str]:
    """('cot-logprob') -> ('cot', 'logprob'). Prompt style and readout from a mode name."""
    prompt_style, _, readout = mode.partition("-")
    if prompt_style not in ("noshot", "fewshot", "cot") or readout not in ("logprob", "token"):
        msg = f"unknown mode {mode!r} (expected {{noshot,fewshot,cot}}-{{logprob,token}})"
        raise ValueError(msg)
    return prompt_style, readout


# Modes whose readout GENERATES text and can fail to produce a parseable/volunteered verdict,
# emitting NaN for that CVE (counted + reported). The logprob readout reads a logit, never fails.
BINARY_FAILURE_MODES = tuple(m for m in MODES if m.endswith("-token"))

PROMPT = (
    "You are a security code reviewer."
    " Analyze the following function for security vulnerabilities.\n\n"
    "```\n{code}\n```\n\n"
    "Does this function contain a security vulnerability?"
    " Answer with exactly one word: YES or NO"
)

COT_PROMPT = (
    "You are a security code reviewer."
    " Analyze the following function for security vulnerabilities.\n\n"
    "```\n{code}\n```\n\n"
    "Think step by step about whether this function contains a security"
    " vulnerability, then end your answer with 'VERDICT: YES' or 'VERDICT: NO'."
)

MAX_INPUT_TOKENS = 8192
CHECKPOINT_EVERY = 25
# CoT FIRST-pass generation budget (NOT a hard truncation cap). If the model has not
# emitted its own "VERDICT:" within the budget, the worker RETRIES with double the budget
# (up to MAX_COT_RETRIES) so reasoning is never cut off mid-thought. A model that still
# never volunteers a verdict scores NaN (a genuine failure, dropped) -- no fabricated verdict.
COT_MAX_NEW_TOKENS = 512
MAX_COT_RETRIES = 3  # budgets: 512 -> 1024 -> 2048 -> 4096, then NaN if no own verdict

# The worker scores one (mode, prompt) -> P(YES) per item. It NEVER parses text for
# the headline; CoT generates reasoning then re-reads logits at "VERDICT:". Raw logits
# only (single forward pass for logprob/fewshot; output_logits for the CoT verdict).
WORKER_SCRIPT = r"""
import json, os, re, sys, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id     = sys.argv[1]
prompts_file = sys.argv[2]   # JSON list of {"text", "cot", "greedy", "cve_id", "pair_idx", "which"}
output_file  = sys.argv[3]   # checkpoint (list of float scores), resume-safe
quant        = sys.argv[4]
max_tokens   = int(sys.argv[5])
ckpt_every   = int(sys.argv[6])
cot_max_new  = int(sys.argv[7])   # FIRST-pass CoT budget; retries double it (no hard cap)
MAX_COT_RETRIES = int(sys.argv[8])
transcript_file = sys.argv[9]   # per-response audit transcript (.jsonl), one line per item
load_dtype_name = sys.argv[10]  # full-precision dtype when not quantized (Gemma -> bfloat16)
load_class_name = sys.argv[11]  # explicit HF class or "none" (Gemma3ForCausalLM -> text-only)

YES_VARIANTS = ["YES","Yes","yes"," YES"," Yes"," yes"]
NO_VARIANTS  = ["NO","No","no"," NO"," No"," no"]

with open(prompts_file) as f:
    items = json.load(f)

scores = []  # CoT/greedy emit NaN on a genuine no-verdict/parse failure (dropped downstream)
if os.path.exists(output_file):
    try:
        with open(output_file) as f:
            scores = list(json.load(f).get("scores", []))
        if len(scores) > len(items):
            scores = []
    except (json.JSONDecodeError, OSError):
        scores = []
start_idx = len(scores)
# Keep the transcript exactly aligned with the checkpoint on resume: truncate it to the
# first start_idx lines so a re-run from a partial checkpoint never double-writes a record.
def truncate_transcript_to(n):
    if not os.path.exists(transcript_file):
        return
    with open(transcript_file) as f:
        lines = f.readlines()
    if len(lines) > n:
        with open(transcript_file, "w") as f:
            f.writelines(lines[:n])
truncate_transcript_to(start_idx)
if start_idx >= len(items):
    print(f"  All {len(items)} items already scored", flush=True); sys.exit(0)
if start_idx > 0:
    print(f"  Resuming from item {start_idx}/{len(items)}", flush=True)

def save_ckpt():
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"scores": scores}, f)
    os.replace(tmp, output_file)

tokenizer = AutoTokenizer.from_pretrained(model_id)
load_kwargs = {"device_map": "auto"}
if quant == "8bit":
    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
elif quant == "4bit":
    # Match the EXTRACTION path's 4bit config (activations.py): nf4 + fp16 compute. Without
    # bnb_4bit_compute_dtype the kernel defaults to fp32, so the prompt arm would run at a
    # different numeric precision than the probe arm on the same (deepseek) checkpoint.
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, load_dtype_name),
        bnb_4bit_quant_type="nf4",
    )
else:
    # Full precision in the spec dtype. Gemma MUST be bf16 (fp16 -> NaN garbage). The
    # prompting load dtype MUST match the probe extraction dtype (same checkpoint, same
    # numerics) so the two arms compare like-for-like.
    load_kwargs["torch_dtype"] = getattr(torch, load_dtype_name)
# Explicit HF class for a multimodal checkpoint's text-only stack (Gemma3ForCausalLM):
# skips the vision tower, keeps the standard model.model.layers path. "none" -> Auto.
if load_class_name != "none":
    import transformers
    _LoaderCls = getattr(transformers, load_class_name)
    model = _LoaderCls.from_pretrained(model_id, **load_kwargs)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
dev = next(model.parameters()).device

# Backend split (verified by tokenizer-only tests): Mistral/Devstral has NO Jinja chat
# template (chat_template is None) and uses mistral-common. Going through HF
# apply_chat_template byte-BPE-encodes the [INST]/<s> markers as literal text, producing
# WRONG ids (control-token soup in generations). For that backend we use mistral-common's
# encode_chat_completion (request -> ids, the path the Devstral model card ships). Vocab
# (131072) matches the HF logits index, so YES/NO logit lookup is valid. Qwen/DeepSeek have
# proper Jinja templates and use the standard path.
USE_MISTRAL_COMMON = tokenizer.chat_template is None
if USE_MISTRAL_COMMON:
    from huggingface_hub import hf_hub_download
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    _mt = MistralTokenizer.from_file(hf_hub_download(repo_id=model_id, filename="tekken.json"))
    _mt_inner = _mt.instruct_tokenizer.tokenizer  # tekken tokenizer (for YES/NO ids + decode)
    print("  using mistral-common encode path (no Jinja chat template)", flush=True)

def _encode_variant(v):
    if USE_MISTRAL_COMMON:
        return _mt_inner.encode(v, bos=False, eos=False)
    return tokenizer.encode(v, add_special_tokens=False)

def resolve_answer_ids(variants):
    # Mirrors src/agentic_sec_probe/logprob_scoring.py:resolve_answer_token_ids. A single-token
    # variant contributes its id; a MULTI-token variant contributes its FIRST sub-token (the
    # slot the model predicts first — discriminative as long as YES/NO differ there). Without
    # the first-sub-token rule, an all-caps "VERDICT: YES" slot whose only form is multi-token
    # (DeepSeek " YES" -> [Y, ES]) yields an EMPTY YES set and floors the score. Ambiguous ids
    # landing on both sides are dropped after collection.
    ids = set()
    for v in variants:
        e = _encode_variant(v)
        if e:
            ids.add(e[0])
    return ids
YES_IDS = resolve_answer_ids(YES_VARIANTS)
NO_IDS  = resolve_answer_ids(NO_VARIANTS)
_ambiguous = YES_IDS & NO_IDS  # a shared leading sub-token cannot discriminate -> drop it
YES_IDS -= _ambiguous
NO_IDS  -= _ambiguous
print(f"  YES ids={sorted(YES_IDS)} NO={sorted(NO_IDS)} amb={sorted(_ambiguous)}", flush=True)
if not YES_IDS or not NO_IDS:
    print("  FATAL: empty YES or NO id set for this tokenizer", flush=True); sys.exit(2)

def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

def yes_no_logmass(row):
    # Aggregate logit mass over the YES vs NO token variants. Returns (yes_logmass,
    # no_logmass) for transcript auditing; the score is sigmoid(yes - no).
    return logsumexp([row[i] for i in YES_IDS]), logsumexp([row[i] for i in NO_IDS])

def score_from_logmass(ly, ln):
    d = ly - ln
    return 1.0/(1.0+math.exp(-d)) if d >= 0 else math.exp(d)/(1.0+math.exp(d))

def yes_score_from_logits(row):
    ly, ln = yes_no_logmass(row)
    return score_from_logmass(ly, ln)

def chat_input_ids(user_prompt):
    # Return [1, seq] input ids for the chat prompt. Mistral/Devstral: mistral-common
    # encode_chat_completion (correct control-token ids; verified to differ from the HF
    # re-tokenize path which mangles [INST]/<s> into literal-text BPE). Others: standard
    # apply_chat_template(tokenize=False) + re-tokenize with add_special_tokens=False (no
    # duplicate BOS). Both truncated to max_tokens.
    if USE_MISTRAL_COMMON:
        toks = _mt.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=user_prompt)])
        ).tokens[:max_tokens]
        return torch.tensor([toks], device=dev)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=max_tokens, return_tensors="pt")
    return enc["input_ids"].to(dev)

def decode_gen(ids_1d):
    # Decode generated token ids with the matching backend (mistral-common for Devstral).
    lst = ids_1d.tolist()
    if USE_MISTRAL_COMMON:
        return _mt_inner.decode(lst)
    return tokenizer.decode(lst, skip_special_tokens=True)

def decode_prompt(ids_1d):
    # Decode the chat-templated PROMPT ids KEEPING special tokens, so the transcript records
    # the exact template the model saw (e.g. <|im_start|> markers) -- this is what catches a
    # template/tokenization bug like the Devstral [INST] mangling. (decode_gen strips them.)
    lst = ids_1d.tolist()
    if USE_MISTRAL_COMMON:
        return _mt_inner.decode(lst)  # tekken decode keeps control tokens as text
    return tokenizer.decode(lst, skip_special_tokens=False)

def plain_ids(text):
    # Tokenize a plain string (no chat template) with the matching backend. Used to build
    # the CoT verdict-marker token patterns to SEARCH for. mistral-common: no bos/eos.
    if USE_MISTRAL_COMMON:
        return _mt_inner.encode(text, bos=False, eos=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]

@torch.no_grad()
def logits_after(input_ids):
    out = model(input_ids=input_ids)
    return out.logits[0, -1, :].float().cpu().tolist()

# Token-space verdict marker: the ids of "VERDICT:" with a leading space (the form a model
# almost always emits mid-stream). We locate THIS subsequence in the generated ids and read
# the logit right after it -- no decode->re-tokenize round-trip (byte-BPE would shift the
# slot). Several encodings are searched because spacing/case varies.
_VERDICT_MARKERS = [plain_ids(t) for t in (" VERDICT:", "VERDICT:", " VERDICT :", "VERDICT :")]

def _find_last_subseq(hay, needle):
    # Return the index ONE PAST the last occurrence of `needle` in `hay` (token lists), or
    # -1. "One past" = the position whose logit predicts the token after the marker.
    n = len(needle)
    if n == 0 or n > len(hay):
        return -1
    for start in range(len(hay) - n, -1, -1):
        if hay[start : start + n] == needle:
            return start + n
    return -1

@torch.no_grad()
def logits_at(input_ids, pos):
    # Distribution over the token AT absolute index `pos` is produced by the logit row at
    # `pos-1`. Clamp into [0, seqlen-1] for safety; callers guarantee pos is in-window.
    out = model(input_ids=input_ids)
    seqlen = out.logits.shape[1]
    row = min(max(pos - 1, 0), seqlen - 1)
    return out.logits[0, row, :].float().cpu().tolist()

@torch.no_grad()
def cot_then_verdict_score(user_prompt):
    # Instruction-induced CoT scored by the YES-vs-NO logit at the model's OWN verdict slot.
    # NO fixed truncation cap and NO force-append: let the model run its due course,
    # then read the verdict it volunteered -- never fabricate one.
    #   1) generate with EOS-based natural stopping under a budget, doubling the budget on
    #      retry while the model has NOT yet emitted its own "VERDICT:" (give it room to
    #      finish, not force an answer);
    #   2) if the model emitted its OWN "VERDICT:" inside the scoring window -> read the
    #      YES/NO logit at the slot right after it, IN TOKEN SPACE from the real generated
    #      ids (no decode->re-tokenize, which byte-BPE can shift);
    #   3) if the model NEVER emitted a reachable verdict -> return NaN (a genuine failure,
    #      dropped from the win-rate and counted, exactly like greedy parse failures). We do
    #      not append a verdict the model did not write.
    ids = chat_input_ids(user_prompt)
    prompt_len = ids.shape[1]
    budget = cot_max_new
    full = None
    for attempt in range(MAX_COT_RETRIES + 1):
        gen = model.generate(
            ids, max_new_tokens=budget, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        full = gen[0:1, :]  # prompt + generated reasoning (ids, no re-tokenize)
        gen_only = gen[0, prompt_len:].tolist()
        has_verdict = any(_find_last_subseq(gen_only, mk) != -1 for mk in _VERDICT_MARKERS)
        stopped_early = gen.shape[1] - prompt_len < budget  # hit EOS before the budget
        if has_verdict or stopped_early or attempt == MAX_COT_RETRIES:
            break
        budget *= 2  # model still mid-reasoning at the budget -> retry with more room

    # Locate the model's OWN verdict marker in TOKEN space (search the generated ids only).
    gen_only = full[0, prompt_len:].tolist()
    best_end = -1
    for mk in _VERDICT_MARKERS:
        end = _find_last_subseq(gen_only, mk)
        if end > best_end:
            best_end = end
    # Every mode returns (score, detail). detail is the per-response audit record: the
    # decoded input prompt (the templated string the model actually saw, post-truncation),
    # any generated text, the raw YES/NO log-mass, n_gen, no_verdict, and max_tokens. Stored
    # one JSON line per item so any score can be re-derived / inspected by hand.
    decoded_prompt = decode_prompt(full[0, :prompt_len])
    gen_text = decode_gen(full[0, prompt_len:])
    n_gen = int(full.shape[1] - prompt_len)
    if best_end != -1:
        verdict_pos = prompt_len + best_end  # absolute index of the slot AFTER the marker
        if verdict_pos < max_tokens:  # slot inside the forward-pass window
            row = logits_at(full[:, :max_tokens], verdict_pos)
            ly, ln = yes_no_logmass(row)
            detail = {"input_prompt": decoded_prompt, "generated_text": gen_text,
                      "yes_logmass": ly, "no_logmass": ln, "n_gen": n_gen,
                      "no_verdict": False, "max_tokens": max_tokens}
            return score_from_logmass(ly, ln), detail
    # No reachable verdict -> NaN (genuine failure; never fabricate one).
    return float("nan"), {"input_prompt": decoded_prompt, "generated_text": gen_text,
                          "yes_logmass": None, "no_logmass": None, "n_gen": n_gen,
                          "no_verdict": True, "max_tokens": max_tokens}

@torch.no_grad()
def cot_greedy_score(user_prompt):
    # DIRECT-TOKEN CoT: greedily generate step-by-step reasoning under the CoT prompt, then
    # PARSE the literal "VERDICT: YES/NO" the model emitted (vs cot_then_verdict_score, which
    # reads the YES/NO logit at the verdict slot). One generation pass, no second logit read,
    # no budget escalation -- the cheapest CoT. Binary score {1.0,0.0} or NaN on parse failure.
    #   1) generate up to cot_max_new tokens (greedy, EOS-stopping);
    #   2) locate the model's OWN "VERDICT:" marker in token space (same markers as the logit
    #      path), decode ONLY the text after it, and word-boundary parse the first YES/NO;
    #   3) no reachable verdict, or no standalone yes/no after it -> NaN (parse failure).
    ids = chat_input_ids(user_prompt)
    prompt_len = ids.shape[1]
    gen = model.generate(ids, max_new_tokens=cot_max_new, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tokenizer.pad_token_id)
    gen_only = gen[0, prompt_len:].tolist()
    decoded_prompt = decode_prompt(gen[0, :prompt_len])
    gen_text = decode_gen(gen[0, prompt_len:])
    n_gen = int(gen.shape[1] - prompt_len)

    # Find the model's last VERDICT marker (token space), then parse the text AFTER it.
    best_end = -1
    for mk in _VERDICT_MARKERS:
        end = _find_last_subseq(gen_only, mk)
        if end > best_end:
            best_end = end
    if best_end != -1:
        after = decode_gen(gen[0, prompt_len + best_end:]).strip().lower()
        y = _YES_RE.search(after)
        n = _NO_RE.search(after)
        if y and not n:
            s = 1.0
        elif n and not y:
            s = 0.0
        elif y and n:
            s = 1.0 if y.start() < n.start() else 0.0  # first standalone yes/no after VERDICT wins
        else:
            s = float("nan")  # marker present but no parseable verdict word
    else:
        s = float("nan")  # model never emitted its own VERDICT marker
    detail = {"input_prompt": decoded_prompt, "generated_text": gen_text,
              "yes_logmass": None, "no_logmass": None, "n_gen": n_gen,
              "no_verdict": bool(math.isnan(s)), "max_tokens": max_tokens}
    return s, detail

@torch.no_grad()
def logprob_score(prompt_text):
    # Read the first answer-token logits after the (correctly tokenized) chat prompt. No
    # generation: the prompt ends with the explicit "YES or NO" ask, so the next-token slot
    # IS the answer. Returns (score, detail) with the raw YES/NO log-mass.
    ids = chat_input_ids(prompt_text)
    row = logits_after(ids)
    ly, ln = yes_no_logmass(row)
    detail = {"input_prompt": decode_prompt(ids[0]), "generated_text": None,
              "yes_logmass": ly, "no_logmass": ln, "n_gen": 0,
              "no_verdict": None, "max_tokens": max_tokens}
    return score_from_logmass(ly, ln), detail

_YES_RE = re.compile(r"\byes\b")
_NO_RE = re.compile(r"\bno\b")

@torch.no_grad()
def greedy_yesno_score(prompt_text):
    # The NAIVE baseline: greedily GENERATE the literal text answer, then parse YES/NO.
    # Returns (score, detail); score = 1.0 (YES), 0.0 (NO), or NaN if unparseable -- the
    # parse-failure mode the logprob primitive was designed to avoid. Uses the correct
    # tokenized chat prompt so Mistral/Devstral does not emit control-token soup. Generates
    # up to 24 tokens and parses with WORD-BOUNDARY regex so "not"/"cannot" do NOT match "no".
    ids = chat_input_ids(prompt_text)
    gen = model.generate(ids, max_new_tokens=24, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
    gen_text = decode_gen(gen[0, ids.shape[1]:])
    ans = gen_text.strip().lower()
    y = _YES_RE.search(ans)
    n = _NO_RE.search(ans)
    if y and not n:
        s = 1.0
    elif n and not y:
        s = 0.0
    elif y and n:
        s = 1.0 if y.start() < n.start() else 0.0  # first standalone yes/no wins
    else:
        s = float("nan")  # neither yes nor no as a standalone word -> parse failure
    detail = {"input_prompt": decode_prompt(ids[0]), "generated_text": gen_text,
              "yes_logmass": None, "no_logmass": None, "n_gen": int(gen.shape[1] - ids.shape[1]),
              "no_verdict": bool(math.isnan(s)), "max_tokens": max_tokens}
    return s, detail

# Transcript: one JSON line per scored item (decoded prompt + output + scoring internals),
# so every result is auditable / re-derivable. Stays on the box + rsynced down (large).
tf = open(transcript_file, "a")

for i in range(start_idx, len(items)):
    it = items[i]
    if it.get("greedy"):
        s, detail = greedy_yesno_score(it["text"])
    elif it.get("cot_greedy"):
        s, detail = cot_greedy_score(it["text"])  # CoT prompt + parse the EMITTED verdict token
    elif it.get("cot"):
        s, detail = cot_then_verdict_score(it["text"])  # NaN if the model volunteered no verdict
    else:
        s, detail = logprob_score(it["text"])
    scores.append(float(s))
    rec = {"i": i, "cve_id": it.get("cve_id"), "pair_idx": it.get("pair_idx"),
           "which": it.get("which"), "score": (None if math.isnan(s) else s), **detail}
    tf.write(json.dumps(rec) + "\n"); tf.flush()
    if (i + 1) % ckpt_every == 0:
        save_ckpt()
        print(f"  [{i+1}/{len(items)}] scored", flush=True)
save_ckpt()
tf.close()
"""


def build_fewshot_prefix() -> str:
    """Four FIXED canonical exemplars for the fewshot mode (2 vulnerable, 2 fixed).

    Deliberately demonstrate CWEs that are NOT in the eval set {CWE-022, 078, 079, 089},
    so the exemplars teach the GENERAL "vulnerable vs fixed" notion without leaking the
    answer pattern for the CWEs the model is actually judged on. Used here:
      - CWE-502 Unsafe Deserialization (pickle.loads on untrusted bytes)
      - CWE-918 Server-Side Request Forgery (unvalidated user-controlled URL)
    These are canonical textbook patterns authored here (not drawn from SVEN, PatchEval,
    or any scraped corpus), so they are provably disjoint from both train and eval and
    cannot contaminate the comparison. Each CWE shows the vulnerable form (YES) and its
    standard fix (NO). The prefix is fixed + identical across all scored CVEs.
    """
    # CWE-502: unsafe deserialization of attacker-controlled bytes.
    deser_vul = (
        "import pickle\n"
        "def load_session(raw_bytes):\n"
        "    return pickle.loads(raw_bytes)  # untrusted input -> arbitrary code exec"
    )
    deser_fix = (
        "import json\n"
        "def load_session(raw_bytes):\n"
        "    return json.loads(raw_bytes)  # data-only deserialization, no code exec"
    )
    # CWE-918: SSRF via an unvalidated user-supplied URL.
    ssrf_vul = (
        "import requests\n"
        "def fetch(user_url):\n"
        "    return requests.get(user_url).text  # user controls the destination host"
    )
    ssrf_fix = (
        "import requests\n"
        "from urllib.parse import urlparse\n"
        "ALLOWED = {'api.example.com'}\n"
        "def fetch(user_url):\n"
        "    if urlparse(user_url).hostname not in ALLOWED:\n"
        "        raise ValueError('host not allowed')\n"
        "    return requests.get(user_url).text"
    )
    return (
        "Example 1:\n```\n" + deser_vul + "\n```\nVulnerable? YES\n\n"
        "Example 2:\n```\n" + deser_fix + "\n```\nVulnerable? NO\n\n"
        "Example 3:\n```\n" + ssrf_vul + "\n```\nVulnerable? YES\n\n"
        "Example 4:\n```\n" + ssrf_fix + "\n```\nVulnerable? NO\n\n"
    )


def build_items(
    cves: list[dict[str, Any]], mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the per-prompt worker items (interleaved vul, fix) + per-CVE metadata."""
    prompt_style, readout = split_mode(mode)
    fewshot_prefix = build_fewshot_prefix() if prompt_style == "fewshot" else ""
    template = COT_PROMPT if prompt_style == "cot" else PROMPT
    # Worker readout flags (mutually exclusive). The dispatch routes on these:
    #   cot          -> cot_then_verdict_score   (cot prompt, logprob at the verdict slot)
    #   greedy       -> greedy_yesno_score       (noshot/fewshot prompt, parse the emitted token)
    #   cot_greedy   -> cot_greedy_score         (cot prompt, parse the emitted VERDICT token)
    #   (none)       -> logprob_score            (noshot/fewshot prompt, logit at the answer slot)
    is_cot = mode == "cot-logprob"
    is_greedy = readout == "token" and prompt_style != "cot"
    is_cot_greedy = mode == "cot-token"

    items: list[dict[str, Any]] = []
    cve_meta: list[dict[str, Any]] = []
    for cve in cves:
        # Pair vul↔fix snippets EXACTLY as the probe arm (07) does, so both arms
        # score identical code per CVE. Emit a (vul, fix) prompt PAIR per aligned pair;
        # the per-CVE risk_gap is the MEAN of pair gaps (CVE = unit of analysis).
        vul, fix = cve["vul_func"], cve["fix_func"]
        stratum = cve_stratum(vul, fix)
        pairs, _unpaired = pair_cve_snippets(vul, fix)
        prompt_pairs = []
        for vi, fi in pairs:
            vul_code = vul[vi].get("snippet", "")
            fix_code = fix[fi].get("snippet", "")
            if not vul_code or not fix_code:
                continue
            prompt_pairs.append((vul_code, fix_code))
        if not prompt_pairs:
            continue  # D_unpairable / empty snippets — no ranking pair
        for pair_idx, (vul_code, fix_code) in enumerate(prompt_pairs):
            for which, code in (("vul", vul_code), ("fix", fix_code)):
                items.append(
                    {
                        "text": fewshot_prefix + template.format(code=code),
                        "cot": is_cot,
                        "greedy": is_greedy,
                        "cot_greedy": is_cot_greedy,
                        "cve_id": cve["cve_id"],  # transcript identity (item -> CVE/pair/side)
                        "pair_idx": pair_idx,
                        "which": which,
                    }
                )
        cve_meta.append(
            {
                "cve_id": cve["cve_id"],
                "cwe": cve_primary_cwe(cve),
                "split": classify_cve(cve),
                "stratum": stratum,
                "n_pairs": len(prompt_pairs),
                "in_trained_repo": cve_in_trained_repo(cve),  # F8 leak flag
            }
        )
    return items, cve_meta


def run_worker(
    model_id: str,
    items: list[dict[str, Any]],
    quantization: str | None,
    checkpoint_path: Path,
    transcript_path: Path,
    load_dtype: str = "float16",
    load_class: str | None = None,
) -> list[float]:
    """Run the scoring worker in a subprocess; return per-item P(YES) scores (NaN where a
    CoT/greedy item produced no parseable verdict). Writes a per-response audit transcript
    (.jsonl) at transcript_path: one line per item with the decoded prompt + output +
    scoring internals, so every result is re-derivable / inspectable. load_dtype/load_class
    carry the spec's full-precision dtype + explicit HF class (Gemma -> bf16 + text-only)."""
    print(f"\n  Loading {model_id} (subprocess)...", flush=True)
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(items, pf)
        items_path = pf.name
    script_path = tempfile.mktemp(suffix=".py")
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            script_path,
            model_id,
            items_path,
            str(checkpoint_path),
            quantization or "none",
            str(MAX_INPUT_TOKENS),
            str(CHECKPOINT_EVERY),
            str(COT_MAX_NEW_TOKENS),
            str(MAX_COT_RETRIES),
            str(transcript_path),
            load_dtype,
            load_class or "none",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=14400,
    )
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}", flush=True)
        msg = f"Worker exit {result.returncode}"
        raise RuntimeError(msg)

    with open(checkpoint_path) as f:
        scores: list[float] = json.load(f)["scores"]
    print(f"  Done in {time.time() - t0:.0f}s ({len(scores)} scores)", flush=True)
    for p in (items_path, script_path):
        with contextlib.suppress(OSError):
            os.unlink(p)
    return scores


def summarize(per_cve: list[dict[str, Any]]) -> dict[str, Any]:
    """Win-rate (Wilson CI + sign test) + AUC, per seen/unseen slice."""

    def slice_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        # Drop rows with a None risk_gap (greedy parse failures) before the win-rate; the
        # dropped count is surfaced so the parse-failure rate stays visible.
        scored = [r for r in rows if r.get("risk_gap") is not None]
        gaps = [r["risk_gap"] for r in scored]
        wr = paired_win_rate(gaps).as_dict()
        wr["n_dropped_unparseable"] = len(rows) - len(scored)
        # Continuous-score AUC: label vul=1 / fix=0 across the slice's scores. Only rows
        # with both scores present (None excluded).
        aucrows = [
            r for r in scored if r.get("vul_score") is not None and r.get("fix_score") is not None
        ]
        y = [1] * len(aucrows) + [0] * len(aucrows)
        s = [r["vul_score"] for r in aucrows] + [r["fix_score"] for r in aucrows]
        wr["auc"] = round(float(roc_auc_score(y, s)), 4) if len(set(y)) > 1 and aucrows else None
        return wr

    unseen = [r for r in per_cve if r["split"] == "unseen"]
    seen = [r for r in per_cve if r["split"] == "seen"]
    # F3 stratified (mirror 07): A_single is the HEADLINE; multi scored per pair.
    unseen_A = [r for r in unseen if r.get("stratum") == "A_single"]
    unseen_multi = [r for r in unseen if r.get("stratum") in ("B_multi_full", "C_partial")]
    # F8 cascade (mirror 07): repo-disjoint is PRIMARY.
    unseen_disjoint = [r for r in unseen if not r.get("in_trained_repo")]
    return {
        "unseen": slice_stats(unseen),
        "seen": slice_stats(seen),
        "overall": slice_stats(per_cve),
        "by_stratum": {
            "unseen_A_single": slice_stats(unseen_A),  # HEADLINE
            "unseen_multi": slice_stats(unseen_multi),
            "unseen_all": slice_stats(unseen),
        },
        "repo_cascade": {
            "unseen_repo_disjoint": slice_stats(unseen_disjoint),  # PRIMARY
            "unseen_full": slice_stats(unseen),
            "n_unseen_dropped_trained_repo": sum(1 for r in unseen if r.get("in_trained_repo")),
        },
    }


def process_model(slug: str, cves: list[dict[str, Any]], mode: str, out_dir: Path) -> None:
    spec = MODEL_REGISTRY[slug]
    items, cve_meta = build_items(cves, mode)
    print(f"  {len(cve_meta)} CVEs, {len(items)} scoring items (mode={mode})", flush=True)

    ckpt = out_dir / f"_ckpt_elicit_{mode}_{slug}.json"
    transcript = out_dir / "transcripts" / f"{mode}_{slug}.jsonl"
    scores = run_worker(
        spec.hf_instruct_id,
        items,
        spec.quantization,
        ckpt,
        transcript,
        load_dtype=spec.dtype,
        load_class=spec.hf_load_class,
    )

    per_cve: list[dict[str, Any]] = []
    cursor = 0  # walks `scores` in [vul, fix] pairs; each CVE consumes 2*n_pairs items
    # greedy parse failures AND cot no-verdict failures both surface as NaN scores; a CVE
    # whose every pair is NaN is a genuine elicitation failure (dropped from the win-rate).
    n_fail = 0
    for meta in cve_meta:
        n_pairs = meta["n_pairs"]
        pair_gaps = []
        vul_scores, fix_scores = [], []
        for _ in range(n_pairs):
            vs = scores[cursor]
            fs = scores[cursor + 1]
            cursor += 2
            pair_gaps.append(vs - fs)
            vul_scores.append(vs)
            fix_scores.append(fs)
        # CVE-level aggregation: mean pair gap / mean score (one observation per CVE).
        # nanmean ignores NaN pairs; a CVE with every pair NaN yields NaN (a failure).
        with np.errstate(invalid="ignore"):
            vul_m = float(np.nanmean(vul_scores)) if vul_scores else float("nan")
            fix_m = float(np.nanmean(fix_scores)) if fix_scores else float("nan")
            gap_m = float(np.nanmean(pair_gaps)) if pair_gaps else float("nan")
        # Modes that emit NaN on a genuine elicitation failure (parse / no-verdict): greedy
        # (parse), cot (no volunteered verdict), cot_greedy (no emitted/parseable verdict).
        if mode in BINARY_FAILURE_MODES and np.isnan(vul_m) and np.isnan(fix_m):
            n_fail += 1
        per_cve.append(
            {
                **meta,
                "vul_score": None if np.isnan(vul_m) else round(vul_m, 6),
                "fix_score": None if np.isnan(fix_m) else round(fix_m, 6),
                "risk_gap": None if np.isnan(gap_m) else round(gap_m, 6),
            }
        )
    if mode in BINARY_FAILURE_MODES:
        # noshot/fewshot-token parse a plain YES/NO; cot-token parses the emitted VERDICT marker.
        kind = "no-verdict" if mode == "cot-token" else "parse"
        print(f"  {mode} {kind} failures: {n_fail}/{len(per_cve)} CVEs", flush=True)

    output = {
        "slug": slug,
        "model_id": spec.hf_instruct_id,
        "mode": mode,
        "n_cves": len(per_cve),
        "n_elicitation_failures": n_fail if mode in BINARY_FAILURE_MODES else None,
        "win_rate": summarize(per_cve),
        "per_cve": per_cve,
    }
    out_path = out_dir / f"patcheval_elicit_{mode}_{slug}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    with contextlib.suppress(OSError):
        ckpt.unlink()

    u = output["win_rate"]["unseen"]
    print(f"\n  Saved: {out_path}", flush=True)
    print(
        f"  unseen win-rate={u['win_rate']:.3f} [{u['ci_lower']:.3f},{u['ci_upper']:.3f}] "
        f"AUC={u['auc']} n={u['n_effective']} p={u['p_value']:.2e}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchEval prompted elicitation suite")
    parser.add_argument("--mode", choices=MODES, default="noshot-logprob")
    parser.add_argument("--models", nargs="*", default=None, help="slugs; default = all that fit")
    args = parser.parse_args()

    cve_path = PROJECT / "data" / "patcheval" / "python_cves.json"
    with open(cve_path) as f:
        cves = json.load(f)
    valid = [c for c in cves if c.get("vul_func") and c.get("fix_func")]
    assert_expected_split(valid)
    print(f"PatchEval CVEs: {len(valid)} valid (mode={args.mode})", flush=True)

    out_dir = PROJECT / "outputs" / "phase1"
    # MODEL_REGISTRY is already scoped at import time by ASP_MODEL_FILTER (models.py), so
    # out-of-scope models (starcoder2/codellama) never appear here. --models narrows further.
    slugs = args.models or sorted(
        MODEL_REGISTRY.keys(), key=lambda s: MODEL_REGISTRY[s].min_vram_gb, reverse=True
    )

    for slug in slugs:
        out_path = out_dir / f"patcheval_elicit_{args.mode}_{slug}.json"
        if out_path.exists():
            print(f"\nSKIP {slug}: {args.mode} results already exist", flush=True)
            continue
        print(f"\n{'=' * 60}\nElicit [{args.mode}]: {slug}\n{'=' * 60}", flush=True)
        try:
            process_model(slug, valid, args.mode, out_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
