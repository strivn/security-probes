# security-probes

Linear and MLP probes on an open-weight reviewer model's activations, for
detecting code security vulnerabilities that the same model's prompted text
output misses.

**Research question.** Can activation-level access to a reviewer model add
detection value beyond what prompting the same model achieves? The pipeline
trains probes on residual-stream activations over paired vulnerable/fixed code,
compares them against forced YES/NO prompting and Semgrep-class static analysis,
and evaluates out-of-distribution transfer to real-world Python CVEs.

## Paper

**Activation Probes Surface Code-Security Signals that the Model's Output Misses**

<!-- Add a link to the paper (arXiv / OpenReview PDF) here once it is public. -->

AI coding agents write a growing share of production code, but human security
review does not scale at the rate code is generated. The agents in widest use
are closed-weight, so a deploying team cannot read their internals. It can
instead run an open-weight model as a *reviewer* over the agent's output, whose
activations are readable. This work asks whether reading those activations
recovers a security signal that simply *asking* the same reviewer misses.

We fit a single linear probe per model on paired vulnerable/fixed Python
functions, then test it without retraining on real disclosed CVEs whose weakness
type the probe never saw in training, across five open-weight reviewer models.
On vulnerabilities fixed by a single-function change, the probe scores the
vulnerable function above its fix in 61–67% of cases for every model, beating
both the 50% chance line and the same model's prompted YES/NO win-rate under
every prompt tried. Asking the model for a written verdict, even with
chain-of-thought, returns the same answer on the vulnerable and fixed function
most of the time and cannot tell them apart.

**Takeaway:** model activations carry a code-security signal that prompting the
same model misses.

## Layout

```
src/agentic_sec_probe/   reusable library
  data.py                paired vulnerable/fixed sample loader
  activations.py         nnsight residual-stream extraction
  models.py              model registry + GPU VRAM guard
  probe.py               logistic-regression probe
  mlp_probe.py           MLP probe (d -> 256 -> 64 -> 1)
  patcheval.py           PatchEval CVE loader
  yesno.py               forced YES/NO prompted scoring
  logprob_scoring.py     logprob-based verdict readout
  vllm_cot.py            chain-of-thought scoring via vLLM (generated-slot readout)
  paired_stats.py        paired win-rate + McNemar significance
  freshness.py           output freshness guard

scripts/phase1/          numbered runnable pipeline (entry points)
tests/                   unit tests
```

## Pipeline (`scripts/phase1/`)

| Step | Purpose |
|------|---------|
| `00_create_split` | Stratified per-CWE train/eval split (seed 42) |
| `01_extract_activations` | Extract residual-stream activations per model |
| `02_layer_sweep` | Probe accuracy across layers |
| `04_mlp_probes` | Train MLP probes per model x scope x strategy |
| `05_prompted_eval` | Forced YES/NO prompted baseline |
| `06_semgrep_eval` | Semgrep static-analysis baseline |
| `07_patcheval_ood` | Probe out-of-distribution eval on PatchEval CVEs |
| `08_analyze_results` | Aggregate JSON outputs into tables |
| `09_probe_eval` | Per-CWE probe metrics + checkpointing |
| `11_patcheval_semgrep` | Semgrep baseline on PatchEval |
| `12_patcheval_prompted` | Prompted baseline on PatchEval |
| `13_patcheval_prompted_elicit` | Stronger prompted baselines (few-shot / CoT) |
| `random_baseline` | Significance floor per scope |
| `task0_vllm_probe`, `vllm_cot_score` | vLLM chain-of-thought scoring |
| `preflight_hooks` | Verify nnsight hook points per model |

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

vLLM chain-of-thought scoring (`task0_vllm_probe`, `vllm_cot_score`) runs in a
separate standalone environment; see `scripts/setup_vllm_venv.sh`.

## Data (not included)

This repository ships **code only**. The pipeline expects two datasets you must
obtain separately:

- **SVEN** (paired vulnerable/fixed code, 9 CWEs). Jingxuan He and Martin
  Vechev, *Large Language Models for Code: Security Hardening and Adversarial
  Testing*, ACM CCS 2023 ([arXiv:2302.05319](https://arxiv.org/abs/2302.05319),
  [eth-sri/sven](https://github.com/eth-sri/sven), MIT-licensed). Place the
  JSONL files under `data/raw/sven_train/` and `data/raw/sven_val/`.
- **PatchEval** (real-world Python CVEs, used for OOD evaluation). Place under
  `data/patcheval/`.

Without these, `scripts/phase1/00_create_split.py` and downstream steps will not
run, and data-dependent tests will fail.

## License

MIT. See [LICENSE](LICENSE). SVEN and PatchEval are the property of their
respective authors under their own licenses.
