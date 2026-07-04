"""PatchEval out-of-distribution validation.

For each model:
  1. Read layer sweep JSON and pick the CV-best (strategy, layer, C) for the
     Python scope — same selection logic as 08/09. This probe is IDENTICAL to
     the main-table Python probe (no hardcoded "mean").
  2. Train LogReg probe on SVEN Python train activations at that (strategy, layer).
  3. For each of 404 PatchEval CVEs:
     - Tokenize with truncation to MAX_INPUT_TOKENS (fixes coverage failures on
       devstral / starcoder2 / deepseek-33b in prior runs where long CVEs
       overflowed the model context).
     - Extract residual stream activations at the CV-best layer, pool with the
       CV-best strategy via src/agentic_sec_probe/activations._pool_tokens.
     - Probe confidence P(vul) on both vul_func and fix_func snippets.
     - Delta = P(vul_code) - P(fix_code).
     - Cache raw vul_act/fix_act vectors to outputs/phase1/patcheval_activations_{slug}.pt
       so future probe changes become a scoring loop, no nnsight re-extraction.
  4. Statistical tests:
     - Paired Wilcoxon signed-rank (two-sided — H0: median delta = 0).
       Two-sided because the direction of the effect was not pre-registered;
       one-sided post-hoc selection inflates type-I error.
     - Cohen's d effect size.
     - Bootstrap 95% CI on mean delta.
     - In-distribution (SVEN Python CWEs) vs OOD CWE breakdown.

Uses first vul/fix function per CVE.

Input:  data/patcheval/python_cves.json
        outputs/phase1/layer_sweep_{slug}.json
Output: outputs/phase1/patcheval_ood_{slug}.json
        outputs/phase1/patcheval_activations_{slug}.pt

Runs on GPU (needs model loaded via nnsight for activation extraction).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

if TYPE_CHECKING:
    import numpy.typing as npt

from agentic_sec_probe.activations import _pool_tokens, _resolve_hook_proxy
from agentic_sec_probe.models import MODEL_REGISTRY, check_gpu_vram
from agentic_sec_probe.paired_stats import paired_win_rate
from agentic_sec_probe.patcheval import (
    TRAINED_CWES,
    assert_expected_split,
    classify_cve,
    cve_in_trained_repo,
    cve_label_set,
    cve_primary_cwe,
    cve_repo,
    cve_stratum,
    pair_cve_snippets,
)

# Seen/unseen-bug-types classification is shared with the semgrep baseline (11) so
# both report on the identical slice. See src/agentic_sec_probe/patcheval.py.
IN_DIST_CWES = TRAINED_CWES  # backward-compat alias for downstream readers

N_BOOTSTRAP = 1000
SEED = 42
CHECKPOINT_EVERY = 25
# Context-length truncation shared with script 05. Prevents position-embedding
# overflow on long PatchEval snippets. 8192 is well within every model's
# max_position_embeddings (smallest in the registry is 16k).
MAX_INPUT_TOKENS = 8192
PROBE_SCOPE = "python"
# Deployed probe = torch, plain Adam, constant weight_decay (identical to 09).
PROBE_WEIGHT_DECAY = 0.01
PROBE_LR = 0.01
PROBE_N_STEPS = 200

# Sentinel key in patcheval_activations_{slug}.pt recording which
# (strategy, layer) the cached vectors were extracted at. A run under a
# different CV-best selection detects the mismatch and starts fresh
# instead of silently mixing vectors from incompatible extraction configs.
ACT_CACHE_META_KEY = "_meta"


def load_cve_dataset(path: Path, *, assert_split: bool = True) -> list[dict[str, Any]]:
    """Load PatchEval CVEs, filtering to those with both vul and fix functions.

    When `assert_split` is True (default for the canonical file), verifies the
    seen/unseen split counts match EXPECTED_SPLIT — a repro guard so the published
    headline n cannot silently drift.
    """
    with open(path) as f:
        cves: list[dict[str, Any]] = json.load(f)
    valid = [c for c in cves if c.get("vul_func") and c.get("fix_func")]
    n_seen = sum(1 for c in valid if classify_cve(c) == "seen")
    print(
        f"  PatchEval: {len(valid)}/{len(cves)} CVEs with both vul and fix functions "
        f"(seen={n_seen}, unseen={len(valid) - n_seen})"
    )
    if assert_split:
        assert_expected_split(valid)
    return valid


def find_cv_best_config(sweep: dict[str, Any], scope_name: str) -> dict[str, Any]:
    """Return CV-best (strategy, layer) for a scope.

    No C (regularization is the constant weight_decay). Iterates all strategies in
    the sweep JSON and picks the one with highest best_cv_auc. Matches 09_probe_eval so
    the OOD probe is identical to the main-table deployed probe.
    """
    best: dict[str, Any] | None = None
    for strat_name, scope_results in sweep["strategies"].items():
        sc = scope_results.get(scope_name, {})
        if "best_cv_auc" not in sc:
            continue
        cv_auc = sc["best_cv_auc"]
        if best is None or cv_auc > best["cv_auc"]:
            best = {
                "strategy": strat_name,
                "layer": sc["best_layer"],
                "cv_auc": cv_auc,
                "eval_auc": sc.get("best_eval_auc"),
            }
    if best is None:
        msg = f"No valid CV-best config for scope={scope_name}"
        raise ValueError(msg)
    return best


class TorchProbe:
    """Single-layer linear probe (torch, plain Adam, constant weight_decay).

    Mirrors 09_probe_eval.train_single_layer_probe exactly (same seed, wd, steps) so the
    OOD scorer uses the IDENTICAL deployed probe family as the main table (torch
    end-to-end). Exposes `predict_proba` so existing call sites are unchanged.
    """

    def __init__(self, w: npt.NDArray[np.float32], b: float) -> None:
        self.w = w  # [d]
        self.b = b

    def predict_proba(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Return [[P(0), P(1)], ...] for scaled inputs x of shape [n, d]."""
        logits = x @ self.w + self.b
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - p1, p1], axis=1)


def train_probe_from_sweep(
    chunks_dir: Path,
    strategy: str,
    layer: int,
) -> tuple[TorchProbe, StandardScaler]:
    """Train the deployed torch probe on SVEN Python train activations.

    Scope by per-sample `language`, not CWE membership. Torch probe at the
    constant WEIGHT_DECAY (same method as the 02 sweep + 09 deploy).
    """
    chunk_files = sorted(chunks_dir.glob("sample_*.pt"))
    vectors, labels = [], []
    act_key = f"resid_pre__{strategy}"

    for cf in chunk_files:
        chunk = torch.load(cf, weights_only=True)  # trusted chunks: tensors + plain types
        if chunk["split"] != "train":
            continue
        if chunk["language"] != "python":
            continue
        vectors.append(chunk[act_key][layer].float().numpy())
        labels.append(chunk["label"])

    X = np.stack(vectors)
    y = np.array(labels, dtype=np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Torch probe — identical config to 09's deployed probe.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = X_scaled.shape[1]
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)
    W = torch.zeros(d, 1, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([W, b], lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    for _ in range(PROBE_N_STEPS):
        logits = (X_t @ W).squeeze(-1) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w_np = W.detach().cpu().numpy().reshape(-1).astype(np.float32)
    b_val = float(b.detach().cpu().numpy()[0])
    probe = TorchProbe(w_np, b_val)

    # sklearn second opinion (the "other test") at the same layer — held-out AUC only.
    clf_sk = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf_sk.fit(X_scaled, y)

    print(
        f"  Trained torch probe: strategy={strategy}, layer={layer}, "
        f"wd={PROBE_WEIGHT_DECAY}, n_train={len(y)}",
        flush=True,
    )
    return probe, scaler


@torch.no_grad()
def extract_single_activation(
    model: Any,
    code: str,
    layer: int,
    strategy: str,
    hook_pattern: str,
    max_length: int = MAX_INPUT_TOKENS,
) -> npt.NDArray[np.float32]:
    """Extract pooled activation at a single layer for one code snippet.

    F5 fix: uses the IDENTICAL hook resolver (`_resolve_hook_proxy`) and pooling
    (`_pool_tokens`) as the SVEN training path in `activations.py`, and feeds the
    truncated TOKEN TENSOR directly to `tracer.invoke` (nnsight 0.6.3 `_tokenize`
    accepts a torch.Tensor of input_ids). This REMOVES the prior tokenize->decode->
    re-tokenize round-trip, which could change the token sequence via byte-BPE
    boundary effects and diverge from the training-path activations.

    Wrapped in `@torch.no_grad()` to match `extract_activations_batch`. Without it,
    autograd graph state accumulates across calls and OOMs the GPU on quantized
    mid-size models within ~80 CVEs.
    """
    token_ids = model.tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )["input_ids"]
    if len(token_ids) == 0:
        msg = "empty tokenization"
        raise ValueError(msg)
    # Pass the pre-tokenized, truncated ids straight to nnsight (no decode round-trip).
    # NOTE: nested `with` form (not the combined `with A, B:` one-liner) — nnsight needs
    # the trace context fully entered before invoke() runs (verified on nnsight 0.6.3;
    # the one-liner raised "model is not executing during interleaving").
    input_ids = torch.tensor(token_ids, dtype=torch.long)

    # noqa SIM117: the nested `with` is REQUIRED — combining into `with A, B:` makes
    # nnsight evaluate invoke() before the trace context is entered ("model is not
    # executing during interleaving"). Verified on nnsight 0.6.3.
    with model.trace() as tracer:  # noqa: SIM117
        with tracer.invoke(input_ids):
            proxy = _resolve_hook_proxy(model, hook_pattern, layer)
            act = proxy.save()  # type: ignore[attr-defined]

    # nnsight's `.input[0]` on input_layernorm resolves to the first batch
    # element of the hidden states, yielding [seq, d_model] (2D) for batch=1.
    # _pool_tokens expects [n_layers, seq, d_model]; we synthesize a singleton
    # layer dim so the pooling is bit-identical to the training path.
    tensor_cpu = act.detach().float().cpu()
    del act
    if tensor_cpu.dim() == 2:
        tensor_3d = tensor_cpu.unsqueeze(0)  # [1, seq, d_model]
    elif tensor_cpu.dim() == 3:
        tensor_3d = tensor_cpu  # [1, seq, d_model] already
    else:
        msg = f"Unexpected activation shape: {tuple(tensor_cpu.shape)}"
        raise ValueError(msg)
    pooled = _pool_tokens(tensor_3d, strategy)  # [1, d_model]
    vec: npt.NDArray[np.float32] = pooled[0].numpy()
    return vec


def process_model(
    slug: str,
    cve_list: list[dict[str, Any]],
    sweep_path: Path,
    chunks_dir: Path,
    out_dir: Path,
) -> None:
    """Run PatchEval OOD validation for one model."""
    spec = MODEL_REGISTRY[slug]

    with open(sweep_path) as f:
        sweep = json.load(f)

    best_cfg = find_cv_best_config(sweep, PROBE_SCOPE)
    strategy = best_cfg["strategy"]
    layer = best_cfg["layer"]
    print(
        f"  CV-best Python probe: {strategy} L{layer} wd={PROBE_WEIGHT_DECAY} "
        f"(sweep CV AUC={best_cfg['cv_auc']:.3f}, held-out AUC={best_cfg['eval_auc']})",
        flush=True,
    )

    clf, scaler = train_probe_from_sweep(chunks_dir, strategy, layer)

    from nnsight import LanguageModel

    # Force all-on-GPU for quantized models. accelerate's device_map="auto" may
    # conservatively spill weights to CPU for models like starcoder2-15b (d=6144),
    # and 8-bit bitsandbytes rejects CPU offload without llm_int8_enable_fp32_cpu_offload.
    load_kwargs: dict[str, Any] = {"dispatch": True}
    if spec.quantization == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["device_map"] = {"": 0}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call, unused-ignore]
    elif spec.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["device_map"] = {"": 0}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)  # type: ignore[no-untyped-call, unused-ignore]
    elif spec.quantization is None:
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16

    # INSTRUCT checkpoint so probe + prompted baseline use the same model.
    print(f"  Loading {spec.probe_model_id}...", flush=True)
    model = LanguageModel(spec.probe_model_id, **load_kwargs)

    # Disable KV cache: extraction is one-shot forward, KV cache wastes VRAM
    # and (worse) accumulates across nnsight trace calls in some setups.
    model._model.config.use_cache = False

    # Checkpoint: resume per-CVE probs + raw activations from prior partial run.
    # The act_cache is tagged with ACT_CACHE_META_KEY = {"strategy", "layer"}
    # so a new run under different CV-best selection detects incompatible
    # cached vectors and starts fresh (old file overwritten on next save).
    # Pre-v2 cache files have no _meta → they always fall through to reset.
    checkpoint_path = out_dir / f"_ckpt_patcheval_ood_{slug}.json"
    activations_path = out_dir / f"patcheval_activations_{slug}.pt"

    # cache_format v2: per-pair vectors (F3 stratified). Bumping invalidates any v1
    # ([0]-snippet) cache so a stale single-snippet cache can't be resumed into the
    # stratified scorer.
    current_meta: dict[str, Any] = {
        "strategy": strategy,
        "layer": layer,
        "cache_format": "v2_pairs",
    }
    per_cve: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    act_cache: dict[str, Any] = {ACT_CACHE_META_KEY: current_meta}

    cache_compatible = False
    if activations_path.exists():
        try:
            # weights_only=False required: this cache stores numpy arrays (vul/fix
            # vectors), which weights_only=True rejects (UnpicklingError, verified).
            # The file is our own freshly-written cache, not untrusted input.
            loaded = torch.load(activations_path, weights_only=False)
            loaded_meta = loaded.get(ACT_CACHE_META_KEY) if isinstance(loaded, dict) else None
            if loaded_meta == current_meta:
                act_cache = loaded
                cache_compatible = True
                n_cached = sum(1 for k in act_cache if k != ACT_CACHE_META_KEY)
                print(f"  Resumed activations: {n_cached} cached", flush=True)
            else:
                print(
                    f"  Activation cache incompatible "
                    f"(cached meta={loaded_meta}, current={current_meta}) "
                    f"— starting fresh; stale file will be overwritten",
                    flush=True,
                )
        except (OSError, RuntimeError) as exc:
            print(f"  Could not load activations ({exc}); starting fresh", flush=True)

    if cache_compatible and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt_data = json.load(f)
            per_cve = list(ckpt_data.get("per_cve", []))
            done_ids = {r["cve_id"] for r in per_cve}
            print(f"  Resumed checkpoint: {len(per_cve)} CVEs done", flush=True)
        except (json.JSONDecodeError, OSError):
            per_cve, done_ids = [], set()
    elif checkpoint_path.exists():
        print(
            "  Checkpoint exists but cache is incompatible "
            "— discarding stale per_cve; stale file will be overwritten",
            flush=True,
        )

    def save_ckpt() -> None:
        tmp_json = str(checkpoint_path) + ".tmp"
        with open(tmp_json, "w") as f:
            json.dump({"per_cve": per_cve}, f)
        os.replace(tmp_json, str(checkpoint_path))

        tmp_pt = str(activations_path) + ".tmp"
        torch.save(act_cache, tmp_pt)
        os.replace(tmp_pt, str(activations_path))

    t_start = time.time()
    n_this_run = 0

    for _ci, cve in enumerate(cve_list):
        cve_id = cve["cve_id"]
        if cve_id in done_ids:
            continue

        primary_cwe = cve_primary_cwe(cve)
        split = classify_cve(cve)  # "seen" or "unseen" (full-label-set rule)

        # Pair vul↔fix snippets within the CVE (file_path + def-name), score EACH
        # pair, aggregate to ONE per-CVE delta (mean of pair deltas). The CVE is the unit
        # of analysis so the downstream win-rate / McNemar are not pseudo-replicated.
        vul, fix = cve["vul_func"], cve["fix_func"]
        stratum = cve_stratum(vul, fix)
        pairs, unpaired = pair_cve_snippets(vul, fix)

        try:
            if not pairs:
                # D_unpairable: no def in any vul snippet — cannot form a ranking pair.
                msg = "unpairable (no paired def snippets)"
                raise ValueError(msg)

            pair_deltas: list[float] = []
            pair_records: list[dict[str, Any]] = []
            cached_vecs: list[dict[str, Any]] = []
            for vi, fi in pairs:
                vul_act = extract_single_activation(
                    model, vul[vi]["snippet"], layer, strategy, spec.hook_pattern
                )
                fix_act = extract_single_activation(
                    model, fix[fi]["snippet"], layer, strategy, spec.hook_pattern
                )
                vp = float(clf.predict_proba(scaler.transform(vul_act.reshape(1, -1)))[0, 1])
                fp = float(clf.predict_proba(scaler.transform(fix_act.reshape(1, -1)))[0, 1])
                pair_deltas.append(vp - fp)
                pair_records.append(
                    {"vi": vi, "fi": fi, "vul_prob": round(vp, 4), "fix_prob": round(fp, 4)}
                )
                cached_vecs.append({"vul": vul_act, "fix": fix_act})

            # CVE-level delta = mean of its pair deltas (one observation per CVE).
            delta = float(np.mean(pair_deltas))

            per_cve.append(
                {
                    "cve_id": cve_id,
                    "cwe": primary_cwe,
                    "label_set": sorted(cve_label_set(cve)),
                    "split": split,  # "seen" / "unseen" (full-label-set rule)
                    "in_distribution": split == "seen",
                    "repo": cve_repo(cve),
                    "in_trained_repo": cve_in_trained_repo(cve),  # F8 leak flag
                    "stratum": stratum,  # A_single / B_multi_full / C_partial
                    "n_pairs": len(pairs),
                    "n_unpaired_vul": len(unpaired),
                    "pairs": pair_records,
                    "delta": round(delta, 4),  # mean pair delta (CVE unit of analysis)
                }
            )
            act_cache[cve_id] = {"pairs": cached_vecs}

        except Exception as exc:
            import traceback

            per_cve.append(
                {
                    "cve_id": cve_id,
                    "cwe": primary_cwe,
                    "split": split,
                    "error": str(exc)[:400],
                    "exc_type": type(exc).__name__,
                    "traceback": traceback.format_exc()[-1200:],
                }
            )

        n_this_run += 1
        # Free any leftover allocations from the trace before the next CVE.
        # Cheap, and prevents fragmentation OOM on quantized 8-bit / 4-bit models.
        torch.cuda.empty_cache()
        if n_this_run % CHECKPOINT_EVERY == 0:
            save_ckpt()
            elapsed = time.time() - t_start
            rate = n_this_run / elapsed * 60 if elapsed > 0 else 0
            n_valid = sum(1 for r in per_cve if "delta" in r)
            n_skip = sum(1 for r in per_cve if "error" in r)
            remaining = len(cve_list) - len(per_cve)
            eta_min = remaining / rate if rate > 0 else 0
            print(
                f"    [{len(per_cve)}/{len(cve_list)}] valid={n_valid} skipped={n_skip}"
                f" rate={rate:.1f} cve/min ETA={int(eta_min // 60)}h{int(eta_min % 60):02d}m",
                flush=True,
            )

    if n_this_run > 0:
        save_ckpt()

    deltas = [r["delta"] for r in per_cve if "delta" in r]
    skipped = sum(1 for r in per_cve if "error" in r)
    print(f"  Processed: {len(deltas)} valid, {skipped} skipped", flush=True)

    deltas_arr = np.array(deltas)
    rng = np.random.RandomState(SEED)

    wilcoxon_alt = "two-sided"
    if len(deltas_arr) >= 10:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(deltas_arr, alternative=wilcoxon_alt)
    else:
        wilcoxon_stat = np.float64(0.0)
        wilcoxon_p = np.float64(1.0)

    mean_delta = float(np.mean(deltas_arr)) if len(deltas_arr) > 0 else 0.0
    std_delta = float(np.std(deltas_arr, ddof=1)) if len(deltas_arr) > 1 else 1.0
    cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0

    boot_means = []
    if len(deltas_arr) > 0:
        for _ in range(N_BOOTSTRAP):
            sample = rng.choice(deltas_arr, size=len(deltas_arr), replace=True)
            boot_means.append(float(np.mean(sample)))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        ci_lower = 0.0
        ci_upper = 0.0

    # Split risk gaps by seen/unseen bug types (full-label-set rule).
    seen_deltas = [r["delta"] for r in per_cve if r.get("split") == "seen" and "delta" in r]
    unseen_deltas = [r["delta"] for r in per_cve if r.get("split") == "unseen" and "delta" in r]

    # Headline metric: paired win-rate (Wilson CI + exact-binomial sign test).
    # The UNSEEN slice is the published headline; seen + overall are corroboration.
    # This replaces the previous mean-delta bootstrap, which was the wrong statistic
    # for a fraction-of-pairs claim and was computed on the pooled set.
    winrate_unseen = paired_win_rate(unseen_deltas)
    winrate_seen = paired_win_rate(seen_deltas)
    winrate_overall = paired_win_rate(deltas)

    # F3 stratified win-rates on the UNSEEN slice (one delta per CVE).
    #   A_single  = the published HEADLINE stratum (single-snippet CVEs, cleanest pairing)
    #   multi     = B_multi_full + C_partial (multi-snippet, scored per aligned pair)
    #   all_strata= every pairable unseen CVE (A + multi) — robustness, same as winrate_unseen
    def _stratum_deltas(strata: set[str]) -> list[float]:
        return [
            r["delta"]
            for r in per_cve
            if r.get("split") == "unseen" and "delta" in r and r.get("stratum") in strata
        ]

    winrate_unseen_A = paired_win_rate(_stratum_deltas({"A_single"}))
    winrate_unseen_multi = paired_win_rate(_stratum_deltas({"B_multi_full", "C_partial"}))

    # F8 cascade: repo-disjoint (PRIMARY) vs full, on the unseen slice. A CVE from a repo
    # seen in SVEN training (in_trained_repo) is dropped from the repo-disjoint subset.
    unseen_disjoint_deltas = [
        r["delta"]
        for r in per_cve
        if r.get("split") == "unseen" and "delta" in r and not r.get("in_trained_repo")
    ]
    winrate_unseen_repo_disjoint = paired_win_rate(unseen_disjoint_deltas)
    n_unseen_in_trained_repo = sum(
        1 for r in per_cve if r.get("split") == "unseen" and r.get("in_trained_repo")
    )

    in_dist_stats = _group_stats(np.array(seen_deltas)) if seen_deltas else None
    ood_stats = _group_stats(np.array(unseen_deltas)) if unseen_deltas else None

    cwe_groups: dict[str, list[float]] = {}
    for r in per_cve:
        if "delta" in r:
            cwe_groups.setdefault(r["cwe"], []).append(r["delta"])

    per_cwe_stats: dict[str, Any] = {}
    for cwe in sorted(cwe_groups):
        vals = cwe_groups[cwe]
        if len(vals) >= 5:
            per_cwe_stats[cwe] = _group_stats(np.array(vals))
            per_cwe_stats[cwe]["n"] = len(vals)
            # Per-CWE paired win-rate with Wilson CI so small-n cells can be
            # flagged n.s. when the CI overlaps the 0.5 chance floor.
            per_cwe_stats[cwe]["win_rate"] = paired_win_rate(vals).as_dict()

    output = {
        "slug": slug,
        "probe_scope": PROBE_SCOPE,
        "probe_strategy": strategy,
        "probe_layer": layer,
        "probe_weight_decay": PROBE_WEIGHT_DECAY,  # torch deploy, no C
        "probe_cv_auc": best_cfg["cv_auc"],  # torch sweep CV-AUC (deployed probe)
        "probe_held_out_auc": best_cfg["eval_auc"],  # torch sweep held-out AUC
        "n_total": len(cve_list),
        "n_valid": len(deltas),
        "n_skipped": skipped,
        "max_input_tokens": MAX_INPUT_TOKENS,
        # Primary metric: paired win-rate with Wilson 95% CI + sign-test p-value.
        "win_rate": {
            "unseen": winrate_unseen.as_dict(),  # all pairable unseen CVEs (robustness)
            "seen": winrate_seen.as_dict(),
            "overall": winrate_overall.as_dict(),
        },
        # F3 stratified win-rates on the unseen slice. A_single is the HEADLINE.
        "win_rate_by_stratum": {
            "unseen_A_single": winrate_unseen_A.as_dict(),  # HEADLINE: single-snippet
            "unseen_multi": winrate_unseen_multi.as_dict(),  # multi-snippet, per-pair
            "unseen_all": winrate_unseen.as_dict(),  # A + multi combined
        },
        # F8 cascade: repo-disjoint is PRIMARY (no SVEN-trained repo leaks in), full
        # alongside. Reported on the unseen slice.
        "win_rate_repo_cascade": {
            "unseen_repo_disjoint": winrate_unseen_repo_disjoint.as_dict(),  # PRIMARY
            "unseen_full": winrate_unseen.as_dict(),
            "n_unseen_dropped_trained_repo": n_unseen_in_trained_repo,
        },
        # Secondary corroboration: mean-risk-gap magnitude + Wilcoxon signed-rank.
        "overall": {
            "mean_delta": round(mean_delta, 4),
            "median_delta": round(float(np.median(deltas_arr)) if len(deltas_arr) > 0 else 0.0, 4),
            "std_delta": round(std_delta, 4),
            "cohens_d": round(cohens_d, 4),
            "wilcoxon_stat": round(float(wilcoxon_stat), 4),
            "wilcoxon_p": float(wilcoxon_p),
            "wilcoxon_alternative": wilcoxon_alt,
            "bootstrap_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "pct_positive_delta": round(
                float(np.mean(deltas_arr > 0)) if len(deltas_arr) > 0 else 0.0, 4
            ),
        },
        # "seen"/"unseen" bug types = full-label-set rule (classify_cve).
        # Keys kept as in_distribution/out_of_distribution for downstream compatibility.
        "in_distribution": {
            "n": len(seen_deltas),
            "cwes": sorted(TRAINED_CWES),
            "stats": in_dist_stats,
        },
        "out_of_distribution": {
            "n": len(unseen_deltas),
            "stats": ood_stats,
        },
        "per_cwe": per_cwe_stats,
        "per_cve": per_cve,
    }

    out_path = out_dir / f"patcheval_ood_{slug}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)
    n_cached = sum(1 for k in act_cache if k != ACT_CACHE_META_KEY)
    print(f"  Cached activations: {activations_path} ({n_cached} CVEs)", flush=True)

    with contextlib.suppress(OSError):
        checkpoint_path.unlink()

    wu = winrate_unseen
    print(
        f"  HEADLINE unseen-bug-types win-rate: {wu.win_rate:.3f} "
        f"[{wu.ci_lower:.3f}, {wu.ci_upper:.3f}] (Wilson 95%), "
        f"n={wu.n_effective} (wins={wu.wins} losses={wu.losses} ties_dropped={wu.ties}), "
        f"sign-test p={wu.p_value:.2e}"
    )
    ws = winrate_seen
    print(
        f"  seen-bug-types win-rate: {ws.win_rate:.3f} "
        f"[{ws.ci_lower:.3f}, {ws.ci_upper:.3f}] n={ws.n_effective} p={ws.p_value:.2e}"
    )
    print(f"  Mean delta (corroboration): {mean_delta:.4f}, Cohen's d={cohens_d:.3f}")
    print(f"  Wilcoxon p ({wilcoxon_alt}): {wilcoxon_p:.4e}")

    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def _group_stats(deltas: npt.NDArray[np.float64]) -> dict[str, float]:
    """Compute summary stats for a group of deltas."""
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    return {
        "mean_delta": round(mean, 4),
        "median_delta": round(float(np.median(deltas)), 4),
        "std_delta": round(std, 4),
        "pct_positive": round(float(np.mean(deltas > 0)), 4),
    }


def main() -> None:
    pe_path = PROJECT / "data" / "patcheval" / "python_cves.json"
    if not pe_path.exists():
        print("ERROR: PatchEval data not found")
        return

    cve_list = load_cve_dataset(pe_path)
    out_dir = PROJECT / "outputs" / "phase1"

    vram = check_gpu_vram()
    print(f"GPU VRAM: {vram:.1f} GB")

    for slug, spec in MODEL_REGISTRY.items():
        if spec.min_vram_gb > vram:
            print(f"\nSKIP {slug}: needs {spec.min_vram_gb} GB, have {vram:.1f} GB")
            continue

        sweep_path = out_dir / f"layer_sweep_{slug}.json"
        if not sweep_path.exists():
            print(f"\nSKIP {slug}: no layer sweep results")
            continue

        chunks_dir = PROJECT / "data" / "activations" / slug / "phase1_chunks"
        if not chunks_dir.exists():
            print(f"\nSKIP {slug}: no activation chunks")
            continue

        ood_path = out_dir / f"patcheval_ood_{slug}.json"
        if ood_path.exists():
            # Only skip if the saved run matches the CURRENT CV-best (strategy, layer).
            # v1 outputs hardcoded strategy="mean" and lack probe_strategy/probe_layer
            # fields — they will not match and must be re-run under v2 selection.
            with open(sweep_path) as f:
                sweep_for_check = json.load(f)
            current_best = find_cv_best_config(sweep_for_check, PROBE_SCOPE)
            with open(ood_path) as f:
                prior = json.load(f)
            prior_strategy = prior.get("probe_strategy")
            prior_layer = prior.get("probe_layer")
            prior_n_valid = prior.get("n_valid", 0)
            config_matches = (
                prior_strategy == current_best["strategy"] and prior_layer == current_best["layer"]
            )
            is_complete = prior_n_valid >= int(0.95 * len(cve_list))
            if config_matches and is_complete:
                print(
                    f"\nSKIP {slug}: PatchEval results already exist "
                    f"(strategy={prior_strategy} L{prior_layer} n_valid={prior_n_valid})"
                )
                continue
            print(
                f"\nRE-RUN {slug}: prior results stale "
                f"(prior strategy={prior_strategy} L{prior_layer} n_valid={prior_n_valid}; "
                f"current CV-best={current_best['strategy']} L{current_best['layer']})"
            )

        print(f"\n{'=' * 60}")
        print(f"PatchEval OOD: {slug}")
        print(f"{'=' * 60}")
        process_model(slug, cve_list, sweep_path, chunks_dir, out_dir)


if __name__ == "__main__":
    main()
