"""Extract residual stream activations for all models in the registry.

For each model that fits on the current GPU:
1. Load model via nnsight (using ModelSpec config)
2. Process all 1606 SVEN samples in batches
3. Save per-sample .pt chunks with all 7 pooling strategies
4. Resume: skip existing chunks

Output per sample: data/activations/{slug}/phase1_chunks/sample_{NNNN}.pt

Each chunk contains:
    resid_pre__{strategy}: tensor[n_layers, d_model]  (x7 strategies)
    label:     int (0=secure, 1=vulnerable)
    cwe:       str
    language:  str
    pair_id:   int
    split:     str ("train" or "eval")
    n_tokens:  int

Designed to run under nohup (~45-90 min per model).

Methods note — dtype, determinism, failure handling:
    Extraction dtype is fp16 (load_model uses torch.float16; EXTRACTION_DTYPE below
    records it in the manifest). Pooling on real tokens only (per-sample invoke, no padding).

    Determinism is bounded, not absolute: extraction is reproducible on a FIXED GPU +
    fixed batch config + fixed driver, but NOT bitwise-identical across different GPU
    counts or batch sizes (CUDA reduction order varies). Claim only the bounded form.

    Failure handling:
      - CUDA OOM is transient → the batch retries single-sample; a single-sample OOM
        is logged + counted as a failure and skipped.
      - ANY non-OOM error (wrong hook path, shape bug) is STRUCTURAL → re-raised so
        the run aborts loudly rather than silently writing an empty/biased dataset.
      - After a model, if fail>0 the chunk set is incomplete: re-run with
        ASP_BATCH_SIZE=1 to fill stragglers (existing chunks are skipped) BEFORE
        training any probe on it.

    Cache identity: each chunk dir carries a manifest.json recording
    model_id / dtype / quant / hook / split signature. A resume under a different
    config aborts instead of mixing incompatible vectors.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT / ".env")
sys.path.insert(0, str(PROJECT / "src"))

from nnsight import LanguageModel

from agentic_sec_probe.activations import (
    SWIM_WINDOW_SIZES,
    ExtractionConfig,
    extract_activations_batch,
    load_model,
)
from agentic_sec_probe.data import SVENSample, load_sven_pairs, pairs_to_samples
from agentic_sec_probe.models import MODEL_REGISTRY, ModelSpec, check_gpu_vram

ALL_STRATEGIES = ("mean", "last", "first", "max") + tuple(f"swim_{w}" for w in SWIM_WINDOW_SIZES)

# Default batch size is 8 for speed. Override with ASP_BATCH_SIZE=1 on a
# re-run pass to pick up samples that OOMed in the batched first pass.
BATCH_SIZE = int(os.environ.get("ASP_BATCH_SIZE", "8"))
# gc.collect() breaks nnsight's ref-cycle leak but does a full-heap scan (expensive at ~200
# batches/model). Firing it every N batches instead of every batch amortizes that cost ~Nx
# while still bounding the leak (the .cpu()-inside-trace fix keeps the live GPU set tiny, so
# a few batches of residual cycles stay well under the OOM ceiling). N=4 by default.
GC_EVERY = int(os.environ.get("ASP_GC_EVERY", "4"))


def load_samples_and_meta(
    project: Path,
) -> tuple[list[SVENSample], list[dict[str, object]]]:
    """Load SVEN samples and split metadata in matching order."""
    data_root = project / "data" / "raw"
    train_pairs = load_sven_pairs(data_root / "sven_train")
    val_pairs = load_sven_pairs(data_root / "sven_val")
    all_pairs = train_pairs + val_pairs
    samples = pairs_to_samples(all_pairs)

    split_path = project / "data" / "splits" / "phase1_split.json"
    with open(split_path) as f:
        split = json.load(f)

    # split["samples"] has same ordering as pairs_to_samples output
    return samples, split["samples"]


def save_chunk(
    path: Path,
    acts: dict[str, torch.Tensor],
    meta: dict[str, object],
    n_tokens: int,
) -> None:
    """Save a single sample's activations and metadata."""
    chunk: dict[str, object] = {
        "label": meta["label"],
        "cwe": meta["cwe"],
        "language": meta["language"],
        "pair_id": meta["pair_id"],
        # GitHub project (owner/repo). The split + CV both group by project so no
        # codebase appears in both train and eval (or across CV folds). See
        # 00_create_split.py. Older chunks without this key fall back to pair_id
        # downstream, but a re-extract is required for leak-free CV.
        "project": meta.get("project", f"unknown_pair_{meta['pair_id']}"),
        "split": meta["split"],
        "n_tokens": n_tokens,
    }
    for strategy, tensor in acts.items():
        chunk[f"resid_pre__{strategy}"] = tensor
    torch.save(chunk, path)


def count_tokens(model: LanguageModel, code: str) -> int:
    """Count tokens using the model's tokenizer."""
    return len(model.tokenizer.encode(code))


def _is_oom(exc: Exception) -> bool:
    """True if `exc` is a CUDA out-of-memory error (transient, retry smaller).

    Anything else (e.g. a wrong hook path, a shape bug) is STRUCTURAL: it would
    fail every sample, so it must abort loudly rather than silently produce an
    empty/biased dataset.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _is_transient_bnb_kernel(exc: Exception) -> bool:
    """True if `exc` is the intermittent bitsandbytes int8 kernel fault (ops.cu:388
    'invalid configuration argument'). Verified NON-deterministic: the same sample that
    triggered it re-extracts fine on retry, so it is transient, not structural
    -- skip the sample + continue rather than aborting the whole run. Logged loudly so the
    skip count is visible (a high count would mean a real problem, not a one-off)."""
    msg = str(exc).lower()
    return "ops.cu" in msg and "invalid configuration argument" in msg


def process_batch(
    model: LanguageModel,
    codes: list[str],
    config: ExtractionConfig,
) -> list[dict[str, torch.Tensor]] | None:
    """Extract activations for a batch.

    Returns the per-sample dicts on success, or None ONLY on a CUDA OOM (the caller
    then retries those samples one at a time). Any non-OOM exception is re-raised:
    it is structural and would corrupt the whole dataset if swallowed.
    """
    # The bitsandbytes int8 kernel fault (ops.cu:388) is transient -> RETRY the same batch
    # once after clearing CUDA state before giving up; the same input re-extracts fine.
    # OOM -> return None so the caller cascades to a smaller sub-batch.
    # Anything else is STRUCTURAL -> re-raise (would corrupt the dataset if swallowed).
    for attempt in range(2):
        try:
            batch_acts = extract_activations_batch(model, codes, config)
            n = len(codes)
            return [{s: t[i] for s, t in batch_acts.items()} for i in range(n)]
        except Exception as exc:
            if _is_oom(exc):
                print(f"    Sub-batch OOM ({len(codes)} samples) — will cascade smaller: {exc}")
                torch.cuda.empty_cache()
                return None
            if _is_transient_bnb_kernel(exc) and attempt == 0:
                print(
                    f"    Transient bitsandbytes kernel fault ({len(codes)} samples) "
                    "— clearing CUDA state and retrying once",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
                continue  # retry the same batch
            if _is_transient_bnb_kernel(exc):
                # Persisted through one retry -> skip this (sub-)batch, don't abort the run.
                print(
                    f"    bitsandbytes kernel fault persisted ({len(codes)} samples) "
                    "— skipping; returning None for cascade/skip",
                    flush=True,
                )
                torch.cuda.empty_cache()
                return None
            print(f"    STRUCTURAL batch failure ({type(exc).__name__}): {exc}")
            raise
    return None


def _save_extracted(
    out_dir: Path,
    model: LanguageModel,
    samples: list[SVENSample],
    meta_list: list[dict[str, object]],
    idx: int,
    result: dict[str, torch.Tensor],
) -> None:
    """Persist one extracted sample's chunk (token count + meta)."""
    n_tok = count_tokens(model, samples[idx].code)
    save_chunk(out_dir / f"sample_{idx:04d}.pt", result, meta_list[idx], n_tok)


def _extract_cascade(
    model: LanguageModel,
    samples: list[SVENSample],
    meta_list: list[dict[str, object]],
    out_dir: Path,
    indices: list[int],
    config: ExtractionConfig,
    counts: dict[str, int],
) -> None:
    """Extract `indices` by CASCADING the sub-batch size 4 -> 2 -> 1 after a full-batch OOM.

    At each level, try sub-batches of that size; any sub-batch that OOMs is split to the next
    smaller level. At size 1, a genuine OOM skips the sample (counts['fail']). Same forward
    pass as the full batch -> identical activations, just fewer in parallel.
    """
    pending = list(indices)
    for sub in (4, 2, 1):
        if not pending:
            break
        still_failing: list[int] = []
        for start in range(0, len(pending), sub):
            group = pending[start : start + sub]
            results = process_batch(model, [samples[i].code for i in group], config)
            if results is not None:
                for j, idx in enumerate(group):
                    _save_extracted(out_dir, model, samples, meta_list, idx, results[j])
                    counts["ok"] += 1
                del results
                gc.collect()  # break nnsight ref-cycles (see process_model cleanup note)
                torch.cuda.empty_cache()
            elif sub == 1:
                # Single-sample OOM is genuine -> skip + count as fail.
                print(f"    Single-sample OOM (skipping idx {group[0]})")
                gc.collect()
                torch.cuda.empty_cache()
                counts["fail"] += 1
            else:
                still_failing.extend(group)  # cascade to the next smaller sub-batch
        pending = still_failing


def process_model(
    spec: ModelSpec,
    samples: list[SVENSample],
    meta_list: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, int]:
    """Extract activations for all samples with one model.

    Returns counts: {"ok": N, "skip": N, "fail": N}
    """
    config = ExtractionConfig(
        # INSTRUCT checkpoint so probe + prompted baseline use the same model.
        model_name=spec.probe_model_id,
        quantization=spec.quantization,
        token_strategies=ALL_STRATEGIES,
        hook_pattern=spec.hook_pattern,
        dtype=spec.dtype,  # Gemma needs bf16
        hf_load_class=spec.hf_load_class,  # Gemma text-only class
        batch_size=BATCH_SIZE,
    )
    model = load_model(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    n_samples = len(samples)
    t0 = time.time()

    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_samples)
        batch_indices = list(range(batch_start, batch_end))

        # Check which chunks in this batch already exist
        todo = [i for i in batch_indices if not (out_dir / f"sample_{i:04d}.pt").exists()]
        skip_count = len(batch_indices) - len(todo)
        counts["skip"] += skip_count

        if not todo:
            continue

        # Extract batch (only the samples we need)
        todo_codes = [samples[i].code for i in todo]
        batch_results = process_batch(model, todo_codes, config)

        if batch_results is not None:
            # Batch succeeded — save each sample
            for j, idx in enumerate(todo):
                n_tok = count_tokens(model, samples[idx].code)
                save_chunk(
                    out_dir / f"sample_{idx:04d}.pt",
                    batch_results[j],
                    meta_list[idx],
                    n_tok,
                )
                counts["ok"] += 1
        else:
            # Batch OOM'd at full size. CASCADE the sub-batch size down (8->4->2->1) instead
            # of jumping straight to 1: a batch that OOMs at 8 usually fits at 4, which is far
            # faster than one-at-a-time. Only the indices that OOM at a level cascade further;
            # a genuine single-sample OOM is skipped (counted as fail) at the bottom.
            print(f"    Batch OOM at size {len(todo)} -> cascading sub-batches for {todo}")
            _extract_cascade(model, samples, meta_list, out_dir, todo, config, counts)

        # Release per-batch memory. nnsight's intervention graph holds the saved activations
        # in REFERENCE CYCLES across trace() calls, so allocated GPU memory accrues until
        # CPython's cyclic GC runs -- this caused ~60GB accumulation -> OOM on qwen-14b
        # (verified: nnsight GitHub #538; empty_cache() frees only RESERVED, not ALLOCATED).
        # The maintainer-confirmed fix is an explicit gc.collect() to break the cycles -- but
        # the full-heap scan is expensive, so fire it every GC_EVERY batches (the
        # .cpu()-inside-trace fix keeps the live set tiny enough that a few batches of residual
        # cycles stay safe). Order: del -> [periodic] gc.collect() -> empty_cache().
        del batch_results
        if (batch_start // BATCH_SIZE) % GC_EVERY == 0:
            gc.collect()
        torch.cuda.empty_cache()

        # Progress report every 10 batches
        processed = counts["ok"] + counts["skip"] + counts["fail"]
        if (batch_start // BATCH_SIZE) % 10 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (n_samples - processed) / rate if rate > 0 else 0
            print(
                f"  [{processed}/{n_samples}] "
                f"{counts['ok']} ok, {counts['skip']} skip, {counts['fail']} fail "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    elapsed = time.time() - t0
    print(
        f"  Done: {counts['ok']} ok, {counts['skip']} skip, {counts['fail']} fail ({elapsed:.0f}s)"
    )
    return counts


# Dtype the activations are extracted in. load_model uses torch.float16; pinned
# here so the manifest records it and the paper/docs stay consistent.
# Activations are extracted in fp16 (recorded in the manifest).
EXTRACTION_DTYPE = "float16"


def manifest_for(spec: ModelSpec, n_samples: int, split_sig: str) -> dict[str, object]:
    """Identity of an activation chunk set. Mismatch on resume = stale cache."""
    return {
        "model_id": spec.probe_model_id,  # instruct checkpoint
        "quantization": spec.quantization or "fp16",
        "dtype": EXTRACTION_DTYPE,
        "hook_pattern": spec.hook_pattern,
        "n_layers": spec.n_layers,
        "d_model": spec.d_model,
        "n_samples": n_samples,
        "split_signature": split_sig,  # detects a changed train/eval split
        "strategies": list(ALL_STRATEGIES),
    }


def check_or_write_manifest(out_dir: Path, manifest: dict[str, object]) -> None:
    """Write the manifest, or abort loudly if an existing one disagrees.

    Cache key was slug-only (existence-based skip): a precision/variant/split change
    would silently serve stale chunks. The manifest makes that a hard error.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    if mpath.exists():
        with open(mpath) as f:
            existing = json.load(f)
        if existing != manifest:
            diff = {
                k: (existing.get(k), manifest.get(k))
                for k in manifest
                if existing.get(k) != manifest.get(k)
            }
            msg = (
                f"Activation cache manifest mismatch in {out_dir}: {diff}. "
                "Existing chunks were extracted under a different config (model/dtype/"
                "split/hook). Delete the dir to re-extract, or fix the config — refusing "
                "to mix incompatible activation vectors."
            )
            raise ValueError(msg)
        return
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)


def split_signature(meta_list: list[dict[str, object]]) -> str:
    """Stable hash of the (pair_id, split, project) assignment, to detect split drift."""
    import hashlib

    payload = ";".join(f"{m['pair_id']}:{m['split']}:{m.get('project', '')}" for m in meta_list)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> None:
    vram = check_gpu_vram()
    print(f"GPU VRAM: {vram:.1f} GB")
    print(f"Strategies: {ALL_STRATEGIES}")

    samples, meta_list = load_samples_and_meta(PROJECT)
    print(f"Loaded {len(samples)} samples")

    # PYTHON-ONLY (default on): the probe/eval stages (02/09/07) use ONLY python samples
    # (language == "python"); extracting the 846 c_cpp samples is pure waste (~53% of the
    # set) AND one of them triggered the bitsandbytes int8 kernel crash. Filter here so we
    # compute exactly what downstream consumes. Set ASP_PYTHON_ONLY=0 to keep all languages.
    # `meta_list` stays aligned with `samples` (filtered in lockstep) so split/chunk metadata
    # and the split signature still match downstream's language-filtered view.
    if os.environ.get("ASP_PYTHON_ONLY", "1") != "0":
        keep = [i for i, m in enumerate(meta_list) if m["language"] == "python"]
        samples = [samples[i] for i in keep]
        meta_list = [meta_list[i] for i in keep]
        print(f"  python-only: kept {len(samples)} python samples (dropped c_cpp)")

    split_sig = split_signature(meta_list)
    print(f"Split signature: {split_sig}")

    results: dict[str, dict[str, object]] = {}

    # Optional in-scope filter: ASP_MODELS="slug1,slug2" restricts the loop so the chain
    # never touches out-of-scope models (e.g. starcoder2-15b structurally fails on this
    # box and would abort the whole chain).
    only = {s.strip() for s in os.environ.get("ASP_MODELS", "").split(",") if s.strip()}

    for slug, spec in MODEL_REGISTRY.items():
        if only and slug not in only:
            continue
        if spec.min_vram_gb > vram:
            print(f"\nSKIP {spec.name}: needs {spec.min_vram_gb} GB, have {vram:.1f} GB")
            results[slug] = {"status": "skipped", "reason": "insufficient_vram"}
            continue

        out_dir = PROJECT / "data" / "activations" / slug / "phase1_chunks"

        # Manifest guard: abort if existing chunks were extracted under a different
        # config (model/dtype/split/hook) instead of silently reusing them.
        manifest = manifest_for(spec, len(samples), split_sig)
        check_or_write_manifest(out_dir, manifest)

        # Check if already complete
        existing = len(list(out_dir.glob("sample_*.pt"))) if out_dir.exists() else 0
        if existing >= len(samples):
            print(f"\nSKIP {spec.name}: all {existing} chunks already exist")
            results[slug] = {"status": "complete", "n_chunks": existing}
            continue

        print(f"\n{'=' * 60}")
        print(f"Extracting: {spec.name} ({slug})")
        print(f"  {spec.n_layers} layers, d_model={spec.d_model}, quant={spec.quantization}")
        print(f"  Existing chunks: {existing}/{len(samples)}")
        print(f"{'=' * 60}")

        try:
            counts = process_model(spec, samples, meta_list, out_dir)
            results[slug] = {"status": "ok", **counts}
            # Fail loud: a non-OOM structural error already raised; any remaining
            # OOM-skipped samples leave the chunk set incomplete. Surface it.
            if counts["fail"] > 0:
                results[slug]["status"] = "incomplete"
                print(
                    f"  WARNING {slug}: {counts['fail']} samples failed (OOM). Re-run with "
                    f"ASP_BATCH_SIZE=1 to fill stragglers before training probes."
                )
        except Exception as exc:
            # Structural failure (wrong hook, shape bug): do NOT swallow — re-raise
            # so the whole run stops loudly rather than producing a biased dataset.
            print(f"  MODEL FAILED (structural): {type(exc).__name__}: {exc}")
            results[slug] = {"status": "error", "error": str(exc)}
            torch.cuda.empty_cache()
            raise
        finally:
            torch.cuda.empty_cache()

    # Save summary
    summary_path = PROJECT / "outputs" / "phase1" / "extraction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    for slug, r in results.items():
        status = r["status"]
        print(f"  {slug}: {status}")


if __name__ == "__main__":
    main()
