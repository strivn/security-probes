"""Per-CWE probe evaluation with full classification metrics and checkpointing.

For each model x scope:
  1. Read best strategy/layer/C from layer_sweep JSON
  2. Extract features at that single layer
  3. Train probe (sklearn LogisticRegression lbfgs — same optimizer as 07/probe.py)
  4. SAVE probe weights (W, b), scaler params, and config
  5. Predict on eval set, SAVE per-sample predictions
  6. Compute full metrics per-scope AND per-CWE:
     AUC, accuracy, precision, recall, F1, confusion matrix

Input:  outputs/phase1/layer_sweep_{slug}.json
        data/activations/{slug}/phase1_chunks/
Output: outputs/phase1/probe_eval_{slug}.json
        models/probes/phase1/{slug}/{scope}_{strategy}_L{layer}.pt

Runs on GPU.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.activations import SWIM_WINDOW_SIZES
from agentic_sec_probe.models import MODEL_REGISTRY

ALL_STRATEGIES = ("mean", "last", "first", "max") + tuple(f"swim_{w}" for w in SWIM_WINDOW_SIZES)
# Scope by per-sample language, not CWE membership. `None` = universal diagnostic.
SCOPES: dict[str, str | None] = {
    "universal": None,
    "python": "python",
}

# The DEPLOYED probe is the torch probe (same method as the 02 sweep) at a constant
# weight_decay. sklearn-lbfgs is also fit at deploy as an independent "other test", not
# tuned to match. These mirror 02's torch config.
WEIGHT_DECAY = 0.01
LR = 0.01
N_STEPS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_chunks_vectorized(
    chunks_dir: Path,
) -> tuple[
    dict[str, npt.NDArray[np.float32]],
    npt.NDArray[np.int64],
    npt.NDArray[np.str_],
    npt.NDArray[np.str_],
    npt.NDArray[np.str_],
    npt.NDArray[np.int64],
    int,
    int,
]:
    """Load all chunks into per-strategy numpy arrays.

    Returns:
        strategy_arrays: {strategy: [n_samples, n_layers, d_model]}
        labels, cwes, languages, splits, pair_ids, n_layers, d_model

    `languages` is the per-sample `language` field (F1: the scope key, not CWE).
    """
    chunk_files = sorted(chunks_dir.glob("sample_*.pt"))
    if not chunk_files:
        msg = f"No chunks found in {chunks_dir}"
        raise FileNotFoundError(msg)

    chunks = [torch.load(cf, weights_only=True) for cf in chunk_files]  # tensors + plain types
    n = len(chunks)

    first_act = chunks[0]["resid_pre__mean"]
    n_layers, d_model = first_act.shape
    print(f"  Loaded {n} chunks, {n_layers} layers, d_model={d_model}", flush=True)

    labels = np.array([c["label"] for c in chunks], dtype=np.int64)
    cwes = np.array([c["cwe"] for c in chunks])
    languages = np.array([c["language"] for c in chunks])
    splits = np.array([c["split"] for c in chunks])
    pair_ids = np.array([c["pair_id"] for c in chunks], dtype=np.int64)

    strategy_arrays: dict[str, npt.NDArray[np.float32]] = {}
    for strategy in ALL_STRATEGIES:
        key = f"resid_pre__{strategy}"
        stacked = torch.stack([c[key].float() for c in chunks]).numpy()
        strategy_arrays[strategy] = stacked

    del chunks
    print("  All strategies vectorized", flush=True)
    return strategy_arrays, labels, cwes, languages, splits, pair_ids, n_layers, d_model


def train_single_layer_probe(
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int64],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train the DEPLOYED linear probe (torch, plain Adam, constant weight_decay).

    This is THE method end-to-end — the 02 sweep selects with the same torch probe
    and 09 deploys it, so there is no search/deploy optimizer mismatch. Returns (W, b) as
    CPU tensors in [d,1]/[1] shape for the checkpoint + predict_probs.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    d = X_train.shape[1]
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    W = torch.zeros(d, 1, device=DEVICE, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([W, b], lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(N_STEPS):
        logits = (X_t @ W).squeeze(-1) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return W.detach().cpu(), b.detach().cpu()


def train_single_layer_sklearn(
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int64],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit sklearn LogisticRegression (defaults) at the same layer as an OTHER TEST.

    Independent second opinion on the deployed torch probe — reported alongside, NOT
    selected between and NOT tuned to agree. sklearn default C=1.0, lbfgs to convergence.
    """
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)
    W = torch.tensor(clf.coef_.reshape(-1, 1), dtype=torch.float32)  # [d, 1]
    b = torch.tensor(clf.intercept_, dtype=torch.float32)  # [1]
    return W, b


def predict_probs(
    X: npt.NDArray[np.float32],
    W: torch.Tensor,
    b: torch.Tensor,
) -> npt.NDArray[np.float64]:
    """Predict P(y=1) for samples. W, b on CPU."""
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = (X_t @ W).squeeze(-1) + b
        probs = torch.sigmoid(logits).numpy()
    return probs


def classification_metrics(
    y_true: npt.NDArray[np.int64],
    y_prob: npt.NDArray[np.float64],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Full classification metrics from ground truth and predicted probabilities."""
    n = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    result: dict[str, Any] = {"n": n}

    # AUC needs both classes
    if len(np.unique(y_true)) > 1:
        result["auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    else:
        result["auc"] = None

    result["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    result["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    result["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    result["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result["tp"] = int(tp)
    result["fp"] = int(fp)
    result["fn"] = int(fn)
    result["tn"] = int(tn)
    result["tpr"] = round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0, 4)
    result["fpr"] = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 4)

    return result


def find_cv_best_config(sweep: dict[str, Any], scope_name: str) -> dict[str, Any] | None:
    """Find best (strategy, layer) for a scope using CV AUC from the layer sweep.

    No C in selection (regularization is the constant weight_decay). Selection MUST
    use best_cv_auc (computed on the training split via GroupKFold) and never
    best_eval_auc — the latter would leak held-out eval information into model selection.
    """
    best_cv_auc = 0.0
    best_strat = ""
    best_layer = 0

    for strat, scopes in sweep["strategies"].items():
        sc = scopes.get(scope_name, {})
        if "best_cv_auc" in sc and sc["best_cv_auc"] > best_cv_auc:
            best_cv_auc = sc["best_cv_auc"]
            best_strat = strat
            best_layer = sc["best_layer"]

    if best_strat == "":
        return None

    return {
        "strategy": best_strat,
        "layer": best_layer,
        "weight_decay": WEIGHT_DECAY,
        "sweep_cv_auc": best_cv_auc,
    }


def process_model(
    slug: str,
    strategy_arrays: dict[str, npt.NDArray[np.float32]],
    labels: npt.NDArray[np.int64],
    cwes: npt.NDArray[np.str_],
    languages: npt.NDArray[np.str_],
    splits: npt.NDArray[np.str_],
    pair_ids: npt.NDArray[np.int64],
    sweep: dict[str, Any],
    probe_dir: Path,
) -> dict[str, Any]:
    """Train, checkpoint, and evaluate probes for one model."""
    results: dict[str, Any] = {
        "slug": slug,
        "n_layers": sweep["n_layers"],
        "d_model": sweep["d_model"],
        "scopes": {},
    }

    for scope_name, lang_filter in SCOPES.items():
        config = find_cv_best_config(sweep, scope_name)
        if config is None:
            print(f"  {scope_name}: no valid config found, skipping", flush=True)
            continue

        strategy = config["strategy"]
        layer = config["layer"]

        print(
            f"\n  {scope_name}: {strategy} L{layer} wd={WEIGHT_DECAY} "
            f"(sweep CV AUC={config['sweep_cv_auc']:.3f})",
            flush=True,
        )

        acts = strategy_arrays[strategy]  # [n_samples, n_layers, d_model]

        # Scope by per-sample language, not CWE membership.
        if lang_filter is not None:
            scope_mask = languages == lang_filter
        else:
            scope_mask = np.ones(len(labels), dtype=bool)

        train_mask = (splits == "train") & scope_mask
        eval_mask = (splits == "eval") & scope_mask

        n_train = int(train_mask.sum())
        n_eval = int(eval_mask.sum())

        if n_train < 10 or n_eval < 4:
            results["scopes"][scope_name] = {"error": "insufficient_samples"}
            continue

        # Extract single layer
        X_train = acts[train_mask, layer, :]
        X_eval = acts[eval_mask, layer, :]
        y_train = labels[train_mask]
        y_eval = labels[eval_mask]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_eval_s = scaler.transform(X_eval)

        # Train the DEPLOYED probe (torch, plain Adam, wd=0.01 — same method as sweep).
        W, b = train_single_layer_probe(X_train_s, y_train)
        # Independent second opinion (sklearn defaults) — NOT used for selection.
        W_sk, b_sk = train_single_layer_sklearn(X_train_s, y_train)

        # ── Save probe checkpoint (weights + scaler params + config) ──
        probe_path = probe_dir / f"{scope_name}_{strategy}_L{layer}.pt"
        torch.save(
            {
                "W": W,
                "b": b,
                "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
                "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
                "scaler_var": torch.tensor(scaler.var_, dtype=torch.float64),
                "strategy": strategy,
                "layer": layer,
                "weight_decay": WEIGHT_DECAY,
                "solver": "torch_adam",  # deployed probe (torch end-to-end)
                "scope": scope_name,
                "slug": slug,
                "n_train": n_train,
                "d_model": sweep["d_model"],
            },
            probe_path,
        )
        print(f"    Saved probe: {probe_path.name}", flush=True)

        # ── Predict on eval set (deployed torch probe) ──
        y_prob = predict_probs(X_eval_s, W, b)
        # Second-opinion eval (sklearn) for the side-by-side AUC.
        y_prob_sk = predict_probs(X_eval_s, W_sk, b_sk)

        # ── Per-sample predictions ──
        eval_cwes = cwes[eval_mask]
        eval_pair_ids = pair_ids[eval_mask]
        per_sample = []
        for i in range(n_eval):
            per_sample.append(
                {
                    "pair_id": int(eval_pair_ids[i]),
                    "cwe": str(eval_cwes[i]),
                    "label": int(y_eval[i]),
                    "prob": round(float(y_prob[i]), 6),
                    "predicted": int(y_prob[i] >= 0.5),
                }
            )

        # ── Scope-level metrics (deployed torch probe) ──
        scope_metrics = classification_metrics(y_eval, y_prob)
        # Second-opinion AUC from the sklearn-default fit (the "other test").
        sklearn_auc = (
            round(float(roc_auc_score(y_eval, y_prob_sk)), 4)
            if len(np.unique(y_eval)) > 1
            else None
        )
        print(
            f"    Scope: torch AUC={scope_metrics['auc']} | sklearn AUC={sklearn_auc} "
            f"acc={scope_metrics['accuracy']:.3f}, "
            f"P={scope_metrics['precision']:.3f}, "
            f"R={scope_metrics['recall']:.3f}, "
            f"F1={scope_metrics['f1']:.3f}",
            flush=True,
        )

        # ── Per-CWE metrics ──
        cwe_groups: dict[str, list[int]] = defaultdict(list)
        for i in range(n_eval):
            cwe_groups[str(eval_cwes[i])].append(i)

        per_cwe: dict[str, dict[str, Any]] = {}
        for cwe, indices in sorted(cwe_groups.items()):
            idx = np.array(indices)
            cwe_metrics = classification_metrics(y_eval[idx], y_prob[idx])
            per_cwe[cwe] = cwe_metrics
            print(
                f"    {cwe} (n={cwe_metrics['n']}): "
                f"AUC={cwe_metrics['auc']}, "
                f"acc={cwe_metrics['accuracy']:.3f}, "
                f"P={cwe_metrics['precision']:.3f}, "
                f"R={cwe_metrics['recall']:.3f}, "
                f"F1={cwe_metrics['f1']:.3f}",
                flush=True,
            )

        results["scopes"][scope_name] = {
            "config": config,
            "n_train": n_train,
            "n_eval": n_eval,
            "metrics": scope_metrics,  # deployed torch probe
            "sklearn_auc": sklearn_auc,  # independent second opinion (not selected on)
            "per_cwe": per_cwe,
            "per_sample": per_sample,
            "probe_file": probe_path.name,
        }

    return results


def main() -> None:
    print(f"Device: {DEVICE}", flush=True)
    out_dir = PROJECT / "outputs" / "phase1"
    probe_dir = PROJECT / "models" / "probes" / "phase1"

    for slug in MODEL_REGISTRY:
        sweep_path = out_dir / f"layer_sweep_{slug}.json"
        if not sweep_path.exists():
            print(f"\nSKIP {slug}: no layer sweep results", flush=True)
            continue

        chunks_dir = PROJECT / "data" / "activations" / slug / "phase1_chunks"
        if not chunks_dir.exists():
            print(f"\nSKIP {slug}: no activation chunks", flush=True)
            continue

        eval_path = out_dir / f"probe_eval_{slug}.json"
        if eval_path.exists():
            print(f"\nSKIP {slug}: probe eval already exists", flush=True)
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"Probe eval: {slug}", flush=True)
        print(f"{'=' * 60}", flush=True)

        with open(sweep_path) as f:
            sweep = json.load(f)

        strategy_arrays, labels, cwes, languages, splits, pair_ids, n_layers, d_model = (
            load_chunks_vectorized(chunks_dir)
        )

        model_probe_dir = probe_dir / slug
        model_probe_dir.mkdir(parents=True, exist_ok=True)

        results = process_model(
            slug,
            strategy_arrays,
            labels,
            cwes,
            languages,
            splits,
            pair_ids,
            sweep,
            model_probe_dir,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {eval_path}", flush=True)

        del strategy_arrays
        torch.cuda.empty_cache()
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
