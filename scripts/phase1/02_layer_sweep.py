"""Layer sweep: find the best probing layer per model × scope × strategy.

For each model:
  For each pooling strategy (mean, last, first, max, swim_16/32/64):
    For each scope (universal, python, c_cpp):
      Train linear probes at ALL layers simultaneously (batched bmm on GPU).
      Sweep C in {0.01, 0.1, 1.0} via 5-fold GroupKFold.
      Evaluate best-C probe on held-out eval split.

Linear probe = batched nn.Linear(d, 1) + BCEWithLogitsLoss + L2 weight decay.
All layers trained in parallel via torch.bmm — one GPU kernel per optimization step.

Output: outputs/phase1/layer_sweep_{slug}.json

Runs on GPU.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.activations import SWIM_WINDOW_SIZES
from agentic_sec_probe.models import MODEL_REGISTRY

ALL_STRATEGIES = ("mean", "last", "first", "max") + tuple(f"swim_{w}" for w in SWIM_WINDOW_SIZES)
N_FOLDS = 5
N_STEPS = 200
LR = 0.01
# F1 fix: scope by per-sample LANGUAGE, not CWE membership (a path-traversal cwe-022
# can be C/C++). `None` = all samples (universal diagnostic); a language string filters
# on the chunk's `language` field. The c_cpp scope is dropped (out of reporting scope).
# Regularization is a single round weight_decay (no C-sweep — with a constant
# wd, C no longer affects the torch probe, so sweeping it is meaningless).
WEIGHT_DECAY = 0.01
SCOPES: dict[str, str | None] = {
    "universal": None,
    "python": "python",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_chunks_vectorized(
    chunks_dir: Path,
) -> tuple[
    dict[str, torch.Tensor],
    npt.NDArray[np.int64],
    npt.NDArray[np.str_],
    npt.NDArray[np.str_],
    npt.NDArray[np.str_],
    npt.NDArray[np.int64],
    npt.NDArray[np.object_],
    int,
    int,
]:
    """Load all chunks and build per-strategy tensors.

    Returns:
        strategy_tensors: {strategy: [n_samples, n_layers, d_model]}
        labels, cwes, languages, splits, pair_ids, projects, n_layers, d_model

    `languages` is the per-sample `language` field ("python"/"c_cpp"), the correct
    scope key (F1 fix). CWE membership is NOT a language proxy.
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
    # Project (owner/repo) is the CV grouping unit so same-repo pairs never split
    # across folds. Fall back to a per-pair sentinel for pre-project chunks.
    projects = np.array(
        [c.get("project", f"unknown_pair_{c['pair_id']}") for c in chunks], dtype=object
    )

    strategy_tensors: dict[str, torch.Tensor] = {}
    for strategy in ALL_STRATEGIES:
        key = f"resid_pre__{strategy}"
        stacked = torch.stack([c[key].float() for c in chunks])
        strategy_tensors[strategy] = stacked

    del chunks
    print("  All strategies loaded", flush=True)
    return strategy_tensors, labels, cwes, languages, splits, pair_ids, projects, n_layers, d_model


def train_batched_probes(
    X: torch.Tensor,
    y: torch.Tensor,
    weight_decay: float,
    n_steps: int = N_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train linear probes for ALL layers simultaneously.

    Args:
        X: [n_samples, n_layers, d_model] on GPU
        y: [n_samples] on GPU
    Returns:
        W: [n_layers, d_model, 1]
        b: [n_layers, 1]
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    n_layers, d = X.shape[1], X.shape[2]
    W = torch.zeros(n_layers, d, 1, device=DEVICE, requires_grad=True)
    b = torch.zeros(n_layers, 1, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([W, b], lr=LR, weight_decay=weight_decay)

    # X transposed for bmm: [n_layers, n_samples, d_model]
    Xt = X.permute(1, 0, 2)
    # y broadcast: [n_layers, n_samples]
    yt = y.unsqueeze(0).expand(n_layers, -1)

    for _ in range(n_steps):
        # [n_layers, n_samples, d] @ [n_layers, d, 1] -> [n_layers, n_samples, 1]
        logits = torch.bmm(Xt, W).squeeze(-1) + b  # [n_layers, n_samples]
        loss = F.binary_cross_entropy_with_logits(logits, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return W.detach(), b.detach()


def predict_batched(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> npt.NDArray[np.float64]:
    """Predict P(y=1) for all layers at once.

    Args:
        X: [n_samples, n_layers, d_model]
        W: [n_layers, d_model, 1]
        b: [n_layers, 1]
    Returns:
        probs: [n_layers, n_samples] numpy array
    """
    Xt = X.permute(1, 0, 2)
    with torch.no_grad():
        logits = torch.bmm(Xt, W).squeeze(-1) + b
        probs: npt.NDArray[np.float64] = torch.sigmoid(logits).cpu().numpy()
    return probs


def scale_all_layers(
    X_np: npt.NDArray[np.float64],
    n_layers: int,
) -> tuple[npt.NDArray[np.float64], list[StandardScaler]]:
    """StandardScale each layer independently. Returns scaled array and scalers."""
    n_samples, _, d = X_np.shape
    scaled = np.empty_like(X_np)
    scalers = []
    for layer in range(n_layers):
        scaler = StandardScaler()
        scaled[:, layer, :] = scaler.fit_transform(X_np[:, layer, :])
        scalers.append(scaler)
    return scaled, scalers


def transform_all_layers(
    X_np: npt.NDArray[np.float64],
    scalers: list[StandardScaler],
    n_layers: int,
) -> npt.NDArray[np.float64]:
    """Apply pre-fit scalers to each layer."""
    scaled = np.empty_like(X_np)
    for layer in range(n_layers):
        scaled[:, layer, :] = scalers[layer].transform(X_np[:, layer, :])
    return scaled


def sweep_cv_batched(
    X_np: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    groups: npt.NDArray[np.object_],
    n_layers: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """GroupKFold CV at a fixed weight_decay, training all layers in parallel.

    No C-sweep. Regularization is the constant `WEIGHT_DECAY`, so there is one
    probe family; selection is over (layer) by CV-AUC only.

    Returns per-layer arrays: (cv_auc, cv_std) each [n_layers].
    """
    n_folds = min(N_FOLDS, len(np.unique(groups)))

    cv_auc = np.zeros(n_layers)
    cv_std = np.zeros(n_layers)

    if n_folds < 2:
        return cv_auc + 0.5, cv_std

    gkf = GroupKFold(n_splits=n_folds)
    fold_aucs = np.zeros((n_folds, n_layers))  # [n_folds, n_layers]
    valid_folds = np.zeros(n_folds, dtype=bool)

    for fi, (train_idx, val_idx) in enumerate(gkf.split(X_np, y, groups)):
        X_tr_np = X_np[train_idx]
        X_val_np = X_np[val_idx]
        y_val = y[val_idx]

        if len(np.unique(y_val)) < 2:
            continue
        valid_folds[fi] = True

        # Scale per layer (scaler fit on fold-train only — no leakage).
        X_tr_scaled, scalers = scale_all_layers(X_tr_np, n_layers)
        X_val_scaled = transform_all_layers(X_val_np, scalers, n_layers)

        X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32, device=DEVICE)
        y_tr_t = torch.tensor(y[train_idx], dtype=torch.float32, device=DEVICE)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=DEVICE)

        W, b = train_batched_probes(X_tr_t, y_tr_t, WEIGHT_DECAY)
        probs = predict_batched(X_val_t, W, b)  # [n_layers, n_val]

        for layer in range(n_layers):
            fold_aucs[fi, layer] = roc_auc_score(y_val, probs[layer])

    if valid_folds.sum() > 0:
        cv_auc = fold_aucs[valid_folds].mean(axis=0)  # [n_layers]
        cv_std = fold_aucs[valid_folds].std(axis=0)

    return cv_auc, cv_std


def eval_batched(
    X_train_np: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    X_eval_np: npt.NDArray[np.float64],
    y_eval: npt.NDArray[np.int64],
    n_layers: int,
) -> npt.NDArray[np.float64]:
    """Eval on held-out at the fixed weight_decay. Returns eval AUC per layer.

    Transparency only — never used for layer selection (that is CV-AUC).
    """
    eval_aucs = np.full(n_layers, 0.5)
    if len(np.unique(y_eval)) < 2:
        return eval_aucs

    X_tr_scaled, scalers = scale_all_layers(X_train_np, n_layers)
    X_ev_scaled = transform_all_layers(X_eval_np, scalers, n_layers)

    X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_ev_t = torch.tensor(X_ev_scaled, dtype=torch.float32, device=DEVICE)

    W, b = train_batched_probes(X_tr_t, y_tr_t, WEIGHT_DECAY)
    probs = predict_batched(X_ev_t, W, b)  # [n_layers, n_eval]

    for layer in range(n_layers):
        eval_aucs[layer] = roc_auc_score(y_eval, probs[layer])

    return eval_aucs


def process_model(slug: str, chunks_dir: Path, out_dir: Path) -> dict[str, object]:
    """Run full layer sweep for one model."""
    strategy_tensors, labels, cwes, languages, splits, pair_ids, projects, n_layers, d_model = (
        load_chunks_vectorized(chunks_dir)
    )

    results: dict[str, object] = {
        "slug": slug,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_chunks": len(labels),
        "strategies": {},
    }

    strategy_results: dict[str, dict[str, object]] = {}

    for strategy in ALL_STRATEGIES:
        print(f"\n  Strategy: {strategy}", flush=True)
        all_acts = strategy_tensors[strategy].numpy()
        scope_results: dict[str, object] = {}

        for scope_name, lang_filter in SCOPES.items():
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
                scope_results[scope_name] = {"error": "insufficient_samples"}
                continue

            y_train = labels[train_mask]
            y_eval = labels[eval_mask]
            # CV group = project (was pair_id). Same-repo pairs stay in one fold so
            # layer/pooling/C selection isn't tuned on leaked near-duplicates.
            g_train = projects[train_mask]

            # [n_samples, n_layers, d_model]
            acts_train = all_acts[train_mask]
            acts_eval = all_acts[eval_mask]

            # CV sweep — all layers trained simultaneously at the fixed weight_decay.
            cv_auc, cv_std = sweep_cv_batched(
                acts_train,
                y_train,
                g_train,
                n_layers,
            )

            # Eval — all layers at once (transparency only).
            eval_aucs = eval_batched(
                acts_train,
                y_train,
                acts_eval,
                y_eval,
                n_layers,
            )

            layer_results = []
            for layer in range(n_layers):
                layer_results.append(
                    {
                        "layer": layer,
                        "weight_decay": WEIGHT_DECAY,
                        "cv_auc": round(float(cv_auc[layer]), 4),
                        "cv_std": round(float(cv_std[layer]), 4),
                        "eval_auc": round(float(eval_aucs[layer]), 4),
                    }
                )

            # Layer selection uses CV AUC on training split only.
            # eval_auc is stored per-layer for transparency but never used for selection.
            best = max(layer_results, key=lambda r: r["cv_auc"])
            print(
                f"    {scope_name} ({n_train}tr/{n_eval}ev): "
                f"best L{best['layer']}, "
                f"cv={best['cv_auc']:.3f}, "
                f"eval={best['eval_auc']:.3f}, wd={WEIGHT_DECAY}",
                flush=True,
            )

            scope_results[scope_name] = {
                "n_train": n_train,
                "n_eval": n_eval,
                "best_layer": best["layer"],
                "best_cv_auc": best["cv_auc"],
                "best_eval_auc": best["eval_auc"],
                "weight_decay": WEIGHT_DECAY,
                "layers": layer_results,
            }

        strategy_results[strategy] = scope_results

    results["strategies"] = strategy_results

    out_path = out_dir / f"layer_sweep_{slug}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)

    return results


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print(f"Device: {DEVICE}", flush=True)
    out_dir = PROJECT / "outputs" / "phase1"

    for slug in MODEL_REGISTRY:
        chunks_dir = PROJECT / "data" / "activations" / slug / "phase1_chunks"
        if not chunks_dir.exists():
            print(f"\nSKIP {slug}: no activation chunks", flush=True)
            continue

        # Coverage gate: ensure extraction actually completed before sweeping. The expected
        # count depends on the extraction scope -- python-only (default, 760 samples) vs all
        # languages (1606). Use a floor at ~90% of the expected python count so a few dropped
        # outliers don't block the sweep, while genuinely-incomplete extraction is caught.
        expected = 1606 if os.environ.get("ASP_PYTHON_ONLY", "1") == "0" else 760
        floor = int(expected * 0.9)
        n_chunks = len(list(chunks_dir.glob("sample_*.pt")))
        if n_chunks < floor:
            print(f"\nSKIP {slug}: only {n_chunks}/{expected} chunks (need >={floor})", flush=True)
            continue

        out_path = out_dir / f"layer_sweep_{slug}.json"
        if out_path.exists():
            print(f"\nSKIP {slug}: results already exist", flush=True)
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"Layer sweep: {slug}", flush=True)
        print(f"{'=' * 60}", flush=True)
        process_model(slug, chunks_dir, out_dir)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
