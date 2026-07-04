"""MLP probes: train non-linear probes at best layer per model x scope.

For each model x scope x strategy:
  1. Read best layer from layer sweep results
  2. Extract features at that layer (vectorized)
  3. Split train 90:10 at pair level for early stopping
  4. Train MLP on GPU, early-stop on val AUC
  5. Test on held-out split

LogReg comparison comes from existing layer_sweep results.

Input:  outputs/phase1/layer_sweep_{slug}.json
        data/activations/{slug}/phase1_chunks/
Output: outputs/phase1/mlp_probes_{slug}.json

Runs on GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.activations import SWIM_WINDOW_SIZES
from agentic_sec_probe.data import C_CPP_CWES, PYTHON_CWES
from agentic_sec_probe.mlp_probe import train_mlp
from agentic_sec_probe.models import MODEL_REGISTRY

ALL_STRATEGIES = ("mean", "last", "first", "max") + tuple(f"swim_{w}" for w in SWIM_WINDOW_SIZES)
MLP_VAL_RATIO = 0.1

SCOPES: dict[str, set[str] | None] = {
    "universal": None,
    "python": PYTHON_CWES,
    "c_cpp": C_CPP_CWES,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_chunks_vectorized(
    chunks_dir: Path,
) -> tuple[
    dict[str, npt.NDArray[np.float32]],
    npt.NDArray[np.int64],
    npt.NDArray[np.str_],
    npt.NDArray[np.str_],
    npt.NDArray[np.int64],
    int,
    int,
]:
    """Load all chunks into per-strategy numpy arrays.

    Returns:
        strategy_arrays: {strategy: [n_samples, n_layers, d_model]}
        labels, cwes, splits, pair_ids, n_layers, d_model
    """
    chunk_files = sorted(chunks_dir.glob("sample_*.pt"))
    if not chunk_files:
        msg = f"No chunks found in {chunks_dir}"
        raise FileNotFoundError(msg)

    chunks = [torch.load(cf, weights_only=False) for cf in chunk_files]
    n = len(chunks)

    first_act = chunks[0]["resid_pre__mean"]
    n_layers, d_model = first_act.shape
    print(f"  Loaded {n} chunks, {n_layers} layers, d_model={d_model}", flush=True)

    labels = np.array([c["label"] for c in chunks], dtype=np.int64)
    cwes = np.array([c["cwe"] for c in chunks])
    splits = np.array([c["split"] for c in chunks])
    pair_ids = np.array([c["pair_id"] for c in chunks], dtype=np.int64)

    strategy_arrays: dict[str, npt.NDArray[np.float32]] = {}
    for strategy in ALL_STRATEGIES:
        key = f"resid_pre__{strategy}"
        stacked = torch.stack([c[key].float() for c in chunks]).numpy()
        strategy_arrays[strategy] = stacked

    del chunks
    print("  All strategies vectorized", flush=True)
    return strategy_arrays, labels, cwes, splits, pair_ids, n_layers, d_model


def train_mlp_probe(
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int64],
    g_train: npt.NDArray[np.int64],
    X_held_out: npt.NDArray[np.float32],
    y_held_out: npt.NDArray[np.int64],
) -> dict[str, float]:
    """Train MLP with 90:10 internal split for early stopping, test on held-out."""
    # Split train into fit/val at pair level
    unique_pairs = np.unique(g_train)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(unique_pairs))
    n_val_pairs = max(1, int(len(unique_pairs) * MLP_VAL_RATIO))
    val_pair_set = set(unique_pairs[perm[:n_val_pairs]])

    fit_mask = np.array([g not in val_pair_set for g in g_train])
    val_mask = ~fit_mask

    if fit_mask.sum() < 10 or val_mask.sum() < 4:
        return {"val_auc": 0.5, "held_out_auc": 0.5}

    # Scale using fit set stats
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_train[fit_mask])
    X_val = scaler.transform(X_train[val_mask])
    X_ho = scaler.transform(X_held_out)

    # Train MLP on GPU
    model, val_auc = train_mlp(
        X_fit,
        y_train[fit_mask],
        X_val,
        y_train[val_mask],
        d_input=X_fit.shape[1],
        device=DEVICE,
    )

    # Test on held-out
    model.train(mode=False)
    X_ho_t = torch.tensor(X_ho, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(X_ho_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    held_out_auc = (
        float(roc_auc_score(y_held_out, probs)) if len(np.unique(y_held_out)) > 1 else 0.5
    )

    return {
        "val_auc": round(val_auc, 4),
        "held_out_auc": round(held_out_auc, 4),
    }


def process_model(
    slug: str,
    strategy_arrays: dict[str, npt.NDArray[np.float32]],
    labels: npt.NDArray[np.int64],
    cwes: npt.NDArray[np.str_],
    splits: npt.NDArray[np.str_],
    pair_ids: npt.NDArray[np.int64],
    sweep: dict[str, Any],
) -> dict[str, Any]:
    """Run MLP probes for one model using pre-vectorized data."""
    results: dict[str, Any] = {
        "slug": slug,
        "n_layers": sweep["n_layers"],
        "d_model": sweep["d_model"],
        "strategies": {},
    }

    for strategy in ALL_STRATEGIES:
        if strategy not in sweep["strategies"]:
            continue
        print(f"\n  Strategy: {strategy}", flush=True)
        acts = strategy_arrays[strategy]  # [n_samples, n_layers, d_model]
        scope_results: dict[str, Any] = {}

        for scope_name, cwe_filter in SCOPES.items():
            strat_scope = sweep["strategies"][strategy].get(scope_name)
            if not strat_scope or "best_layer" not in strat_scope:
                continue

            best_layer = strat_scope["best_layer"]

            # Build masks
            if cwe_filter is not None:
                cwe_mask = np.array([c in cwe_filter for c in cwes])
            else:
                cwe_mask = np.ones(len(labels), dtype=bool)

            train_mask = (splits == "train") & cwe_mask
            eval_mask = (splits == "eval") & cwe_mask

            n_train = int(train_mask.sum())
            n_eval = int(eval_mask.sum())

            if n_train < 10 or n_eval < 4:
                scope_results[scope_name] = {"error": "insufficient_samples"}
                continue

            # Extract single layer via numpy indexing
            X_train = acts[train_mask, best_layer, :]
            X_held_out = acts[eval_mask, best_layer, :]
            y_train = labels[train_mask]
            y_held_out = labels[eval_mask]
            g_train = pair_ids[train_mask]

            print(f"    {scope_name}: layer {best_layer} ({n_train}tr/{n_eval}ev)", flush=True)

            mlp_result = train_mlp_probe(X_train, y_train, g_train, X_held_out, y_held_out)
            print(
                f"      MLP: val={mlp_result['val_auc']:.3f}, "
                f"held-out={mlp_result['held_out_auc']:.3f}",
                flush=True,
            )

            scope_results[scope_name] = {
                "layer": best_layer,
                "n_train": n_train,
                "n_eval": n_eval,
                "mlp": mlp_result,
            }

        results["strategies"][strategy] = scope_results

    return results


def main() -> None:
    print(f"Device: {DEVICE}", flush=True)
    out_dir = PROJECT / "outputs" / "phase1"

    for slug in MODEL_REGISTRY:
        sweep_path = out_dir / f"layer_sweep_{slug}.json"
        if not sweep_path.exists():
            print(f"\nSKIP {slug}: no layer sweep results", flush=True)
            continue

        chunks_dir = PROJECT / "data" / "activations" / slug / "phase1_chunks"
        if not chunks_dir.exists():
            print(f"\nSKIP {slug}: no activation chunks", flush=True)
            continue

        mlp_path = out_dir / f"mlp_probes_{slug}.json"
        if mlp_path.exists():
            print(f"\nSKIP {slug}: MLP results already exist", flush=True)
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"MLP probes: {slug}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Load once, vectorize
        strategy_arrays, labels, cwes, splits, pair_ids, n_layers, d_model = load_chunks_vectorized(
            chunks_dir
        )

        with open(sweep_path) as f:
            sweep = json.load(f)

        results = process_model(slug, strategy_arrays, labels, cwes, splits, pair_ids, sweep)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(mlp_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {mlp_path}", flush=True)

        del strategy_arrays
        torch.cuda.empty_cache()
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
