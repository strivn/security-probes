"""Probe training, saving, loading, and inference.

Consolidates probe logic into reusable functions.
Probes are LogisticRegression classifiers trained on mean-pooled residual
stream activations at a specific layer.

Artifacts are saved as .joblib files (sklearn model + scaler) with a
companion metadata.json describing the training configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """Result of a single probe prediction."""

    is_vulnerable: bool
    confidence: float  # probability of vulnerable class


@dataclass
class ProbeArtifact:
    """A trained probe ready for inference."""

    model: LogisticRegression
    scaler: StandardScaler
    metadata: dict


def load_activations(
    chunks_dir: Path,
    act_key: str = "resid_pre__mean",
    layer: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load activations at a specific layer from per-sample chunks.

    Returns: (X, labels, cwes, langs, pair_ids, projects)

    `langs` is the per-sample `language` field ("python"/"c_cpp") read directly from
    the chunk (F1 fix: NOT derived from CWE membership — a cwe-022 sample can be C/C++).
    `projects` is the GitHub repo per sample and is the correct GroupKFold group
    (no codebase spans CV folds). `pair_ids` is kept for backward compatibility;
    prefer `projects` for cross-validation grouping.
    """
    chunk_files = sorted(chunks_dir.glob("sample_*.pt"))
    if not chunk_files:
        msg = f"No chunks found in {chunks_dir}"
        raise FileNotFoundError(msg)

    vectors, labels, cwes, langs, pair_ids, projects = [], [], [], [], [], []
    for cf in chunk_files:
        chunk = torch.load(cf, weights_only=True)  # chunks hold tensors + plain types
        vectors.append(chunk[act_key][layer].float().numpy())
        labels.append(chunk["label"])
        cwes.append(chunk["cwe"])
        langs.append(chunk["language"])
        pair_ids.append(chunk["pair_id"])
        projects.append(chunk.get("project", f"unknown_pair_{chunk['pair_id']}"))

    X = np.stack(vectors)
    labels_arr = np.array(labels)
    cwes_arr = np.array(cwes)
    langs_arr = np.array(langs)
    pair_ids_arr = np.array(pair_ids)
    projects_arr = np.array(projects, dtype=object)
    return X, labels_arr, cwes_arr, langs_arr, pair_ids_arr, projects_arr


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float = 1.0,
    n_folds: int = 5,
) -> tuple[LogisticRegression, StandardScaler, float, float]:
    """Train a LogReg probe with GroupKFold CV. Returns (model, scaler, auc, auc_std).

    The returned model and scaler are fit on ALL data (not a single fold).
    The AUC is the cross-validated estimate.

    `groups` should be the per-sample PROJECT (from load_activations) so no codebase
    spans CV folds. Passing pair_ids reintroduces same-repo cross-fold leakage.
    """
    n_folds = min(n_folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_folds)

    fold_aucs = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_tr, y[train_idx])
        fold_aucs.append(roc_auc_score(y[val_idx], clf.predict_proba(X_val)[:, 1]))

    # Fit final model on all data
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
    final_model.fit(X_scaled, y)

    return final_model, final_scaler, float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def evaluate_probe(
    probe: ProbeArtifact,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained probe on held-out data.

    Returns: {"auc": float, "accuracy": float, "f1": float}
    """
    from sklearn.metrics import accuracy_score, f1_score

    X_scaled = probe.scaler.transform(X_eval)
    y_prob = probe.model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    auc = float(roc_auc_score(y_eval, y_prob)) if len(np.unique(y_eval)) > 1 else 0.5
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0.0)),
    }


def save_probe(
    out_dir: Path,
    model: LogisticRegression,
    scaler: StandardScaler,
    *,
    name: str,
    model_slug: str,
    layer: int,
    d_model: int,
    auc: float,
    auc_std: float,
    C: float,
    n_samples: int,
    cwes_used: list[str],
) -> Path:
    """Save probe artifacts to disk.

    Creates:
        {out_dir}/{name}_probe.joblib  — sklearn model
        {out_dir}/{name}_scaler.joblib — StandardScaler
        {out_dir}/{name}_metadata.json — training config
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / f"{name}_probe.joblib")
    joblib.dump(scaler, out_dir / f"{name}_scaler.joblib")

    metadata = {
        "name": name,
        "model_slug": model_slug,
        "layer": layer,
        "d_model": d_model,
        "auc": auc,
        "auc_std": auc_std,
        "C": C,
        "n_samples": n_samples,
        "cwes_used": cwes_used,
    }
    with open(out_dir / f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out_dir


def load_probe(probe_dir: Path, name: str) -> ProbeArtifact:
    """Load a saved probe from disk."""
    probe_dir = Path(probe_dir)

    model = joblib.load(probe_dir / f"{name}_probe.joblib")
    scaler = joblib.load(probe_dir / f"{name}_scaler.joblib")
    with open(probe_dir / f"{name}_metadata.json") as f:
        metadata = json.load(f)

    return ProbeArtifact(model=model, scaler=scaler, metadata=metadata)


def predict(probe: ProbeArtifact, activations: np.ndarray) -> ProbeResult:
    """Run probe inference on a single activation vector.

    Args:
        probe: loaded ProbeArtifact
        activations: shape [d_model] — mean-pooled activations at the probe's layer

    Returns:
        ProbeResult with is_vulnerable and confidence
    """
    X = activations.reshape(1, -1)
    X_scaled = probe.scaler.transform(X)
    prob = probe.model.predict_proba(X_scaled)[0, 1]
    return ProbeResult(is_vulnerable=prob > 0.5, confidence=float(prob))


def predict_batch(probe: ProbeArtifact, activations: np.ndarray) -> list[ProbeResult]:
    """Run probe inference on a batch of activation vectors.

    Args:
        probe: loaded ProbeArtifact
        activations: shape [N, d_model]

    Returns:
        list of ProbeResult
    """
    X_scaled = probe.scaler.transform(activations)
    probs = probe.model.predict_proba(X_scaled)[:, 1]
    return [ProbeResult(is_vulnerable=p > 0.5, confidence=float(p)) for p in probs]
