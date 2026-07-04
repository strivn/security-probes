"""Tests for MLPProbe forward pass and shape."""

import numpy as np
import torch

from agentic_sec_probe.mlp_probe import MLPProbe, train_mlp


def test_mlp_probe_forward_shape() -> None:
    probe = MLPProbe(d_input=128)
    x = torch.randn(16, 128)
    out = probe(x)
    assert out.shape == (16,)


def test_mlp_probe_single_sample() -> None:
    probe = MLPProbe(d_input=64)
    # BatchNorm needs train(False) for single sample
    probe.train(False)
    x = torch.randn(1, 64)
    out = probe(x)
    assert out.shape == (1,)


def test_mlp_probe_custom_dims() -> None:
    probe = MLPProbe(d_input=512, d_hidden1=128, d_hidden2=32, dropout=0.1)
    x = torch.randn(8, 512)
    out = probe(x)
    assert out.shape == (8,)


def test_train_mlp_on_synthetic_data() -> None:
    """Train on linearly separable synthetic data, expect AUC > 0.7."""
    rng = np.random.RandomState(42)
    n = 200
    d = 32

    # Linearly separable: class 0 centered at -1, class 1 at +1
    X = rng.randn(n, d).astype(np.float64)
    y = np.zeros(n, dtype=np.int64)
    y[n // 2 :] = 1
    X[: n // 2] -= 1.0
    X[n // 2 :] += 1.0

    # Shuffle then split 80:20 (sequential split would put all class 1 in val)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]

    model, auc = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        d_input=d,
        n_epochs=30,
        batch_size=32,
        device="cpu",
    )
    assert isinstance(model, MLPProbe)
    assert auc > 0.7, f"Expected AUC > 0.7 on separable data, got {auc:.3f}"
