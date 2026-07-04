"""MLP probe: non-linear probe on residual stream activations.

Architecture: d_input -> 256 -> 64 -> 1 with BatchNorm, ReLU, Dropout.
Trained with BCEWithLogitsLoss, AdamW, cosine LR schedule, early stopping.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class MLPProbe(nn.Module):
    """2-layer MLP probe with batch norm and dropout."""

    def __init__(
        self,
        d_input: int,
        d_hidden1: int = 256,
        d_hidden2: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden1),
            nn.BatchNorm1d(d_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden1, d_hidden2),
            nn.BatchNorm1d(d_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out


def train_mlp(
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    X_val: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.int64],
    *,
    d_input: int,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.3,
    patience: int = 15,
    device: str = "cuda",
) -> tuple[MLPProbe, float]:
    """Train MLP probe with early stopping. Returns (best_model, best_val_auc).

    Validates every 5 epochs and restores best checkpoint.
    """
    model = MLPProbe(d_input, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_auc = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i : i + batch_size]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.train(False)
            with torch.no_grad():
                val_logits = model(X_v)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()

            if len(np.unique(y_val)) < 2:
                continue

            auc = float(roc_auc_score(y_val, val_probs))

            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.train(False)
    with torch.no_grad():
        val_logits = model(X_v)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    final_auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.5
    return model, final_auc
