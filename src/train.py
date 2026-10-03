"""
CNG403 Assignment 1 — Scratch Implementation: Training Pipeline
===============================================================
This script trains and evaluates your from-scratch FFNN.

Usage:
    python train.py --config ../configs/scratch_config.json

After running reference.py and train.py, compare the learning curves in the
notebook to verify your implementation produces consistent results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import your scratch modules
from layers import CrossEntropyLoss
from network import FFNN
from optimizers import SGD, BatchGD

# Import shared utilities
from data_utils import set_seed, load_and_extract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def save_checkpoint(net: FFNN, metrics: dict, path: str) -> None:
    """Save weights of all Linear layers and training metrics."""
    state = {
        "weights": [(l.W.clone(), l.b.clone()) for l in net.linear_layers],
        "metrics": metrics,
    }
    torch.save(state, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(net: FFNN, path: str) -> dict:
    """Restore Linear layer weights from a checkpoint."""
    state = torch.load(path, weights_only=True)
    for layer, (W, b) in zip(net.linear_layers, state["weights"]):
        layer.W = W.clone()
        layer.b = b.clone()
    return state["metrics"]


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    net: FFNN,
    loader: DataLoader,
    loss_fn: CrossEntropyLoss,
    optimizer,
    l2_lambda: float,
) -> tuple[float, float]:
    """
    Run one full pass over the training DataLoader.

    Args:
        net:       your FFNN instance.
        loader:    DataLoader yielding (X_batch, y_batch) tuples.
        loss_fn:   CrossEntropyLoss instance.
        optimizer: SGD or BatchGD instance.
        l2_lambda: L2 regularisation coefficient.

    Returns:
        avg_loss (float): mean loss across all samples.
        acc      (float): accuracy across all samples.
    """
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        # Forward
        logits = net(X_batch)
        loss   = loss_fn(logits, y_batch)

        # Backward
        grad = loss_fn.backward()
        net.backward(grad)

        # L2 regularisation gradient
        if l2_lambda > 0:
            net.l2_grad(l2_lambda)

        optimizer.step()

        # Track metrics (detach loss to avoid keeping computation graph)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: FFNN,
    loader: DataLoader,
    loss_fn: CrossEntropyLoss,
) -> tuple[float, float]:
    """
    Evaluate the network on a DataLoader (no parameter updates).

    Returns:
        avg_loss (float), accuracy (float)
    """
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        logits      = net(X_batch)
        loss        = loss_fn(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    set_seed(cfg["training"]["seed"])

    # -- Data ----------------------------------------------------------------
    data_root = "data/"
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract(data_root)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    tcfg       = cfg["training"]
    batch_size = tcfg["batch_size"]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=256)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=256)

    # -- Model ---------------------------------------------------------------
    input_dim  = X_train.shape[1]
    mcfg       = cfg["model"]
    net = FFNN(
        input_dim    = input_dim,
        hidden_sizes = mcfg["hidden_sizes"],
        num_classes  = 10,
        activation   = mcfg["activation"],
    )

    # -- Optimiser -----------------------------------------------------------
    optimizer_name = tcfg["optimizer"].lower()
    if optimizer_name == "sgd":
        optimizer = SGD(net, lr=tcfg["learning_rate"], momentum=tcfg.get("momentum", 0.0))
    elif optimizer_name == "batch_gd":
        # Batch GD: set batch_size = full dataset in the config
        optimizer = BatchGD(net, lr=tcfg["learning_rate"])
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Choose 'sgd' or 'batch_gd'.")

    loss_fn   = CrossEntropyLoss()
    l2_lambda = tcfg["l2_lambda"]

    # -- Training loop -------------------------------------------------------
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(ckpt_dir / "scratch_best.pt")

    best_val_acc = 0.0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, tcfg["epochs"] + 1):
        tr_loss, tr_acc     = train_one_epoch(net, train_loader, loss_fn, optimizer, l2_lambda)
        val_loss, val_acc   = evaluate(net, val_loader, loss_fn)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(net, {"epoch": epoch, "val_acc": val_acc}, best_ckpt)

    # -- Test ----------------------------------------------------------------
    load_checkpoint(net, best_ckpt)
    test_loss, test_acc = evaluate(net, test_loader, loss_fn)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Save history for notebook comparison
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(log_dir / "scratch_history.npy"), history)
    print(f"Training history saved → {log_dir / 'scratch_history.npy'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNG403 A1 — Scratch FFNN Training")
    parser.add_argument(
        "--config",
        default="../config.json",
        help="Path to config.json",
    )
    args = parser.parse_args()
    run(args.config)
