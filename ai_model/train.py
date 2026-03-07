"""
ai_model/train.py - Training loop for the Transformer signal model.

Features
--------
* AdamW optimiser with cosine annealing learning-rate schedule.
* Weighted cross-entropy loss (handles BUY/HOLD/SELL class imbalance).
* Early stopping based on validation loss.
* Automatic model checkpointing (best validation accuracy).
* Training/validation metrics logged every epoch.
* GPU / CPU device selection with ``torch.device`` auto-detection.

Usage::

    python -m ai_model.train                          # uses default ModelConfig
    python -m ai_model.train --epochs 30 --lr 5e-4   # override via CLI flags
"""

import argparse
import logging
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ai_model.config import ModelConfig
from ai_model.dataset import build_dataloaders
from ai_model.model import TransformerSignalModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    """Run one full pass over *loader*.

    Args:
        model:     The :class:`TransformerSignalModel`.
        loader:    DataLoader for the current split.
        criterion: Loss function (weighted CrossEntropyLoss).
        optimizer: Optimiser (``None`` in eval mode).
        device:    Computation device.
        train:     If ``True``, runs backward pass and weight updates.

    Returns:
        Tuple of ``(avg_loss, accuracy)`` for the epoch.
    """
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    cfg: Optional[ModelConfig] = None,
    processed_csv: Optional[str] = None,
) -> TransformerSignalModel:
    """Train the Transformer model end-to-end.

    Args:
        cfg:           Model/training configuration (defaults to
                       :class:`~ai_model.config.ModelConfig`).
        processed_csv: Path to the processed CSV.  Falls back to
                       ``cfg.processed_data_file`` when ``None``.

    Returns:
        The trained :class:`~ai_model.model.TransformerSignalModel` loaded
        with the best checkpoint weights.
    """
    if cfg is None:
        cfg = ModelConfig()

    device = get_device()
    torch.manual_seed(cfg.random_seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, _, class_weights = build_dataloaders(
        processed_csv=processed_csv, cfg=cfg
    )

    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError(
            "Training dataset is empty. Ensure the processed CSV contains "
            "enough data (at least seq_len + label_horizon rows per ticker)."
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TransformerSignalModel(cfg).to(device)
    logger.info(
        "Model: %s | Parameters: %d", type(model).__name__, model.count_parameters()
    )

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f train_acc=%.3f | "
            "val_loss=%.4f val_acc=%.3f | %.1fs",
            epoch, cfg.num_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            elapsed,
        )

        # Checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "cfg": cfg,
                },
                cfg.best_model_path,
            )
            logger.info("  ✓ Checkpoint saved (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch, cfg.patience,
                )
                break

    # ------------------------------------------------------------------
    # Load best weights and return
    # ------------------------------------------------------------------
    checkpoint = torch.load(cfg.best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Training complete. Best model from epoch %d (val_loss=%.4f, val_acc=%.3f).",
        checkpoint["epoch"], checkpoint["val_loss"], checkpoint["val_acc"],
    )
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PSE Transformer signal model.")
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = ModelConfig()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len

    train(cfg=cfg, processed_csv=args.data)
