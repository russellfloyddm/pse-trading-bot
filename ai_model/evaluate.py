"""
ai_model/evaluate.py - Evaluation metrics for the Transformer signal model.

Provides two levels of evaluation:

1. **Classification metrics** – accuracy, per-class precision/recall/F1,
   confusion matrix.  Computed directly on the test-set model predictions.

2. **Trading metrics** – Sharpe ratio, maximum drawdown, and win rate
   derived from a simple backtest that converts model predictions into
   a long-only equity curve.

Usage::

    from ai_model.evaluate import evaluate_model, print_report
    report = evaluate_model(model, test_loader, cfg, device)
    print_report(report)
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ai_model.config import ModelConfig

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Label names for display
LABEL_NAMES = {0: "BUY", 1: "HOLD", 2: "SELL"}


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_all(
    model: "nn.Module",
    loader: "DataLoader",
    device: "torch.device",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on an entire DataLoader.

    Args:
        model:  Trained :class:`~ai_model.model.TransformerSignalModel`.
        loader: DataLoader for the evaluation split.
        device: Computation device.

    Returns:
        Tuple of ``(all_preds, all_labels, all_probs)`` as NumPy arrays.
        *all_probs* has shape ``(n_samples, num_classes)``.
    """
    import torch

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 3,
) -> Dict:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix.

    Avoids depending on scikit-learn so the module works even before ML
    dependencies are installed (gracefully falls back to manual computation).

    Args:
        preds:       Predicted class indices (1-D array).
        labels:      Ground-truth class indices (1-D array).
        num_classes: Total number of classes.

    Returns:
        Dictionary with keys: ``accuracy``, ``per_class``, ``confusion_matrix``.
    """
    n = len(labels)
    if n == 0:
        return {"accuracy": 0.0, "per_class": {}, "confusion_matrix": []}

    accuracy = float((preds == labels).mean())

    # Confusion matrix (rows = true, cols = predicted)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for cls in range(num_classes):
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum()) - tp
        fn = int(cm[cls, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[LABEL_NAMES[cls]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[cls, :].sum()),
        }

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Trading simulation metrics
# ---------------------------------------------------------------------------

def trading_metrics(
    preds: np.ndarray,
    close_prices: Optional[np.ndarray] = None,
) -> Dict:
    """Compute Sharpe ratio, max drawdown, and win rate from predictions.

    Uses a simplified long-only simulation:
    * **BUY (0)**: open a long position (buy at current price).
    * **HOLD (1)**: maintain current position.
    * **SELL (2)**: close the position (sell at current price).

    If *close_prices* is ``None``, returns placeholder values of 0.0.

    Args:
        preds:        Array of predicted signals (0=BUY, 1=HOLD, 2=SELL).
        close_prices: Array of close prices aligned with *preds*.

    Returns:
        Dictionary with keys: ``sharpe_ratio``, ``max_drawdown``, ``win_rate``,
        ``num_trades``.
    """
    if close_prices is None or len(close_prices) < 2:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "num_trades": 0}

    equity = 1.0
    position_price: Optional[float] = None
    trade_returns: list[float] = []
    equity_curve: list[float] = [equity]

    for i, signal in enumerate(preds):
        price = float(close_prices[i])
        if signal == 0 and position_price is None:       # BUY
            position_price = price
        elif signal == 2 and position_price is not None: # SELL
            ret = (price - position_price) / position_price
            trade_returns.append(ret)
            equity *= 1 + ret
            position_price = None
        equity_curve.append(equity)

    # Mark-to-market if still in a position at end
    if position_price is not None and len(close_prices) > 0:
        last_price = float(close_prices[-1])
        ret = (last_price - position_price) / position_price
        trade_returns.append(ret)

    # Sharpe ratio (annualised, 1-min: 252*390 bars/year)
    equity_arr = np.array(equity_curve, dtype=float)
    bar_returns = np.diff(equity_arr) / equity_arr[:-1]
    if bar_returns.std() > 0:
        sharpe = float(bar_returns.mean() / bar_returns.std() * np.sqrt(252 * 390))
    else:
        sharpe = 0.0

    # Maximum drawdown
    roll_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - roll_max) / roll_max
    max_dd = float(drawdowns.min())

    # Win rate
    wins = sum(1 for r in trade_returns if r > 0)
    num_trades = len(trade_returns)
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    return {
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "num_trades": num_trades,
    }


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: "nn.Module",
    loader: "DataLoader",
    cfg: ModelConfig,
    device: "torch.device",
    close_prices: Optional[np.ndarray] = None,
) -> Dict:
    """Run full evaluation: classification + trading metrics.

    Args:
        model:        Trained model.
        loader:       DataLoader for the evaluation split.
        cfg:          Model configuration.
        device:       Computation device.
        close_prices: Optional close-price array for trading simulation.

    Returns:
        Nested dictionary with ``classification`` and ``trading`` sub-dicts,
        plus raw ``predictions`` and ``labels`` arrays.
    """
    preds, labels, probs = predict_all(model, loader, device)

    cls_metrics = classification_metrics(preds, labels, num_classes=cfg.num_classes)
    trd_metrics = trading_metrics(preds, close_prices)

    return {
        "classification": cls_metrics,
        "trading": trd_metrics,
        "predictions": preds,
        "labels": labels,
        "probabilities": probs,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_report(report: Dict) -> None:
    """Print a human-readable evaluation summary.

    Args:
        report: Dictionary returned by :func:`evaluate_model`.
    """
    cls = report.get("classification", {})
    trd = report.get("trading", {})

    print("\n" + "=" * 55)
    print("  CLASSIFICATION METRICS")
    print("=" * 55)
    print(f"  Overall Accuracy : {cls.get('accuracy', 0):.2%}")
    print()
    per_class = cls.get("per_class", {})
    header = f"  {'Class':<6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}"
    print(header)
    print("  " + "-" * 48)
    for name, m in per_class.items():
        print(
            f"  {name:<6}  {m['precision']:>10.4f}  {m['recall']:>8.4f}"
            f"  {m['f1']:>8.4f}  {m['support']:>8d}"
        )

    print()
    print("=" * 55)
    print("  TRADING METRICS (long-only simulation)")
    print("=" * 55)
    print(f"  Sharpe Ratio     : {trd.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown     : {trd.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate         : {trd.get('win_rate', 0):.2%}")
    print(f"  # Trades         : {trd.get('num_trades', 0)}")
    print("=" * 55 + "\n")
