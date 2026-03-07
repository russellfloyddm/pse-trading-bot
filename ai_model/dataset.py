"""
ai_model/dataset.py - PyTorch Dataset for windowed trading sequences.

``TradingSequenceDataset`` converts the normalised DataFrame (output of the
feature pipeline) into overlapping fixed-length sequences of shape
``(seq_len, num_features)`` paired with a scalar label (BUY=0, HOLD=1, SELL=2).

Design decisions
----------------
* Sequences are built **per-ticker** so that windows never span across
  different stocks.
* The dataset optionally accepts a ``class_weights`` tensor computed with
  :func:`compute_class_weights`, which can be passed directly to
  ``torch.nn.CrossEntropyLoss(weight=...)`` to handle label imbalance.
* A helper :func:`build_dataloaders` creates ``DataLoader`` objects for
  train / val / test splits in one call.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ai_model.config import ModelConfig
from ai_model.features import build_feature_pipeline, split_data

logger = logging.getLogger(__name__)


class TradingSequenceDataset(Dataset):
    """Windowed sequence dataset for Transformer training.

    Each ``__getitem__`` call returns a tuple ``(x, y)`` where:

    * ``x`` – FloatTensor of shape ``(seq_len, num_features)``
    * ``y`` – LongTensor scalar (0=BUY, 1=HOLD, 2=SELL)

    Args:
        df:  Normalised DataFrame produced by the feature pipeline. Must
             contain all feature columns, a *Label* column, a *Ticker* column,
             and be sorted by [Datetime, Ticker].
        cfg: :class:`~ai_model.config.ModelConfig` instance.
    """

    def __init__(self, df: pd.DataFrame, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []
        self._build(df)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, df: pd.DataFrame) -> None:
        """Slide a window across each ticker's time series to build sequences."""
        feature_cols = self.cfg.feature_columns
        seq_len = self.cfg.seq_len

        num_before = 0
        for ticker, group in df.groupby("Ticker", sort=False):
            g = group.sort_values("Datetime").reset_index(drop=True)
            valid = g.dropna(subset=["Label"] + feature_cols)

            if len(valid) < seq_len + 1:
                logger.debug(
                    "Ticker %s has only %d valid rows – need at least %d; skipping.",
                    ticker, len(valid), seq_len + 1,
                )
                continue

            feats = valid[feature_cols].to_numpy(dtype=np.float32)
            labels = valid["Label"].to_numpy(dtype=np.int64)

            for i in range(seq_len, len(valid)):
                self.sequences.append(feats[i - seq_len : i])
                self.labels.append(int(labels[i]))

            num_before += len(valid)

        logger.info(
            "Dataset built: %d sequences from %d valid rows (seq_len=%d).",
            len(self.sequences), num_before, seq_len,
        )

    # ------------------------------------------------------------------
    # Class distribution helper
    # ------------------------------------------------------------------

    def label_distribution(self) -> dict:
        """Return a count mapping {0: n_buy, 1: n_hold, 2: n_sell}."""
        labels_arr = np.array(self.labels)
        return {cls: int((labels_arr == cls).sum()) for cls in range(self.cfg.num_classes)}


# ---------------------------------------------------------------------------
# Class-weight computation (for imbalanced datasets)
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: TradingSequenceDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for ``CrossEntropyLoss``.

    Typical PSE 1-minute data is heavily skewed toward HOLD signals.  Using
    inverse-frequency weights gives the minority BUY/SELL classes more
    influence during training.

    Args:
        dataset: A :class:`TradingSequenceDataset` instance.

    Returns:
        FloatTensor of shape ``(num_classes,)`` with weights summing to
        ``num_classes``.
    """
    dist = dataset.label_distribution()
    total = sum(dist.values())
    num_classes = dataset.cfg.num_classes
    weights = []
    for cls in range(num_classes):
        count = dist.get(cls, 0)
        w = total / (num_classes * count) if count > 0 else 1.0
        weights.append(w)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(
        "Class weights – BUY: %.3f  HOLD: %.3f  SELL: %.3f",
        weight_tensor[0].item(), weight_tensor[1].item(), weight_tensor[2].item(),
    )
    return weight_tensor


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    processed_csv: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    cfg: Optional[ModelConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Build train / val / test DataLoaders from either a CSV path or a DataFrame.

    At least one of *processed_csv* or *df* must be supplied.

    Args:
        processed_csv: Path to the processed CSV file produced by the main
                       pipeline (``indicators.add_indicators()`` output).
        df:            Pre-loaded DataFrame (used if *processed_csv* is None).
        cfg:           Model configuration (defaults to :class:`ModelConfig`).

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, class_weights)``.
    """
    if cfg is None:
        cfg = ModelConfig()

    if df is None:
        if processed_csv is None:
            processed_csv = cfg.processed_data_file
        logger.info("Loading data from %s", processed_csv)
        df = pd.read_csv(processed_csv, parse_dates=["Datetime"])

    processed_df, _ = build_feature_pipeline(df, cfg)
    train_df, val_df, test_df = split_data(processed_df, cfg)

    train_ds = TradingSequenceDataset(train_df, cfg)
    val_ds = TradingSequenceDataset(val_df, cfg)
    test_ds = TradingSequenceDataset(test_df, cfg)

    class_weights = compute_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights
