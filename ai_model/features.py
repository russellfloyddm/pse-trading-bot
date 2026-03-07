"""
ai_model/features.py - Feature engineering pipeline for the Transformer model.

Converts the combined OHLCV + indicator DataFrame (output of
``indicators.add_indicators()``) into per-ticker normalised feature matrices
ready for sequence construction.

Pipeline
--------
1. Select the configured feature columns.
2. Drop rows where *any* feature or the Close price is NaN.
3. Per-ticker z-score normalisation (fit on training data, applied to
   val/test data using the same statistics).
4. Expose helper functions for splitting and persisting scaler statistics.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ai_model.config import ModelConfig

logger = logging.getLogger(__name__)

# Scaler statistics type: maps ticker → {"mean": Series, "std": Series}
ScalerStats = Dict[str, Dict[str, pd.Series]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_features(df: pd.DataFrame, cfg: ModelConfig) -> None:
    """Raise ``ValueError`` if any required feature column is absent.

    Args:
        df:  Processed DataFrame.
        cfg: :class:`~ai_model.config.ModelConfig` instance.

    Raises:
        ValueError: If one or more feature columns are missing.
    """
    missing = [c for c in cfg.feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns in DataFrame: {missing}. "
            "Run indicators.add_indicators() before calling feature functions."
        )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_features(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Drop rows with NaN in feature or Close columns.

    Indicators computed with rolling windows produce NaN for the first
    ``window - 1`` rows.  These rows are removed per-ticker to avoid padding
    the sequence model with zeros.

    Args:
        df:  DataFrame with feature columns and a *Ticker* column.
        cfg: Model configuration.

    Returns:
        Cleaned DataFrame (copy) sorted by [Datetime, Ticker].
    """
    validate_features(df, cfg)
    required = cfg.feature_columns + ["Close"]
    subset = [c for c in required if c in df.columns]
    cleaned = df.dropna(subset=subset).copy()
    dropped = len(df) - len(cleaned)
    if dropped:
        logger.info("Dropped %d rows with NaN values in feature columns.", dropped)
    return cleaned.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-ticker z-score normalisation
# ---------------------------------------------------------------------------

def fit_scalers(df: pd.DataFrame, cfg: ModelConfig) -> ScalerStats:
    """Compute per-ticker z-score statistics (mean, std) from *training* data.

    Args:
        df:  Training split of the processed DataFrame.
        cfg: Model configuration.

    Returns:
        ``ScalerStats`` dictionary mapping ticker → {"mean": …, "std": …}.
    """
    stats: ScalerStats = {}
    for ticker, group in df.groupby("Ticker"):
        feat_df = group[cfg.feature_columns]
        mean = feat_df.mean()
        std = feat_df.std().replace(0, 1)   # avoid division by zero
        stats[str(ticker)] = {"mean": mean, "std": std}
        logger.debug("Scaler fitted for %s (%d rows).", ticker, len(group))
    return stats


def apply_scalers(
    df: pd.DataFrame,
    cfg: ModelConfig,
    stats: ScalerStats,
) -> pd.DataFrame:
    """Apply pre-fitted z-score normalisation to a DataFrame.

    Tickers absent from *stats* (unseen during training) are normalised with
    the global mean/std computed across all available tickers.

    Args:
        df:    DataFrame to normalise (can be val, test, or live data).
        cfg:   Model configuration.
        stats: Output of :func:`fit_scalers` called on training data.

    Returns:
        DataFrame copy with feature columns replaced by normalised values.
    """
    if not stats:
        logger.warning("Empty scaler stats passed – returning unnormalised data.")
        return df.copy()

    # Compute global fallback statistics (mean of per-ticker means/stds)
    all_means = pd.concat([s["mean"] for s in stats.values()], axis=1).mean(axis=1)
    all_stds = pd.concat([s["std"] for s in stats.values()], axis=1).mean(axis=1)

    frames = []
    for ticker, group in df.groupby("Ticker"):
        g = group.copy()
        ticker_stats = stats.get(str(ticker), {"mean": all_means, "std": all_stds})
        g[cfg.feature_columns] = (
            g[cfg.feature_columns].sub(ticker_stats["mean"]).div(ticker_stats["std"])
        )
        frames.append(g)

    if not frames:
        return df.copy()

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["Datetime", "Ticker"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def make_labels(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Compute look-ahead return labels per ticker.

    Label encoding:
        * **0 – BUY**:  future return ≥ ``label_threshold``
        * **1 – HOLD**: |future return| < ``label_threshold``
        * **2 – SELL**: future return ≤ ``-label_threshold``

    The last ``label_horizon`` rows per ticker receive ``NaN`` labels because
    the future close is unavailable; those rows should be dropped before
    building sequences.

    Args:
        df:  Cleaned DataFrame with a *Close* column.
        cfg: Model configuration.

    Returns:
        DataFrame copy with an additional integer *Label* column.
    """
    frames = []
    for _, group in df.groupby("Ticker", sort=False):
        g = group.copy().sort_values("Datetime").reset_index(drop=True)
        future_close = g["Close"].shift(-cfg.label_horizon)
        future_ret = (future_close - g["Close"]) / g["Close"]

        label = np.where(
            future_ret >= cfg.label_threshold, 0,        # BUY
            np.where(future_ret <= -cfg.label_threshold, 2, 1),  # SELL / HOLD
        )
        g["Label"] = label.astype(float)
        g.loc[g.index[-cfg.label_horizon:], "Label"] = np.nan
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["Datetime", "Ticker"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------------------------------------------------------
# Chronological train / validation / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    cfg: ModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train / validation / test sets.

    The split is time-based (no shuffling) to preserve temporal ordering and
    avoid look-ahead bias.  The split fractions are taken from ``cfg``.

    Args:
        df:  Full cleaned DataFrame sorted by Datetime.
        cfg: Model configuration.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    test_cut = int(n * (1 - cfg.test_split))
    val_cut = int(test_cut * (1 - cfg.val_split))

    train_df = df.iloc[:val_cut].copy()
    val_df = df.iloc[val_cut:test_cut].copy()
    test_df = df.iloc[test_cut:].copy()

    logger.info(
        "Data split: train=%d, val=%d, test=%d rows.", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (convenience wrapper)
# ---------------------------------------------------------------------------

def build_feature_pipeline(
    df: pd.DataFrame,
    cfg: ModelConfig,
    scaler_stats: Optional[ScalerStats] = None,
) -> Tuple[pd.DataFrame, ScalerStats]:
    """End-to-end feature pipeline: clean → label → split → scale.

    Fits scalers on the training split and applies them to all three splits.

    Args:
        df:           Raw processed DataFrame from ``add_indicators()``.
        cfg:          Model configuration.
        scaler_stats: Pre-fitted scaler stats (pass for inference; ``None``
                      to fit fresh stats on the training split).

    Returns:
        Tuple of (processed_df_with_labels, scaler_stats).  The returned
        DataFrame contains normalised features and a *Label* column.
    """
    df = clean_features(df, cfg)
    df = make_labels(df, cfg)
    df = df.dropna(subset=["Label"]).reset_index(drop=True)

    if scaler_stats is None:
        train_df, _, _ = split_data(df, cfg)
        scaler_stats = fit_scalers(train_df, cfg)

    df = apply_scalers(df, cfg, scaler_stats)
    return df, scaler_stats
