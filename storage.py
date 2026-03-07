"""
storage.py - Data persistence helpers for the PSE Trading Bot.

Saves and loads DataFrames to/from CSV and Parquet formats.
"""

import logging
import os

import pandas as pd

import config

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to persist.
        path: Absolute file path (including ``.csv`` extension).
    """
    _ensure_dir(path)
    df.to_csv(path, index=False)
    logger.info("Saved CSV  → %s  (%d rows)", path, len(df))


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a Parquet file.

    Args:
        df: DataFrame to persist.
        path: Absolute file path (including ``.parquet`` extension).
    """
    _ensure_dir(path)
    df.to_parquet(path, index=False)
    logger.info("Saved Parquet → %s  (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load a DataFrame from a CSV file.

    Args:
        path: Absolute file path.

    Returns:
        Loaded DataFrame, or empty DataFrame if the file does not exist.
    """
    if not os.path.exists(path):
        logger.warning("CSV file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info("Loaded CSV  ← %s  (%d rows)", path, len(df))
    return df


def load_parquet(path: str) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file.

    Args:
        path: Absolute file path.

    Returns:
        Loaded DataFrame, or empty DataFrame if the file does not exist.
    """
    if not os.path.exists(path):
        logger.warning("Parquet file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info("Loaded Parquet ← %s  (%d rows)", path, len(df))
    return df


# ---------------------------------------------------------------------------
# Convenience wrappers for standard bot file paths
# ---------------------------------------------------------------------------

def save_raw_data(df: pd.DataFrame) -> None:
    """Save raw market data to the configured raw data CSV path."""
    save_csv(df, config.RAW_DATA_FILE)


def save_processed_data(df: pd.DataFrame) -> None:
    """Save processed (indicator-enriched) data to the configured path."""
    save_csv(df, config.PROCESSED_DATA_FILE)
    # Also save a Parquet copy for faster re-loading
    parquet_path = config.PROCESSED_DATA_FILE.replace(".csv", ".parquet")
    save_parquet(df, parquet_path)


def load_raw_data() -> pd.DataFrame:
    """Load raw market data from the configured CSV path."""
    return load_csv(config.RAW_DATA_FILE)


def load_processed_data() -> pd.DataFrame:
    """Load processed data, preferring Parquet if available."""
    parquet_path = config.PROCESSED_DATA_FILE.replace(".csv", ".parquet")
    if os.path.exists(parquet_path):
        return load_parquet(parquet_path)
    return load_csv(config.PROCESSED_DATA_FILE)


def save_trade_log(df: pd.DataFrame) -> None:
    """Append the trade log DataFrame to the configured CSV file."""
    save_csv(df, config.TRADE_LOG_FILE)


def save_portfolio_log(df: pd.DataFrame) -> None:
    """Save a portfolio snapshot DataFrame to the configured CSV file."""
    save_csv(df, config.PORTFOLIO_LOG_FILE)
