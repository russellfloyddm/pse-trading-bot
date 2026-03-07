"""
data_pipeline.py - Market data fetching for the PSE Trading Bot.

Fetches 1-minute interval OHLCV data from Yahoo Finance for all configured
PSE tickers and combines them into a single DataFrame.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


def fetch_ticker_data(
    ticker: str,
    period: str = config.DATA_PERIOD,
    interval: str = config.DATA_INTERVAL,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a single ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "BDO.PS").
        period: Look-back period string (e.g. "1d", "5d").
        interval: Candle interval string (e.g. "1m", "5m").

    Returns:
        DataFrame with columns [Datetime, Open, High, Low, Close, Volume, Ticker]
        or None if the fetch fails.
    """
    try:
        raw = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            logger.warning("No data returned for %s", ticker)
            return None

        # Flatten MultiIndex columns that yfinance may produce
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only the standard OHLCV columns
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        df = raw[cols].copy()

        # Ensure the index is named "Datetime" and becomes a regular column
        df.index.name = "Datetime"
        df = df.reset_index()
        df["Ticker"] = ticker

        # Ensure Datetime is tz-naive for consistent storage
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)

        logger.info("Fetched %d rows for %s", len(df), ticker)
        return df

    except Exception as exc:
        logger.error("Error fetching data for %s: %s", ticker, exc)
        return None


def fetch_all_tickers(
    tickers: list[str] = config.TICKERS,
    period: str = config.DATA_PERIOD,
    interval: str = config.DATA_INTERVAL,
) -> pd.DataFrame:
    """Fetch OHLCV data for all tickers and combine into one DataFrame.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        period: Look-back period string.
        interval: Candle interval string.

    Returns:
        Combined DataFrame sorted by Datetime and Ticker.
        Empty DataFrame if all fetches fail.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = fetch_ticker_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        logger.error("Failed to fetch data for any ticker.")
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"])

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["Datetime", "Ticker"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    logger.info("Combined DataFrame: %d rows across %d tickers", len(combined), len(frames))
    return combined


def validate_ticker(ticker: str) -> bool:
    """Check whether a ticker symbol is valid on Yahoo Finance.

    Makes a lightweight daily data request (last 5 days) to confirm the
    symbol exists and returns non-empty data.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "BDO.PS", "AAPL").

    Returns:
        True if the ticker is valid and returns data, False otherwise.
    """
    try:
        raw = yf.download(
            tickers=ticker,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        return raw is not None and not raw.empty
    except Exception as exc:
        logger.warning("Ticker validation failed for %s: %s", ticker, exc)
        return False


def get_latest_candles(
    tickers: list[str] = config.TICKERS,
    interval: str = config.DATA_INTERVAL,
) -> pd.DataFrame:
    """Fetch the most recent candles for live-update mode.

    Fetches the last trading day and returns only the latest candle per ticker.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        interval: Candle interval string.

    Returns:
        DataFrame with the latest candle for each ticker.
    """
    df = fetch_all_tickers(tickers=tickers, period="1d", interval=interval)
    if df.empty:
        return df
    # Return the last available candle per ticker
    return df.groupby("Ticker").last().reset_index()
