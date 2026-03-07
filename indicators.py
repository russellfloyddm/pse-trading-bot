"""
indicators.py - Technical indicator computation for the PSE Trading Bot.

Computes EMA, RSI, Bollinger Bands, returns, and volatility on a per-ticker
basis from a combined OHLCV DataFrame.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level indicator functions (operate on a single Series / DataFrame)
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Compute the Exponential Moving Average.

    Args:
        series: Numeric pandas Series (e.g. Close prices).
        period: Look-back period.

    Returns:
        EMA Series aligned with the input index.
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Args:
        series: Numeric pandas Series of prices.
        period: Look-back period (default 14).

    Returns:
        RSI Series (0–100) aligned with the input index.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series,
    period: int = config.BOLLINGER_PERIOD,
    num_std: float = config.BOLLINGER_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper, middle SMA, lower).

    Args:
        series: Numeric pandas Series of prices.
        period: Rolling window period.
        num_std: Number of standard deviations for the bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band) Series.
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def returns(series: pd.Series) -> pd.Series:
    """Compute simple period-over-period percentage returns.

    Args:
        series: Numeric pandas Series of prices.

    Returns:
        Returns Series (decimal, e.g. 0.01 = 1%).
    """
    return series.pct_change()


def volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling annualised volatility (std of returns).

    Args:
        series: Numeric pandas Series of prices.
        window: Rolling window in candles.

    Returns:
        Volatility Series (annualised using sqrt(252*390) for 1-min candles).
    """
    ret = returns(series)
    # For 1-minute data: 252 trading days * 390 minutes/day
    ann_factor = np.sqrt(252 * 390)
    return ret.rolling(window=window).std() * ann_factor


# ---------------------------------------------------------------------------
# High-level function: add all indicators to a combined DataFrame
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all technical indicators to the market data DataFrame.

    Processes each ticker independently so that rolling calculations do not
    bleed across symbols.

    Args:
        df: Combined OHLCV DataFrame with columns
            [Datetime, Open, High, Low, Close, Volume, Ticker].

    Returns:
        DataFrame with additional columns:
            EMA_fast, EMA_slow, RSI, BB_upper, BB_middle, BB_lower,
            Returns, Volatility.
    """
    if df.empty:
        logger.warning("add_indicators called with an empty DataFrame.")
        return df

    results: list[pd.DataFrame] = []

    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.copy().sort_values("Datetime").reset_index(drop=True)
        close = g["Close"]

        g[f"EMA_{config.EMA_FAST}"] = ema(close, config.EMA_FAST)
        g[f"EMA_{config.EMA_SLOW}"] = ema(close, config.EMA_SLOW)
        g["RSI"] = rsi(close, config.RSI_PERIOD)

        bb_upper, bb_mid, bb_lower = bollinger_bands(
            close, config.BOLLINGER_PERIOD, config.BOLLINGER_STD
        )
        g["BB_upper"] = bb_upper
        g["BB_middle"] = bb_mid
        g["BB_lower"] = bb_lower

        g["Returns"] = returns(close)
        g["Volatility"] = volatility(close)

        results.append(g)
        logger.debug("Indicators computed for %s (%d rows)", ticker, len(g))

    out = pd.concat(results, ignore_index=True)
    out.sort_values(["Datetime", "Ticker"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    logger.info("Indicators added. DataFrame shape: %s", out.shape)
    return out
