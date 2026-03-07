"""
trading_agent.py - Trading strategy implementation for the PSE Trading Bot.

Implements three rule-based strategies and provides an extensible base class
so that additional strategies (including AI/ML models) can be plugged in by
subclassing ``BaseStrategy``.

Strategies
----------
* :class:`EMACrossoverStrategy` – Golden / death cross on fast and slow EMAs.
* :class:`RSIStrategy` – Mean-reversion entries via RSI threshold crossovers.
* :class:`BollingerBandStrategy` – Mean-reversion entries via Bollinger Band
  crossovers.

The :data:`STRATEGY_REGISTRY` mapping provides a convenient way to look up a
strategy instance by its human-readable name.
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd

import config
import risk_management as rm
from portfolio import Portfolio

logger = logging.getLogger(__name__)

Signal = Literal["BUY", "SELL", "HOLD"]


# ---------------------------------------------------------------------------
# Abstract base strategy – all strategies must implement generate_signal()
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Subclass this and implement :meth:`generate_signal` to add new strategies.
    Override :attr:`lag_columns` to declare which indicator columns need a
    one-period lag added by :meth:`TradingAgent.prepare_signals_df`.
    """

    @property
    def lag_columns(self) -> list[str]:
        """Indicator column names that require a one-period lag.

        :meth:`TradingAgent.prepare_signals_df` shifts each column listed here
        and stores the result in a ``prev_<col>`` column so that crossover
        detection works correctly.

        Returns:
            List of column name strings (default: empty list).
        """
        return []

    @abstractmethod
    def generate_signal(self, row: pd.Series, ticker: str) -> Signal:
        """Generate a trading signal for a single candle.

        Args:
            row: A single row from the processed DataFrame (must contain all
                 indicator columns).
            ticker: Ticker symbol for the current row.

        Returns:
            "BUY", "SELL", or "HOLD".
        """
        ...


# ---------------------------------------------------------------------------
# EMA crossover strategy
# ---------------------------------------------------------------------------

class EMACrossoverStrategy(BaseStrategy):
    """Simple EMA crossover strategy.

    **BUY**  when the fast EMA crosses *above* the slow EMA.
    **SELL** when the fast EMA crosses *below* the slow EMA.
    **HOLD** otherwise.

    Args:
        fast_period: Fast EMA period (default: EMA_FAST from config).
        slow_period: Slow EMA period (default: EMA_SLOW from config).
    """

    def __init__(
        self,
        fast_period: int = config.EMA_FAST,
        slow_period: int = config.EMA_SLOW,
    ) -> None:
        self.fast_col = f"EMA_{fast_period}"
        self.slow_col = f"EMA_{slow_period}"

    @property
    def lag_columns(self) -> list[str]:
        return [self.fast_col, self.slow_col]

    def generate_signal(self, row: pd.Series, ticker: str) -> Signal:
        """Generate a signal based on EMA crossover.

        Requires columns *EMA_fast*, *EMA_slow*, *prev_EMA_fast*, and
        *prev_EMA_slow* to detect a crossover.  Falls back to HOLD if any
        column is missing or NaN.

        Args:
            row: Current candle row (must contain the EMA columns and their
                 lagged *prev_* counterparts).
            ticker: Ticker symbol (used for logging).

        Returns:
            "BUY", "SELL", or "HOLD".
        """
        try:
            fast = row[self.fast_col]
            slow = row[self.slow_col]
            prev_fast = row[f"prev_{self.fast_col}"]
            prev_slow = row[f"prev_{self.slow_col}"]
        except KeyError:
            return "HOLD"

        if any(pd.isna(v) for v in [fast, slow, prev_fast, prev_slow]):
            return "HOLD"

        # Golden cross – fast crosses above slow
        if prev_fast <= prev_slow and fast > slow:
            logger.debug("BUY signal for %s (EMA crossover up)", ticker)
            return "BUY"

        # Death cross – fast crosses below slow
        if prev_fast >= prev_slow and fast < slow:
            logger.debug("SELL signal for %s (EMA crossover down)", ticker)
            return "SELL"

        return "HOLD"


# ---------------------------------------------------------------------------
# RSI mean-reversion strategy
# ---------------------------------------------------------------------------

class RSIStrategy(BaseStrategy):
    """RSI mean-reversion strategy.

    **BUY**  when the RSI crosses back *above* the oversold threshold
             (price recovering from an oversold dip).
    **SELL** when the RSI crosses back *below* the overbought threshold
             (price retreating from an overbought peak).
    **HOLD** otherwise.

    Args:
        oversold: RSI level considered oversold (default: RSI_OVERSOLD from config).
        overbought: RSI level considered overbought (default: RSI_OVERBOUGHT from config).
    """

    def __init__(
        self,
        oversold: float = config.RSI_OVERSOLD,
        overbought: float = config.RSI_OVERBOUGHT,
    ) -> None:
        self.oversold = oversold
        self.overbought = overbought

    @property
    def lag_columns(self) -> list[str]:
        return ["RSI"]

    def generate_signal(self, row: pd.Series, ticker: str) -> Signal:
        """Generate a signal based on RSI threshold crossovers.

        Requires columns *RSI* and *prev_RSI*.  Falls back to HOLD if any
        column is missing or NaN.

        Args:
            row: Current candle row.
            ticker: Ticker symbol (used for logging).

        Returns:
            "BUY", "SELL", or "HOLD".
        """
        try:
            rsi_val = row["RSI"]
            prev_rsi = row["prev_RSI"]
        except KeyError:
            return "HOLD"

        if any(pd.isna(v) for v in [rsi_val, prev_rsi]):
            return "HOLD"

        # Recovery from oversold: RSI crosses back above the oversold threshold
        if prev_rsi <= self.oversold < rsi_val:
            logger.debug("BUY signal for %s (RSI recovery: %.1f → %.1f)", ticker, prev_rsi, rsi_val)
            return "BUY"

        # Retreat from overbought: RSI crosses back below the overbought threshold
        if prev_rsi >= self.overbought > rsi_val:
            logger.debug("SELL signal for %s (RSI retreat: %.1f → %.1f)", ticker, prev_rsi, rsi_val)
            return "SELL"

        return "HOLD"


# ---------------------------------------------------------------------------
# Bollinger Bands mean-reversion strategy
# ---------------------------------------------------------------------------

class BollingerBandStrategy(BaseStrategy):
    """Bollinger Bands mean-reversion strategy.

    **BUY**  when the Close price crosses back *above* the lower Bollinger Band
             (price recovering from an oversold dip below the band).
    **SELL** when the Close price crosses back *below* the upper Bollinger Band
             (price retreating from an overbought peak above the band).
    **HOLD** otherwise.
    """

    @property
    def lag_columns(self) -> list[str]:
        return ["Close", "BB_upper", "BB_lower"]

    def generate_signal(self, row: pd.Series, ticker: str) -> Signal:
        """Generate a signal based on Bollinger Band crossovers.

        Requires columns *Close*, *BB_upper*, *BB_lower* and their ``prev_``
        counterparts.  Falls back to HOLD if any column is missing or NaN.

        Args:
            row: Current candle row.
            ticker: Ticker symbol (used for logging).

        Returns:
            "BUY", "SELL", or "HOLD".
        """
        try:
            close = row["Close"]
            bb_upper = row["BB_upper"]
            bb_lower = row["BB_lower"]
            prev_close = row["prev_Close"]
            prev_bb_upper = row["prev_BB_upper"]
            prev_bb_lower = row["prev_BB_lower"]
        except KeyError:
            return "HOLD"

        if any(pd.isna(v) for v in [close, bb_upper, bb_lower, prev_close, prev_bb_upper, prev_bb_lower]):
            return "HOLD"

        # Recovery: close was below the lower band and crosses back above it
        if prev_close < prev_bb_lower and close >= bb_lower:
            logger.debug(
                "BUY signal for %s (BB recovery: %.4f → %.4f, lower=%.4f)",
                ticker, prev_close, close, bb_lower,
            )
            return "BUY"

        # Retreat: close was above the upper band and crosses back below it
        if prev_close > prev_bb_upper and close <= bb_upper:
            logger.debug(
                "SELL signal for %s (BB retreat: %.4f → %.4f, upper=%.4f)",
                ticker, prev_close, close, bb_upper,
            )
            return "SELL"

        return "HOLD"


# ---------------------------------------------------------------------------
# Strategy registry – maps display name → strategy instance
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, BaseStrategy] = {
    "EMA Crossover": EMACrossoverStrategy(),
    "RSI Mean-Reversion": RSIStrategy(),
    "Bollinger Bands": BollingerBandStrategy(),
}


# ---------------------------------------------------------------------------
# Trading agent – wraps strategy + risk management
# ---------------------------------------------------------------------------

class TradingAgent:
    """Orchestrates signal generation, risk checks, and order placement.

    Args:
        portfolio: Active :class:`~portfolio.Portfolio` instance.
        strategy: A :class:`BaseStrategy` instance (default: EMA crossover).
    """

    def __init__(
        self,
        portfolio: Portfolio,
        strategy: BaseStrategy | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.strategy = strategy or EMACrossoverStrategy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_signals_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged columns required by the active strategy.

        Reads :attr:`BaseStrategy.lag_columns` from the current strategy and
        shifts each listed column by one period (per-ticker) to produce
        ``prev_<col>`` columns used for crossover detection.

        Args:
            df: Processed DataFrame with indicator columns.

        Returns:
            DataFrame with additional ``prev_`` columns for each lag column.
        """
        lag_cols = self.strategy.lag_columns
        if not lag_cols:
            return df

        missing = [c for c in lag_cols if c not in df.columns]
        if missing:
            logger.warning(
                "Columns missing for strategy lag: %s – run indicators.add_indicators() first.",
                missing,
            )
            return df

        out_frames = []
        for _, group in df.groupby("Ticker", sort=False):
            g = group.sort_values("Datetime").copy()
            for col in lag_cols:
                g[f"prev_{col}"] = g[col].shift(1)
            out_frames.append(g)

        result = pd.concat(out_frames, ignore_index=True)
        result.sort_values(["Datetime", "Ticker"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the strategy row-by-row on the prepared DataFrame.

        For each candle the agent:
        1. Generates a raw signal from the strategy.
        2. Applies risk management overrides (stop-loss, daily loss limit).
        3. Executes a virtual BUY or SELL via the portfolio.

        Args:
            df: Processed and signal-ready DataFrame (output of
                :meth:`prepare_signals_df`).

        Returns:
            Original DataFrame with an added *Signal* column.
        """
        df = df.copy()
        df["Signal"] = "HOLD"
        halted = False

        for idx, row in df.iterrows():
            ticker = row["Ticker"]
            price = row["Close"]
            ts = row["Datetime"]

            if halted:
                df.at[idx, "Signal"] = "HALT"
                continue

            # 1. Strategy signal
            raw_signal: Signal = self.strategy.generate_signal(row, ticker)

            # 2. Risk override
            risk_action = rm.apply_risk_checks(ticker, self.portfolio, price, ts)

            if risk_action == "HALT":
                halted = True
                df.at[idx, "Signal"] = "HALT"
                logger.warning("Daily loss limit reached – halting all trades.")
                continue

            if risk_action == "SELL":
                final_signal: Signal = "SELL"
            else:
                final_signal = raw_signal

            df.at[idx, "Signal"] = final_signal

            # 3. Execute trade
            if final_signal == "BUY" and ticker not in self.portfolio.positions:
                shares = rm.compute_position_size(self.portfolio.cash, price)
                if shares > 0:
                    self.portfolio.buy(ticker, shares, price, ts, notes=type(self.strategy).__name__)

            elif final_signal == "SELL" and ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                self.portfolio.sell(ticker, pos.shares, price, ts, notes=risk_action)

        return df
