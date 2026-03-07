"""
trading_agent.py - Trading strategy implementation for the PSE Trading Bot.

Currently implements a rule-based EMA crossover strategy.  The design is
intentionally extensible so that AI/ML strategies can be plugged in later by
subclassing ``BaseStrategy``.
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
    """

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
        """Add lagged EMA columns required by the EMA crossover strategy.

        Groups by ticker, sorts by Datetime, and adds *prev_EMA_fast* and
        *prev_EMA_slow* shift columns.

        Args:
            df: Processed DataFrame with indicator columns.

        Returns:
            DataFrame with additional *prev_* columns.
        """
        fast_col = f"EMA_{config.EMA_FAST}"
        slow_col = f"EMA_{config.EMA_SLOW}"
        if fast_col not in df.columns or slow_col not in df.columns:
            logger.warning("EMA columns missing – run indicators.add_indicators() first.")
            return df

        out_frames = []
        for _, group in df.groupby("Ticker", sort=False):
            g = group.sort_values("Datetime").copy()
            g[f"prev_{fast_col}"] = g[fast_col].shift(1)
            g[f"prev_{slow_col}"] = g[slow_col].shift(1)
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
                    self.portfolio.buy(ticker, shares, price, ts, notes="EMA crossover")

            elif final_signal == "SELL" and ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                self.portfolio.sell(ticker, pos.shares, price, ts, notes=risk_action)

        return df
