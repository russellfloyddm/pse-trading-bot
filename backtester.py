"""
backtester.py - Historical backtesting engine for the PSE Trading Bot.

Simulates the full trade loop on historical minute-level data and produces
performance metrics: total return, Sharpe ratio, maximum drawdown, and a
per-trade summary.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
import indicators
import storage
from portfolio import Portfolio
from trading_agent import EMACrossoverStrategy, TradingAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 252 * 390) -> float:
    """Annualised Sharpe ratio (assumes risk-free rate = 0).

    Args:
        returns: Per-candle portfolio return series.
        periods_per_year: Number of candles per year (default: 1-min candles).

    Returns:
        Sharpe ratio (float).  Returns 0.0 if std is zero.
    """
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown from an equity curve.

    Args:
        equity_curve: Series of portfolio total values over time.

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.05 = -5%).
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Simulate the trading strategy on historical data and report performance.

    Args:
        initial_capital: Starting virtual capital in PHP.
        strategy: Optional custom strategy instance.  Defaults to EMA crossover.
    """

    def __init__(
        self,
        initial_capital: float = config.INITIAL_CAPITAL,
        strategy: Optional[object] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.strategy = strategy or EMACrossoverStrategy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> dict:
        """Run the backtest on a processed DataFrame.

        Args:
            df: Processed DataFrame with all indicator columns
                (output of :func:`indicators.add_indicators`).
                Must contain columns: Datetime, Close, Ticker, EMA_fast,
                EMA_slow.

        Returns:
            Dictionary with keys:
                - ``trade_log``: DataFrame of all executed trades.
                - ``equity_curve``: Series of portfolio value per unique timestamp.
                - ``total_return_pct``: Total return as a percentage.
                - ``sharpe_ratio``: Annualised Sharpe ratio.
                - ``max_drawdown``: Maximum drawdown (negative decimal).
                - ``total_trades``: Number of executed trades.
                - ``winning_trades``: Number of trades with positive P&L.
                - ``win_rate``: Win rate as a decimal.
                - ``summary_df``: Per-ticker summary DataFrame.
        """
        portfolio = Portfolio(self.initial_capital)
        agent = TradingAgent(portfolio, self.strategy)

        # Prepare lagged EMA columns for crossover detection
        df_ready = agent.prepare_signals_df(df)

        # Run the strategy
        df_signals = agent.run(df_ready)

        # Build equity curve (portfolio value at each minute)
        equity_points = self._build_equity_curve(df_signals, portfolio)

        # Compute metrics
        trade_log = portfolio.to_trade_log_df()
        equity_series = pd.Series(equity_points)
        ret_series = equity_series.pct_change().dropna()

        final_value = equity_series.iloc[-1] if len(equity_series) else self.initial_capital
        total_return_pct = (final_value - self.initial_capital) / self.initial_capital * 100

        metrics = {
            "trade_log": trade_log,
            "equity_curve": equity_series,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": _sharpe_ratio(ret_series),
            "max_drawdown": _max_drawdown(equity_series),
            "total_trades": len(trade_log),
            "winning_trades": int((trade_log["pnl"] > 0).sum()) if not trade_log.empty else 0,
            "win_rate": (
                float((trade_log["pnl"] > 0).mean()) if not trade_log.empty else 0.0
            ),
            "summary_df": self._per_ticker_summary(trade_log),
        }

        logger.info(
            "Backtest complete: return=%.2f%%  Sharpe=%.3f  MaxDD=%.2f%%  trades=%d",
            total_return_pct,
            metrics["sharpe_ratio"],
            metrics["max_drawdown"] * 100,
            metrics["total_trades"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_equity_curve(
        self, df: pd.DataFrame, portfolio: Portfolio
    ) -> list[float]:
        """Build an equity curve by tracking portfolio value at each timestamp.

        Args:
            df: Signal DataFrame (output of :meth:`TradingAgent.run`).
            portfolio: Executed portfolio (after agent.run).

        Returns:
            List of portfolio values, one per unique Datetime in the data.
        """
        equity: list[float] = []
        for ts, group in df.groupby("Datetime", sort=True):
            prices = dict(zip(group["Ticker"], group["Close"]))
            equity.append(portfolio.market_value(prices))
        return equity

    def _per_ticker_summary(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        """Build a per-ticker trade summary.

        Args:
            trade_log: Trade log DataFrame from :meth:`Portfolio.to_trade_log_df`.

        Returns:
            DataFrame summarising trade counts and P&L per ticker.
        """
        if trade_log.empty:
            return pd.DataFrame(
                columns=["ticker", "total_trades", "winning_trades", "total_pnl"]
            )
        sells = trade_log[trade_log["action"] == "SELL"]
        summary = (
            sells.groupby("ticker")
            .agg(
                total_trades=("pnl", "count"),
                winning_trades=("pnl", lambda x: (x > 0).sum()),
                total_pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        return summary

    def save_report(self, metrics: dict) -> None:
        """Persist backtest results to the configured reports directory.

        Saves the per-ticker summary CSV and the trade log CSV.

        Args:
            metrics: Dictionary returned by :meth:`run`.
        """
        summary = metrics.get("summary_df")
        if summary is not None and not summary.empty:
            storage.save_csv(summary, config.BACKTEST_REPORT_FILE)

        trade_log = metrics.get("trade_log")
        if trade_log is not None and not trade_log.empty:
            storage.save_trade_log(trade_log)

        logger.info("Backtest report saved to %s", config.REPORTS_DIR)

    def print_report(self, metrics: dict) -> None:
        """Print a human-readable summary of backtest performance.

        Args:
            metrics: Dictionary returned by :meth:`run`.
        """
        print("\n" + "=" * 60)
        print("  PSE Trading Bot – Backtest Report")
        print("=" * 60)
        print(f"  Initial Capital : PHP {self.initial_capital:,.2f}")
        print(f"  Total Return    : {metrics['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio    : {metrics['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown    : {metrics['max_drawdown'] * 100:.2f}%")
        print(f"  Total Trades    : {metrics['total_trades']}")
        print(f"  Winning Trades  : {metrics['winning_trades']}")
        win_rate = metrics['win_rate'] * 100
        print(f"  Win Rate        : {win_rate:.1f}%")
        print("-" * 60)
        summary = metrics.get("summary_df")
        if summary is not None and not summary.empty:
            print("  Per-Ticker Summary:")
            print(summary.to_string(index=False))
        print("=" * 60 + "\n")
