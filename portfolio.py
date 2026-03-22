"""
portfolio.py - Virtual portfolio management for the PSE Trading Bot.

Tracks cash balance, open positions, trade history, and computes realized /
unrealized P&L.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position for a single ticker.

    Attributes:
        ticker: Stock ticker symbol.
        shares: Number of shares held (positive = long).
        avg_cost: Average cost per share (PHP).
        entry_time: Timestamp of the first purchase.
    """
    ticker: str
    shares: float
    avg_cost: float
    entry_time: datetime


@dataclass
class TradeRecord:
    """A single executed trade log entry.

    Attributes:
        timestamp: When the trade was executed.
        ticker: Stock ticker.
        action: "BUY" or "SELL".
        shares: Number of shares traded.
        price: Execution price per share.
        cost: Total cost/proceeds (price * shares, negative for buys).
        pnl: Realized P&L for this trade (0 for buys).
        notes: Optional notes (e.g. "stop-loss triggered").
    """
    timestamp: datetime
    ticker: str
    action: str
    shares: float
    price: float
    cost: float
    pnl: float
    notes: str = ""


class Portfolio:
    """Manages virtual cash, positions, and trade history.

    Args:
        initial_capital: Starting virtual capital in PHP.
    """

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL) -> None:
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_log: list[TradeRecord] = []
        self._daily_realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions at avg cost as fallback)."""
        pos_value = sum(p.shares * p.avg_cost for p in self.positions.values())
        return self.cash + pos_value

    def market_value(self, market_prices: dict[str, float]) -> float:
        """Total portfolio value using live market prices.

        Args:
            market_prices: Mapping of ticker -> current price.

        Returns:
            Total portfolio value in PHP.
        """
        pos_value = sum(
            p.shares * market_prices.get(p.ticker, p.avg_cost)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def unrealized_pnl(self, market_prices: dict[str, float]) -> dict[str, float]:
        """Compute unrealized P&L per open position.

        Args:
            market_prices: Mapping of ticker -> current price.

        Returns:
            Dict mapping ticker -> unrealized P&L in PHP.
        """
        result: dict[str, float] = {}
        for ticker, pos in self.positions.items():
            current_price = market_prices.get(ticker, pos.avg_cost)
            result[ticker] = (current_price - pos.avg_cost) * pos.shares
        return result

    @property
    def daily_realized_pnl(self) -> float:
        """Net realized P&L for the current trading session."""
        return self._daily_realized_pnl

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        timestamp: datetime,
        notes: str = "",
    ) -> bool:
        """Execute a virtual BUY order.

        Args:
            ticker: Stock ticker symbol.
            shares: Number of shares to buy.
            price: Execution price per share.
            timestamp: Trade timestamp.
            notes: Optional descriptive notes.

        Returns:
            True if the order was executed, False if insufficient cash.
        """
        cost = shares * price
        if cost > self.cash:
            logger.warning(
                "BUY rejected for %s: cost %.2f > available cash %.2f",
                ticker, cost, self.cash,
            )
            return False

        self.cash -= cost

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
        else:
            self.positions[ticker] = Position(
                ticker=ticker, shares=shares, avg_cost=price, entry_time=timestamp
            )

        record = TradeRecord(
            timestamp=timestamp,
            ticker=ticker,
            action="BUY",
            shares=shares,
            price=price,
            cost=-cost,
            pnl=0.0,
            notes=notes,
        )
        self.trade_log.append(record)
        logger.info("BUY  %s  %.0f shares @ %.4f  (cash left: %.2f)", ticker, shares, price, self.cash)
        return True

    def sell(
        self,
        ticker: str,
        shares: float,
        price: float,
        timestamp: datetime,
        notes: str = "",
    ) -> bool:
        """Execute a virtual SELL order.

        Args:
            ticker: Stock ticker symbol.
            shares: Number of shares to sell.
            price: Execution price per share.
            timestamp: Trade timestamp.
            notes: Optional descriptive notes.

        Returns:
            True if the order was executed, False if insufficient shares.
        """
        if ticker not in self.positions or self.positions[ticker].shares < shares:
            logger.warning(
                "SELL rejected for %s: not enough shares (have %.0f, need %.0f)",
                ticker,
                self.positions.get(ticker, Position(ticker, 0, 0, timestamp)).shares,
                shares,
            )
            return False

        pos = self.positions[ticker]
        proceeds = shares * price
        realized = (price - pos.avg_cost) * shares

        self.cash += proceeds
        self._daily_realized_pnl += realized

        pos.shares -= shares
        if pos.shares <= 0:
            del self.positions[ticker]

        record = TradeRecord(
            timestamp=timestamp,
            ticker=ticker,
            action="SELL",
            shares=shares,
            price=price,
            cost=proceeds,
            pnl=realized,
            notes=notes,
        )
        self.trade_log.append(record)
        logger.info(
            "SELL %s  %.0f shares @ %.4f  realized P&L: %.2f  (cash: %.2f)",
            ticker, shares, price, realized, self.cash,
        )
        return True

    def reset_daily_pnl(self) -> None:
        """Reset the daily realized P&L counter (call at start of new day)."""
        self._daily_realized_pnl = 0.0

    def set_daily_realized_pnl(self, value: float) -> None:
        """Set the daily realized P&L counter (used when restoring persisted state).

        Args:
            value: The daily realized P&L value to restore.
        """
        self._daily_realized_pnl = value

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_trade_log_df(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame(
                columns=["timestamp", "ticker", "action", "shares", "price", "cost", "pnl", "notes"]
            )
        return pd.DataFrame([vars(t) for t in self.trade_log])

    def summary(self, market_prices: Optional[dict[str, float]] = None) -> dict:
        """Return a snapshot of current portfolio state.

        Args:
            market_prices: Optional mapping of ticker -> current price for
                computing unrealized P&L.

        Returns:
            Dict with cash, positions, realized P&L, and total value.
        """
        prices = market_prices or {}
        total_realized = sum(t.pnl for t in self.trade_log)
        return {
            "cash": self.cash,
            "initial_capital": self.initial_capital,
            "total_realized_pnl": total_realized,
            "daily_realized_pnl": self._daily_realized_pnl,
            "open_positions": {t: {"shares": p.shares, "avg_cost": p.avg_cost}
                               for t, p in self.positions.items()},
            "unrealized_pnl": self.unrealized_pnl(prices) if prices else {},
            "market_value": self.market_value(prices) if prices else self.total_value,
            "total_return_pct": (
                (self.market_value(prices) if prices else self.total_value)
                - self.initial_capital
            ) / self.initial_capital * 100,
        }
