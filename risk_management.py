"""
risk_management.py - Risk management for the PSE Trading Bot.

Implements position sizing, stop-loss checking, and daily loss limits.
"""

import logging
from datetime import datetime

import config
from portfolio import Portfolio

logger = logging.getLogger(__name__)


def compute_position_size(
    capital: float,
    price: float,
    max_pct: float = config.MAX_POSITION_PCT,
) -> float:
    """Compute the number of shares to buy based on capital allocation.

    Limits the trade to *max_pct* of available capital and returns the whole
    number of shares that can be purchased.

    Args:
        capital: Available capital in PHP.
        price: Current price per share.
        max_pct: Maximum fraction of capital to allocate (default 5%).

    Returns:
        Number of shares (floored to nearest whole share). 0 if price <= 0.
    """
    if price <= 0:
        return 0.0
    max_spend = capital * max_pct
    shares = max_spend / price
    return float(int(shares))  # whole shares only


def check_stop_loss(
    entry_price: float,
    current_price: float,
    stop_loss_pct: float = config.STOP_LOSS_PCT,
) -> bool:
    """Return True if the current price has fallen below the stop-loss level.

    Args:
        entry_price: Average cost / entry price per share.
        current_price: Latest market price per share.
        stop_loss_pct: Stop-loss threshold (e.g. 0.02 = 2%).

    Returns:
        True if stop-loss is triggered, False otherwise.
    """
    stop_level = entry_price * (1 - stop_loss_pct)
    triggered = current_price <= stop_level
    if triggered:
        logger.debug(
            "Stop-loss triggered: entry=%.4f  current=%.4f  level=%.4f",
            entry_price, current_price, stop_level,
        )
    return triggered


def check_take_profit(
    entry_price: float,
    current_price: float,
    take_profit_pct: float = config.TAKE_PROFIT_PCT,
) -> bool:
    """Return True if the current price has reached the take-profit level.

    Args:
        entry_price: Average cost / entry price per share.
        current_price: Latest market price per share.
        take_profit_pct: Take-profit threshold (e.g. 0.04 = 4%).

    Returns:
        True if take-profit is triggered, False otherwise.
    """
    tp_level = entry_price * (1 + take_profit_pct)
    triggered = current_price >= tp_level
    if triggered:
        logger.debug(
            "Take-profit triggered: entry=%.4f  current=%.4f  level=%.4f",
            entry_price, current_price, tp_level,
        )
    return triggered


def check_daily_loss_limit(
    portfolio: Portfolio,
    max_daily_loss_pct: float = config.MAX_DAILY_LOSS_PCT,
) -> bool:
    """Return True if the maximum daily loss limit has been exceeded.

    Compares the daily realized P&L against a fraction of initial capital.

    Args:
        portfolio: Active Portfolio instance.
        max_daily_loss_pct: Maximum tolerated daily loss as a fraction of
            initial capital (e.g. 0.03 = 3%).

    Returns:
        True if trading should be halted for the day, False otherwise.
    """
    limit = -portfolio.initial_capital * max_daily_loss_pct
    exceeded = portfolio.daily_realized_pnl <= limit
    if exceeded:
        logger.warning(
            "Daily loss limit reached: realized P&L=%.2f  limit=%.2f",
            portfolio.daily_realized_pnl, limit,
        )
    return exceeded


def apply_risk_checks(
    ticker: str,
    portfolio: Portfolio,
    current_price: float,
    timestamp: datetime,
) -> str:
    """Apply all risk rules and return a forced action if necessary.

    Checks (in order):
    1. Daily loss limit → HALT (no new trades allowed).
    2. Stop-loss for each open position → SELL.
    3. Take-profit for each open position → SELL.

    Args:
        ticker: Ticker to evaluate.
        portfolio: Active Portfolio instance.
        current_price: Latest price for the ticker.
        timestamp: Current candle timestamp.

    Returns:
        "HALT", "SELL", or "NONE" (no forced action required).
    """
    # 1. Daily loss limit
    if check_daily_loss_limit(portfolio):
        return "HALT"

    # 2 & 3. Check open position for this ticker
    pos = portfolio.positions.get(ticker)
    if pos is not None:
        if check_stop_loss(pos.avg_cost, current_price):
            logger.info("Risk: stop-loss SELL for %s @ %.4f", ticker, current_price)
            return "SELL"
        if check_take_profit(pos.avg_cost, current_price):
            logger.info("Risk: take-profit SELL for %s @ %.4f", ticker, current_price)
            return "SELL"

    return "NONE"
