"""
db.py - SQLite3 database manager for the PSE Trading Bot.

Provides persistent storage for portfolio state, open positions, trade log,
and custom tickers across Streamlit session resets.  An optional
Google Cloud Storage (GCS) sync layer can be activated by setting the
``GCS_BUCKET_NAME`` environment variable (see ``gcs_sync.py``).

Usage::

    from db import DatabaseManager
    db = DatabaseManager()          # uses config.DB_FILE by default
    db.save_portfolio(portfolio)
    portfolio = db.load_portfolio()
    db.close()
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import config
from portfolio import Portfolio, Position, TradeRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS portfolio_state (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    initial_capital     REAL    NOT NULL,
    cash                REAL    NOT NULL,
    daily_realized_pnl  REAL    NOT NULL DEFAULT 0.0,
    updated_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    ticker      TEXT    PRIMARY KEY,
    shares      REAL    NOT NULL,
    avg_cost    REAL    NOT NULL,
    entry_time  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    shares      REAL    NOT NULL,
    price       REAL    NOT NULL,
    cost        REAL    NOT NULL,
    pnl         REAL    NOT NULL,
    notes       TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS custom_tickers (
    ticker  TEXT PRIMARY KEY
);
"""


class DatabaseManager:
    """SQLite3-backed persistence layer for the trading bot.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``config.DB_FILE``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or config.DB_FILE
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()
        logger.info("DatabaseManager initialised: %s", self._db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables if they do not already exist."""
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def save_portfolio(self, portfolio: Portfolio) -> None:
        """Persist the full portfolio state (positions + trade log + cash).

        Args:
            portfolio: The :class:`~portfolio.Portfolio` instance to save.
        """
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()

        # --- portfolio_state (single-row table, id=1) ---
        cur.execute(
            """
            INSERT INTO portfolio_state (id, initial_capital, cash, daily_realized_pnl, updated_at)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                initial_capital    = excluded.initial_capital,
                cash               = excluded.cash,
                daily_realized_pnl = excluded.daily_realized_pnl,
                updated_at         = excluded.updated_at
            """,
            (
                portfolio.initial_capital,
                portfolio.cash,
                portfolio.daily_realized_pnl,
                now,
            ),
        )

        # --- positions (replace-all strategy) ---
        cur.execute("DELETE FROM positions")
        cur.executemany(
            "INSERT INTO positions (ticker, shares, avg_cost, entry_time) VALUES (?, ?, ?, ?)",
            [
                (ticker, pos.shares, pos.avg_cost, pos.entry_time.isoformat())
                for ticker, pos in portfolio.positions.items()
            ],
        )

        # --- trade_log (replace-all strategy) ---
        cur.execute("DELETE FROM trade_log")
        cur.executemany(
            """
            INSERT INTO trade_log (timestamp, ticker, action, shares, price, cost, pnl, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rec.timestamp.isoformat(),
                    rec.ticker,
                    rec.action,
                    rec.shares,
                    rec.price,
                    rec.cost,
                    rec.pnl,
                    rec.notes,
                )
                for rec in portfolio.trade_log
            ],
        )

        self._conn.commit()
        logger.info("Portfolio saved to DB (%d positions, %d trades)", len(portfolio.positions), len(portfolio.trade_log))

    def load_portfolio(self) -> Optional[Portfolio]:
        """Restore a Portfolio from the database.

        Returns:
            A fully-restored :class:`~portfolio.Portfolio`, or ``None`` if no
            state has been saved yet.
        """
        cur = self._conn.cursor()
        row = cur.execute("SELECT initial_capital, cash, daily_realized_pnl FROM portfolio_state WHERE id = 1").fetchone()
        if row is None:
            return None

        initial_capital, cash, daily_pnl = row
        portfolio = Portfolio(initial_capital)
        portfolio.cash = cash
        portfolio.set_daily_realized_pnl(daily_pnl)

        # --- positions ---
        for ticker, shares, avg_cost, entry_time_str in cur.execute(
            "SELECT ticker, shares, avg_cost, entry_time FROM positions"
        ):
            portfolio.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=avg_cost,
                entry_time=datetime.fromisoformat(entry_time_str),
            )

        # --- trade_log ---
        for row in cur.execute(
            "SELECT timestamp, ticker, action, shares, price, cost, pnl, notes FROM trade_log ORDER BY id"
        ):
            ts_str, ticker, action, shares, price, cost, pnl, notes = row
            portfolio.trade_log.append(
                TradeRecord(
                    timestamp=datetime.fromisoformat(ts_str),
                    ticker=ticker,
                    action=action,
                    shares=shares,
                    price=price,
                    cost=cost,
                    pnl=pnl,
                    notes=notes,
                )
            )

        logger.info(
            "Portfolio loaded from DB (cash=%.2f, %d positions, %d trades)",
            portfolio.cash,
            len(portfolio.positions),
            len(portfolio.trade_log),
        )
        return portfolio

    # ------------------------------------------------------------------
    # Custom tickers
    # ------------------------------------------------------------------

    def save_custom_tickers(self, tickers: list[str]) -> None:
        """Persist the user-added custom ticker list.

        Args:
            tickers: List of Yahoo Finance ticker symbols.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM custom_tickers")
        cur.executemany("INSERT INTO custom_tickers (ticker) VALUES (?)", [(t,) for t in tickers])
        self._conn.commit()
        logger.info("Custom tickers saved: %s", tickers)

    def load_custom_tickers(self) -> list[str]:
        """Load the persisted custom ticker list.

        Returns:
            List of ticker symbols, or an empty list if none have been saved.
        """
        rows = self._conn.execute("SELECT ticker FROM custom_tickers ORDER BY ticker").fetchall()
        tickers = [r[0] for r in rows]
        logger.info("Custom tickers loaded: %s", tickers)
        return tickers

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> str:
        """Absolute path to the underlying SQLite file."""
        return self._db_path

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
        logger.info("DatabaseManager closed: %s", self._db_path)
