"""
config.py - Central configuration for the PSE Trading Bot.

Contains tickers, API settings, data intervals, file paths, virtual capital,
and risk management parameters.
"""

import os

# ---------------------------------------------------------------------------
# Target PSE tickers (Yahoo Finance format)
# ---------------------------------------------------------------------------
TICKERS = [
    "BDO.PS",   # BDO Unibank, Inc.
    "SM.PS",    # SM Investments Corporation
    "ALI.PS",   # Ayala Land, Inc.
    "JFC.PS",   # Jollibee Foods Corporation
    "AC.PS",    # Ayala Corporation
    "TEL.PS",   # PLDT, Inc.
]

# ---------------------------------------------------------------------------
# Market data settings
# ---------------------------------------------------------------------------
DATA_INTERVAL = "1m"          # 1-minute candles
DATA_PERIOD = "1d"            # Fetch last trading day by default
BACKTEST_PERIOD = "5d"        # Historical period for backtesting (max 7d for 1m)
DATA_SOURCE = "yfinance"      # Market data provider

# ---------------------------------------------------------------------------
# Backtest date range (used when IS_BACKTEST = True)
# ---------------------------------------------------------------------------
BACKTEST_START_DATE = None    # Simulation start date (date object or None)
BACKTEST_END_DATE = None      # Simulation end date   (date object or None)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "market_data_raw.csv")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "market_data_processed.csv")
TRADE_LOG_FILE = os.path.join(REPORTS_DIR, "trade_log.csv")
PORTFOLIO_LOG_FILE = os.path.join(REPORTS_DIR, "portfolio_log.csv")
BACKTEST_REPORT_FILE = os.path.join(REPORTS_DIR, "backtest_report.csv")

# SQLite database (persists portfolio state and custom tickers across sessions)
DB_DIR = os.path.join(DATA_DIR, "db")
DB_FILE = os.path.join(DB_DIR, "trading_bot.db")

# ---------------------------------------------------------------------------
# Google Cloud Storage (optional – set GCS_BUCKET_NAME to enable)
# ---------------------------------------------------------------------------
GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME", "")
GCS_DB_BLOB_NAME: str = os.environ.get("GCS_DB_BLOB_NAME", "pse_trading_bot.db")

# ---------------------------------------------------------------------------
# Virtual portfolio settings
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000.0   # PHP 1,000,000 virtual starting capital

# ---------------------------------------------------------------------------
# Risk management parameters
# ---------------------------------------------------------------------------
MAX_POSITION_PCT = 0.05         # Maximum 5% of capital per single trade
STOP_LOSS_PCT = 0.02            # Stop-loss at 2% below entry price
TAKE_PROFIT_PCT = 0.04          # Take-profit at 4% above entry price
MAX_DAILY_LOSS_PCT = 0.03       # Halt trading if daily loss exceeds 3% of capital

# ---------------------------------------------------------------------------
# Trading strategy parameters (EMA crossover)
# ---------------------------------------------------------------------------
EMA_FAST = 9                    # Fast EMA period
EMA_SLOW = 21                   # Slow EMA period
RSI_PERIOD = 14                 # RSI look-back period
RSI_OVERBOUGHT = 70             # RSI overbought threshold
RSI_OVERSOLD = 30               # RSI oversold threshold
BOLLINGER_PERIOD = 20           # Bollinger Bands look-back period
BOLLINGER_STD = 2.0             # Bollinger Bands standard deviation multiplier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "trading_bot.log")
