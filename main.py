"""
main.py - Orchestration entry point for the PSE Trading Bot.

Runs the full pipeline:
  1. Fetch market data via yfinance
  2. Compute technical indicators
  3. Save raw and processed data
  4. Run the trading agent (EMA crossover + risk management)
  5. Update and log the virtual portfolio
  6. Optionally run a backtest on recent historical data
  7. Print a summary report

Usage:
    python main.py                        # fetch today's data + run agent
    python main.py --backtest             # fetch 5-day history + run backtest
    python main.py --live                 # live-update mode (1 candle per minute)
"""

import argparse
import logging
import time
from datetime import datetime

import config
import data_pipeline
import indicators
import storage
from backtester import Backtester
from portfolio import Portfolio
from trading_agent import TradingAgent

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def fetch_and_save(period: str = config.DATA_PERIOD) -> "pd.DataFrame":
    """Fetch raw market data, persist it, and return the DataFrame.

    Args:
        period: Yahoo Finance period string (e.g. "1d", "5d").

    Returns:
        Raw OHLCV DataFrame.
    """
    import pandas as pd  # local import to keep module-level imports clean

    logger.info("Fetching market data (period=%s, interval=%s)…", period, config.DATA_INTERVAL)
    raw_df = data_pipeline.fetch_all_tickers(period=period)

    if raw_df.empty:
        logger.error("No market data fetched. Exiting pipeline.")
        return raw_df

    storage.save_raw_data(raw_df)
    logger.info("Raw data saved: %d rows", len(raw_df))
    return raw_df


def compute_and_save(raw_df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute indicators on raw data, persist, and return the result.

    Args:
        raw_df: Raw OHLCV DataFrame.

    Returns:
        Processed DataFrame with indicator columns added.
    """
    logger.info("Computing technical indicators…")
    processed_df = indicators.add_indicators(raw_df)
    storage.save_processed_data(processed_df)
    logger.info("Processed data saved: %d rows, %d columns", *processed_df.shape)
    return processed_df


def run_agent(processed_df: "pd.DataFrame", portfolio: Portfolio) -> "pd.DataFrame":
    """Run the trading agent on the processed data.

    Args:
        processed_df: Indicator-enriched DataFrame.
        portfolio: Virtual portfolio instance.

    Returns:
        DataFrame with *Signal* column appended.
    """
    logger.info("Running trading agent…")
    agent = TradingAgent(portfolio)
    df_ready = agent.prepare_signals_df(processed_df)
    df_signals = agent.run(df_ready)

    buy_count = (df_signals["Signal"] == "BUY").sum()
    sell_count = (df_signals["Signal"] == "SELL").sum()
    logger.info("Signals generated – BUY: %d  SELL: %d", buy_count, sell_count)
    return df_signals


def print_portfolio_summary(portfolio: Portfolio, market_prices: dict) -> None:
    """Print a formatted portfolio snapshot to stdout.

    Args:
        portfolio: Active portfolio instance.
        market_prices: Mapping of ticker -> latest close price.
    """
    summary = portfolio.summary(market_prices)
    print("\n" + "=" * 60)
    print("  PSE Trading Bot – Portfolio Summary")
    print("=" * 60)
    print(f"  Cash             : PHP {summary['cash']:>15,.2f}")
    print(f"  Market Value     : PHP {summary['market_value']:>15,.2f}")
    print(f"  Initial Capital  : PHP {summary['initial_capital']:>15,.2f}")
    print(f"  Total Return     : {summary['total_return_pct']:>+.2f}%")
    print(f"  Realized P&L     : PHP {summary['total_realized_pnl']:>+15,.2f}")
    print("-" * 60)
    if summary["open_positions"]:
        print("  Open Positions:")
        upnl = summary.get("unrealized_pnl", {})
        for ticker, pos_info in summary["open_positions"].items():
            upnl_val = upnl.get(ticker, 0.0)
            print(
                f"    {ticker:<10} shares={pos_info['shares']:.0f}  "
                f"avg_cost={pos_info['avg_cost']:.4f}  "
                f"unrealized_pnl={upnl_val:+.2f}"
            )
    else:
        print("  No open positions.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Live-update mode
# ---------------------------------------------------------------------------

def live_mode(portfolio: Portfolio) -> None:
    """Continuously fetch the latest candle every 60 seconds and trade.

    Args:
        portfolio: Active portfolio instance.
    """
    logger.info("Entering live-update mode. Press Ctrl-C to stop.")
    agent = TradingAgent(portfolio)

    while True:
        try:
            candles = data_pipeline.get_latest_candles()
            if candles.empty:
                logger.warning("Live fetch returned no data.")
            else:
                candles_with_indicators = indicators.add_indicators(candles)
                df_ready = agent.prepare_signals_df(candles_with_indicators)
                agent.run(df_ready)
                prices = dict(zip(candles["Ticker"], candles["Close"]))
                print_portfolio_summary(portfolio, prices)

            logger.info("Sleeping 60 seconds until next candle…")
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Live mode stopped by user.")
            break
        except Exception as exc:
            logger.error("Unexpected error in live mode: %s", exc)
            time.sleep(60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSE Trading Bot – virtual capital simulator"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on recent historical data instead of today's session.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live-update mode: fetch a new candle every minute.",
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Override the data fetch period (e.g. '1d', '5d').",
    )
    args = parser.parse_args()

    logger.info("PSE Trading Bot started at %s", datetime.now().isoformat())

    portfolio = Portfolio(config.INITIAL_CAPITAL)

    # ------------------------------------------------------------------
    # Backtest mode
    # ------------------------------------------------------------------
    if args.backtest:
        period = args.period or config.BACKTEST_PERIOD
        raw_df = fetch_and_save(period=period)
        if raw_df.empty:
            return
        processed_df = compute_and_save(raw_df)
        backtester = Backtester(config.INITIAL_CAPITAL)
        metrics = backtester.run(processed_df)
        backtester.print_report(metrics)
        backtester.save_report(metrics)
        return

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------
    if args.live:
        live_mode(portfolio)
        return

    # ------------------------------------------------------------------
    # Standard single-run mode
    # ------------------------------------------------------------------
    period = args.period or config.DATA_PERIOD
    raw_df = fetch_and_save(period=period)
    if raw_df.empty:
        return

    processed_df = compute_and_save(raw_df)
    df_signals = run_agent(processed_df, portfolio)

    # Persist trade log
    trade_log_df = portfolio.to_trade_log_df()
    if not trade_log_df.empty:
        storage.save_trade_log(trade_log_df)

    # Latest prices for unrealized P&L
    latest_prices: dict[str, float] = (
        processed_df.groupby("Ticker")["Close"].last().to_dict()
    )
    print_portfolio_summary(portfolio, latest_prices)

    logger.info("PSE Trading Bot run complete.")


if __name__ == "__main__":
    main()
