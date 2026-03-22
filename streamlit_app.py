"""
streamlit_app.py - Streamlit testing interface for the PSE Trading Bot.

Provides an interactive UI to test all major system components:
  - Trading Simulation (IS_BACKTEST / CURRENT_DATE / STRATEGY)
  - Data Pipeline    (live or synthetic OHLCV data)
  - Indicators       (EMA, RSI, Bollinger Bands)
  - Trading Signals  (configurable strategy)
  - Portfolio        (cash, positions, P&L)
  - Backtest         (equity curve, performance metrics)
  - Risk Management  (stop-loss, take-profit, daily loss limit)

Run with:
    streamlit run streamlit_app.py
"""

import sys
import os

# Ensure the project root is on sys.path so local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import indicators
import risk_management as rm
from backtester import Backtester
from data_pipeline import validate_ticker
from db import DatabaseManager
import gcs_sync
from indicators import add_indicators_custom
from optimizer import STRATEGY_PARAM_BOUNDS, StrategyOptimizer
from portfolio import Portfolio
from trading_agent import (
    STRATEGY_REGISTRY,
    BollingerBandStrategy,
    EMACrossoverStrategy,
    RSIStrategy,
    TradingAgent,
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PSE Trading Bot – Test Console",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGES = [
    "🏠 Dashboard",
    "🎮 Trading Simulation",
    "📡 Data Pipeline",
    "📊 Indicators",
    "🤖 Trading Signals",
    "💼 Portfolio",
    "🔬 Backtest",
    "⚠️ Risk Management",
    "🔧 Parameter Optimizer",
]


def _make_synthetic_ohlcv(
    tickers: list[str],
    n_candles: int = 200,
    base_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for offline testing.

    Args:
        tickers: List of ticker symbols.
        n_candles: Number of 1-minute candles per ticker.
        base_price: Starting close price.
        seed: Random seed for reproducibility.

    Returns:
        Combined OHLCV DataFrame with columns:
        [Datetime, Open, High, Low, Close, Volume, Ticker].
    """
    rng = np.random.default_rng(seed)
    now = datetime(2024, 1, 15, 9, 30)
    frames = []
    for i, ticker in enumerate(tickers):
        price = base_price * (1 + 0.1 * i)
        prices = [price]
        for _ in range(n_candles - 1):
            price = prices[-1] * (1 + rng.normal(0, 0.003))
            prices.append(max(price, 1.0))

        close = np.array(prices)
        spread = close * 0.002
        high = close + rng.uniform(0, 1, n_candles) * spread
        low = close - rng.uniform(0, 1, n_candles) * spread
        open_ = close + rng.uniform(-0.5, 0.5, n_candles) * spread
        volume = rng.integers(1_000, 50_000, n_candles).astype(float)
        timestamps = [now + timedelta(minutes=j) for j in range(n_candles)]

        frames.append(
            pd.DataFrame(
                {
                    "Datetime": timestamps,
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "Ticker": ticker,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["Datetime", "Ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner="Fetching live market data…")
def _fetch_live_data(tickers: tuple[str, ...], period: str, interval: str) -> pd.DataFrame:
    """Cached wrapper around data_pipeline.fetch_all_tickers."""
    from data_pipeline import fetch_all_tickers

    return fetch_all_tickers(tickers=list(tickers), period=period, interval=interval)


@st.cache_data(show_spinner="Fetching live market data for backtest window…")
def _fetch_live_data_range(tickers: tuple[str, ...], start_date, end_date, interval: str) -> pd.DataFrame:
    """Cached wrapper around data_pipeline.fetch_all_tickers_range.

    Uses explicit start/end dates so the full backtest window is always
    covered, regardless of how many calendar days ago it started.
    """
    from data_pipeline import fetch_all_tickers_range

    return fetch_all_tickers_range(
        tickers=list(tickers), start_date=start_date, end_date=end_date, interval=interval
    )


def _format_php(value: float) -> str:
    return f"₱{value:,.2f}"


def _color_pnl(val: float) -> str:
    return "🟢" if val >= 0 else "🔴"


# ---------------------------------------------------------------------------
# Database initialisation (persists data across session resets)
# ---------------------------------------------------------------------------

if "db" not in st.session_state:
    # On first run: try to pull the latest DB from GCS so state survives
    # re-deployments and server restarts.
    gcs_sync.download_db(config.DB_FILE)
    st.session_state["db"] = DatabaseManager()

_db: DatabaseManager = st.session_state["db"]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "portfolio" not in st.session_state:
    _saved_portfolio = _db.load_portfolio()
    st.session_state["portfolio"] = (
        _saved_portfolio if _saved_portfolio is not None else Portfolio(config.INITIAL_CAPITAL)
    )

if "custom_tickers" not in st.session_state:
    st.session_state["custom_tickers"] = _db.load_custom_tickers()

# ---------------------------------------------------------------------------
# Sidebar – navigation & global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("PSE Trading Bot")
    st.caption("Test Console")
    st.divider()

    page = st.radio("Navigate to", PAGES, index=0)
    st.divider()

    st.subheader("⚙️ Data Source")
    use_live = st.toggle("Use Live Yahoo Finance data", value=False)

    # Build the full list of available tickers (defaults + user-added)
    _all_tickers = list(dict.fromkeys(config.TICKERS + st.session_state["custom_tickers"]))
    _default_selection = [t for t in config.TICKERS[:3] if t in _all_tickers]

    selected_tickers = st.multiselect(
        "Tickers",
        _all_tickers,
        default=_default_selection,
    )
    if not selected_tickers:
        selected_tickers = _all_tickers[:2]

    with st.expander("➕ Add Ticker"):
        _new_ticker = st.text_input(
            "Yahoo Finance symbol (e.g. BPI.PS, AAPL)",
            placeholder="Enter ticker",
            key="new_ticker_input",
        ).strip().upper()
        if st.button("Validate & Add", key="validate_add_btn"):
            if not _new_ticker:
                st.warning("Please enter a ticker symbol.")
            elif _new_ticker in _all_tickers:
                st.warning(f"**{_new_ticker}** is already in the list.")
            else:
                with st.spinner(f"Validating {_new_ticker} on Yahoo Finance…"):
                    if validate_ticker(_new_ticker):
                        st.session_state["custom_tickers"].append(_new_ticker)
                        _db.save_custom_tickers(st.session_state["custom_tickers"])
                        gcs_sync.upload_db(_db.db_path)
                        st.success(f"✅ **{_new_ticker}** added!")
                        st.rerun()
                    else:
                        st.error(f"❌ **{_new_ticker}** is not a valid Yahoo Finance ticker.")

    n_candles = st.slider(
        "Candles (synthetic)", 100, 500, 200, step=50,
        disabled=use_live,
    )

    st.divider()

    st.subheader("🎮 Simulation Settings")
    is_backtest = st.toggle(
        "IS_BACKTEST",
        value=False,
        help="When enabled, data is filtered to the START_DATE – END_DATE window for simulation.",
    )
    _default_end = datetime(2024, 1, 15).date() if not use_live else datetime.today().date()
    _default_start = (datetime(2024, 1, 15) - timedelta(days=7)).date() if not use_live else (datetime.today() - timedelta(days=7)).date()
    start_date = st.date_input(
        "START_DATE",
        value=_default_start,
        help="Simulation start date (inclusive). Only used when IS_BACKTEST is enabled.",
        disabled=not is_backtest,
    )
    end_date = st.date_input(
        "END_DATE",
        value=_default_end,
        help="Simulation end date (inclusive). Only used when IS_BACKTEST is enabled.",
        disabled=not is_backtest,
    )
    strategy_name = st.selectbox(
        "STRATEGY",
        list(STRATEGY_REGISTRY.keys()),
        index=0,
        help="Trading strategy used for signal generation and backtesting.",
    )

    st.divider()

    # ---- Capital & Risk Settings ----
    with st.expander("💰 Capital & Risk Settings"):
        sidebar_initial_capital = st.number_input(
            "Initial Capital (PHP)",
            min_value=10_000,
            max_value=100_000_000,
            value=int(config.INITIAL_CAPITAL),
            step=100_000,
            help="Virtual starting capital used for all simulations and backtests.",
        )
        sidebar_max_position_pct = st.slider(
            "Max Position Size (%)", 1, 20,
            int(config.MAX_POSITION_PCT * 100),
            help="Maximum fraction of capital deployed per trade.",
        ) / 100.0
        sidebar_stop_loss_pct = st.slider(
            "Stop-Loss (%)", 1, 15,
            int(config.STOP_LOSS_PCT * 100),
            help="Sell position when price falls this % below entry.",
        ) / 100.0
        sidebar_take_profit_pct = st.slider(
            "Take-Profit (%)", 1, 25,
            int(config.TAKE_PROFIT_PCT * 100),
            help="Sell position when price rises this % above entry.",
        ) / 100.0
        sidebar_max_daily_loss_pct = st.slider(
            "Max Daily Loss (%)", 1, 15,
            int(config.MAX_DAILY_LOSS_PCT * 100),
            help="Halt all trading if daily loss exceeds this % of initial capital.",
        ) / 100.0

    # ---- Strategy Parameters ----
    with st.expander("📐 Strategy Parameters"):
        if strategy_name == "EMA Crossover":
            sidebar_ema_fast = st.slider(
                "Fast EMA Period", 2, 20, config.EMA_FAST,
                help="Short look-back period for the fast EMA.",
            )
            sidebar_ema_slow = st.slider(
                "Slow EMA Period", 10, 100, config.EMA_SLOW,
                help="Long look-back period for the slow EMA.",
            )
            # Ensure fast < slow; auto-clamp to avoid invalid configuration
            if sidebar_ema_fast >= sidebar_ema_slow:
                st.warning(
                    "Fast EMA period must be less than Slow EMA period. "
                    "Slow EMA will be clamped to Fast EMA + 1."
                )
                sidebar_ema_slow = sidebar_ema_fast + 1
            sidebar_rsi_period = config.RSI_PERIOD
            sidebar_rsi_oversold = config.RSI_OVERSOLD
            sidebar_rsi_overbought = config.RSI_OVERBOUGHT
            sidebar_bollinger_period = config.BOLLINGER_PERIOD
            sidebar_bollinger_std = config.BOLLINGER_STD
        elif strategy_name == "RSI Mean-Reversion":
            sidebar_rsi_period = st.slider(
                "RSI Period", 5, 30, config.RSI_PERIOD,
                help="Look-back period for RSI calculation.",
            )
            sidebar_rsi_oversold = st.slider(
                "RSI Oversold Threshold", 10, 45, int(config.RSI_OVERSOLD),
                help="RSI level considered oversold (BUY signal).",
            )
            sidebar_rsi_overbought = st.slider(
                "RSI Overbought Threshold", 55, 90, int(config.RSI_OVERBOUGHT),
                help="RSI level considered overbought (SELL signal).",
            )
            sidebar_ema_fast = config.EMA_FAST
            sidebar_ema_slow = config.EMA_SLOW
            sidebar_bollinger_period = config.BOLLINGER_PERIOD
            sidebar_bollinger_std = config.BOLLINGER_STD
        else:  # Bollinger Bands
            sidebar_bollinger_period = st.slider(
                "Bollinger Period", 5, 50, config.BOLLINGER_PERIOD,
                help="Rolling window for Bollinger Bands.",
            )
            sidebar_bollinger_std = st.slider(
                "Bollinger Std Dev", 1.0, 3.5, float(config.BOLLINGER_STD), step=0.1,
                help="Number of standard deviations for the band width.",
            )
            sidebar_ema_fast = config.EMA_FAST
            sidebar_ema_slow = config.EMA_SLOW
            sidebar_rsi_period = config.RSI_PERIOD
            sidebar_rsi_oversold = config.RSI_OVERSOLD
            sidebar_rsi_overbought = config.RSI_OVERBOUGHT

    st.divider()

    if use_live:
        period = st.selectbox(
            "Period",
            ["1d", "2d", "5d", "7d"],
            index=0,
            disabled=is_backtest,
            help=(
                "Look-back period for live data. Disabled when IS_BACKTEST is "
                "enabled — the full START_DATE → END_DATE window is fetched automatically."
            ),
        )
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h","1d", "1wk"], index=0)
    else:
        period = "synthetic"
        interval = "1m"

    st.divider()
    if st.button("🔄 Reset Portfolio", use_container_width=True):
        st.session_state["portfolio"] = Portfolio(sidebar_initial_capital)
        _db.save_portfolio(st.session_state["portfolio"])
        gcs_sync.upload_db(_db.db_path)
        st.success("Portfolio reset.")


# ---------------------------------------------------------------------------
# Load data (shared across pages)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating synthetic data…", ttl=60)
def _get_synthetic(tickers_key: str, n: int) -> pd.DataFrame:
    tickers = tickers_key.split(",")
    return _make_synthetic_ohlcv(tickers, n_candles=n)


def load_data() -> pd.DataFrame:
    if use_live:
        if is_backtest:
            # Use explicit date-range fetch so the full START_DATE→END_DATE
            # window is retrieved regardless of the relative-period cap.
            df = _fetch_live_data_range(tuple(selected_tickers), start_date, end_date, interval)
        else:
            df = _fetch_live_data(tuple(selected_tickers), period, interval)
    else:
        key = ",".join(selected_tickers)
        df = _get_synthetic(key, n_candles)
    # Ensure Datetime column has a proper datetime dtype regardless of source
    # (empty DataFrames from failed live fetches carry object-dtype Datetime)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    # Apply simulation date range when IS_BACKTEST is enabled
    if is_backtest and not df.empty:
        df = df[
            (df["Datetime"].dt.date >= start_date) &
            (df["Datetime"].dt.date <= end_date)
        ].copy()
    return df


def _get_strategy():
    """Return the strategy instance built from the current sidebar parameters."""
    if strategy_name == "EMA Crossover":
        return EMACrossoverStrategy(
            fast_period=sidebar_ema_fast,
            slow_period=sidebar_ema_slow,
        )
    if strategy_name == "RSI Mean-Reversion":
        return RSIStrategy(
            oversold=sidebar_rsi_oversold,
            overbought=sidebar_rsi_overbought,
        )
    return BollingerBandStrategy()


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators using the current sidebar parameter values."""
    return add_indicators_custom(
        df,
        ema_fast=sidebar_ema_fast,
        ema_slow=sidebar_ema_slow,
        rsi_period=sidebar_rsi_period,
        bollinger_period=sidebar_bollinger_period,
        bollinger_std=sidebar_bollinger_std,
    )


def _make_backtester(initial_capital: float | None = None) -> Backtester:
    """Create a Backtester configured with the current sidebar risk parameters."""
    return Backtester(
        initial_capital=initial_capital if initial_capital is not None else sidebar_initial_capital,
        strategy=_get_strategy(),
        max_position_pct=sidebar_max_position_pct,
        stop_loss_pct=sidebar_stop_loss_pct,
        take_profit_pct=sidebar_take_profit_pct,
        max_daily_loss_pct=sidebar_max_daily_loss_pct,
    )


def _make_agent(portfolio: Portfolio) -> TradingAgent:
    """Create a TradingAgent configured with the current sidebar risk parameters."""
    return TradingAgent(
        portfolio,
        _get_strategy(),
        max_position_pct=sidebar_max_position_pct,
        stop_loss_pct=sidebar_stop_loss_pct,
        take_profit_pct=sidebar_take_profit_pct,
        max_daily_loss_pct=sidebar_max_daily_loss_pct,
    )


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================

if page == PAGES[0]:
    st.title("🏠 PSE Trading Bot – Test Console")
    st.markdown(
        "Welcome to the interactive testing interface. "
        "Use the sidebar to navigate between test modules."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Capital", _format_php(sidebar_initial_capital))
    col2.metric("Tickers Available", len(_all_tickers))
    col3.metric("Data Interval", config.DATA_INTERVAL)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Strategy Parameters")
        params = {
            "Fast EMA": sidebar_ema_fast,
            "Slow EMA": sidebar_ema_slow,
            "RSI Period": sidebar_rsi_period,
            "RSI Overbought": sidebar_rsi_overbought,
            "RSI Oversold": sidebar_rsi_oversold,
            "Bollinger Period": sidebar_bollinger_period,
            "Bollinger Std": sidebar_bollinger_std,
        }
        st.table(pd.DataFrame(params.items(), columns=["Parameter", "Value"]))

    with c2:
        st.subheader("Risk Parameters")
        risk_params = {
            "Max Position %": f"{sidebar_max_position_pct * 100:.0f}%",
            "Stop-Loss %": f"{sidebar_stop_loss_pct * 100:.0f}%",
            "Take-Profit %": f"{sidebar_take_profit_pct * 100:.0f}%",
            "Max Daily Loss %": f"{sidebar_max_daily_loss_pct * 100:.0f}%",
        }
        st.table(pd.DataFrame(risk_params.items(), columns=["Parameter", "Value"]))

    st.divider()
    st.subheader("Configured Tickers")
    _default_names = {
        "BDO.PS": "BDO Unibank, Inc.",
        "SM.PS": "SM Investments Corporation",
        "ALI.PS": "Ayala Land, Inc.",
        "JFC.PS": "Jollibee Foods Corporation",
        "AC.PS": "Ayala Corporation",
        "TEL.PS": "PLDT, Inc.",
    }
    ticker_data = [
        {"Ticker": t, "Company": _default_names.get(t, t)}
        for t in _all_tickers
    ]
    st.dataframe(pd.DataFrame(ticker_data), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: Trading Simulation
# ===========================================================================

elif page == PAGES[1]:
    st.title("🎮 Trading Simulation")

    # ---- Simulation settings banner ----
    if is_backtest:
        st.info(
            f"📅 **IS_BACKTEST = True** — simulating from **{start_date}** to **{end_date}**  "
            f"| Strategy: **{strategy_name}**  "
            f"| Data between {start_date} and {end_date} (inclusive) will be used."
        )
    else:
        st.info(
            f"📅 **IS_BACKTEST = False** — using all available data (live mode)  "
            f"| Strategy: **{strategy_name}**"
        )

    st.markdown(
        "Configure **IS_BACKTEST**, **START_DATE**, **END_DATE**, and **STRATEGY** in the sidebar, "
        "then click **Run Simulation** to see what the bot would do."
    )

    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner("Loading data…"):
            sim_raw = load_data()

        if sim_raw.empty:
            st.error(
                "No data available for the selected settings. "
                "Try adjusting START_DATE / END_DATE or switch to synthetic data."
            )
            st.stop()

        with st.spinner("Computing indicators…"):
            sim_ind = _add_indicators(sim_raw)

        portfolio = Portfolio(sidebar_initial_capital)
        agent = _make_agent(portfolio)
        with st.spinner("Running strategy…"):
            sim_ready = agent.prepare_signals_df(sim_ind)
            sim_signals = agent.run(sim_ready)

        # Build equity curve
        eq_points: list[float] = []
        for _, grp in sim_signals.groupby("Datetime", sort=True):
            prices = dict(zip(grp["Ticker"], grp["Close"]))
            eq_points.append(portfolio.market_value(prices))

        st.session_state["sim_signals"] = sim_signals
        st.session_state["sim_portfolio"] = portfolio
        st.session_state["sim_equity"] = eq_points
        st.session_state["sim_strategy"] = strategy_name
        st.session_state["sim_date"] = f"{start_date} → {end_date}" if is_backtest else "live"
        # Persist simulation portfolio so results survive a session reset.
        # Also sync to the main portfolio key so the Portfolio page reflects it.
        st.session_state["portfolio"] = portfolio
        _db.save_portfolio(portfolio)
        gcs_sync.upload_db(_db.db_path)
        st.success("Simulation complete!")

    # ---- Results ----
    sim_signals = st.session_state.get("sim_signals")
    sim_portfolio: Portfolio | None = st.session_state.get("sim_portfolio")

    if sim_signals is None or sim_portfolio is None:
        st.info("Configure settings in the sidebar and click **▶ Run Simulation**.")
        st.stop()

    # Show which settings were used for the cached results
    _sim_strat_used = st.session_state.get("sim_strategy", "—")
    _sim_date_used = st.session_state.get("sim_date", "—")
    st.caption(f"Results: strategy = **{_sim_strat_used}** | date range = **{_sim_date_used}**")

    # --- Market snapshot ---
    st.divider()
    st.subheader("📌 Market Snapshot (latest price per ticker)")
    latest = sim_signals.groupby("Ticker").last()[["Close", "Signal"]].reset_index()
    latest.columns = ["Ticker", "Latest Close (PHP)", "Latest Signal"]
    st.dataframe(latest, use_container_width=True, hide_index=True)

    # --- Portfolio summary ---
    st.divider()
    st.subheader("💼 Portfolio State")
    mkt_prices = dict(zip(latest["Ticker"], latest["Latest Close (PHP)"]))
    summary = sim_portfolio.summary(mkt_prices)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Cash", _format_php(summary["cash"]))
    s2.metric("Market Value", _format_php(summary["market_value"]))
    pnl = summary["total_realized_pnl"]
    s3.metric("Realized P&L", f"{_color_pnl(pnl)} {_format_php(pnl)}")
    s4.metric("Total Return", f"{summary['total_return_pct']:+.2f}%")

    if summary["open_positions"]:
        pos_rows = []
        for ticker, pos in summary["open_positions"].items():
            current = mkt_prices.get(ticker, pos["avg_cost"])
            unrealized = (current - pos["avg_cost"]) * pos["shares"]
            pos_rows.append({
                "Ticker": ticker,
                "Shares": f"{pos['shares']:.0f}",
                "Avg Cost": _format_php(pos["avg_cost"]),
                "Current": _format_php(current),
                "Unrealized P&L": _format_php(unrealized),
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    # --- Equity curve ---
    eq_points = st.session_state.get("sim_equity", [])
    if eq_points:
        st.divider()
        st.subheader("📈 Equity Curve")
        eq_df = pd.DataFrame({"Candle": range(len(eq_points)), "Portfolio Value (PHP)": eq_points})
        fig_eq = px.line(eq_df, x="Candle", y="Portfolio Value (PHP)", title="Portfolio Value Over Time")
        fig_eq.add_hline(
            y=sidebar_initial_capital, line_dash="dash", line_color="grey",
            annotation_text="Initial Capital",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # --- Signal chart (per ticker) ---
    st.divider()
    st.subheader("🤖 Signals Chart")
    unique_tickers = sorted(sim_signals["Ticker"].unique())
    chart_ticker = st.selectbox("Select Ticker", unique_tickers, key="sim_chart_ticker")
    t_df = sim_signals[sim_signals["Ticker"] == chart_ticker].copy()
    buys = t_df[t_df["Signal"] == "BUY"]
    sells = t_df[t_df["Signal"] == "SELL"]

    fast_col = f"EMA_{sidebar_ema_fast}"
    slow_col = f"EMA_{sidebar_ema_slow}"

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=t_df["Datetime"], y=t_df["Close"], name="Close",
        line=dict(color="royalblue", width=1),
    ))
    if fast_col in t_df.columns:
        fig_sig.add_trace(go.Scatter(
            x=t_df["Datetime"], y=t_df[fast_col], name=f"EMA {sidebar_ema_fast}",
            line=dict(color="orange", width=1.5, dash="dash"),
        ))
    if slow_col in t_df.columns:
        fig_sig.add_trace(go.Scatter(
            x=t_df["Datetime"], y=t_df[slow_col], name=f"EMA {sidebar_ema_slow}",
            line=dict(color="red", width=1.5, dash="dash"),
        ))
    if "BB_upper" in t_df.columns:
        fig_sig.add_trace(go.Scatter(
            x=t_df["Datetime"], y=t_df["BB_upper"], name="BB Upper",
            line=dict(color="grey", width=1, dash="dot"),
        ))
        fig_sig.add_trace(go.Scatter(
            x=t_df["Datetime"], y=t_df["BB_lower"], name="BB Lower",
            line=dict(color="grey", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
        ))
    if not buys.empty:
        fig_sig.add_trace(go.Scatter(
            x=buys["Datetime"], y=buys["Close"], mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", color="green", size=12),
        ))
    if not sells.empty:
        fig_sig.add_trace(go.Scatter(
            x=sells["Datetime"], y=sells["Close"], mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", color="red", size=12),
        ))
    fig_sig.update_layout(
        title=f"{chart_ticker} – {_sim_strat_used} Signals",
        height=450, xaxis_title="Datetime", yaxis_title="Price (PHP)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    # --- Trade log ---
    st.divider()
    st.subheader("📋 Trade Log")
    trade_df = sim_portfolio.to_trade_log_df()
    if trade_df.empty:
        st.info("No trades were executed during this simulation.")
    else:
        st.metric("Total Trades", len(trade_df))
        st.dataframe(trade_df, use_container_width=True)

    # --- Decision log ---
    st.divider()
    st.subheader("📊 Decision Log")
    st.markdown(
        "Every candle in the simulation window with its strategy decision and "
        "key indicator values. Use this table to verify the bot's behaviour "
        "even when no trades were executed."
    )

    # Strategy-specific indicator columns to display alongside the signal
    _strategy_indicator_cols: dict[str, list[str]] = {
        "EMA Crossover": [f"EMA_{sidebar_ema_fast}", f"EMA_{sidebar_ema_slow}"],
        "RSI Mean-Reversion": ["RSI"],
        "Bollinger Bands": ["BB_upper", "BB_middle", "BB_lower"],
    }
    _indicator_cols = _strategy_indicator_cols.get(_sim_strat_used, [])
    _base_cols = ["Datetime", "Ticker", "Close", "Signal"]
    _display_cols = _base_cols + [c for c in _indicator_cols if c in sim_signals.columns]

    decision_log = sim_signals[_display_cols].copy()
    decision_log = decision_log.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)

    # Summary counts
    _n_buy = int((decision_log["Signal"] == "BUY").sum())
    _n_sell = int((decision_log["Signal"] == "SELL").sum())
    _n_hold = int((decision_log["Signal"] == "HOLD").sum())
    _n_halt = int((decision_log["Signal"] == "HALT").sum())

    dl1, dl2, dl3, dl4, dl5 = st.columns(5)
    dl1.metric("Total Candles", f"{len(decision_log):,}")
    dl2.metric("🟢 BUY", _n_buy)
    dl3.metric("🔴 SELL", _n_sell)
    dl4.metric("⚪ HOLD", _n_hold)
    dl5.metric("🛑 HALT", _n_halt)

    st.dataframe(decision_log, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: Data Pipeline
# ===========================================================================

elif page == PAGES[2]:
    st.title("📡 Data Pipeline")
    st.markdown(
        "Fetch or generate OHLCV market data for the selected tickers. "
        "Toggle *Live Yahoo Finance data* in the sidebar to switch between "
        "live and synthetic data."
    )

    if st.button("▶ Load Data", type="primary"):
        with st.spinner("Loading data…"):
            df = load_data()
        st.session_state["raw_df"] = df

    df = st.session_state.get("raw_df")

    if df is None:
        st.info("Click **Load Data** to fetch or generate market data.")
        st.stop()

    if df.empty:
        st.error("No data returned. Check your internet connection or try synthetic mode.")
        st.stop()

    # Summary metrics
    unique_tickers = df["Ticker"].unique()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Tickers", len(unique_tickers))
    col3.metric("Date Range", f"{df['Datetime'].min().strftime('%Y-%m-%d')} → {df['Datetime'].max().strftime('%Y-%m-%d')}")
    col4.metric("Candles/Ticker", f"{len(df) // max(len(unique_tickers), 1):,}")

    st.divider()

    # Ticker selector for chart
    chart_ticker = st.selectbox("Select Ticker to Chart", sorted(unique_tickers))
    t_df = df[df["Ticker"] == chart_ticker].copy()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=t_df["Datetime"],
                open=t_df["Open"],
                high=t_df["High"],
                low=t_df["Low"],
                close=t_df["Close"],
                name=chart_ticker,
            )
        ]
    )
    fig.update_layout(
        title=f"{chart_ticker} – OHLCV",
        xaxis_title="Datetime",
        yaxis_title="Price (PHP)",
        xaxis_rangeslider_visible=False,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Raw Data (first 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)


# ===========================================================================
# PAGE: Indicators
# ===========================================================================

elif page == PAGES[3]:
    st.title("📊 Indicators")
    st.markdown(
        "Compute technical indicators (EMA, RSI, Bollinger Bands, Returns, "
        "Volatility) on market data."
    )

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        with st.spinner("Loading data…"):
            raw_df = load_data()
        st.session_state["raw_df"] = raw_df

    with st.spinner("Computing indicators…"):
        ind_df = _add_indicators(raw_df)

    st.session_state["ind_df"] = ind_df

    unique_tickers = sorted(ind_df["Ticker"].unique())
    chart_ticker = st.selectbox("Select Ticker", unique_tickers)
    t_df = ind_df[ind_df["Ticker"] == chart_ticker].copy()

    fast_col = f"EMA_{sidebar_ema_fast}"
    slow_col = f"EMA_{sidebar_ema_slow}"

    # --- Price + EMA chart ---
    st.subheader(f"{chart_ticker} – Price & EMAs")
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df["Close"], name="Close", line=dict(color="royalblue", width=1))
    )
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df[fast_col], name=f"EMA {sidebar_ema_fast}", line=dict(color="orange", width=1.5, dash="dash"))
    )
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df[slow_col], name=f"EMA {sidebar_ema_slow}", line=dict(color="red", width=1.5, dash="dash"))
    )
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df["BB_upper"], name="BB Upper", line=dict(color="grey", width=1, dash="dot"))
    )
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df["BB_lower"], name="BB Lower", line=dict(color="grey", width=1, dash="dot"),
                   fill="tonexty", fillcolor="rgba(150,150,150,0.1)")
    )
    fig_price.update_layout(height=400, xaxis_title="Datetime", yaxis_title="Price (PHP)", legend=dict(orientation="h"))
    st.plotly_chart(fig_price, use_container_width=True)

    # --- RSI chart ---
    st.subheader(f"{chart_ticker} – RSI ({sidebar_rsi_period})")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df["RSI"], name="RSI", line=dict(color="purple")))
    fig_rsi.add_hline(y=sidebar_rsi_overbought, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=sidebar_rsi_oversold, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(height=250, yaxis=dict(range=[0, 100]), xaxis_title="Datetime", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- Volatility chart ---
    st.subheader(f"{chart_ticker} – Rolling Volatility (20-period annualised)")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df["Volatility"], name="Volatility", line=dict(color="teal")))
    fig_vol.update_layout(height=200, xaxis_title="Datetime", yaxis_title="Volatility")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()
    st.subheader("Indicator Data (first 50 rows)")
    display_cols = ["Datetime", "Ticker", "Close", fast_col, slow_col, "RSI", "BB_upper", "BB_middle", "BB_lower", "Returns", "Volatility"]
    st.dataframe(ind_df[display_cols].head(50), use_container_width=True)


# ===========================================================================
# PAGE: Trading Signals
# ===========================================================================

elif page == PAGES[4]:
    st.title("🤖 Trading Signals")
    st.markdown(
        "Run the trading agent on market data. "
        "BUY / SELL signals are generated by the **STRATEGY** selected in the sidebar."
    )

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        with st.spinner("Loading data…"):
            raw_df = load_data()
        st.session_state["raw_df"] = raw_df

    ind_df = st.session_state.get("ind_df")
    if ind_df is None:
        with st.spinner("Computing indicators…"):
            ind_df = _add_indicators(raw_df)
        st.session_state["ind_df"] = ind_df

    if st.button("▶ Generate Signals", type="primary"):
        portfolio = Portfolio(sidebar_initial_capital)
        agent = _make_agent(portfolio)
        with st.spinner("Running trading agent…"):
            ready_df = agent.prepare_signals_df(ind_df)
            signals_df = agent.run(ready_df)
        st.session_state["signals_df"] = signals_df
        st.session_state["signal_portfolio"] = portfolio
        st.session_state["signals_strategy"] = strategy_name
        # Keep main portfolio in sync so the Portfolio page and DB reflect it.
        st.session_state["portfolio"] = portfolio
        _db.save_portfolio(portfolio)
        gcs_sync.upload_db(_db.db_path)
        st.success(f"Signals generated using **{strategy_name}**!")

    signals_df = st.session_state.get("signals_df")
    if signals_df is None:
        st.info("Click **Generate Signals** to run the trading agent.")
        st.stop()

    # Signal counts
    counts = signals_df["Signal"].value_counts()
    cols = st.columns(len(counts))
    for col, (sig, cnt) in zip(cols, counts.items()):
        col.metric(sig, cnt)

    st.divider()

    unique_tickers = sorted(signals_df["Ticker"].unique())
    chart_ticker = st.selectbox("Select Ticker", unique_tickers)
    t_df = signals_df[signals_df["Ticker"] == chart_ticker].copy()

    fast_col = f"EMA_{sidebar_ema_fast}"
    slow_col = f"EMA_{sidebar_ema_slow}"

    buys = t_df[t_df["Signal"] == "BUY"]
    sells = t_df[t_df["Signal"] == "SELL"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df["Close"], name="Close", line=dict(color="royalblue", width=1)))
    if fast_col in t_df.columns:
        fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df[fast_col], name=f"EMA {sidebar_ema_fast}", line=dict(color="orange", width=1.5, dash="dash")))
    if slow_col in t_df.columns:
        fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df[slow_col], name=f"EMA {sidebar_ema_slow}", line=dict(color="red", width=1.5, dash="dash")))
    if "BB_upper" in t_df.columns and st.session_state.get("signals_strategy") in ("Bollinger Bands", None):
        fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df["BB_upper"], name="BB Upper", line=dict(color="grey", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df["BB_lower"], name="BB Lower", line=dict(color="grey", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(150,150,150,0.1)"))
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["Datetime"], y=buys["Close"],
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", color="green", size=12),
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["Datetime"], y=sells["Close"],
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", color="red", size=12),
        ))
    used_strategy = st.session_state.get("signals_strategy", strategy_name)
    fig.update_layout(
        title=f"{chart_ticker} – {used_strategy} Signals",
        height=450,
        xaxis_title="Datetime",
        yaxis_title="Price (PHP)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Signal DataFrame (first 100 rows)")
    sig_cols = ["Datetime", "Ticker", "Close", "Signal"]
    for col in [fast_col, slow_col, "RSI", "BB_upper", "BB_lower"]:
        if col in signals_df.columns:
            sig_cols.append(col)
    st.dataframe(signals_df[sig_cols].head(100), use_container_width=True)


# ===========================================================================
# PAGE: Portfolio
# ===========================================================================

elif page == PAGES[5]:
    st.title("💼 Portfolio")
    st.markdown(
        "Inspect the virtual portfolio state. The portfolio is shared with the "
        "Trading Signals page when signals are generated. Use the **Reset Portfolio** "
        "button in the sidebar to start fresh."
    )

    portfolio: Portfolio = st.session_state.get(
        "signal_portfolio", st.session_state["portfolio"]
    )

    # Get latest prices from signals df if available
    sig_df = st.session_state.get("signals_df")
    market_prices: dict = {}
    if sig_df is not None and not sig_df.empty:
        latest = sig_df.groupby("Ticker")["Close"].last()
        market_prices = latest.to_dict()

    summary = portfolio.summary(market_prices)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cash", _format_php(summary["cash"]))
    col2.metric("Market Value", _format_php(summary["market_value"]))
    pnl = summary["total_realized_pnl"]
    col3.metric("Total Realized P&L", f"{_color_pnl(pnl)} {_format_php(pnl)}")
    ret_pct = summary["total_return_pct"]
    col4.metric("Total Return", f"{ret_pct:+.2f}%", delta=f"{ret_pct:+.2f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Open Positions")
        if summary["open_positions"]:
            pos_rows = []
            for ticker, pos in summary["open_positions"].items():
                current = market_prices.get(ticker, pos["avg_cost"])
                unrealized = (current - pos["avg_cost"]) * pos["shares"]
                pos_rows.append({
                    "Ticker": ticker,
                    "Shares": f"{pos['shares']:.0f}",
                    "Avg Cost": _format_php(pos["avg_cost"]),
                    "Current Price": _format_php(current),
                    "Unrealized P&L": _format_php(unrealized),
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")

    with c2:
        st.subheader("Portfolio Allocation")
        if summary["open_positions"] or summary["cash"] > 0:
            labels, values = ["Cash"], [summary["cash"]]
            for ticker, pos in summary["open_positions"].items():
                price = market_prices.get(ticker, pos["avg_cost"])
                labels.append(ticker)
                values.append(pos["shares"] * price)
            fig_pie = px.pie(values=values, names=labels, title="Portfolio Allocation")
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("Trade Log")
    trade_df = portfolio.to_trade_log_df()
    if trade_df.empty:
        st.info("No trades executed yet. Generate signals first.")
    else:
        st.metric("Total Trades", len(trade_df))
        st.dataframe(
            trade_df.style.map(
                lambda v: "color: green" if v == "BUY" else ("color: red" if v == "SELL" else ""),
                subset=["action"],
            ),
            use_container_width=True,
        )

        # P&L bar chart (SELL trades only)
        sells = trade_df[trade_df["action"] == "SELL"].copy()
        if not sells.empty:
            sells["color"] = sells["pnl"].apply(lambda x: "Profit" if x >= 0 else "Loss")
            fig_pnl = px.bar(
                sells,
                x="timestamp",
                y="pnl",
                color="color",
                color_discrete_map={"Profit": "green", "Loss": "red"},
                labels={"pnl": "Realized P&L (PHP)", "timestamp": "Trade Time"},
                title="Realized P&L per SELL Trade",
            )
            st.plotly_chart(fig_pnl, use_container_width=True)


# ===========================================================================
# PAGE: Backtest
# ===========================================================================

elif page == PAGES[6]:
    st.title("🔬 Backtest")
    st.markdown(
        "Run a full historical backtest on the loaded market data using the "
        "**STRATEGY** selected in the sidebar. Results include performance metrics and an "
        "equity curve."
    )

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        with st.spinner("Loading data…"):
            raw_df = load_data()
        st.session_state["raw_df"] = raw_df

    ind_df = st.session_state.get("ind_df")
    if ind_df is None:
        with st.spinner("Computing indicators…"):
            ind_df = _add_indicators(raw_df)
        st.session_state["ind_df"] = ind_df

    st.info(
        f"Capital and risk settings are configured in the **💰 Capital & Risk Settings** "
        f"expander in the sidebar.  "
        f"Current: Initial Capital = **{_format_php(sidebar_initial_capital)}** | "
        f"Stop-Loss = **{sidebar_stop_loss_pct * 100:.0f}%** | "
        f"Take-Profit = **{sidebar_take_profit_pct * 100:.0f}%**"
    )

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Running backtest…"):
            bt = _make_backtester()
            metrics = bt.run(ind_df)
        st.session_state["bt_metrics"] = metrics
        st.success(f"Backtest complete using **{strategy_name}**!")

    metrics = st.session_state.get("bt_metrics")
    if metrics is None:
        st.info("Click **Run Backtest** to start the simulation.")
        st.stop()

    # Key metrics
    st.divider()
    st.subheader("Performance Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Return", f"{metrics['total_return_pct']:+.2f}%")
    m2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}")
    m3.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    m4.metric("Total Trades", metrics["total_trades"])
    m5.metric("Winning Trades", metrics["winning_trades"])
    m6.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")

    # Equity curve
    st.divider()
    st.subheader("Equity Curve")
    eq = metrics["equity_curve"]
    if len(eq) > 0:
        eq_df = pd.DataFrame({"Candle": range(len(eq)), "Portfolio Value (PHP)": eq.values})
        fig_eq = px.line(eq_df, x="Candle", y="Portfolio Value (PHP)", title="Portfolio Value Over Time")
        fig_eq.add_hline(y=sidebar_initial_capital, line_dash="dash", line_color="grey", annotation_text="Initial Capital")
        st.plotly_chart(fig_eq, use_container_width=True)

    # Per-ticker summary
    st.divider()
    st.subheader("Per-Ticker Summary")
    summary_df = metrics.get("summary_df")
    if summary_df is not None and not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        fig_ticker = px.bar(
            summary_df,
            x="ticker",
            y="total_pnl",
            color="total_pnl",
            color_continuous_scale=["red", "green"],
            labels={"total_pnl": "Total P&L (PHP)", "ticker": "Ticker"},
            title="Total P&L per Ticker",
        )
        st.plotly_chart(fig_ticker, use_container_width=True)
    else:
        st.info("No completed trades during the backtest period.")

    # Trade log
    st.divider()
    st.subheader("Trade Log")
    trade_log = metrics.get("trade_log")
    if trade_log is not None and not trade_log.empty:
        st.dataframe(trade_log, use_container_width=True)
    else:
        st.info("No trades in the log.")


# ===========================================================================
# PAGE: Risk Management
# ===========================================================================

elif page == PAGES[7]:
    st.title("⚠️ Risk Management")
    st.markdown(
        "Interactively test the risk management functions: position sizing, "
        "stop-loss, take-profit, and the daily loss limit."
    )

    st.subheader("Position Sizing")
    col1, col2, col3 = st.columns(3)
    ps_capital = col1.number_input("Available Capital (PHP)", value=1_000_000.0, step=50_000.0)
    ps_price = col2.number_input("Share Price (PHP)", value=100.0, step=1.0, min_value=0.01)
    ps_pct = col3.slider("Max Position %", 1, 20, int(config.MAX_POSITION_PCT * 100)) / 100.0
    shares = rm.compute_position_size(ps_capital, ps_price, max_pct=ps_pct)
    spend = shares * ps_price
    st.success(
        f"**{shares:.0f} shares** @ ₱{ps_price:.2f}  →  "
        f"Total spend: **{_format_php(spend)}**  "
        f"({spend / ps_capital * 100:.2f}% of capital)"
    )

    st.divider()

    col_sl, col_tp = st.columns(2)

    with col_sl:
        st.subheader("Stop-Loss Check")
        sl_entry = st.number_input("Entry Price (PHP)", value=100.0, step=1.0, key="sl_entry", min_value=0.01)
        sl_current = st.number_input("Current Price (PHP)", value=97.0, step=0.5, key="sl_current", min_value=0.01)
        sl_pct = st.slider("Stop-Loss %", 1, 10, int(config.STOP_LOSS_PCT * 100), key="sl_pct") / 100.0
        triggered_sl = rm.check_stop_loss(sl_entry, sl_current, sl_pct)
        stop_level = sl_entry * (1 - sl_pct)
        if triggered_sl:
            st.error(f"🔴 STOP-LOSS TRIGGERED  (level: ₱{stop_level:.4f})")
        else:
            st.success(f"✅ Stop-loss not triggered  (level: ₱{stop_level:.4f})")

    with col_tp:
        st.subheader("Take-Profit Check")
        tp_entry = st.number_input("Entry Price (PHP)", value=100.0, step=1.0, key="tp_entry", min_value=0.01)
        tp_current = st.number_input("Current Price (PHP)", value=105.0, step=0.5, key="tp_current", min_value=0.01)
        tp_pct = st.slider("Take-Profit %", 1, 15, int(config.TAKE_PROFIT_PCT * 100), key="tp_pct") / 100.0
        triggered_tp = rm.check_take_profit(tp_entry, tp_current, tp_pct)
        tp_level = tp_entry * (1 + tp_pct)
        if triggered_tp:
            st.success(f"🟢 TAKE-PROFIT TRIGGERED  (level: ₱{tp_level:.4f})")
        else:
            st.info(f"ℹ️ Take-profit not triggered  (level: ₱{tp_level:.4f})")

    st.divider()
    st.subheader("Daily Loss Limit Check")
    dl_initial = st.number_input("Initial Capital (PHP)", value=1_000_000.0, step=50_000.0, key="dl_initial")
    dl_pnl = st.number_input("Daily Realized P&L (PHP)", value=-25_000.0, step=1_000.0, key="dl_pnl")
    dl_pct = st.slider("Max Daily Loss %", 1, 10, int(config.MAX_DAILY_LOSS_PCT * 100), key="dl_pct") / 100.0

    test_portfolio = Portfolio(dl_initial)
    test_portfolio._daily_realized_pnl = dl_pnl
    limit_value = dl_initial * dl_pct
    triggered_dl = rm.check_daily_loss_limit(test_portfolio, dl_pct)

    if triggered_dl:
        st.error(
            f"🔴 DAILY LOSS LIMIT EXCEEDED  "
            f"(P&L: {_format_php(dl_pnl)}  |  Limit: -{_format_php(limit_value)})"
        )
    else:
        remaining = limit_value + dl_pnl
        st.success(
            f"✅ Within daily loss limit  "
            f"(P&L: {_format_php(dl_pnl)}  |  Limit: -{_format_php(limit_value)}  |  "
            f"Remaining buffer: {_format_php(remaining)})"
        )

    st.divider()
    st.subheader("Risk Parameter Reference")
    ref = {
        "Parameter": [
            "Max Position Size",
            "Stop-Loss Threshold",
            "Take-Profit Threshold",
            "Max Daily Loss",
        ],
        "Config Value": [
            f"{config.MAX_POSITION_PCT * 100:.0f}% of capital",
            f"{config.STOP_LOSS_PCT * 100:.0f}% below entry",
            f"{config.TAKE_PROFIT_PCT * 100:.0f}% above entry",
            f"{config.MAX_DAILY_LOSS_PCT * 100:.0f}% of initial capital",
        ],
    }
    st.table(pd.DataFrame(ref))


# ===========================================================================
# PAGE: Parameter Optimizer
# ===========================================================================

elif page == PAGES[8]:
    st.title("🔧 Parameter Optimizer")
    st.markdown(
        "Automatically search for **strategy and risk parameters** that maximise "
        "total return on the loaded data using an iterative hill-climbing algorithm.\n\n"
        "The optimizer runs repeated backtests, each time tweaking the parameters "
        "slightly and keeping changes that improve the return.  A random exploration "
        "step prevents getting stuck in local optima.  All explored parameter sets "
        "are constrained to human-reasonable bounds."
    )

    # ---- Configuration ----
    st.subheader("⚙️ Optimizer Settings")

    opt_col1, opt_col2, opt_col3 = st.columns(3)

    _opt_strategy_keys = list(STRATEGY_PARAM_BOUNDS.keys())
    with opt_col1:
        opt_strategy = st.selectbox(
            "Strategy to Optimise",
            _opt_strategy_keys,
            index=_opt_strategy_keys.index(strategy_name)
            if strategy_name in _opt_strategy_keys else 0,
            help="Select the strategy whose parameters will be optimised.",
        )

    with opt_col2:
        opt_iterations = st.number_input(
            "Number of Iterations",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help=(
                "How many optimisation steps to run.  More iterations explore more "
                "of the parameter space but take longer.  50–100 is a good starting point."
            ),
        )

    with opt_col3:
        opt_capital = st.number_input(
            "Initial Capital (PHP)",
            min_value=10_000,
            max_value=100_000_000,
            value=int(sidebar_initial_capital),
            step=100_000,
            help="Virtual capital used during each backtest evaluation.",
        )

    # Parameter bounds reference table
    with st.expander("📋 Parameter Bounds for Selected Strategy", expanded=False):
        bounds = STRATEGY_PARAM_BOUNDS[opt_strategy]
        bounds_data = [
            {
                "Parameter": b.name,
                "Min": b.min_val,
                "Max": b.max_val,
                "Default": b.initial,
                "Type": "integer" if b.is_int else "float",
            }
            for b in bounds
        ]
        st.dataframe(pd.DataFrame(bounds_data), use_container_width=True, hide_index=True)

    # ---- Data loading ----
    raw_df_opt = st.session_state.get("raw_df")
    if raw_df_opt is None:
        with st.spinner("Loading data for optimisation…"):
            raw_df_opt = load_data()
        st.session_state["raw_df"] = raw_df_opt

    if raw_df_opt.empty:
        st.error(
            "No data available.  "
            "Switch to synthetic data or adjust the date range in the sidebar."
        )
        st.stop()

    # ---- Run optimisation ----
    st.divider()
    if st.button("▶ Run Optimizer", type="primary"):
        progress_bar = st.progress(0, text="Initialising…")
        status_text = st.empty()

        def _update_progress(i: int, total: int, best_ret: float, best_p: dict) -> None:
            pct = int(i / total * 100)
            progress_bar.progress(pct, text=f"Iteration {i}/{total} — best return: {best_ret:+.3f}%")
            status_text.caption(f"Best params so far: {best_p}")

        with st.spinner("Running optimisation…"):
            opt = StrategyOptimizer(
                df_raw=raw_df_opt,
                strategy_name=opt_strategy,
                initial_capital=float(opt_capital),
                n_iterations=int(opt_iterations),
                seed=42,
            )
            result = opt.run(progress_callback=_update_progress)

        progress_bar.progress(100, text="✅ Optimisation complete!")
        status_text.empty()
        st.session_state["opt_result"] = result
        st.success(
            f"Optimisation finished in **{result.n_evaluations}** backtest evaluations.  "
            f"Best return: **{result.best_return_pct:+.3f}%**"
        )

    # ---- Results ----
    result = st.session_state.get("opt_result")
    if result is None:
        st.info(
            "Configure the settings above and click **▶ Run Optimizer** to start the search."
        )
        st.stop()

    if result.strategy_name != opt_strategy:
        st.warning(
            f"Displayed results are for **{result.strategy_name}**, "
            f"not the currently selected **{opt_strategy}**.  "
            "Re-run the optimizer to get fresh results."
        )

    # ---- Key metrics comparison ----
    st.divider()
    st.subheader("📊 Results Summary")

    improvement = result.best_return_pct - result.initial_return_pct
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Initial Return (default params)",
        f"{result.initial_return_pct:+.3f}%",
    )
    r2.metric(
        "Best Return (optimised params)",
        f"{result.best_return_pct:+.3f}%",
        delta=f"{improvement:+.3f}%",
    )
    r3.metric("Iterations", result.n_iterations)
    r4.metric("Evaluations", result.n_evaluations)

    # ---- Parameter comparison table ----
    st.divider()
    st.subheader("🔬 Parameter Comparison")

    bounds_map = {b.name: b for b in STRATEGY_PARAM_BOUNDS[result.strategy_name]}
    # Parameters whose values represent percentages (stored as decimals)
    _pct_params = {"max_position_pct", "stop_loss_pct", "take_profit_pct", "max_daily_loss_pct"}
    comparison_rows = []
    for name in result.initial_params:
        b = bounds_map.get(name)
        initial_val = result.initial_params[name]
        best_val = result.best_params[name]
        if name in _pct_params:
            fmt_val = lambda v: f"{v * 100:.2f}%"  # noqa: E731
        elif b and b.is_int:
            fmt_val = lambda v: f"{v:.0f}"  # noqa: E731
        else:
            fmt_val = lambda v: f"{v:.4f}"  # noqa: E731
        comparison_rows.append({
            "Parameter": name,
            "Default Value": fmt_val(initial_val),
            "Optimised Value": fmt_val(best_val),
            "Range": (
                f"[{b.min_val * 100:.1f}%, {b.max_val * 100:.1f}%]"
                if name in _pct_params and b
                else (f"[{b.min_val}, {b.max_val}]" if b else "—")
            ),
            "Changed": "✅" if abs(best_val - initial_val) > 1e-9 else "—",
        })
    st.dataframe(
        pd.DataFrame(comparison_rows),
        use_container_width=True,
        hide_index=True,
    )

    # ---- Convergence chart ----
    st.divider()
    st.subheader("📈 Convergence Chart")
    st.markdown(
        "The best return found so far plotted against the iteration number.  "
        "A flat line after some point indicates convergence."
    )
    history_df = pd.DataFrame(
        [(i, r) for i, r, _ in result.iteration_history],
        columns=["Iteration", "Best Return (%)"],
    )
    fig_conv = px.line(
        history_df,
        x="Iteration",
        y="Best Return (%)",
        title=f"{result.strategy_name} – Optimisation Convergence",
        markers=False,
    )
    fig_conv.add_hline(
        y=result.initial_return_pct,
        line_dash="dash",
        line_color="grey",
        annotation_text="Default params baseline",
    )
    fig_conv.update_layout(height=350)
    st.plotly_chart(fig_conv, use_container_width=True)

    # ---- Verify best params with a full backtest ----
    st.divider()
    st.subheader("✅ Verification Backtest (Best Parameters)")
    st.markdown(
        "Run a full backtest using the optimised parameters to see detailed metrics "
        "and the equity curve."
    )

    if st.button("▶ Run Verification Backtest", key="opt_verify_btn"):
        bp = result.best_params
        with st.spinner("Building strategy and running backtest…"):
            df_ind_opt = add_indicators_custom(
                raw_df_opt,
                ema_fast=int(bp.get("ema_fast", config.EMA_FAST)),
                ema_slow=int(bp.get("ema_slow", config.EMA_SLOW)),
                rsi_period=int(bp.get("rsi_period", config.RSI_PERIOD)),
                bollinger_period=int(bp.get("bollinger_period", config.BOLLINGER_PERIOD)),
                bollinger_std=float(bp.get("bollinger_std", config.BOLLINGER_STD)),
            )

            if result.strategy_name == "EMA Crossover":
                verify_strategy = EMACrossoverStrategy(
                    fast_period=int(bp["ema_fast"]),
                    slow_period=int(bp["ema_slow"]),
                )
            elif result.strategy_name == "RSI Mean-Reversion":
                verify_strategy = RSIStrategy(
                    oversold=float(bp["rsi_oversold"]),
                    overbought=float(bp["rsi_overbought"]),
                )
            else:
                verify_strategy = BollingerBandStrategy()

            verify_bt = Backtester(
                initial_capital=float(opt_capital),
                strategy=verify_strategy,
                max_position_pct=float(bp.get("max_position_pct", config.MAX_POSITION_PCT)),
                stop_loss_pct=float(bp.get("stop_loss_pct", config.STOP_LOSS_PCT)),
                take_profit_pct=float(bp.get("take_profit_pct", config.TAKE_PROFIT_PCT)),
            )
            verify_metrics = verify_bt.run(df_ind_opt)

        st.session_state["opt_verify_metrics"] = verify_metrics
        st.success("Verification backtest complete!")

    verify_metrics = st.session_state.get("opt_verify_metrics")
    if verify_metrics is not None:
        vm = verify_metrics
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        v1.metric("Total Return", f"{vm['total_return_pct']:+.2f}%")
        v2.metric("Sharpe Ratio", f"{vm['sharpe_ratio']:.4f}")
        v3.metric("Max Drawdown", f"{vm['max_drawdown'] * 100:.2f}%")
        v4.metric("Total Trades", vm["total_trades"])
        v5.metric("Winning Trades", vm["winning_trades"])
        v6.metric("Win Rate", f"{vm['win_rate'] * 100:.1f}%")

        eq = vm["equity_curve"]
        if len(eq) > 0:
            st.divider()
            eq_df = pd.DataFrame({"Candle": range(len(eq)), "Portfolio Value (PHP)": eq.values})
            fig_eq = px.line(
                eq_df, x="Candle", y="Portfolio Value (PHP)",
                title="Equity Curve – Optimised Parameters",
            )
            fig_eq.add_hline(
                y=float(opt_capital), line_dash="dash", line_color="grey",
                annotation_text="Initial Capital",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        st.divider()
        st.subheader("Per-Ticker Summary")
        summary_df = vm.get("summary_df")
        if summary_df is not None and not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("No completed trades during the backtest period.")
