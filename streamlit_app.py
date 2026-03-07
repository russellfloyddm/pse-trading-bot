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

    return fetch_all_tickers(tickers=tickers, period=period, interval=interval)


def _format_php(value: float) -> str:
    return f"₱{value:,.2f}"


def _color_pnl(val: float) -> str:
    return "🟢" if val >= 0 else "🔴"


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = Portfolio(config.INITIAL_CAPITAL)

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

    selected_tickers = st.multiselect(
        "Tickers",
        config.TICKERS,
        default=config.TICKERS[:3],
    )
    if not selected_tickers:
        selected_tickers = config.TICKERS[:2]

    n_candles = st.slider(
        "Candles (synthetic)", 100, 500, 200, step=50,
        disabled=use_live,
    )

    if use_live:
        period = st.selectbox("Period", ["1d", "2d", "5d"], index=0)
        interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=0)
    else:
        period = "synthetic"
        interval = "1m"

    st.divider()

    st.subheader("🎮 Simulation Settings")
    is_backtest = st.toggle(
        "IS_BACKTEST",
        value=False,
        help="When enabled, data is filtered to CURRENT_DATE so the bot acts as if that is today.",
    )
    _default_date = datetime(2024, 1, 15).date() if not use_live else datetime.today().date()
    current_date = st.date_input(
        "CURRENT_DATE",
        value=_default_date,
        help="Acts as the current trading date when IS_BACKTEST is enabled.",
        disabled=not is_backtest,
    )
    strategy_name = st.selectbox(
        "STRATEGY",
        list(STRATEGY_REGISTRY.keys()),
        index=0,
        help="Trading strategy used for signal generation and backtesting.",
    )

    st.divider()
    if st.button("🔄 Reset Portfolio", use_container_width=True):
        st.session_state["portfolio"] = Portfolio(config.INITIAL_CAPITAL)
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
        df = _fetch_live_data(tuple(selected_tickers), period, interval)
    else:
        key = ",".join(selected_tickers)
        df = _get_synthetic(key, n_candles)
    # Ensure Datetime column has a proper datetime dtype regardless of source
    # (empty DataFrames from failed live fetches carry object-dtype Datetime)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    # Apply simulation date cutoff when IS_BACKTEST is enabled
    if is_backtest and not df.empty:
        df = df[df["Datetime"].dt.date <= current_date].copy()
    return df


def _get_strategy():
    """Return the strategy instance selected in the sidebar."""
    return STRATEGY_REGISTRY[strategy_name]


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
    col1.metric("Initial Capital", _format_php(config.INITIAL_CAPITAL))
    col2.metric("Tickers Loaded", len(config.TICKERS))
    col3.metric("Data Interval", config.DATA_INTERVAL)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Strategy Parameters")
        params = {
            "Fast EMA": config.EMA_FAST,
            "Slow EMA": config.EMA_SLOW,
            "RSI Period": config.RSI_PERIOD,
            "RSI Overbought": config.RSI_OVERBOUGHT,
            "RSI Oversold": config.RSI_OVERSOLD,
            "Bollinger Period": config.BOLLINGER_PERIOD,
            "Bollinger Std": config.BOLLINGER_STD,
        }
        st.table(pd.DataFrame(params.items(), columns=["Parameter", "Value"]))

    with c2:
        st.subheader("Risk Parameters")
        risk_params = {
            "Max Position %": f"{config.MAX_POSITION_PCT * 100:.0f}%",
            "Stop-Loss %": f"{config.STOP_LOSS_PCT * 100:.0f}%",
            "Take-Profit %": f"{config.TAKE_PROFIT_PCT * 100:.0f}%",
            "Max Daily Loss %": f"{config.MAX_DAILY_LOSS_PCT * 100:.0f}%",
        }
        st.table(pd.DataFrame(risk_params.items(), columns=["Parameter", "Value"]))

    st.divider()
    st.subheader("Configured Tickers")
    ticker_data = [
        {"Ticker": "BDO.PS", "Company": "BDO Unibank, Inc."},
        {"Ticker": "SM.PS", "Company": "SM Investments Corporation"},
        {"Ticker": "ALI.PS", "Company": "Ayala Land, Inc."},
        {"Ticker": "JFC.PS", "Company": "Jollibee Foods Corporation"},
        {"Ticker": "AC.PS", "Company": "Ayala Corporation"},
        {"Ticker": "TEL.PS", "Company": "PLDT, Inc."},
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
            f"📅 **IS_BACKTEST = True** — simulating as of **{current_date}**  "
            f"| Strategy: **{strategy_name}**  "
            f"| Data up to and including {current_date} will be used."
        )
    else:
        st.info(
            f"📅 **IS_BACKTEST = False** — using all available data (live mode)  "
            f"| Strategy: **{strategy_name}**"
        )

    st.markdown(
        "Configure **IS_BACKTEST**, **CURRENT_DATE**, and **STRATEGY** in the sidebar, "
        "then click **Run Simulation** to see what the bot would do."
    )

    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner("Loading data…"):
            sim_raw = load_data()

        if sim_raw.empty:
            st.error(
                "No data available for the selected settings. "
                "Try a later CURRENT_DATE or switch to synthetic data."
            )
            st.stop()

        with st.spinner("Computing indicators…"):
            sim_ind = indicators.add_indicators(sim_raw)

        portfolio = Portfolio(config.INITIAL_CAPITAL)
        strategy = _get_strategy()
        agent = TradingAgent(portfolio, strategy)
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
        st.session_state["sim_date"] = str(current_date) if is_backtest else "live"
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
    st.caption(f"Results: strategy = **{_sim_strat_used}** | data cutoff = **{_sim_date_used}**")

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
            y=config.INITIAL_CAPITAL, line_dash="dash", line_color="grey",
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

    fast_col = f"EMA_{config.EMA_FAST}"
    slow_col = f"EMA_{config.EMA_SLOW}"

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=t_df["Datetime"], y=t_df["Close"], name="Close",
        line=dict(color="royalblue", width=1),
    ))
    if fast_col in t_df.columns:
        fig_sig.add_trace(go.Scatter(
            x=t_df["Datetime"], y=t_df[fast_col], name=f"EMA {config.EMA_FAST}",
            line=dict(color="orange", width=1.5, dash="dash"),
        ))
    if slow_col in t_df.columns:
        fig_sig.add_trace(go.Scatter(
            x=t_df["Datetime"], y=t_df[slow_col], name=f"EMA {config.EMA_SLOW}",
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
        ind_df = indicators.add_indicators(raw_df)

    st.session_state["ind_df"] = ind_df

    unique_tickers = sorted(ind_df["Ticker"].unique())
    chart_ticker = st.selectbox("Select Ticker", unique_tickers)
    t_df = ind_df[ind_df["Ticker"] == chart_ticker].copy()

    fast_col = f"EMA_{config.EMA_FAST}"
    slow_col = f"EMA_{config.EMA_SLOW}"

    # --- Price + EMA chart ---
    st.subheader(f"{chart_ticker} – Price & EMAs")
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df["Close"], name="Close", line=dict(color="royalblue", width=1))
    )
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df[fast_col], name=f"EMA {config.EMA_FAST}", line=dict(color="orange", width=1.5, dash="dash"))
    )
    fig_price.add_trace(
        go.Scatter(x=t_df["Datetime"], y=t_df[slow_col], name=f"EMA {config.EMA_SLOW}", line=dict(color="red", width=1.5, dash="dash"))
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
    st.subheader(f"{chart_ticker} – RSI ({config.RSI_PERIOD})")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df["RSI"], name="RSI", line=dict(color="purple")))
    fig_rsi.add_hline(y=config.RSI_OVERBOUGHT, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=config.RSI_OVERSOLD, line_dash="dash", line_color="green", annotation_text="Oversold")
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
            ind_df = indicators.add_indicators(raw_df)
        st.session_state["ind_df"] = ind_df

    if st.button("▶ Generate Signals", type="primary"):
        portfolio = Portfolio(config.INITIAL_CAPITAL)
        agent = TradingAgent(portfolio, _get_strategy())
        with st.spinner("Running trading agent…"):
            ready_df = agent.prepare_signals_df(ind_df)
            signals_df = agent.run(ready_df)
        st.session_state["signals_df"] = signals_df
        st.session_state["signal_portfolio"] = portfolio
        st.session_state["signals_strategy"] = strategy_name
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

    fast_col = f"EMA_{config.EMA_FAST}"
    slow_col = f"EMA_{config.EMA_SLOW}"

    buys = t_df[t_df["Signal"] == "BUY"]
    sells = t_df[t_df["Signal"] == "SELL"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df["Close"], name="Close", line=dict(color="royalblue", width=1)))
    if fast_col in t_df.columns:
        fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df[fast_col], name=f"EMA {config.EMA_FAST}", line=dict(color="orange", width=1.5, dash="dash")))
    if slow_col in t_df.columns:
        fig.add_trace(go.Scatter(x=t_df["Datetime"], y=t_df[slow_col], name=f"EMA {config.EMA_SLOW}", line=dict(color="red", width=1.5, dash="dash")))
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
            ind_df = indicators.add_indicators(raw_df)
        st.session_state["ind_df"] = ind_df

    col_a, col_b = st.columns(2)
    initial_capital = col_a.number_input(
        "Initial Capital (PHP)",
        min_value=10_000,
        max_value=100_000_000,
        value=int(config.INITIAL_CAPITAL),
        step=100_000,
    )

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Running backtest…"):
            bt = Backtester(initial_capital=initial_capital, strategy=_get_strategy())
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
        fig_eq.add_hline(y=initial_capital, line_dash="dash", line_color="grey", annotation_text="Initial Capital")
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
