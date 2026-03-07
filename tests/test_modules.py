"""
tests/test_modules.py - Unit tests for the PSE Trading Bot modules.

Tests cover indicators, portfolio logic, risk management, trading agent
signals, storage helpers, and the backtester – all without requiring a live
internet connection.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on the path when running from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import indicators
import storage
from backtester import Backtester, _max_drawdown, _sharpe_ratio
from portfolio import Portfolio
from risk_management import (
    check_daily_loss_limit,
    check_stop_loss,
    check_take_profit,
    compute_position_size,
)
from trading_agent import EMACrossoverStrategy, TradingAgent
from trading_agent import RSIStrategy, BollingerBandStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 100, start: float = 100.0, seed: int = 42) -> pd.Series:
    """Generate a deterministic price series for testing."""
    rng = np.random.default_rng(seed)
    changes = rng.normal(0, 0.5, size=n)
    prices = start + np.cumsum(changes)
    return pd.Series(np.clip(prices, 1.0, None))


def _make_ohlcv_df(
    tickers: list[str] = ("BDO.PS", "SM.PS"),
    n_candles: int = 50,
) -> pd.DataFrame:
    """Build a minimal multi-ticker OHLCV DataFrame for testing."""
    base_time = datetime(2024, 1, 2, 9, 30)
    frames = []
    for i, ticker in enumerate(tickers):
        prices = _make_price_series(n=n_candles, start=100.0 + i * 10, seed=i)
        datetimes = [base_time + timedelta(minutes=j) for j in range(n_candles)]
        df = pd.DataFrame(
            {
                "Datetime": datetimes,
                "Open": prices * 0.999,
                "High": prices * 1.002,
                "Low": prices * 0.997,
                "Close": prices.values,
                "Volume": np.full(n_candles, 10000),
                "Ticker": ticker,
            }
        )
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["Datetime", "Ticker"], inplace=True)
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# indicators tests
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_ema_length(self):
        s = _make_price_series(50)
        result = indicators.ema(s, 9)
        assert len(result) == len(s)

    def test_ema_values_not_all_nan(self):
        s = _make_price_series(50)
        result = indicators.ema(s, 9)
        assert result.notna().any()

    def test_rsi_range(self):
        s = _make_price_series(100)
        result = indicators.rsi(s, 14).dropna()
        assert ((result >= 0) & (result <= 100)).all()

    def test_bollinger_bands_shape(self):
        s = _make_price_series(50)
        upper, mid, lower = indicators.bollinger_bands(s, 20, 2.0)
        assert len(upper) == len(s)
        # Upper should be >= middle >= lower (where not NaN)
        valid = ~(upper.isna() | mid.isna() | lower.isna())
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()

    def test_returns_first_is_nan(self):
        s = _make_price_series(30)
        ret = indicators.returns(s)
        assert pd.isna(ret.iloc[0])

    def test_add_indicators_columns(self):
        df = _make_ohlcv_df()
        result = indicators.add_indicators(df)
        for col in [f"EMA_{config.EMA_FAST}", f"EMA_{config.EMA_SLOW}", "RSI",
                    "BB_upper", "BB_middle", "BB_lower", "Returns", "Volatility"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_indicators_row_count(self):
        df = _make_ohlcv_df(n_candles=40)
        result = indicators.add_indicators(df)
        assert len(result) == len(df)

    def test_add_indicators_empty(self):
        empty = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"])
        result = indicators.add_indicators(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# portfolio tests
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(1_000_000)
        assert p.cash == 1_000_000
        assert p.positions == {}
        assert p.trade_log == []

    def test_buy_reduces_cash(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        assert p.cash == pytest.approx(1_000_000 - 100 * 50.0)

    def test_buy_creates_position(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        assert "BDO.PS" in p.positions
        assert p.positions["BDO.PS"].shares == 100

    def test_buy_insufficient_cash_rejected(self):
        p = Portfolio(100)
        result = p.buy("BDO.PS", 100, 50.0, datetime.now())
        assert result is False
        assert "BDO.PS" not in p.positions

    def test_sell_increases_cash(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        cash_after_buy = p.cash
        p.sell("BDO.PS", 100, 60.0, datetime.now())
        assert p.cash > cash_after_buy

    def test_sell_computes_realized_pnl(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        p.sell("BDO.PS", 100, 60.0, datetime.now())
        assert p.daily_realized_pnl == pytest.approx(1000.0)

    def test_sell_removes_position(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        p.sell("BDO.PS", 100, 60.0, datetime.now())
        assert "BDO.PS" not in p.positions

    def test_sell_no_position_rejected(self):
        p = Portfolio(1_000_000)
        result = p.sell("BDO.PS", 100, 60.0, datetime.now())
        assert result is False

    def test_market_value(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        mv = p.market_value({"BDO.PS": 55.0})
        assert mv == pytest.approx(1_000_000 - 5000 + 100 * 55.0)

    def test_unrealized_pnl(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        upnl = p.unrealized_pnl({"BDO.PS": 55.0})
        assert upnl["BDO.PS"] == pytest.approx(500.0)

    def test_to_trade_log_df(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        p.sell("BDO.PS", 100, 60.0, datetime.now())
        df = p.to_trade_log_df()
        assert len(df) == 2
        assert set(df["action"]) == {"BUY", "SELL"}

    def test_summary_keys(self):
        p = Portfolio(1_000_000)
        summary = p.summary()
        for key in ["cash", "initial_capital", "total_realized_pnl", "open_positions"]:
            assert key in summary


# ---------------------------------------------------------------------------
# risk_management tests
# ---------------------------------------------------------------------------

class TestRiskManagement:
    def test_compute_position_size_basic(self):
        shares = compute_position_size(1_000_000, 100.0, max_pct=0.05)
        assert shares == 500.0  # 5% of 1M / 100 per share

    def test_compute_position_size_zero_price(self):
        assert compute_position_size(1_000_000, 0.0) == 0.0

    def test_stop_loss_triggered(self):
        assert check_stop_loss(100.0, 97.0, 0.02) is True  # 3% drop > 2% threshold

    def test_stop_loss_not_triggered(self):
        assert check_stop_loss(100.0, 99.0, 0.02) is False  # 1% drop < 2% threshold

    def test_take_profit_triggered(self):
        assert check_take_profit(100.0, 105.0, 0.04) is True  # 5% gain > 4% threshold

    def test_take_profit_not_triggered(self):
        assert check_take_profit(100.0, 102.0, 0.04) is False  # 2% gain < 4%

    def test_daily_loss_limit_exceeded(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 1000, 50.0, datetime.now())
        p.sell("BDO.PS", 1000, 19.0, datetime.now())  # large loss: -31000
        assert check_daily_loss_limit(p, max_daily_loss_pct=0.03) is True

    def test_daily_loss_limit_not_exceeded(self):
        p = Portfolio(1_000_000)
        p.buy("BDO.PS", 100, 50.0, datetime.now())
        p.sell("BDO.PS", 100, 49.0, datetime.now())  # tiny loss
        assert check_daily_loss_limit(p, max_daily_loss_pct=0.03) is False


# ---------------------------------------------------------------------------
# trading_agent tests
# ---------------------------------------------------------------------------

class TestTradingAgent:
    def _make_crossover_row(self, fast: float, slow: float, prev_fast: float, prev_slow: float) -> pd.Series:
        return pd.Series({
            f"EMA_{config.EMA_FAST}": fast,
            f"EMA_{config.EMA_SLOW}": slow,
            f"prev_EMA_{config.EMA_FAST}": prev_fast,
            f"prev_EMA_{config.EMA_SLOW}": prev_slow,
        })

    def test_buy_signal_on_golden_cross(self):
        strategy = EMACrossoverStrategy()
        row = self._make_crossover_row(fast=102, slow=100, prev_fast=99, prev_slow=100)
        assert strategy.generate_signal(row, "BDO.PS") == "BUY"

    def test_sell_signal_on_death_cross(self):
        strategy = EMACrossoverStrategy()
        row = self._make_crossover_row(fast=98, slow=100, prev_fast=101, prev_slow=100)
        assert strategy.generate_signal(row, "BDO.PS") == "SELL"

    def test_hold_signal_no_crossover(self):
        strategy = EMACrossoverStrategy()
        row = self._make_crossover_row(fast=102, slow=100, prev_fast=103, prev_slow=100)
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_hold_signal_nan_values(self):
        strategy = EMACrossoverStrategy()
        row = self._make_crossover_row(fast=float("nan"), slow=100, prev_fast=99, prev_slow=100)
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_agent_adds_signal_column(self):
        p = Portfolio(1_000_000)
        agent = TradingAgent(p)
        df = _make_ohlcv_df(n_candles=50)
        df_ind = indicators.add_indicators(df)
        df_ready = agent.prepare_signals_df(df_ind)
        df_signals = agent.run(df_ready)
        assert "Signal" in df_signals.columns

    def test_agent_signals_are_valid(self):
        p = Portfolio(1_000_000)
        agent = TradingAgent(p)
        df = _make_ohlcv_df(n_candles=50)
        df_ind = indicators.add_indicators(df)
        df_ready = agent.prepare_signals_df(df_ind)
        df_signals = agent.run(df_ready)
        valid = {"BUY", "SELL", "HOLD", "HALT"}
        assert set(df_signals["Signal"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# storage tests
# ---------------------------------------------------------------------------

class TestStorage:
    def test_save_and_load_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = str(tmp_path / "test.csv")
        storage.save_csv(df, path)
        loaded = storage.load_csv(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_csv_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.csv")
        result = storage.load_csv(path)
        assert result.empty

    def test_save_and_load_parquet(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        path = str(tmp_path / "test.parquet")
        storage.save_parquet(df, path)
        loaded = storage.load_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_parquet_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.parquet")
        result = storage.load_parquet(path)
        assert result.empty


# ---------------------------------------------------------------------------
# backtester metric tests
# ---------------------------------------------------------------------------

class TestBacktesterMetrics:
    def test_sharpe_ratio_positive_returns(self):
        returns = pd.Series([0.001] * 100)
        sr = _sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_ratio_zero_std(self):
        returns = pd.Series([0.0] * 100)
        sr = _sharpe_ratio(returns)
        assert sr == 0.0

    def test_max_drawdown_flat(self):
        equity = pd.Series([100.0] * 50)
        dd = _max_drawdown(equity)
        assert dd == pytest.approx(0.0)

    def test_max_drawdown_declining(self):
        equity = pd.Series([100.0, 90.0, 80.0])
        dd = _max_drawdown(equity)
        assert dd == pytest.approx(-0.2)

    def test_backtester_run_returns_dict(self):
        df = _make_ohlcv_df(n_candles=60)
        df_ind = indicators.add_indicators(df)
        bt = Backtester(initial_capital=100_000)
        metrics = bt.run(df_ind)
        for key in ["trade_log", "equity_curve", "total_return_pct",
                    "sharpe_ratio", "max_drawdown", "total_trades", "win_rate"]:
            assert key in metrics

    def test_backtester_equity_curve_length(self):
        df = _make_ohlcv_df(n_candles=30)
        df_ind = indicators.add_indicators(df)
        bt = Backtester(initial_capital=100_000)
        metrics = bt.run(df_ind)
        # Equity curve should have one entry per unique Datetime
        n_timestamps = df["Datetime"].nunique()
        assert len(metrics["equity_curve"]) == n_timestamps


# ---------------------------------------------------------------------------
# RSIStrategy tests
# ---------------------------------------------------------------------------

class TestRSIStrategy:
    def _make_rsi_row(self, rsi: float, prev_rsi: float) -> pd.Series:
        return pd.Series({"RSI": rsi, "prev_RSI": prev_rsi})

    def test_buy_signal_on_oversold_recovery(self):
        strategy = RSIStrategy(oversold=30, overbought=70)
        row = self._make_rsi_row(rsi=31.0, prev_rsi=29.0)  # crosses back above 30
        assert strategy.generate_signal(row, "BDO.PS") == "BUY"

    def test_sell_signal_on_overbought_retreat(self):
        strategy = RSIStrategy(oversold=30, overbought=70)
        row = self._make_rsi_row(rsi=69.0, prev_rsi=71.0)  # crosses back below 70
        assert strategy.generate_signal(row, "BDO.PS") == "SELL"

    def test_hold_signal_no_crossover(self):
        strategy = RSIStrategy(oversold=30, overbought=70)
        row = self._make_rsi_row(rsi=50.0, prev_rsi=49.0)
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_hold_signal_nan_values(self):
        strategy = RSIStrategy()
        row = self._make_rsi_row(rsi=float("nan"), prev_rsi=29.0)
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_hold_signal_missing_column(self):
        strategy = RSIStrategy()
        row = pd.Series({"RSI": 31.0})  # prev_RSI missing
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_lag_columns(self):
        assert RSIStrategy().lag_columns == ["RSI"]

    def test_prepare_signals_df_adds_prev_rsi(self):
        p = Portfolio(1_000_000)
        agent = TradingAgent(p, RSIStrategy())
        df = _make_ohlcv_df(n_candles=50)
        df_ind = indicators.add_indicators(df)
        df_ready = agent.prepare_signals_df(df_ind)
        assert "prev_RSI" in df_ready.columns

    def test_agent_with_rsi_strategy_signals_valid(self):
        p = Portfolio(1_000_000)
        agent = TradingAgent(p, RSIStrategy())
        df = _make_ohlcv_df(n_candles=50)
        df_ind = indicators.add_indicators(df)
        df_ready = agent.prepare_signals_df(df_ind)
        df_signals = agent.run(df_ready)
        valid = {"BUY", "SELL", "HOLD", "HALT"}
        assert set(df_signals["Signal"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# BollingerBandStrategy tests
# ---------------------------------------------------------------------------

class TestBollingerBandStrategy:
    def _make_bb_row(
        self,
        close: float,
        prev_close: float,
        bb_upper: float,
        bb_lower: float,
        prev_bb_upper: float,
        prev_bb_lower: float,
    ) -> pd.Series:
        return pd.Series({
            "Close": close,
            "prev_Close": prev_close,
            "BB_upper": bb_upper,
            "BB_lower": bb_lower,
            "prev_BB_upper": prev_bb_upper,
            "prev_BB_lower": prev_bb_lower,
        })

    def test_buy_signal_on_lower_band_recovery(self):
        strategy = BollingerBandStrategy()
        # prev_close below lower band, close crosses back above it
        row = self._make_bb_row(
            close=99.0, prev_close=98.0,
            bb_upper=110.0, bb_lower=99.0,
            prev_bb_upper=110.0, prev_bb_lower=99.0,
        )
        assert strategy.generate_signal(row, "BDO.PS") == "BUY"

    def test_sell_signal_on_upper_band_retreat(self):
        strategy = BollingerBandStrategy()
        # prev_close above upper band, close crosses back below it
        row = self._make_bb_row(
            close=110.0, prev_close=111.0,
            bb_upper=110.0, bb_lower=90.0,
            prev_bb_upper=110.0, prev_bb_lower=90.0,
        )
        assert strategy.generate_signal(row, "BDO.PS") == "SELL"

    def test_hold_signal_inside_bands(self):
        strategy = BollingerBandStrategy()
        row = self._make_bb_row(
            close=100.0, prev_close=100.0,
            bb_upper=110.0, bb_lower=90.0,
            prev_bb_upper=110.0, prev_bb_lower=90.0,
        )
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_hold_signal_nan_values(self):
        strategy = BollingerBandStrategy()
        row = self._make_bb_row(
            close=float("nan"), prev_close=98.0,
            bb_upper=110.0, bb_lower=99.0,
            prev_bb_upper=110.0, prev_bb_lower=99.0,
        )
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_hold_signal_missing_column(self):
        strategy = BollingerBandStrategy()
        row = pd.Series({"Close": 100.0, "BB_upper": 110.0})  # many columns missing
        assert strategy.generate_signal(row, "BDO.PS") == "HOLD"

    def test_lag_columns(self):
        assert BollingerBandStrategy().lag_columns == ["Close", "BB_upper", "BB_lower"]

    def test_prepare_signals_df_adds_prev_bb_columns(self):
        p = Portfolio(1_000_000)
        agent = TradingAgent(p, BollingerBandStrategy())
        df = _make_ohlcv_df(n_candles=50)
        df_ind = indicators.add_indicators(df)
        df_ready = agent.prepare_signals_df(df_ind)
        for col in ["prev_Close", "prev_BB_upper", "prev_BB_lower"]:
            assert col in df_ready.columns, f"Missing column: {col}"

    def test_agent_with_bb_strategy_signals_valid(self):
        p = Portfolio(1_000_000)
        agent = TradingAgent(p, BollingerBandStrategy())
        df = _make_ohlcv_df(n_candles=50)
        df_ind = indicators.add_indicators(df)
        df_ready = agent.prepare_signals_df(df_ind)
        df_signals = agent.run(df_ready)
        valid = {"BUY", "SELL", "HOLD", "HALT"}
        assert set(df_signals["Signal"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# load_data date-filter regression tests
# (mirrors the logic in streamlit_app.py::load_data to guard against the
# AttributeError that occurs when the Datetime column has object dtype)
# ---------------------------------------------------------------------------

class TestLoadDataDateFilter:
    """
    Regression tests for the IS_BACKTEST date-filter logic in load_data().

    When live data fetches fail, fetch_all_tickers returns an empty DataFrame
    whose Datetime column has object dtype.  Calling .dt.date on an object
    Series raises AttributeError.  The fix is to call pd.to_datetime() before
    the filter.  These tests verify that behaviour is correct for both the
    empty-DataFrame case and the normal populated case.
    """

    @staticmethod
    def _apply_backtest_filter(df: pd.DataFrame, cutoff_date) -> pd.DataFrame:
        """Reproduces the fixed load_data() filter logic."""
        df = df.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        if not df.empty:
            df = df[df["Datetime"].dt.date <= cutoff_date].copy()
        return df

    def test_empty_object_dtype_datetime_does_not_raise(self):
        """Empty DataFrame with object-dtype Datetime must not raise AttributeError."""
        empty = pd.DataFrame(
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"]
        )
        assert empty["Datetime"].dtype == object  # pre-condition: object dtype
        cutoff = datetime(2026, 2, 19).date()
        result = self._apply_backtest_filter(empty, cutoff)
        assert result.empty

    def test_populated_object_dtype_datetime_filters_correctly(self):
        """Object-dtype Datetime strings must be coerced and filtered correctly."""
        df = pd.DataFrame({
            "Datetime": ["2024-01-15 09:30:00", "2024-01-16 09:30:00", "2024-01-17 09:30:00"],
            "Close": [100.0, 101.0, 102.0],
            "Ticker": ["BDO.PS"] * 3,
        })
        assert df["Datetime"].dtype == object  # pre-condition: object dtype
        cutoff = datetime(2024, 1, 16).date()
        result = self._apply_backtest_filter(df, cutoff)
        assert len(result) == 2
        assert all(result["Datetime"].dt.date <= cutoff)

    def test_datetime64_dtype_still_works(self):
        """datetime64 columns (normal synthetic/live path) continue to work."""
        df = _make_ohlcv_df(n_candles=10)
        # Confirm the fixture produces datetime64 (or promote it so the test is explicit)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df["Datetime"])
        # Synthetic data starts 2024-01-02 09:30; cutoff 2024-01-15 includes all rows
        cutoff = datetime(2024, 1, 15).date()
        result = self._apply_backtest_filter(df, cutoff)
        assert len(result) == len(df)
        assert all(result["Datetime"].dt.date <= cutoff)

    def test_backtest_cutoff_excludes_future_rows(self):
        """Rows after CURRENT_DATE are excluded when IS_BACKTEST is True."""
        # Build a dataset that spans two calendar days
        df = pd.DataFrame({
            "Datetime": [
                "2024-01-02 09:30:00",  # day 1 — should be kept
                "2024-01-02 10:00:00",  # day 1 — should be kept
                "2024-01-03 09:30:00",  # day 2 — should be excluded
                "2024-01-04 09:30:00",  # day 3 — should be excluded
            ],
            "Close": [100.0, 101.0, 102.0, 103.0],
            "Ticker": ["BDO.PS"] * 4,
        })
        cutoff = datetime(2024, 1, 2).date()
        result = self._apply_backtest_filter(df, cutoff)
        assert len(result) == 2
        assert all(result["Datetime"].dt.date <= cutoff)


# ---------------------------------------------------------------------------
# load_data date-range filter tests (START_DATE + END_DATE)
# ---------------------------------------------------------------------------

class TestLoadDataDateRangeFilter:
    """
    Tests for the IS_BACKTEST START_DATE / END_DATE range filter in load_data().

    When IS_BACKTEST = True the bot should only process candles whose date
    falls within [start_date, end_date] (both inclusive).
    """

    @staticmethod
    def _apply_range_filter(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
        """Reproduces the load_data() range-filter logic."""
        df = df.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        if not df.empty:
            df = df[
                (df["Datetime"].dt.date >= start_date) &
                (df["Datetime"].dt.date <= end_date)
            ].copy()
        return df

    def test_empty_dataframe_does_not_raise(self):
        """Empty DataFrame with object-dtype Datetime must not raise AttributeError."""
        empty = pd.DataFrame(
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"]
        )
        start = datetime(2026, 3, 3).date()
        end = datetime(2026, 3, 6).date()
        result = self._apply_range_filter(empty, start, end)
        assert result.empty

    def test_rows_before_start_date_are_excluded(self):
        """Rows before START_DATE are excluded."""
        df = pd.DataFrame({
            "Datetime": [
                "2026-03-01 09:30:00",  # before start — excluded
                "2026-03-02 09:30:00",  # before start — excluded
                "2026-03-03 09:30:00",  # start date   — kept
                "2026-03-04 09:30:00",  # within range — kept
                "2026-03-06 09:30:00",  # end date     — kept
            ],
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Ticker": ["BDO.PS"] * 5,
        })
        start = datetime(2026, 3, 3).date()
        end = datetime(2026, 3, 6).date()
        result = self._apply_range_filter(df, start, end)
        assert len(result) == 3
        assert all(result["Datetime"].dt.date >= start)
        assert all(result["Datetime"].dt.date <= end)

    def test_rows_after_end_date_are_excluded(self):
        """Rows after END_DATE are excluded."""
        df = pd.DataFrame({
            "Datetime": [
                "2026-03-03 09:30:00",  # start date   — kept
                "2026-03-05 09:30:00",  # within range — kept
                "2026-03-07 09:30:00",  # after end    — excluded
                "2026-03-08 09:30:00",  # after end    — excluded
            ],
            "Close": [100.0, 101.0, 102.0, 103.0],
            "Ticker": ["BDO.PS"] * 4,
        })
        start = datetime(2026, 3, 3).date()
        end = datetime(2026, 3, 6).date()
        result = self._apply_range_filter(df, start, end)
        assert len(result) == 2
        assert all(result["Datetime"].dt.date >= start)
        assert all(result["Datetime"].dt.date <= end)

    def test_start_and_end_dates_are_inclusive(self):
        """Both START_DATE and END_DATE boundary rows must be retained."""
        df = pd.DataFrame({
            "Datetime": [
                "2026-03-02 09:30:00",  # before — excluded
                "2026-03-03 09:30:00",  # == start — kept
                "2026-03-06 15:59:00",  # == end   — kept
                "2026-03-07 09:30:00",  # after    — excluded
            ],
            "Close": [100.0, 101.0, 102.0, 103.0],
            "Ticker": ["BDO.PS"] * 4,
        })
        start = datetime(2026, 3, 3).date()
        end = datetime(2026, 3, 6).date()
        result = self._apply_range_filter(df, start, end)
        assert len(result) == 2
        dates = result["Datetime"].dt.date.tolist()
        assert datetime(2026, 3, 3).date() in dates
        assert datetime(2026, 3, 6).date() in dates

    def test_no_rows_in_range_returns_empty(self):
        """Filter returns empty DataFrame when no candles fall in the date range."""
        df = pd.DataFrame({
            "Datetime": [
                "2026-02-01 09:30:00",
                "2026-04-01 09:30:00",
            ],
            "Close": [100.0, 101.0],
            "Ticker": ["BDO.PS"] * 2,
        })
        start = datetime(2026, 3, 3).date()
        end = datetime(2026, 3, 6).date()
        result = self._apply_range_filter(df, start, end)
        assert result.empty

    def test_all_rows_in_range_are_kept(self):
        """All rows are kept when the entire dataset falls within the date range."""
        df = pd.DataFrame({
            "Datetime": [
                "2026-03-03 09:30:00",
                "2026-03-04 09:30:00",
                "2026-03-05 09:30:00",
                "2026-03-06 09:30:00",
            ],
            "Close": [100.0, 101.0, 102.0, 103.0],
            "Ticker": ["BDO.PS"] * 4,
        })
        start = datetime(2026, 3, 3).date()
        end = datetime(2026, 3, 6).date()
        result = self._apply_range_filter(df, start, end)
        assert len(result) == 4

    def test_problem_statement_example(self):
        """
        Mirrors the problem statement example:
          CURRENT_DATE=20260303, Period=7d, END_DATE=20260306
          => bot trades within 20260303 to 20260306 only.
        """
        df = pd.DataFrame({
            "Datetime": [
                "2026-02-24 09:30:00",  # 7 days before start — excluded
                "2026-03-03 09:30:00",  # START_DATE          — kept
                "2026-03-04 09:30:00",  # within range        — kept
                "2026-03-05 09:30:00",  # within range        — kept
                "2026-03-06 09:30:00",  # END_DATE            — kept
                "2026-03-07 09:30:00",  # after END_DATE      — excluded
            ],
            "Close": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "Ticker": ["BDO.PS"] * 6,
        })
        start = datetime(2026, 3, 3).date()   # CURRENT_DATE in the example
        end = datetime(2026, 3, 6).date()     # END_DATE in the example
        result = self._apply_range_filter(df, start, end)
        assert len(result) == 4
        assert all(result["Datetime"].dt.date >= start)
        assert all(result["Datetime"].dt.date <= end)


# ---------------------------------------------------------------------------
# validate_ticker tests
# ---------------------------------------------------------------------------

class TestValidateTicker:
    """Tests for data_pipeline.validate_ticker() without a live network."""

    def test_valid_ticker_returns_true(self):
        """validate_ticker returns True when yfinance returns non-empty data."""
        from unittest.mock import patch
        from data_pipeline import validate_ticker

        mock_df = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2))
        with patch("data_pipeline.yf.download", return_value=mock_df):
            assert validate_ticker("BDO.PS") is True

    def test_empty_response_returns_false(self):
        """validate_ticker returns False when yfinance returns an empty DataFrame."""
        from unittest.mock import patch
        from data_pipeline import validate_ticker

        with patch("data_pipeline.yf.download", return_value=pd.DataFrame()):
            assert validate_ticker("INVALID.PS") is False

    def test_none_response_returns_false(self):
        """validate_ticker returns False when yfinance returns None."""
        from unittest.mock import patch
        from data_pipeline import validate_ticker

        with patch("data_pipeline.yf.download", return_value=None):
            assert validate_ticker("INVALID.PS") is False

    def test_exception_returns_false(self):
        """validate_ticker returns False when yfinance raises an exception."""
        from unittest.mock import patch
        from data_pipeline import validate_ticker

        with patch("data_pipeline.yf.download", side_effect=Exception("Network error")):
            assert validate_ticker("ERROR.PS") is False
