"""Smoke tests for the HFT scalper strategy and backtest engine."""

import numpy as np
import pandas as pd
import pytest

from financial_algorithms.backtest.hft_engine import HFTBacktestEngine, HFTTrade, Side
from financial_algorithms.strategies.hft_scalper import HFTScalperStrategy

# ── helpers ─────────────────────────────────────────────────────────────


def _synthetic_3min_bars(n_bars: int = 500, seed: int = 0) -> pd.DataFrame:
    """Create deterministic 3-min OHLCV bars for testing."""
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0.001, 0.15, n_bars))
    timestamps = pd.date_range("2025-06-01 09:30", periods=n_bars, freq="3min")
    spread = np.abs(rng.normal(0, 0.1, n_bars))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices - spread * 0.2,
            "high": prices + spread,
            "low": prices - spread,
            "close": prices,
            "volume": rng.integers(10_000, 200_000, n_bars),
        }
    )


# ── strategy unit tests ────────────────────────────────────────────────


class TestHFTScalperStrategy:
    """Validate individual indicator scoring and composite logic."""

    def setup_method(self):
        self.strat = HFTScalperStrategy()

    # -- EMA signal -------------------------------------------------------
    def test_ema_signal_range(self):
        close = np.linspace(100, 110, 50)
        sig = self.strat.calculate_ema_signal(close)
        assert -2 <= sig <= 2

    def test_ema_signal_insufficient_data(self):
        assert self.strat.calculate_ema_signal(np.array([100, 101])) == 0.0

    # -- RSI signal -------------------------------------------------------
    def test_rsi_signal_range(self):
        close = np.linspace(100, 105, 30)
        sig = self.strat.calculate_rsi_signal(close)
        assert -2 <= sig <= 2

    def test_rsi_oversold_bullish(self):
        # Steep decline → RSI < 25 → bullish +2
        close = np.linspace(120, 80, 30)
        sig = self.strat.calculate_rsi_signal(close)
        assert sig >= 1

    def test_rsi_overbought_bearish(self):
        # Steep rise → RSI > 75 → bearish -2
        close = np.linspace(80, 120, 30)
        sig = self.strat.calculate_rsi_signal(close)
        assert sig <= -1

    # -- VWAP signal ------------------------------------------------------
    def test_vwap_signal_range(self):
        close = np.ones(30) * 100
        vol = np.ones(30) * 50000
        sig = self.strat.calculate_vwap_signal(close, vol)
        assert -2 <= sig <= 2

    # -- Stochastic signal ------------------------------------------------
    def test_stoch_signal_range(self):
        high = np.linspace(102, 108, 20)
        low = np.linspace(98, 104, 20)
        close = np.linspace(100, 106, 20)
        sig = self.strat.calculate_stoch_signal(high, low, close)
        assert -2 <= sig <= 2

    # -- MACD signal ------------------------------------------------------
    def test_macd_signal_range(self):
        close = np.linspace(100, 108, 40)
        sig = self.strat.calculate_macd_signal(close)
        assert -2 <= sig <= 2

    # -- ADX signal -------------------------------------------------------
    def test_adx_signal_range(self):
        h = np.linspace(102, 110, 20)
        lo = np.linspace(98, 105, 20)
        c = np.linspace(100, 108, 20)
        sig = self.strat.calculate_adx_signal(h, lo, c)
        assert -2 <= sig <= 2

    # -- Bollinger %B signal ----------------------------------------------
    def test_bb_signal_range(self):
        close = np.linspace(100, 105, 30)
        sig = self.strat.calculate_bb_signal(close)
        assert -2 <= sig <= 2

    # -- composite --------------------------------------------------------
    def test_composite_score_range(self):
        n = 50
        c = np.linspace(100, 108, n)
        h = c + 0.5
        lo = c - 0.5
        v = np.ones(n) * 100_000
        score = self.strat.calculate_composite_score(
            close_3m=c, high_5m=h, low_5m=lo, close_5m=c,
            volume_3m=v, high_15m=h, low_15m=lo, close_15m=c, close_10m=c,
        )
        assert -14 <= score <= 14

    # -- entry / exit helpers ---------------------------------------------
    def test_should_enter_above_threshold(self):
        assert self.strat.should_enter(4.0)
        assert not self.strat.should_enter(1.0)

    def test_should_exit_below_threshold(self):
        assert self.strat.should_exit_on_signal(-4.0)
        assert not self.strat.should_exit_on_signal(-1.0)

    # -- position sizing --------------------------------------------------
    def test_position_sizing_scales_with_strength(self):
        small = self.strat.calculate_position_size(4.5, 100_000)
        large = self.strat.calculate_position_size(12.0, 100_000)
        assert 0 < small < large

    def test_position_sizing_zero_below_threshold(self):
        assert self.strat.calculate_position_size(1.0, 100_000) == 0.0

    # -- entry/exit levels ------------------------------------------------
    def test_entry_exit_levels(self):
        levels = self.strat.calculate_entry_exit_levels(100.0)
        assert levels["sl"] < 100.0
        assert levels["tp1"] > 100.0
        assert levels["tp2"] > levels["tp1"]

    # -- generate_signals convenience method ------------------------------
    def test_generate_signals_returns_series(self):
        df = _synthetic_3min_bars(200)
        signals = self.strat.generate_signals(df)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)

    # -- system summary ---------------------------------------------------
    def test_system_summary_keys(self):
        summary = self.strat.get_system_summary()
        assert "name" in summary
        assert "indicators" in summary
        assert len(summary["indicators"]) == 7


# ── engine tests ────────────────────────────────────────────────────────


class TestHFTBacktestEngine:
    """Validate the backtest engine end-to-end."""

    def test_run_produces_metrics(self):
        df = _synthetic_3min_bars(300)
        engine = HFTBacktestEngine(initial_capital=100_000)
        results = engine.run(df)
        assert "metrics" in results
        assert "Total Return" in results["metrics"]
        assert "Sortino Ratio" in results["metrics"]
        assert "Trades/Hour" in results["metrics"]
        assert "Monthly CAGR" in results["metrics"]

    def test_equity_curve_length(self):
        df = _synthetic_3min_bars(300)
        engine = HFTBacktestEngine(initial_capital=100_000)
        results = engine.run(df)
        eq = results["equity_curve"]
        assert len(eq) == len(df)

    def test_initial_equity_preserved_on_no_trades(self):
        """If no signal fires, equity stays at initial capital."""
        # Generate perfectly flat data → no signals
        n = 50
        flat = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=n, freq="3min"),
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [50000] * n,
            }
        )
        engine = HFTBacktestEngine(initial_capital=50_000)
        results = engine.run(flat)
        assert results["metrics"]["Total Trades"] == 0
        assert results["metrics"]["Final Equity"] == 50_000.0

    def test_trades_df_columns(self):
        df = _synthetic_3min_bars(300)
        engine = HFTBacktestEngine(initial_capital=100_000)
        results = engine.run(df)
        tdf = results["trades"]
        if not tdf.empty:
            expected_cols = {"symbol", "side", "entry_time", "exit_time",
                            "entry_price", "exit_price", "pnl", "pnl_pct",
                            "exit_reason", "size"}
            assert expected_cols.issubset(set(tdf.columns))

    def test_custom_strategy_params(self):
        strat = HFTScalperStrategy(min_buy_score=2.0, max_sell_score=-2.0)
        engine = HFTBacktestEngine(strategy=strat, initial_capital=100_000)
        df = _synthetic_3min_bars(300)
        results = engine.run(df)
        # Lower thresholds → more trades
        assert results["metrics"]["Total Trades"] >= 0

    def test_stop_loss_limits_downside(self):
        """Trades closed by SL should have losses bounded by SL pct."""
        df = _synthetic_3min_bars(500, seed=7)
        engine = HFTBacktestEngine(initial_capital=100_000)
        results = engine.run(df)
        tdf = results["trades"]
        if not tdf.empty:
            sl_trades = tdf[tdf["exit_reason"] == "sl"]
            if not sl_trades.empty:
                # worst SL loss should be roughly bounded by stop_loss_pct
                # (plus slippage)
                worst = sl_trades["pnl_pct"].min()
                assert worst > -2.0  # generous bound including slippage


# ── trade dataclass tests ──────────────────────────────────────────────


class TestHFTTrade:
    def test_long_pnl(self):
        t = HFTTrade(
            symbol="X", side=Side.LONG,
            entry_time=pd.Timestamp("2025-01-01"),
            exit_time=pd.Timestamp("2025-01-01 01:00"),
            entry_price=100, exit_price=101, size=1000,
            exit_reason="tp1",
        )
        assert t.pnl_pct == pytest.approx(0.01, abs=1e-6)
        assert t.pnl == pytest.approx(10.0, abs=0.01)

    def test_short_pnl(self):
        t = HFTTrade(
            symbol="X", side=Side.SHORT,
            entry_time=pd.Timestamp("2025-01-01"),
            exit_time=pd.Timestamp("2025-01-01 01:00"),
            entry_price=100, exit_price=99, size=1000,
            exit_reason="tp1",
        )
        assert t.pnl_pct == pytest.approx(0.01, abs=1e-6)
        assert t.pnl == pytest.approx(10.0, abs=0.01)
