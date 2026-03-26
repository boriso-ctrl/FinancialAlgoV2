"""Tests for Deep Learning strategies (DL-1, DL-2, DL-3)."""

import numpy as np
import pandas as pd
import pytest
import torch

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.strategies.dl_strategies import (
    TemporalCNNAlpha,
    TemporalCNNConfig,
    LSTMRegimeDetector,
    LSTMRegimeConfig,
    AttentionCrossSectionalRanker,
    AttentionRankerConfig,
    _TemporalCNNNet,
    _LSTMRegimeNet,
    _CrossAssetAttentionNet,
    _safe_tensor,
    _momentum_df,
    _realized_vol_df,
    _rsi_df,
    _zscore_df,
    _above_sma,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_prices(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic prices — enough for DL warm-up (504 min_train_days)."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    tickers = [
        "SPY", "QQQ", "IWM", "GLD", "TLT", "IEF", "UUP",
        "XLE", "EFA", "EEM", "SLV", "DBC", "VNQ",
        "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
        "HYG", "LQD",
    ]
    data = {}
    for t in tickers:
        ret = rng.normal(0.0003, 0.015, n)
        data[t] = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame(data, index=dates)


def _small_prices(n: int = 100, seed: int = 7) -> pd.DataFrame:
    """Tiny price set — not enough for warm-up, should return zeros."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["SPY", "QQQ", "GLD", "TLT"]
    data = {t: 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))) for t in tickers}
    return pd.DataFrame(data, index=dates)


# ── Shared Helpers ───────────────────────────────────────────────────

class TestSharedHelpers:
    def test_momentum_df(self):
        p = _small_prices()
        mom = _momentum_df(p, 21)
        assert isinstance(mom, pd.DataFrame)
        assert mom.shape == p.shape
        assert not mom.isna().any().any()

    def test_realized_vol_df(self):
        p = _small_prices()
        vol = _realized_vol_df(p, 20)
        assert vol.shape == p.shape
        assert (vol >= 0).all().all()

    def test_rsi_df(self):
        p = _small_prices()
        rsi = _rsi_df(p, 14)
        assert rsi.shape == p.shape
        assert rsi.min().min() >= 0
        assert rsi.max().max() <= 100

    def test_zscore_df(self):
        p = _small_prices()
        z = _zscore_df(p, 60)
        assert z.shape == p.shape
        assert not z.isna().any().any()

    def test_above_sma(self):
        p = _small_prices()
        above = _above_sma(p, 50)
        assert above.shape == p.shape
        assert set(above.values.flatten()).issubset({0.0, 1.0})

    def test_safe_tensor(self):
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        t = _safe_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        assert not torch.isnan(t).any()
        assert not torch.isinf(t).any()


# ── Raw NN Modules ───────────────────────────────────────────────────

class TestNNModules:
    def test_temporal_cnn_forward(self):
        net = _TemporalCNNNet(n_features=7, window=60)
        x = torch.randn(4, 7, 60)
        out = net(x)
        assert out.shape == (4,)

    def test_lstm_regime_forward(self):
        net = _LSTMRegimeNet(n_features=8, hidden_size=16, n_regimes=3)
        x = torch.randn(4, 30, 8)
        out = net(x)
        assert out.shape == (4, 3)

    def test_cross_asset_attention_forward(self):
        net = _CrossAssetAttentionNet(n_features=8, d_model=16, n_heads=2)
        x = torch.randn(4, 10, 8)  # 10 assets
        out = net(x)
        assert out.shape == (4, 10)

    def test_cnn_deterministic(self):
        """Same input => same output (eval mode, no dropout)."""
        net = _TemporalCNNNet(7, 60)
        net.eval()
        x = torch.randn(2, 7, 60)
        with torch.no_grad():
            y1 = net(x)
            y2 = net(x)
        assert torch.allclose(y1, y2)


# ── DL-1: TemporalCNNAlpha ──────────────────────────────────────────

class TestTemporalCNNAlpha:
    def test_instantiate(self):
        strat = TemporalCNNAlpha()
        assert strat.name == "DL1-TemporalCNNAlpha"

    def test_custom_config(self):
        cfg = TemporalCNNConfig(top_n=3, leverage=1.0)
        strat = TemporalCNNAlpha(config=cfg)
        assert strat.cfg.top_n == 3
        assert strat.cfg.leverage == 1.0

    def test_generate_weights_shape(self):
        """Weights DataFrame has correct shape and no NaN."""
        prices = _make_prices(n=550)
        # Fast config: small warm-up, fewer epochs
        cfg = TemporalCNNConfig(
            min_train_days=300, retrain_freq=200,
            epochs=2, rebalance_freq=50, window=30,
        )
        strat = TemporalCNNAlpha(config=cfg)
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape[0] == len(prices)
        assert not w.isna().any().any()

    def test_insufficient_tickers_returns_zeros(self):
        """If too few tickers available, should return all-zero weights."""
        prices = pd.DataFrame(
            {"AAA": np.linspace(100, 150, 200)},
            index=pd.bdate_range("2020-01-01", periods=200),
        )
        strat = TemporalCNNAlpha()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_warm_up_period_zeros(self):
        """Before min_train_days, weights should be zero."""
        prices = _make_prices(n=550)
        cfg = TemporalCNNConfig(min_train_days=400, epochs=1, retrain_freq=200)
        strat = TemporalCNNAlpha(config=cfg)
        w = strat.generate_weights(prices)
        # First 400 rows should be all zero
        assert (w.iloc[:400] == 0).all().all()


# ── DL-2: LSTMRegimeDetector ────────────────────────────────────────

class TestLSTMRegimeDetector:
    def test_instantiate(self):
        strat = LSTMRegimeDetector()
        assert strat.name == "DL2-LSTMRegimeDetector"

    def test_custom_config(self):
        cfg = LSTMRegimeConfig(seq_len=20, leverage=1.0)
        strat = LSTMRegimeDetector(config=cfg)
        assert strat.cfg.seq_len == 20

    def test_generate_weights_shape(self):
        prices = _make_prices(n=550)
        cfg = LSTMRegimeConfig(
            min_train_days=300, retrain_freq=200,
            epochs=2, rebalance_freq=50, seq_len=20,
        )
        strat = LSTMRegimeDetector(config=cfg)
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape[0] == len(prices)
        assert not w.isna().any().any()

    def test_no_spy_returns_zeros(self):
        """If SPY is missing, should return all-zero weights."""
        prices = pd.DataFrame(
            {"AAA": np.linspace(100, 150, 200), "BBB": np.linspace(90, 130, 200)},
            index=pd.bdate_range("2020-01-01", periods=200),
        )
        strat = LSTMRegimeDetector()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_warm_up_period_zeros(self):
        prices = _make_prices(n=550)
        cfg = LSTMRegimeConfig(min_train_days=400, epochs=1, retrain_freq=200)
        strat = LSTMRegimeDetector(config=cfg)
        w = strat.generate_weights(prices)
        expected_warmup = min(
            cfg.min_train_days,
            max(cfg.seq_len + 60, int(0.40 * len(prices))),
        )
        assert (w.iloc[:expected_warmup] == 0).all().all()

    def test_build_market_features(self):
        """Check market features are built with correct columns."""
        prices = _make_prices(n=200)
        strat = LSTMRegimeDetector()
        feat = strat._build_market_features(prices)
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) == len(prices)
        expected_cols = {"spy_vol", "spy_ret5", "spy_ret20", "spy_rsi",
                         "credit_z", "spy_dd", "vov", "breadth"}
        assert set(feat.columns) == expected_cols
        assert not feat.isna().any().any()

    def test_short_slice_adaptive_warmup_produces_non_zero_weights(self):
        """DL2 should trade on short slices even when min_train_days is large."""
        prices = _make_prices(n=320)
        cfg = LSTMRegimeConfig(
            min_train_days=504,
            epochs=2,
            retrain_freq=999,
            rebalance_freq=5,
        )
        strat = LSTMRegimeDetector(config=cfg)
        w = strat.generate_weights(prices)

        active_days = int((w.abs().sum(axis=1) > 0).sum())
        assert active_days > 0, "Expected non-zero allocations on short slice"
        assert not np.isinf(w.values).any()
        assert not w.isna().any().any()

    def test_short_slice_backtest_sharpe_positive_net(self):
        """Deterministic short-slice validation with net costs should be tradable."""
        rng = np.random.RandomState(123)
        n = 340
        dates = pd.bdate_range("2022-01-03", periods=n)
        cols = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLE", "XLK", "XLF", "HYG", "LQD"]

        data = {
            t: 100.0
            * np.exp(
                np.cumsum(
                    rng.normal(
                        0.0012 if t in {"SPY", "QQQ", "IWM", "XLE", "XLK", "XLF"} else 0.00005,
                        0.008 if t in {"SPY", "QQQ", "IWM", "XLE", "XLK", "XLF"} else 0.006,
                        n,
                    )
                )
            )
            for t in cols
        }
        prices = pd.DataFrame(data, index=dates)

        strat = LSTMRegimeDetector()
        weights = strat.backtest_weights(prices)
        bt_cfg = BacktestConfig(
            tx_cost_bps=5.0,
            leverage_cost_annual=0.015,
            short_cost_annual=0.005,
            initial_capital=1_000_000.0,
            vol_target=0.20,
            max_drawdown_trigger=-0.25,
            drawdown_recovery_rate=0.10,
        )
        metrics = backtest(prices, weights, bt_cfg)["metrics"]

        assert (weights.abs().sum(axis=1) > 0).any(), "Expected active DL2 allocations"
        assert metrics.get("sharpe", 0.0) > 0.20


# ── DL-3: AttentionCrossSectionalRanker ──────────────────────────────

class TestAttentionCrossSectionalRanker:
    def test_instantiate(self):
        strat = AttentionCrossSectionalRanker()
        assert strat.name == "DL3-AttentionRanker"

    def test_custom_config(self):
        cfg = AttentionRankerConfig(long_n=3, short_n=2)
        strat = AttentionCrossSectionalRanker(config=cfg)
        assert strat.cfg.long_n == 3

    def test_generate_weights_shape(self):
        prices = _make_prices(n=550)
        cfg = AttentionRankerConfig(
            min_train_days=300, retrain_freq=200,
            epochs=2, rebalance_freq=50,
        )
        strat = AttentionCrossSectionalRanker(config=cfg)
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape[0] == len(prices)
        assert not w.isna().any().any()

    def test_insufficient_tickers_returns_zeros(self):
        prices = pd.DataFrame(
            {"AAA": np.linspace(100, 150, 200)},
            index=pd.bdate_range("2020-01-01", periods=200),
        )
        strat = AttentionCrossSectionalRanker()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_warm_up_period_zeros(self):
        prices = _make_prices(n=550)
        cfg = AttentionRankerConfig(min_train_days=400, epochs=1, retrain_freq=200)
        strat = AttentionCrossSectionalRanker(config=cfg)
        w = strat.generate_weights(prices)
        assert (w.iloc[:400] == 0).all().all()

    def test_has_short_positions(self):
        """DL-3 is long/short — should produce negative weights."""
        prices = _make_prices(n=550)
        cfg = AttentionRankerConfig(
            min_train_days=300, retrain_freq=200,
            epochs=2, rebalance_freq=50,
        )
        strat = AttentionCrossSectionalRanker(config=cfg)
        w = strat.generate_weights(prices)
        # After warm-up, there should be some negative weights
        post_warmup = w.iloc[350:]
        assert (post_warmup < 0).any().any(), "Expected short positions in DL-3"


# ── Integration ──────────────────────────────────────────────────────

class TestDLStrategyIntegration:
    def test_all_strategies_in_init(self):
        """All 3 DL strategies should be importable from strategies package."""
        from financial_algo.strategies import (
            TemporalCNNAlpha,
            LSTMRegimeDetector,
            AttentionCrossSectionalRanker,
        )
        assert TemporalCNNAlpha.name == "DL1-TemporalCNNAlpha"
        assert LSTMRegimeDetector.name == "DL2-LSTMRegimeDetector"
        assert AttentionCrossSectionalRanker.name == "DL3-AttentionRanker"

    def test_weights_no_inf(self):
        """Verify no inf values sneak through for any DL strategy."""
        prices = _make_prices(n=550)
        for StratClass, CfgClass in [
            (TemporalCNNAlpha, TemporalCNNConfig),
            (LSTMRegimeDetector, LSTMRegimeConfig),
            (AttentionCrossSectionalRanker, AttentionRankerConfig),
        ]:
            cfg = CfgClass(min_train_days=300, retrain_freq=200, epochs=1)
            strat = StratClass(config=cfg)
            w = strat.generate_weights(prices)
            assert not np.isinf(w.values).any(), f"{strat.name} has inf in weights"
