"""Tests for intraday HFT strategy research pack.

Each strategy is tested for:
1. Shape correctness — output index matches input index.
2. dtype — float64 output.
3. No NaN values in signal.
4. No inf values in signal.
5. Values bounded in [-1.0, 1.0].
6. Look-ahead safety — shuffling future bars does not change past signal values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_algo.strategies.intraday_research_pack import (
    BollingerSnapback,
    GapRegimeSelector,
    KSTTSIOpeningBurst,
    MACDHistogramAcceleration,
    RangeExpansionExhaustion,
    RealizedVolImpulseFade,
    RelativeVolContinuation,
    SqueezeReleaseBreakout,
    VolRegimeRouter,
    VWAPVolNormalizedFade,
    WickRejectionFade,
)


# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_ohlcv(n_bars: int = 1000, start: str = "2024-01-02 09:31:00") -> pd.DataFrame:
    """Synthetic 1-min OHLCV with realistic structure (no look-ahead)."""
    idx = pd.date_range(start, periods=n_bars, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(_RNG.standard_normal(n_bars) * 0.05)
    returns = np.diff(close, prepend=close[0]) / close * 0.5
    high = close + _RNG.uniform(0.01, 0.20, n_bars)
    low = close - _RNG.uniform(0.01, 0.20, n_bars)
    open_ = close - returns * _RNG.uniform(0.3, 0.7, n_bars)
    volume = _RNG.integers(500, 50_000, n_bars).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


OHLCV = _make_ohlcv()


# ---------------------------------------------------------------------------
# Strategy instances under test
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    MACDHistogramAcceleration(),
    VWAPVolNormalizedFade(),
    RealizedVolImpulseFade(),
    KSTTSIOpeningBurst(),
    RelativeVolContinuation(),
    WickRejectionFade(),
    BollingerSnapback(),
    SqueezeReleaseBreakout(),
    RangeExpansionExhaustion(),
    VolRegimeRouter(),
    GapRegimeSelector(),
]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
class TestSignalProperties:
    """Property tests applied to every strategy uniformly."""

    def test_shape_matches_input(self, strategy):
        sig = strategy.generate_signal(OHLCV)
        assert sig.shape[0] == OHLCV.shape[0], (
            f"{strategy.name}: signal length {sig.shape[0]} != ohlcv length {OHLCV.shape[0]}"
        )

    def test_index_matches_input(self, strategy):
        sig = strategy.generate_signal(OHLCV)
        assert sig.index.equals(OHLCV.index), f"{strategy.name}: index mismatch"

    def test_dtype_float(self, strategy):
        sig = strategy.generate_signal(OHLCV)
        assert sig.dtype in (np.float64, np.float32), (
            f"{strategy.name}: unexpected dtype {sig.dtype}"
        )

    def test_no_nan(self, strategy):
        sig = strategy.generate_signal(OHLCV)
        n_nan = sig.isna().sum()
        assert n_nan == 0, f"{strategy.name}: {n_nan} NaN values in signal"

    def test_no_inf(self, strategy):
        sig = strategy.generate_signal(OHLCV)
        n_inf = np.isinf(sig.values).sum()
        assert n_inf == 0, f"{strategy.name}: {n_inf} inf values in signal"

    def test_bounded(self, strategy):
        sig = strategy.generate_signal(OHLCV)
        assert sig.min() >= -1.0 - 1e-9, (
            f"{strategy.name}: signal below -1.0: {sig.min()}"
        )
        assert sig.max() <= 1.0 + 1e-9, (
            f"{strategy.name}: signal above 1.0: {sig.max()}"
        )

    def test_look_ahead_safety(self, strategy):
        """Shuffling future bars must not change past signals.

        We take the first 500 bars and verify that shuffling bars 501-1000
        produces an identical signal for bars 0-499. This confirms no future
        information leaks backward.
        """
        pivot = 500
        past = OHLCV.iloc[:pivot].copy()
        future = OHLCV.iloc[pivot:].copy()
        shuffled_future = future.sample(frac=1, random_state=99)
        shuffled_future.index = future.index  # keep original index

        combined = pd.concat([past, shuffled_future])
        sig_base = strategy.generate_signal(OHLCV)
        sig_shuffled = strategy.generate_signal(combined)

        past_base = sig_base.iloc[:pivot]
        past_shuffled = sig_shuffled.iloc[:pivot]
        assert (past_base.values == past_shuffled.values).all(), (
            f"{strategy.name}: look-ahead bias detected — past signals changed "
            "after shuffling future bars"
        )


# ---------------------------------------------------------------------------
# Feature-library unit tests
# ---------------------------------------------------------------------------


class TestFeatureLibrary:
    """Smoke tests for individual feature functions."""

    def setup_method(self):
        self.close = OHLCV["close"]
        self.high = OHLCV["high"]
        self.low = OHLCV["low"]
        self.volume = OHLCV["volume"]

    def test_feat_ema_no_nan(self):
        from financial_algo.strategies.intraday_base import feat_ema

        out = feat_ema(self.close, 12)
        assert out.isna().sum() == 0

    def test_feat_macd_histogram(self):
        from financial_algo.strategies.intraday_base import feat_macd

        _, _, hist = feat_macd(self.close)
        assert hist.isna().sum() == 0
        assert np.isinf(hist.values).sum() == 0

    def test_feat_rsi_bounded(self):
        from financial_algo.strategies.intraday_base import feat_rsi

        rsi = feat_rsi(self.close)
        assert rsi.min() >= 0.0 - 1e-9
        assert rsi.max() <= 100.0 + 1e-9

    def test_feat_bollinger_pct_b(self):
        from financial_algo.strategies.intraday_base import feat_bollinger

        _, _, _, bw, pct_b = feat_bollinger(self.close)
        assert pct_b.isna().sum() == 0
        assert bw.isna().sum() == 0

    def test_feat_vwap_no_nan(self):
        from financial_algo.strategies.intraday_base import feat_vwap

        vwap = feat_vwap(self.close, self.volume)
        assert vwap.isna().sum() == 0

    def test_feat_realized_vol_no_nan(self):
        from financial_algo.strategies.intraday_base import feat_realized_vol

        rv = feat_realized_vol(self.close)
        assert rv.isna().sum() == 0

    def test_feat_atr_no_nan(self):
        from financial_algo.strategies.intraday_base import feat_atr

        atr = feat_atr(self.high, self.low, self.close)
        assert atr.isna().sum() == 0

    def test_feat_kst_no_nan(self):
        from financial_algo.strategies.intraday_base import feat_kst

        kst = feat_kst(self.close)
        assert kst.isna().sum() == 0

    def test_feat_tsi_no_nan(self):
        from financial_algo.strategies.intraday_base import feat_tsi

        tsi = feat_tsi(self.close)
        assert tsi.isna().sum() == 0
        # TSI in (-100, 100) by construction
        assert tsi.max() <= 100.0 + 1e-9
        assert tsi.min() >= -100.0 - 1e-9
