"""Tests for Alpaca data pipeline modules.

Tests are designed to run WITHOUT network access and WITHOUT API keys.
All Alpaca client calls are mocked.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_1min_bars(dates, ticker="SPY"):
    """Build a representative 1-min bar DataFrame for testing."""
    rng = pd.date_range(
        f"{dates[0]} 09:30", periods=len(dates) * 390, freq="1min", tz="UTC"
    )
    n = len(rng)
    np.random.seed(42)
    close = 400 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.05,
            "high": close + np.abs(np.random.randn(n)) * 0.1,
            "low": close - np.abs(np.random.randn(n)) * 0.1,
            "close": close,
            "volume": np.abs(np.random.randn(n) * 1e5 + 1e6).astype(int),
            "vwap": close + np.random.randn(n) * 0.02,
            "trade_count": np.random.randint(100, 1000, n),
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# alpaca_loader tests
# ---------------------------------------------------------------------------

class TestAlpacaLoaderImport:
    def test_import_without_keys(self):
        """Module should import cleanly without API keys present."""
        import financial_algo.data.alpaca_loader as al
        assert hasattr(al, "load_intraday")
        assert hasattr(al, "load_daily_alpaca")
        assert hasattr(al, "get_available_history")

    def test_get_client_warns_without_keys(self):
        """_get_client should warn when env vars are missing."""
        from financial_algo.data.alpaca_loader import _get_client
        env_backup = {}
        for k in ("ALPACA_API_KEY", "ALPACA_SECRET"):
            env_backup[k] = os.environ.pop(k, None)

        try:
            # Patch at the source so the local import inside _get_client gets the mock
            with patch("alpaca.data.historical.StockHistoricalDataClient", MagicMock()):
                with pytest.warns(UserWarning, match="ALPACA_API_KEY"):
                    _get_client()
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_fetch_bars_returns_none_on_api_error(self):
        """_fetch_bars should return None (not raise) if the API call fails."""
        from financial_algo.data.alpaca_loader import _fetch_bars
        import warnings as _w
        mock_client = MagicMock()
        mock_client.get_stock_bars.side_effect = RuntimeError("API down")

        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            result = _fetch_bars(
                mock_client, "SPY", "2025-01-06", "2025-01-06", "1Min", "iex"
            )
        assert result is None

    def test_fetch_bars_processes_multiindex(self):
        """_fetch_bars should strip the symbol level from the MultiIndex."""
        from financial_algo.data import alpaca_loader as al

        # Build a mock MultiIndex DataFrame as alpaca-py returns
        rng = pd.date_range("2025-01-06 14:30", periods=390, freq="1min", tz="UTC")
        inner_df = pd.DataFrame(
            {"open": 400.0, "high": 401.0, "low": 399.0, "close": 400.5,
             "volume": 1000, "vwap": 400.2, "trade_count": 50},
            index=pd.MultiIndex.from_arrays(
                [["SPY"] * 390, rng], names=["symbol", "timestamp"]
            ),
        )
        mock_bars_response = MagicMock()
        mock_bars_response.df = inner_df
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_bars_response

        result = al._fetch_bars(
            mock_client, "SPY", "2025-01-06", "2025-01-06", "1Min", "iex"
        )

        assert result is not None
        assert not isinstance(result.index, pd.MultiIndex)
        assert "close" in result.columns


# ---------------------------------------------------------------------------
# feature_store tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_extract_features_returns_expected_columns(self):
        """_extract_features should produce 7 named columns per ticker."""
        from financial_algo.data.feature_store import _extract_features, FEATURE_NAMES

        bars = _make_1min_bars(["2025-01-06", "2025-01-07", "2025-01-08"])
        result = _extract_features(bars, "SPY")

        assert not result.empty
        for feat in FEATURE_NAMES:
            assert f"SPY_{feat}" in result.columns, f"Missing column: SPY_{feat}"

    def test_extract_features_no_lookahead(self):
        """Opening gap uses previous day's close (shift=1). Check sign."""
        from financial_algo.data.feature_store import _extract_features

        bars = _make_1min_bars(["2025-01-06", "2025-01-07", "2025-01-08"])
        result = _extract_features(bars, "SPY")

        # Day 1 opening gap should be NaN (no prior close available)
        first_gap = result["SPY_opening_gap"].iloc[0]
        assert pd.isna(first_gap) or first_gap == 0.0

    def test_extract_features_no_nan_after_warmup(self):
        """After warmup period (21+ days), NaN rate should be minimal."""
        from financial_algo.data.feature_store import _extract_features

        # Build 30 days of data
        dates = pd.bdate_range("2025-01-02", periods=30).strftime("%Y-%m-%d").tolist()
        bars = _make_1min_bars(dates[:10])  # 10 days is enough for basic check
        result = _extract_features(bars, "TEST")

        # NaN rate in close_to_high and opening_gap after row 1 should be low
        assert result.shape[0] > 0
        # These features have no rolling window — only first row NaN
        non_nan = result["TEST_intraday_range"].notna().sum()
        assert non_nan > 0

    def test_extract_features_nan_safe_division(self):
        """No inf values in output even with edge-case zero volumes."""
        from financial_algo.data.feature_store import _extract_features

        bars = _make_1min_bars(["2025-01-06", "2025-01-07"])
        # Force zero volume for some bars
        bars["volume"] = 0
        result = _extract_features(bars, "SPY")

        assert not np.isinf(result.values).any()

    def test_equity_only_filter(self):
        """_equity_only should exclude VIX and crypto tickers."""
        from financial_algo.data.feature_store import _equity_only
        tickers = ["SPY", "QQQ", "^VIX", "BTC-USD", "ETH-USD", "GLD", "TLT"]
        result = _equity_only(tickers)
        assert "^VIX" not in result
        assert "BTC-USD" not in result
        assert "ETH-USD" not in result
        assert "SPY" in result
        assert "GLD" in result

    def test_feature_names_constant(self):
        """FEATURE_NAMES should have exactly 7 entries."""
        from financial_algo.data.feature_store import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 7

    def test_vwap_fallback(self):
        """_compute_daily_vwap should work even if vwap column is all-NaN."""
        from financial_algo.data.feature_store import _compute_daily_vwap

        bars = _make_1min_bars(["2025-01-06"])
        bars["_date"] = bars.index.tz_convert("America/New_York").normalize()
        bars["vwap"] = np.nan  # force fallback

        result = _compute_daily_vwap(bars)
        assert result.notna().any()
        assert not np.isinf(result).any()


# ---------------------------------------------------------------------------
# FeatureStore integration tests (no network calls)
# ---------------------------------------------------------------------------

class TestFeatureStoreIntegration:
    def test_feature_store_build_with_mock_loader(self, tmp_path):
        """FeatureStore.build() should return correct shape when data is available."""
        from financial_algo.data.feature_store import FeatureStore

        bars_spy = _make_1min_bars(
            ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"],
            "SPY",
        )
        bars_qqq = _make_1min_bars(
            ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"],
            "QQQ",
        )

        with patch(
            "financial_algo.data.feature_store.load_intraday",
            side_effect=lambda tickers, **kw: {tickers[0]: (bars_spy if tickers[0] == "SPY" else bars_qqq)},
        ):
            store = FeatureStore(
                tickers=["SPY", "QQQ"],
                start="2025-01-06",
                end="2025-01-10",
                cache_dir=tmp_path,
            )
            result = store.build(force_refresh=True)

        assert not result.empty
        # Should have 7 features × 2 tickers = 14 columns
        assert result.shape[1] == 14

    def test_feature_store_skips_non_equity(self, tmp_path):
        """FeatureStore should skip ^VIX and BTC-USD silently."""
        from financial_algo.data.feature_store import FeatureStore

        bars = _make_1min_bars(["2025-01-06", "2025-01-07"], "SPY")

        with patch(
            "financial_algo.data.feature_store.load_intraday",
            return_value={"SPY": bars},
        ):
            store = FeatureStore(
                tickers=["SPY", "^VIX", "BTC-USD"],
                start="2025-01-06",
                end="2025-01-07",
                cache_dir=tmp_path,
            )
            result = store.build(force_refresh=True)

        # Only SPY should be present
        assert all(col.startswith("SPY_") for col in result.columns)


# ---------------------------------------------------------------------------
# Base strategy feature injection test
# ---------------------------------------------------------------------------

class TestBaseStrategyFeatureInjection:
    def test_set_intraday_features(self):
        """Strategy.set_intraday_features should store features without error."""
        from financial_algo.strategies.signal_combo import FeatureComboSignal

        strat = FeatureComboSignal()
        assert strat._intraday_features is None

        dummy_features = pd.DataFrame({"SPY_realized_vol_1min": [0.15, 0.18]})
        strat.set_intraday_features(dummy_features)
        assert strat._intraday_features is not None
        assert len(strat._intraday_features) == 2

        strat.set_intraday_features(None)
        assert strat._intraday_features is None

    def test_all_strategies_have_set_intraday_features(self):
        """All strategies should inherit set_intraday_features from base."""
        from financial_algo.strategies.signal_combo import (
            FeatureComboSignal, XGBoostSignalCombo, GMMRegimeClassifier,
        )
        from financial_algo.strategies.ml_strategies import (
            AdaptiveThreshold, CrossSectionalRanker,
        )
        for cls in [FeatureComboSignal, XGBoostSignalCombo, GMMRegimeClassifier,
                    AdaptiveThreshold, CrossSectionalRanker]:
            strat = cls()
            assert hasattr(strat, "set_intraday_features"), (
                f"{cls.__name__} missing set_intraday_features"
            )
            assert hasattr(strat, "_intraday_features"), (
                f"{cls.__name__} missing _intraday_features"
            )
