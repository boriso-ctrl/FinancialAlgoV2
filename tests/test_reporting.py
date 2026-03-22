"""Tests for financial_algo.reporting — QuantStats tear sheet wrapper."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from financial_algo.reporting import generate_metrics_snapshot, generate_tearsheet


def _mock_backtest_result() -> dict:
    """Create a minimal backtest result dict with random returns."""
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx, name="returns")
    return {"returns": returns}


class TestGenerateTearsheet:
    """Tests for generate_tearsheet()."""

    def test_import_error_when_quantstats_missing(self, tmp_path):
        """Should raise ImportError with helpful message when quantstats absent."""
        result = _mock_backtest_result()
        with patch.dict("sys.modules", {"quantstats": None}):
            with pytest.raises(ImportError, match="quantstats is not installed"):
                generate_tearsheet(result, output_path=tmp_path / "test.html")

    def test_type_error_on_dataframe_returns(self, tmp_path):
        """Should raise TypeError when returns is a DataFrame, not Series."""
        idx = pd.bdate_range("2020-01-01", periods=10)
        bad_result = {"returns": pd.DataFrame({"a": np.zeros(10)}, index=idx)}
        # The TypeError should fire before the quantstats import is even needed,
        # but reporting.py checks type after import — so we mock quantstats too.
        fake_qs = type("FakeQS", (), {"reports": None, "stats": None})()
        with patch.dict("sys.modules", {"quantstats": fake_qs}):
            with pytest.raises(TypeError, match="pd.Series"):
                generate_tearsheet(bad_result, output_path=tmp_path / "test.html")


class TestGenerateMetricsSnapshot:
    """Tests for generate_metrics_snapshot()."""

    def test_import_error_when_quantstats_missing(self):
        """Should raise ImportError with helpful message when quantstats absent."""
        result = _mock_backtest_result()
        with patch.dict("sys.modules", {"quantstats": None}):
            with pytest.raises(ImportError, match="quantstats is not installed"):
                generate_metrics_snapshot(result)
