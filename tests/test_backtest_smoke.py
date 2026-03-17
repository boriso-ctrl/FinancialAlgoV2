"""Smoke test for the simple backtest engine."""

import numpy as np
import pandas as pd
import pytest

from financial_algorithms.backtest import blend_signals, run_backtest


def _synthetic_prices(n: int = 60):
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    data = {
        'AAA': np.linspace(100, 110, n),
        'BBB': np.linspace(50, 55, n),
    }
    return pd.DataFrame(data, index=idx)


def test_backtest_returns_equity_and_metrics():
    prices = _synthetic_prices()
    # simple always-on signal
    signals = pd.DataFrame(1, index=prices.index, columns=prices.columns)
    results = run_backtest(prices, signals, initial_capital=100000)

    assert 'equity_curve' in results
    assert 'metrics' in results
    assert len(results['equity_curve']) == len(prices)
    assert 'Total Return' in results['metrics']


def test_backtest_penalizes_trading_costs():
    prices = _synthetic_prices()
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    # Flip allocation between tickers to force turnover
    signals.loc[signals.index[::2], 'AAA'] = 1
    signals.loc[signals.index[1::2], 'BBB'] = 1

    frictionless = run_backtest(prices, signals, initial_capital=100000, cost_bps=0)
    with_costs = run_backtest(prices, signals, initial_capital=100000, cost_bps=50)

    assert with_costs['equity_curve'].iloc[-1] < frictionless['equity_curve'].iloc[-1]


def test_backtest_respects_conviction_weighting():
    prices = _synthetic_prices(10)
    # Two assets: strong conviction on AAA, weak on BBB
    signals = pd.DataFrame({
        'AAA': [5] * len(prices),
        'BBB': [1] * len(prices),
    }, index=prices.index)

    results = run_backtest(prices, signals, initial_capital=100000, max_signal_abs=5)
    weights = results['weights']
    # Expect AAA weight about 5/6 and BBB about 1/6 after normalization
    aaa_weight = weights['AAA'].iloc[-1]
    bbb_weight = weights['BBB'].iloc[-1]
    assert aaa_weight > bbb_weight
    assert np.isclose(aaa_weight, 5/6, atol=0.05)


def test_blend_signals_weighting():
    idx = pd.date_range('2020-01-01', periods=5, freq='D')
    base = pd.DataFrame({'AAA': [1, 2, 3, 4, 5]}, index=idx)
    sig_a = base * 1  # e.g., Bollinger core
    sig_b = base * 0.5  # e.g., secondary indicator

    blended = blend_signals({'a': sig_a, 'b': sig_b}, {'a': 2.0, 'b': 0.5}, max_signal_abs=5)

    # Weighting: (2*sig_a + 0.5*sig_b) / (|2|+|0.5|) = (2x + 0.5*0.5x)/2.5
    expected = (2 * sig_a + 0.5 * sig_b) / 2.5
    assert np.allclose(blended.values, expected.values)


def test_strict_data_rejects_duplicates():
    prices = _synthetic_prices(5)
    # introduce duplicate date
    dup_idx = prices.index.insert(2, prices.index[2])
    prices_dup = prices.reindex(dup_idx)
    signals = pd.DataFrame(1, index=prices_dup.index, columns=prices_dup.columns)

    with pytest.raises(ValueError):
        run_backtest(prices_dup, signals, strict_data=True)
