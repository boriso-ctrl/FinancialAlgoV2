import numpy as np
import pandas as pd

from financial_algorithms.backtest import blend_signals, run_backtest
from scripts.demo_blend import build_price_signal, build_volume_signal


def _synthetic_prices(n: int = 60, tickers=None):
    tickers = tickers or ['AAA', 'BBB']
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    data = {t: np.linspace(100 + i, 110 + i, n) for i, t in enumerate(tickers)}
    return pd.DataFrame(data, index=idx)


def test_demo_blend_pipeline():
    prices = _synthetic_prices()
    price_sig = build_price_signal(prices, fast=5, slow=10)
    volume = pd.Series(1_000_000, index=prices.index)  # flat volume proxy
    vol_sig = build_volume_signal(prices, fast=5, slow=10)

    blended = blend_signals({'price': price_sig, 'volume': vol_sig}, {'price': 2.0, 'volume': 1.0}, max_signal_abs=5)
    assert blended.shape == prices.shape

    results = run_backtest(prices, blended, initial_capital=100_000, cost_bps=0)
    assert 'metrics' in results
    assert 'Sharpe Ratio' in results['metrics']
