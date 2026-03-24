"""Quick test for gradual DD overlay and O6-TailHedgeOverlay."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from financial_algo.backtest import BacktestConfig, backtest


def test_gradual_dd_overlay():
    rng = np.random.RandomState(42)
    n = 252
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = np.concatenate([
        rng.normal(0.001, 0.008, 100),
        rng.normal(-0.01, 0.02, 50),
        rng.normal(0.002, 0.01, 102),
    ])
    spy = 100 * np.exp(np.cumsum(rets))
    prices = pd.DataFrame({"SPY": spy}, index=dates)
    weights = pd.DataFrame({"SPY": 1.0}, index=dates)

    # Without overlay
    cfg_off = BacktestConfig()
    r_off = backtest(prices, weights, cfg_off)
    dd_off = r_off["metrics"]["max_drawdown"]

    # With overlay
    cfg_on = BacktestConfig(strategy_dd_scale_start=-0.05, strategy_dd_scale_end=-0.15)
    r_on = backtest(prices, weights, cfg_on)
    dd_on = r_on["metrics"]["max_drawdown"]

    print(f"Without overlay: MaxDD={dd_off:.4f}, CAGR={r_off['metrics']['cagr']:.4f}")
    print(f"With overlay:    MaxDD={dd_on:.4f}, CAGR={r_on['metrics']['cagr']:.4f}")
    assert dd_on > dd_off, "DD overlay should reduce drawdown (less negative)"
    print("PASS: Gradual DD overlay reduces max drawdown")
    print()

    # Verify disabled by default (None)
    cfg_default = BacktestConfig()
    assert cfg_default.strategy_dd_scale_start is None
    assert cfg_default.strategy_dd_scale_end is None
    print("PASS: DD overlay disabled by default")


def test_tail_hedge_overlay():
    from financial_algo.strategies.tail_risk import TailHedgeOverlay
    from financial_algo.regimes import Regime

    rng = np.random.RandomState(42)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["SPY", "QQQ", "GLD", "TLT", "UUP", "IWM"]
    data = {t: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))) for t in tickers}
    prices = pd.DataFrame(data, index=dates)

    cycle = (
        [Regime.NORMAL] * 80
        + [Regime.ELEVATED] * 40
        + [Regime.GENERAL_CRISIS] * 40
        + [Regime.RECOVERY] * 40
    )
    regimes = (cycle * 3)[:n]
    regime = pd.Series(regimes, index=dates)

    strat = TailHedgeOverlay()
    w = strat.generate_weights(prices, regime)

    # Basic checks
    assert w.shape[0] == n
    assert not w.isna().any().any(), "No NaN"
    assert not np.isinf(w.values).any(), "No inf"

    # Pure hedge: no equity columns
    eq_cols = [c for c in ["SPY", "QQQ", "IWM"] if c in w.columns]
    assert len(eq_cols) == 0, f"Should not have equity columns, got {eq_cols}"

    # Regime scaling
    normal_alloc = w.loc[regime.isin({Regime.NORMAL})].sum(axis=1).mean()
    elevated_alloc = w.loc[regime.isin({Regime.ELEVATED})].sum(axis=1).mean()
    crisis_alloc = w.loc[regime.isin({Regime.GENERAL_CRISIS})].sum(axis=1).mean()

    print(f"Normal alloc:   {normal_alloc:.4f}")
    print(f"Elevated alloc: {elevated_alloc:.4f}")
    print(f"Crisis alloc:   {crisis_alloc:.4f}")

    assert elevated_alloc > normal_alloc, "Elevated > Normal"
    assert crisis_alloc > elevated_alloc, "Crisis > Elevated"
    assert crisis_alloc >= 0.20, "Crisis should be >= 20%"
    print("PASS: TailHedgeOverlay regime scaling correct")
    print()

    # Backtest it
    bw = strat.backtest_weights(prices, regime)
    result = backtest(prices[["GLD", "TLT", "UUP"]], bw, BacktestConfig())
    m = result["metrics"]
    print(f"O6 Backtest: CAGR={m['cagr']:.4f}, Sharpe={m['sharpe']:.4f}, MaxDD={m['max_drawdown']:.4f}")
    print("PASS: TailHedgeOverlay backtests cleanly")


if __name__ == "__main__":
    test_gradual_dd_overlay()
    print()
    test_tail_hedge_overlay()
    print()
    print("ALL TESTS PASSED")
