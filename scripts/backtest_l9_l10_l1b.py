"""Standalone backtest for L9, L10, L1b VIX-adaptive vol strategies.

Usage:
    .venv\\Scripts\\python.exe scripts/backtest_l9_l10_l1b.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.volatility_strats import (
    VIXAdaptiveCarry,
    DynamicVolRegimeSwitch,
    VolRiskPremiumAdaptive,
    VolRiskPremium,
)

TICKERS = [
    "SPY", "QQQ", "TLT", "GLD", "SLV", "EFA", "EEM", "IWM",
    "XLE", "XLF", "HYG", "LQD", "SHY", "IEF", "DBC", "UUP",
    "^VIX",
]

def main() -> None:
    print("Loading prices...")
    prices = load_prices(TICKERS, start="2010-01-01")
    print(f"  Loaded {len(prices)} days, {len(prices.columns)} tickers")

    regime = detect_regime(prices)

    strategies = [
        ("L9-VIXAdaptiveCarry", VIXAdaptiveCarry()),
        ("L10-DynamicVolRegimeSwitch", DynamicVolRegimeSwitch()),
        ("L1b-VolRiskPremiumAdaptive", VolRiskPremiumAdaptive()),
        ("L1-VolRiskPremium (original)", VolRiskPremium()),
    ]

    cfg = BacktestConfig(
        tx_cost_bps=5.0,
        leverage_cost_annual=0.015,
        short_cost_annual=0.005,
    )

    results = {}
    for name, strat in strategies:
        w = strat.backtest_weights(prices, regime)
        res = backtest(prices, w, cfg)
        m = res["metrics"]
        results[name] = m
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  CAGR:     {m.get('cagr', 0):.4f}")
        print(f"  Sharpe:   {m.get('sharpe', 0):.4f}")
        print(f"  Sortino:  {m.get('sortino', 0):.4f}")
        print(f"  Max DD:   {m.get('max_drawdown', 0):.4f}")
        print(f"  Calmar:   {m.get('calmar', 0):.4f}")
        print(f"  Win Rate: {m.get('win_rate', 0):.4f}")
        print(f"  Ann Vol:  {m.get('annual_vol', 0):.4f}")

    # Correlation matrix between strategy returns
    print(f"\n{'='*60}")
    print("  Return Correlations")
    print(f"{'='*60}")
    ret_dict = {}
    for name, strat in strategies:
        w = strat.backtest_weights(prices, regime)
        res = backtest(prices, w, cfg)
        ret_dict[name] = res["returns"]

    ret_df = pd.DataFrame(ret_dict)
    corr = ret_df.corr()
    print(corr.to_string())

    # Summary table
    print(f"\n{'='*60}")
    print("  Summary Table")
    print(f"{'='*60}")
    header = f"{'Strategy':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'WinRate':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = (
            f"{name:<35} "
            f"{m.get('cagr', 0):>8.4f} "
            f"{m.get('sharpe', 0):>8.4f} "
            f"{m.get('sortino', 0):>8.4f} "
            f"{m.get('max_drawdown', 0):>8.4f} "
            f"{m.get('calmar', 0):>8.4f} "
            f"{m.get('win_rate', 0):>8.4f}"
        )
        print(row)


if __name__ == "__main__":
    main()
