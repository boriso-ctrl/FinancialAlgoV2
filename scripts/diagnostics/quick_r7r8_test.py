"""Quick backtest for R7-VolExplosionAlpha and R8-VolRegimeSwitcher."""

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
from financial_algo.strategies.regime_hardening import VolExplosionAlpha, VolRegimeSwitcher

warnings.filterwarnings("ignore", category=FutureWarning)

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "TLT", "GLD", "IEF", "UUP", "SHY",
    "XLE", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "EFA", "EEM", "SLV", "DBC",
]))

BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=-0.25,
    drawdown_recovery_rate=0.10,
)

WINDOWS = {
    "Full Period (2010-2025)": ("2010-01-01", "2025-12-31"),
    "Oil Crash (2014-2016)":   ("2014-06-01", "2016-06-01"),
    "Volmageddon 2018":        ("2018-01-01", "2019-01-01"),
    "COVID-19 2020":           ("2020-01-01", "2021-01-01"),
    "Russia-Ukraine 2022":     ("2022-01-01", "2023-01-01"),
}


def run_one(strategy, prices, regime, config, window_prices):
    try:
        w = strategy.backtest_weights(window_prices, regime)
        result = backtest(window_prices, w, config)
        return result["metrics"]
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("Loading data...")
    prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
    print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} tickers")

    regime = detect_regime(prices)

    strategies = [
        ("R7-VolExplosionAlpha", VolExplosionAlpha()),
        ("R8-VolRegimeSwitcher", VolRegimeSwitcher()),
    ]

    for sname, strat in strategies:
        print(f"\n{'='*70}")
        print(f"  {sname}")
        print(f"{'='*70}")
        print(f"{'Window':<30s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>8s}")
        print("-" * 70)

        for wname, (wstart, wend) in WINDOWS.items():
            wp = prices.loc[wstart:wend]
            rg = regime.loc[wp.index]
            m = run_one(strat, prices, rg, BT_CONFIG, wp)
            if m:
                print(f"{wname:<30s} {m['cagr']:>8.2%} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['max_drawdown']:>8.2%} {m['calmar']:>8.2f}")
            else:
                print(f"{wname:<30s}  ERROR")


if __name__ == "__main__":
    main()
