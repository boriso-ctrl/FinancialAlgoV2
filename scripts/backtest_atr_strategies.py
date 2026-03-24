"""Quick backtest of the 3 new ATR-based strategies.

Runs each strategy over multiple crisis windows and reports metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime

from financial_algo.strategies.tail_risk import ATRCrisisAlpha
from financial_algo.strategies.crash_hedge import AdaptiveStopTrend
from financial_algo.strategies.crypto_crisis import CryptoRecoverySurgeATR


TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "SLV", "TLT", "IEF", "SHY", "UUP",
    "XLE", "USO", "XOP",
    "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "XBI", "XLC", "XLRE",
    "DBC", "DBA",
    "HYG", "LQD", "TIP", "AGG", "EMB",
    "FXI", "VGK", "EWJ", "INDA",
    "VNQ",
    "BTC-USD", "ETH-USD",
]))
VIX_TICKER = "^VIX"


CRISIS_WINDOWS = {
    "Full Period 2010-2025": ("2010-01-01", "2025-12-31"),
    "COVID-19 2020": ("2020-01-01", "2020-12-31"),
    "Russia-Ukraine 2022": ("2022-01-01", "2022-12-31"),
    "Oil Crash 2014-16": ("2014-06-01", "2016-06-30"),
    "Volmageddon 2018": ("2018-01-01", "2018-12-31"),
    "Recovery 2023-25": ("2023-01-01", "2025-03-24"),
}

CONFIG = BacktestConfig(tx_cost_bps=5.0)


def run_one(name, strategy, prices, regime, config):
    """Run a single strategy backtest, return metrics dict or None."""
    try:
        weights = strategy.backtest_weights(prices, regime)
        result = backtest(prices, weights, config)
        return result["metrics"]
    except Exception as e:
        print(f"  [WARN] {name}: {e}")
        return None


def fmt(m):
    """Format metrics dict for display."""
    if m is None:
        return "ERR"
    return (
        f"CAGR={m['cagr']:+.2%}  Sharpe={m['sharpe']:.2f}  "
        f"Sortino={m['sortino']:.2f}  MaxDD={m['max_drawdown']:.2%}  "
        f"Calmar={m['calmar']:.2f}"
    )


def main():
    print("=" * 70)
    print("ATR Strategy Backtest Suite")
    print("=" * 70)

    # Load full-period data
    print("\nLoading prices...")
    prices_full = load_prices(TICKERS + [VIX_TICKER])
    regime_full = detect_regime(prices_full)
    print(f"  Loaded {len(prices_full)} days, {len(prices_full.columns)} tickers")

    strategies = [
        ("O9-ATRCrisisAlpha", ATRCrisisAlpha(), True),
        ("D4-AdaptiveStopTrend", AdaptiveStopTrend(), False),
        ("F2b-CryptoRecovSurgeATR", CryptoRecoverySurgeATR(), True),
    ]

    for window_name, (start, end) in CRISIS_WINDOWS.items():
        prices = prices_full.loc[start:end]
        regime = regime_full.loc[prices.index]

        if len(prices) < 20:
            print(f"\n--- {window_name}: SKIPPED (too few days) ---")
            continue

        print(f"\n--- {window_name} ({len(prices)} days) ---")

        for sname, strat, needs_regime in strategies:
            if needs_regime:
                m = run_one(sname, strat, prices, regime, CONFIG)
            else:
                # D4 doesn't require regime but accepts it
                m = run_one(sname, strat, prices, regime, CONFIG)
            print(f"  {sname:30s} | {fmt(m)}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
