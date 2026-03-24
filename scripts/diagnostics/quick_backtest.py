"""Quick full-period backtest of all strategies. Outputs performance table."""
from __future__ import annotations
import sys, warnings
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime

# Re-use config from main script
from scripts.run_crisis_backtest import (
    build_strategy_registry, BT_CONFIG, TICKERS, VIX_TICKER
)

def main():
    print("Loading data...")
    prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
    try:
        vix_df = load_prices([VIX_TICKER], start="2009-01-01", end="2025-12-31")
        vix = vix_df[VIX_TICKER]
    except Exception:
        vix = None
    regime = detect_regime(prices, vix=vix)

    mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
    p = prices.loc[mask]
    r = regime.loc[mask]
    print(f"Data: {len(p)} days, {p.shape[1]} tickers")

    reg = build_strategy_registry()
    total = sum(len(v) for v in reg.values())
    print(f"Running {total} strategies...")

    results = []
    for cat, strats in reg.items():
        for name, strat, needs_regime in strats:
            try:
                if needs_regime:
                    w = strat.backtest_weights(p, r)
                else:
                    w = strat.backtest_weights(p)
                res = backtest(p, w, BT_CONFIG)
                m = res["metrics"]
                results.append({
                    "Strategy": name,
                    "CAGR": m["cagr"] * 100,
                    "Sharpe": m["sharpe"],
                    "MaxDD": m["max_drawdown"] * 100,
                    "Sortino": m["sortino"],
                    "Calmar": m["calmar"],
                    "AnnVol": m["annual_vol"] * 100,
                })
            except Exception as e:
                print(f"  ERR {name}: {e}")
                results.append({
                    "Strategy": name,
                    "CAGR": 0, "Sharpe": 0, "MaxDD": 0,
                    "Sortino": 0, "Calmar": 0, "AnnVol": 0,
                })

    # Sort by Sharpe
    results.sort(key=lambda x: x["Sharpe"], reverse=True)

    # Print table
    header = f"{'Strategy':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Sortino':>8} {'Calmar':>7}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['Strategy']:<30} {r['CAGR']:>6.1f}% {r['Sharpe']:>7.2f} "
            f"{r['MaxDD']:>7.1f}% {r['Sortino']:>8.2f} {r['Calmar']:>7.2f}"
        )

    print(f"\nTotal strategies: {len(results)}")
    pos = sum(1 for r in results if r["Sharpe"] > 0)
    print(f"Positive Sharpe: {pos}/{len(results)}")
    avg_sharpe = np.mean([r["Sharpe"] for r in results])
    print(f"Average Sharpe: {avg_sharpe:.2f}")

if __name__ == "__main__":
    main()
