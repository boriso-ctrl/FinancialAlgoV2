"""Quick backtest of DL strategies only."""
from __future__ import annotations
import sys, warnings
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.dl_strategies import (
    TemporalCNNAlpha, LSTMRegimeDetector, AttentionCrossSectionalRanker
)

TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "SLV", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "ITA", "LMT", "RTX", "HYG", "LQD", "DBC", "VNQ", "BTC-USD",
]

def main():
    print("Loading data...")
    prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
    try:
        vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
        vix = vix_df["^VIX"]
    except Exception:
        vix = None
    regime = detect_regime(prices, vix=vix)

    mask = (prices.index >= "2020-01-01") & (prices.index <= "2025-12-31")
    p = prices.loc[mask]
    r = regime.loc[mask]
    print(f"Data: {len(p)} days, {p.shape[1]} tickers")

    cfg = BacktestConfig(
        tx_cost_bps=5.0,
        leverage_cost_annual=0.015,
        short_cost_annual=0.005,
        initial_capital=1_000_000.0,
    )

    strats = [
        ("DL1-TemporalCNNAlpha", TemporalCNNAlpha()),
        ("DL2-LSTMRegimeDetector", LSTMRegimeDetector()),
        ("DL3-AttentionRanker", AttentionCrossSectionalRanker()),
    ]

    results = []
    for name, strat in strats:
        print(f"\nRunning {name}...")
        try:
            w = strat.backtest_weights(p)
            res = backtest(p, w, cfg)
            m = res["metrics"]
            results.append({
                "Strategy": name,
                "CAGR": m["cagr"] * 100,
                "Sharpe": m["sharpe"],
                "Sortino": m["sortino"],
                "MaxDD": m["max_drawdown"] * 100,
                "Calmar": m["calmar"],
                "AnnVol": m["annual_vol"] * 100,
            })
            print(f"  CAGR:    {m['cagr']*100:.2f}%")
            print(f"  Sharpe:  {m['sharpe']:.2f}")
            print(f"  Sortino: {m['sortino']:.2f}")
            print(f"  MaxDD:   {m['max_drawdown']*100:.2f}%")
            print(f"  Calmar:  {m['calmar']:.2f}")
            print(f"  AnnVol:  {m['annual_vol']*100:.2f}%")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 75)
    header = f"{'Strategy':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Sortino':>8} {'Calmar':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['Strategy']:<30} {r['CAGR']:>6.1f}% {r['Sharpe']:>7.2f} "
            f"{r['MaxDD']:>7.1f}% {r['Sortino']:>8.2f} {r['Calmar']:>7.2f}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
