"""Evaluate all 8 new R-category strategies across full period and weak years.

Reports on:
1. Full-period (2010-2025) performance
2. Weak year performance (2015, 2018, 2022)
3. Highlights which candidates pass the ensemble bar
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.regime_hardening import (
    BearMarketAlpha,
    CrisisHedgeAdaptive,
    RatesTighteningAlpha,
    BondEquityHedge,
    DefensiveRotationR3,
    AdaptiveRiskBudget,
    VolExplosionAlpha,
    VolRegimeSwitcher,
    MultiAssetCTATrend,
)

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "SLV", "TLT", "IEF", "SHY", "UUP",
    "XLE", "USO", "XOP", "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "XBI", "XLC", "XLRE", "DBC", "DBA",
    "HYG", "LQD", "TIP", "AGG", "EMB",
    "FXI", "VGK", "EWJ", "INDA", "VNQ",
    "BTC-USD", "ETH-USD",
]))

BT_CFG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=None,
)

WINDOWS = {
    "Full (2010-2025)": ("2010-01-01", "2025-12-31"),
    "2015": ("2015-01-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}

NEW_STRATS = [
    ("R1-BearMktAlpha",   BearMarketAlpha()),
    ("R2-CrisisHedge",    CrisisHedgeAdaptive()),
    ("R3-DefRotation",    DefensiveRotationR3()),
    ("R4-AdaptRiskBudget", AdaptiveRiskBudget()),
    ("R5-RatesTightening", RatesTighteningAlpha()),
    ("R6-BondEqHedge",    BondEquityHedge()),
    ("R7-VolExplosion",   VolExplosionAlpha()),
    ("R8-VolRegime",      VolRegimeSwitcher()),
    ("R9-CTATrend",       MultiAssetCTATrend()),
]


def eval_strat(strat, p, r):
    try:
        w = strat.backtest_weights(p, r)
        res = backtest(p, w, BT_CFG)
        m = res["metrics"]
        return m["cagr"], m["sharpe"], m["max_drawdown"]
    except Exception as e:
        return float("nan"), float("nan"), float("nan")


def main():
    print("Loading data...")
    prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
    try:
        vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
        vix = vix_df["^VIX"]
    except Exception:
        vix = None
    regime = detect_regime(prices, vix=vix)
    print()

    # SPY benchmark per window
    spy_bench = {}
    for wname, (ws, we) in WINDOWS.items():
        mask = (prices.index >= ws) & (prices.index <= we)
        p = prices.loc[mask]
        r = regime.loc[mask]
        w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        w["SPY"] = 1.0
        try:
            res = backtest(p, w.shift(1).fillna(0), BT_CFG)
            spy_bench[wname] = res["metrics"]["sharpe"]
        except Exception:
            spy_bench[wname] = float("nan")

    print(f"{'Strategy':<22s} | {'Full Sharpe':>11s} | {'Full CAGR':>9s} | {'Full DD':>8s} | {'2015 Shr':>9s} | {'2018 Shr':>9s} | {'2022 Shr':>9s}")
    print(f"{'-'*22}-+-{'-'*11}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    results = {}
    for sname, strat in NEW_STRATS:
        row = {}
        for wname, (ws, we) in WINDOWS.items():
            mask = (prices.index >= ws) & (prices.index <= we)
            p = prices.loc[mask]
            r = regime.loc[mask]
            c, s, d = eval_strat(strat, p, r)
            row[wname] = (c, s, d)
        results[sname] = row

        full = row["Full (2010-2025)"]
        y15 = row["2015"][1]
        y18 = row["2018"][1]
        y22 = row["2022"][1]

        flag = ""
        if pd.notna(full[1]) and full[1] > 0.3:
            if pd.notna(y15) and pd.notna(y18) and pd.notna(y22):
                pos_count = sum([y15 > 0, y18 > 0, y22 > 0])
                if pos_count >= 2:
                    flag = " <-- CANDIDATE"
                elif pos_count >= 1:
                    flag = " (partial)"

        full_shr = f"{full[1]:.2f}" if pd.notna(full[1]) else "ERR"
        full_cagr = f"{full[0]:.2%}" if pd.notna(full[0]) else "ERR"
        full_dd = f"{full[2]:.2%}" if pd.notna(full[2]) else "ERR"
        s15 = f"{y15:.2f}" if pd.notna(y15) else "ERR"
        s18 = f"{y18:.2f}" if pd.notna(y18) else "ERR"
        s22 = f"{y22:.2f}" if pd.notna(y22) else "ERR"

        print(f"{sname:<22s} | {full_shr:>11s} | {full_cagr:>9s} | {full_dd:>8s} | {s15:>9s} | {s18:>9s} | {s22:>9s}{flag}")

    print()
    print(f"SPY benchmarks - Full: {spy_bench['Full (2010-2025)']:.2f}, 2015: {spy_bench['2015']:.2f}, 2018: {spy_bench['2018']:.2f}, 2022: {spy_bench['2022']:.2f}")
    print()
    print("ENSEMBLE CANDIDATES (Full Sharpe>0.3 AND positive in >=2 weak years):")
    for sname, row in results.items():
        full = row["Full (2010-2025)"]
        y15 = row["2015"][1]
        y18 = row["2018"][1]
        y22 = row["2022"][1]
        if pd.notna(full[1]) and full[1] > 0.3:
            if pd.notna(y15) and pd.notna(y18) and pd.notna(y22):
                pos_count = sum([y15 > 0, y18 > 0, y22 > 0])
                if pos_count >= 2:
                    print(f"  {sname}: Full Sharpe={full[1]:.2f}, 2015={y15:.2f}, 2018={y18:.2f}, 2022={y22:.2f} ({pos_count}/3 positive)")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
