"""Diagnose which ensemble members bleed / help in weak WF years.

Outputs per-strategy Sharpe for 2015, 2018, 2022 to find:
1. Which strategies are destroying alpha
2. Which strategies are helping (if any)
3. What regime conditions dominate each year
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

# Import all 24 ensemble members + extras
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.crypto_crisis import CryptoRecoverySurge, CryptoGoldDivergence
from financial_algo.strategies.multi_freq import MonthlyMacroRegime, WeeklyMomentumRotation
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium, VolOfVolRegime, VolTermStructure, VolSpreadHarvest, VolSpikeRecovery,
)
from financial_algo.strategies.tail_risk import TailRiskParity, TailHedgeOverlay
from financial_algo.strategies.quality_trend import QualityTrend, MultiAssetTrend, MomentumCrashFilter
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite, ValueFactor
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.momentum import DualMomentum
from financial_algo.fundamental.strategies import (
    FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence,
)
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig

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

WEAK_YEARS = {
    "2015": ("2015-01-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}

ENSEMBLE_STRATS = [
    ("P1-FeatureCombo",    FeatureComboSignal()),
    ("D2-CrashHedgeQQQ",   CrashHedgeQQQ()),
    ("F2-CryptoRecovery",  CryptoRecoverySurge()),
    ("MF2-MonthlyMacro",   MonthlyMacroRegime()),
    ("L1-VolRiskPremium",  VolRiskPremium()),
    ("L4-VolOfVolRegime",  VolOfVolRegime()),
    ("O1-TailRiskParity",  TailRiskParity()),
    ("D3-VolCarry",        VolCarry()),
    ("F3-CryptoGoldDiv",   CryptoGoldDivergence()),
    ("Q2-MultiAssetTrend", MultiAssetTrend()),
    ("G2-FearGreed",       FearGreedContrarian()),
    ("L3-VolTermStruct",   VolTermStructure()),
    ("L2-VolSpreadHarv",   VolSpreadHarvest()),
    ("K1-LowVolFactor",    LowVolFactor()),
    ("K2-MultiFactor",     MultiFactorComposite()),
    ("L5-VolSpikeRecov",   VolSpikeRecovery()),
    ("Q1-QualityTrend",    QualityTrend()),
    ("Q3-MomCrashFilter",  MomentumCrashFilter()),
    ("K4-ValueFactor",     ValueFactor()),
    ("G1-SentCrisisAlpha", SentimentCrisisAlpha()),
    ("N1-Seasonal",        SeasonalStrategy()),
    ("MF1-WeeklyMom",      WeeklyMomentumRotation()),
    ("G3-SentDivergence",  SentimentDivergence()),
    ("O6-TailHedge",       TailHedgeOverlay()),
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
    print()

    for year_name, (start, end) in WEAK_YEARS.items():
        mask = (prices.index >= start) & (prices.index <= end)
        p = prices.loc[mask].copy()
        r = regime.loc[mask].copy()

        if len(p) < 20:
            print(f"[SKIP] {year_name}: not enough data")
            continue

        print("=" * 70)
        print(f"  WEAK YEAR: {year_name} ({len(p)} days)")
        print("=" * 70)

        # Regime distribution
        rc = r.value_counts()
        print("  Regime distribution:")
        for rv, cnt in rc.items():
            print(f"    {rv.value:20s}: {cnt:4d} days ({cnt/len(r)*100:.1f}%)")
        print()

        # SPY benchmark
        w_spy = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        w_spy["SPY"] = 1.0
        try:
            spy_res = backtest(p, w_spy.shift(1).fillna(0), BT_CFG)
            spy_m = spy_res["metrics"]
            print(f"  SPY Buy-Hold: CAGR={spy_m['cagr']:.2%}, Sharpe={spy_m['sharpe']:.2f}, MaxDD={spy_m['max_drawdown']:.2%}")
        except Exception as e:
            print(f"  SPY: ERR {e}")
        print()

        # Per-strategy attribution
        results = []
        for sname, strat in ENSEMBLE_STRATS:
            try:
                w = strat.backtest_weights(p, r)
                res = backtest(p, w, BT_CFG)
                m = res["metrics"]
                results.append({
                    "Strategy": sname,
                    "CAGR": m["cagr"],
                    "Sharpe": m["sharpe"],
                    "MaxDD": m["max_drawdown"],
                    "AnnVol": m["annual_vol"],
                })
            except Exception as e:
                results.append({
                    "Strategy": sname,
                    "CAGR": float("nan"),
                    "Sharpe": float("nan"),
                    "MaxDD": float("nan"),
                    "AnnVol": float("nan"),
                })

        df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
        print(f"  {'Strategy':<22s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'AnnVol':>8s}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in df.iterrows():
            cagr_s = f"{row['CAGR']:.2%}" if pd.notna(row['CAGR']) else "ERR"
            shr_s = f"{row['Sharpe']:.2f}" if pd.notna(row['Sharpe']) else "ERR"
            mdd_s = f"{row['MaxDD']:.2%}" if pd.notna(row['MaxDD']) else "ERR"
            vol_s = f"{row['AnnVol']:.2%}" if pd.notna(row['AnnVol']) else "ERR"
            marker = " <-- BLEEDING" if pd.notna(row['Sharpe']) and row['Sharpe'] < -0.1 else ""
            marker = " <-- STRONG" if pd.notna(row['Sharpe']) and row['Sharpe'] > 0.5 else marker
            print(f"  {row['Strategy']:<22s} {cagr_s:>8s} {shr_s:>8s} {mdd_s:>8s} {vol_s:>8s}{marker}")

        # Summary stats
        sharpes = df["Sharpe"].dropna()
        bleeders = sharpes[sharpes < -0.1].count()
        strong = sharpes[sharpes > 0.5].count()
        print(f"\n  Summary: {bleeders} bleeders (Sharpe < -0.1), {strong} strong (Sharpe > 0.5)")
        print(f"  Median Sharpe: {sharpes.median():.2f}, Mean: {sharpes.mean():.2f}")
        print()

    print("DONE")


if __name__ == "__main__":
    main()
