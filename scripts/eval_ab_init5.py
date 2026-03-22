"""A/B test ensemble variants for Initiative 5.

Tests:
- v7-Baseline:  24 members (current)
- v8a (+R9):    25 members + MultiAssetCTATrend
- v8b (weights-tuned): 24 members with boosted MF1/MF2/F2/F3 weights
- v8c (combined): 25 members + weight tuning + tighter circuit breaker
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy
from financial_algo.strategies.regime_hardening import MultiAssetCTATrend, CommodityMacroOverlay

# --- Import all 24 baseline ensemble members ---
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
from financial_algo.fundamental.strategies import (
    FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence,
)

from financial_algo.strategies.ml_strategies import AdaptiveThreshold


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
    strategy_dd_scale_start=-0.15,
    strategy_dd_scale_end=-0.25,
)

WINDOWS = {
    "Full (2010-2025)": ("2010-01-01", "2025-12-31"),
    "2015": ("2015-01-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}

# -------------------------------------------------------------------------
# Baseline 25-member ensemble (production v7 — matches run_crisis_backtest.py)
# -------------------------------------------------------------------------
def make_baseline():
    members = [
        FeatureComboSignal(),    # P1 Sharpe 0.97
        CrashHedgeQQQ(),         # D2 Sharpe 0.94
        CryptoRecoverySurge(),   # F2 Sharpe 0.93
        MonthlyMacroRegime(),    # MF2 Sharpe 0.93
        VolRiskPremium(),        # L1 Sharpe 0.92
        VolOfVolRegime(),        # L4 Sharpe 0.92
        TailRiskParity(),        # O1 Sharpe 0.90
        VolCarry(),              # D3 Sharpe 0.89
        CryptoGoldDivergence(),  # F3 Sharpe 0.89
        MultiAssetTrend(),       # Q2 Sharpe 0.88
        FearGreedContrarian(),   # G2 Sharpe 0.87
        VolTermStructure(),      # L3 Sharpe 0.86
        VolSpreadHarvest(),      # L2 Sharpe 0.85
        LowVolFactor(),          # K1 Sharpe 0.84
        MultiFactorComposite(),  # K2 Sharpe 0.84
        VolSpikeRecovery(),      # L5 Sharpe 0.84
        QualityTrend(),          # Q1 Sharpe 0.84
        MomentumCrashFilter(),   # Q3 Sharpe 0.83
        ValueFactor(),           # K4 Sharpe 0.81
        SentimentCrisisAlpha(),  # G1 Sharpe 0.80
        SeasonalStrategy(),      # N1 Sharpe 0.79
        WeeklyMomentumRotation(), # MF1 Sharpe 0.78
        SentimentDivergence(),   # G3 Sharpe 0.78
        TailHedgeOverlay(),      # O6 hedge  (Sharpe 0.50 score)
        AdaptiveThreshold(),     # P4 Sharpe 0.75
    ]
    sharpes = [0.97, 0.94, 0.93, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89, 0.88,
               0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.81, 0.80,
               0.79, 0.78, 0.78, 0.50, 0.75]
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    return members, cfg, sharpes


def make_v8a(baseline_members, baseline_sharpes):
    """v8a: baseline + R9-CTATrend (Sharpe 1.14)."""
    members = baseline_members + [MultiAssetCTATrend()]
    sharpes = baseline_sharpes + [1.14]
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    return members, cfg


def make_v8b(baseline_members):
    """v8b: same 24 members but BOOST MF1, MF2, F2, F3 weights
    (the strategies that work in weak years) and REDUCE pure-vol strategies."""
    # Boost the heroes of weak years (3x their normal Sharpe^2)
    # MF2 (2018 hero: +0.63), MF1 (2022 hero: +0.56), F2 (2015 hero: +1.32)
    sharpes = [0.97, 0.94, 1.30, 1.40, 0.92, 0.92, 0.90, 0.89, 1.10,
               0.88, 0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.81, 0.80,
               0.79, 1.20, 0.78, 0.50, 0.75]
    # ^^^^: F2=1.30, MF2=1.40, F3=1.10, MF1=1.20 (boosted), P4=0.75
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    return baseline_members, cfg


def make_v8c(baseline_members):
    """v8c: boosted weights + tighter circuit breaker (dd_scale_start=-0.08)."""
    sharpes = [0.97, 0.94, 1.30, 1.40, 0.92, 0.92, 0.90, 0.89, 1.10,
               0.88, 0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.81, 0.80,
               0.79, 1.20, 0.78, 0.50, 0.75]
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.08,   # tighter - start scaling at -8% DD
        dd_scale_end=-0.18,     # fully flat at -18%
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.60,  # more aggressive correlation hedge
        correlation_hedge_max=0.30,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.18,  # more sensitive vol detection
        vol_crisis_threshold=0.28,
        leverage_elevated=1.6,
        leverage_crisis=1.0,
    )
    return baseline_members, cfg


def make_v8d(baseline_members, baseline_sharpes):
    """v8d: baseline + R10-CommodityMacroOverlay (score 1.0, activates in commodity crises)."""
    members = baseline_members + [CommodityMacroOverlay()]
    sharpes = baseline_sharpes + [1.0]
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    return members, cfg


def make_v8e(baseline_members, baseline_sharpes):
    """v8e: baseline + R9-CTATrend + R10-CommodityMacroOverlay (best of both)."""
    members = baseline_members + [MultiAssetCTATrend(), CommodityMacroOverlay()]
    sharpes = baseline_sharpes + [1.14, 1.0]
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    return members, cfg


def make_v8f(baseline_members, baseline_sharpes):
    """v8f: R9 + R10 with high priority weight for R10 (score=3.0)
    to test if 20%+ allocation to commodity overlay materially improves 2022."""
    members = baseline_members + [MultiAssetCTATrend(), CommodityMacroOverlay()]
    # Give R10 a very high score so it takes ~20% of ensemble weight
    sharpes = baseline_sharpes + [1.14, 3.0]
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.25,   # allow R10 to approach 20-25%
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    return members, cfg


def run_ensemble(members, cfg, prices, regime):
    """Run ensemble backtest."""
    ens = EnsembleStrategy(members, cfg)
    w = ens.backtest_weights(prices, regime)
    res = backtest(prices, w, BT_CFG)
    return res["metrics"]


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

    baseline_members, baseline_cfg, baseline_sharpes = make_baseline()

    v8a_members, v8a_cfg = make_v8a(baseline_members, baseline_sharpes)
    v8d_members, v8d_cfg = make_v8d(baseline_members, baseline_sharpes)
    v8e_members, v8e_cfg = make_v8e(baseline_members, baseline_sharpes)
    v8f_members, v8f_cfg = make_v8f(baseline_members, baseline_sharpes)

    variants = [
        ("v7-Baseline (25)",     baseline_members, baseline_cfg),
        ("v8a +R9 (26)",         v8a_members,      v8a_cfg),
        ("v8d +R10 (26)",        v8d_members,      v8d_cfg),
        ("v8e +R9+R10 (27)",     v8e_members,      v8e_cfg),
        ("v8f R9+R10-MaxWt (27)", v8f_members,     v8f_cfg),
    ]

    print(f"{'Variant':<25s} | {'Window':<22s} | {'Sharpe':>7s} | {'CAGR':>8s} | {'MaxDD':>8s}")
    print(f"{'-'*25}-+-{'-'*22}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    BASELINE_KEY = "v7-Baseline (25)"
    summary = {}
    for vname, members, cfg in variants:
        row = {}
        for wname, (ws, we) in WINDOWS.items():
            mask = (prices.index >= ws) & (prices.index <= we)
            p = prices.loc[mask].copy()
            r = regime.loc[mask].copy()
            if len(p) < 30:
                continue
            try:
                m = run_ensemble(members, cfg, p, r)
                row[wname] = m
                print(f"{vname:<25s} | {wname:<22s} | {m['sharpe']:>7.2f} | {m['cagr']:>8.2%} | {m['max_drawdown']:>8.2%}")
            except Exception as e:
                print(f"{vname:<25s} | {wname:<22s} | ERR: {e}")
                row[wname] = None
        summary[vname] = row
        print()

    # Delta vs baseline
    print("=" * 78)
    print("DELTA vs v7-Baseline (25 members)")
    print("=" * 78)
    base_full = (summary.get(BASELINE_KEY) or {}).get("Full (2010-2025)")
    if base_full:
        for vname, row in summary.items():
            if vname == BASELINE_KEY:
                continue
            full = (row or {}).get("Full (2010-2025)")
            if full:
                delta = full["sharpe"] - base_full["sharpe"]
                delta_dd = full["max_drawdown"] - base_full["max_drawdown"]
                winner = "IMPROVEMENT" if delta > 0.005 else ("NEUTRAL" if delta > -0.005 else "REGRESSION")
                print(f"  {vname:<25s}: dSharpe={delta:+.3f}, dMaxDD={delta_dd:+.3f} -> {winner}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
