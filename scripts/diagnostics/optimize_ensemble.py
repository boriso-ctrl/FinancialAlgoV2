"""Quick ensemble optimizer — tests multiple configurations to find the best Sharpe.

Run: .venv\Scripts\python.exe scripts/diagnostics/optimize_ensemble.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import RegimeConfig, detect_regime
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig

# All candidate strategies (Sharpe > 0.5 on full period)
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry, FourStateTactical
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium, VolSpreadHarvest, VolTermStructure, VolOfVolRegime, VolSpikeRecovery,
)
from financial_algo.strategies.tail_risk import TailRiskParity
from financial_algo.strategies.quality_trend import QualityTrend, MultiAssetTrend, MomentumCrashFilter
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite, ValueFactor
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.momentum import TimeSeriesMomentum, MomentumVolScaled
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.macro import DollarCarry, GoldDollarInverse, CommodityMomentum, RatesRegimeTrade
from financial_algo.strategies.mean_reversion import OvernightGapFade
from financial_algo.strategies.fixed_income import CreditSpreadMeanRev
from financial_algo.fundamental.strategies import (
    FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence,
)
from financial_algo.fundamental import build_synthetic_sentiment

# ------------------------------------------------------------------
TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP",
    "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "HYG", "LQD",
    "BTC-USD",
]))

BT_ENS = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=None,
)

BT_ENS_NOVT = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=None,          # no vol target -- let natural diversification work
    max_drawdown_trigger=None,
)

BT_ENS_HI = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.25,          # higher vol target
    max_drawdown_trigger=None,
)


def main():
    print("[1/3] Loading data ...")
    prices = load_prices(TICKERS, "2009-01-01", "2025-12-31")
    vix = load_prices(["^VIX"], "2009-01-01", "2025-12-31")
    vix_series = vix["^VIX"] if "^VIX" in vix.columns else None
    prices = prices.reindex(columns=[t for t in prices.columns if t != "^VIX"])
    if vix_series is not None:
        prices["^VIX"] = vix_series
    regime = detect_regime(prices, vix_series, RegimeConfig())

    # Full period slice
    mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
    p = prices.loc[mask].copy()
    r = regime.loc[mask].copy()

    # Build sentiment
    sent = build_synthetic_sentiment(p)

    print("[2/3] Pre-computing strategy weights ...")
    # Candidate pool with name + Sharpe + instance
    candidates = [
        ("P1", 0.97, FeatureComboSignal()),
        ("D2", 0.94, CrashHedgeQQQ()),
        ("L1", 0.92, VolRiskPremium()),
        ("O1", 0.90, TailRiskParity()),
        ("D3", 0.89, VolCarry()),
        ("Q2", 0.88, MultiAssetTrend()),
        ("G2", 0.87, FearGreedContrarian()),
        ("L2", 0.85, VolSpreadHarvest()),
        ("K1", 0.84, LowVolFactor()),
        ("L5", 0.84, VolSpikeRecovery()),
        ("Q1", 0.84, QualityTrend()),
        ("Q3", 0.83, MomentumCrashFilter()),
        ("N1", 0.79, SeasonalStrategy()),
        ("K2", 0.73, MultiFactorComposite()),
        ("I1", 0.70, TimeSeriesMomentum()),
        ("I4", 0.70, MomentumVolScaled()),
        ("M1", 0.70, DollarCarry()),
        ("G1", 0.68, SentimentCrisisAlpha()),
        ("M2", 0.68, GoldDollarInverse()),
        ("K4", 0.68, ValueFactor()),
        ("G3", 0.65, SentimentDivergence()),
        ("L3", 0.64, VolTermStructure()),
        ("D1", 0.63, FourStateTactical()),
        ("L4", 0.63, VolOfVolRegime()),
    ]

    # Pre-compute all strategy weights (expensive step, but only once)
    precomputed = {}
    for name, sharpe, strat in candidates:
        try:
            w = strat.backtest_weights(p, r)
            precomputed[name] = (sharpe, strat, w)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    print(f"  {len(precomputed)} strategies ready")

    print("[3/3] Testing ensemble configurations ...")

    # Test multiple configs
    results = []

    # Config variations: (name, inv_vol, sharpe_pow, max_lev, dd_start, dd_end, max_w)
    configs = [
        ("fixed-sharpe2-3x-nocb",  False, 2, 3.0, -0.50, -0.80, 0.25),
        ("fixed-sharpe3-3x-nocb",  False, 3, 3.0, -0.50, -0.80, 0.25),
        ("invvol-sharpe2-3x-nocb", True, 2, 3.0, -0.50, -0.80, 0.25),
    ]

    # Top N member counts
    member_counts = [8, 10, 12]

    sorted_names = sorted(precomputed.keys(), key=lambda n: precomputed[n][0], reverse=True)

    # Test with different backtest configs (vol target variations)
    bt_configs = [
        ("vt20", BT_ENS),
        ("vt25", BT_ENS_HI),
        ("noVT", BT_ENS_NOVT),
    ]

    for n_members in member_counts:
        members_names = sorted_names[:n_members]
        members = [precomputed[n][1] for n in members_names]
        sharpes = [precomputed[n][0] for n in members_names]

        for cfg_name, inv_vol, sharpe_pow, max_lev, dd_start, dd_end, max_w in configs:
            priors = [s ** sharpe_pow for s in sharpes]
            cfg = EnsembleConfig(
                use_inverse_vol=inv_vol,
                max_gross_leverage=max_lev,
                max_single_weight=max_w,
                dd_scale_start=dd_start,
                dd_scale_end=dd_end,
                prior_weights=priors,
            )
            for bt_name, bt_cfg in bt_configs:
                try:
                    ens = EnsembleStrategy(members, cfg)
                    ens_w = ens.backtest_weights(p, r)
                    res = backtest(p, ens_w, bt_cfg)
                    m = res["metrics"]
                    results.append({
                        "n": n_members,
                        "config": f"{cfg_name}|{bt_name}",
                        "members": ",".join(members_names),
                        "CAGR": m["cagr"],
                        "Sharpe": m["sharpe"],
                        "Sortino": m["sortino"],
                        "MaxDD": m["max_drawdown"],
                        "Calmar": m["calmar"],
                        "AnnVol": m["annual_vol"],
                    })
                except Exception as e:
                    print(f"  [ERR] n={n_members} {cfg_name}|{bt_name}: {e}")

    # Sort by Sharpe
    results.sort(key=lambda r: r["Sharpe"], reverse=True)

    print()
    print("=" * 120)
    print("TOP 20 ENSEMBLE CONFIGURATIONS (by Sharpe)")
    print("=" * 120)
    print(f"{'N':>3} {'Config':>35} {'CAGR':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'AnnVol':>7}  Members")
    print("-" * 130)
    for r in results[:20]:
        print(f"{r['n']:>3} {r['config']:>35} {r['CAGR']:>7.2%} {r['Sharpe']:>7.2f} {r['Sortino']:>8.2f} {r['MaxDD']:>7.2%} {r['Calmar']:>7.2f} {r['AnnVol']:>6.2%}  {r['members']}")

    print()
    print("=" * 120)
    print("WORST 5 (avoid these)")
    print("=" * 120)
    for r in results[-5:]:
        print(f"{r['n']:>3} {r['config']:>35} {r['CAGR']:>7.2%} {r['Sharpe']:>7.2f} {r['Sortino']:>8.2f} {r['MaxDD']:>7.2%} {r['Calmar']:>7.2f} {r['AnnVol']:>6.2%}  {r['members']}")


if __name__ == "__main__":
    main()
