"""Initiative 7 — A/B test unvalidated R-strategies for ensemble v10.

Tests R1, R3, R4, R5, R6, R7, R8 individually and in smart combos.
Baseline: 26-member ensemble (v9).
Decision criterion: WF Sharpe improvement AND correlation check.

Run:
    .venv\Scripts\python.exe scripts/eval_init7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy

# -- Baseline 26-member ensemble imports --
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium, VolOfVolRegime, VolTermStructure,
    VolSpreadHarvest, VolSpikeRecovery,
)
from financial_algo.strategies.tail_risk import TailRiskParity, TailHedgeOverlay
from financial_algo.strategies.crypto_crisis import (
    CryptoRecoverySurge, CryptoGoldDivergence,
)
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.ml_strategies import AdaptiveThreshold
from financial_algo.strategies.quality_trend import (
    QualityTrend, MultiAssetTrend, MomentumCrashFilter,
)
from financial_algo.strategies.factor import (
    LowVolFactor, MultiFactorComposite, ValueFactor,
)
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.multi_freq import (
    WeeklyMomentumRotation, MonthlyMacroRegime,
)
from financial_algo.fundamental.strategies import (
    SentimentCrisisAlpha, FearGreedContrarian, SentimentDivergence,
)
from financial_algo.strategies.regime_hardening import (
    BearMarketAlpha, DefensiveRotationR3, AdaptiveRiskBudget,
    RatesTighteningAlpha, BondEquityHedge, VolExplosionAlpha,
    VolRegimeSwitcher, MultiAssetCTATrend,
)

# =========================================================================
DATA_START = "2009-01-01"
DATA_END   = "2025-12-31"
FULL_START = "2010-01-01"
VIX_TICKER = "^VIX"

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP", "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "HYG", "LQD", "BTC-USD",
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

# Baseline Sharpe^2 prior weights (26 members)
_BASELINE_SHARPES = [
    0.97, 0.94, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89,
    0.88, 0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84,
    0.83, 0.81, 0.80, 0.79, 0.78, 0.50, 0.75, 0.78, 0.93, 1.14,
]

ENS_CFG_TEMPLATE = EnsembleConfig(
    use_inverse_vol=False,
    max_gross_leverage=2.5,
    max_single_weight=0.20,
    dd_scale_start=-0.12,
    dd_scale_end=-0.22,
    correlation_hedge_enabled=True,
    correlation_hedge_threshold=0.65,
    correlation_hedge_max=0.25,
    vol_regime_scaling=True,
    vol_elevated_threshold=0.20,
    vol_crisis_threshold=0.30,
    leverage_elevated=1.8,
    leverage_crisis=1.2,
)


# =========================================================================
# Baseline 26-member strategy list (same order as run_walk_forward.py)
# =========================================================================
def build_baseline_members():
    return [
        FeatureComboSignal(),
        CrashHedgeQQQ(),
        CryptoRecoverySurge(),
        VolRiskPremium(),
        VolOfVolRegime(),
        TailRiskParity(),
        VolCarry(),
        CryptoGoldDivergence(),
        MultiAssetTrend(),
        FearGreedContrarian(),
        VolTermStructure(),
        VolSpreadHarvest(),
        LowVolFactor(),
        MultiFactorComposite(),
        VolSpikeRecovery(),
        QualityTrend(),
        MomentumCrashFilter(),
        ValueFactor(),
        SentimentCrisisAlpha(),
        SeasonalStrategy(),
        SentimentDivergence(),
        TailHedgeOverlay(),
        AdaptiveThreshold(),
        WeeklyMomentumRotation(),
        MonthlyMacroRegime(),
        MultiAssetCTATrend(),
    ]


def make_ensemble(strategies, sharpe_priors):
    cfg = EnsembleConfig(
        use_inverse_vol=ENS_CFG_TEMPLATE.use_inverse_vol,
        max_gross_leverage=ENS_CFG_TEMPLATE.max_gross_leverage,
        max_single_weight=ENS_CFG_TEMPLATE.max_single_weight,
        dd_scale_start=ENS_CFG_TEMPLATE.dd_scale_start,
        dd_scale_end=ENS_CFG_TEMPLATE.dd_scale_end,
        prior_weights=[s ** 2 for s in sharpe_priors],
        correlation_hedge_enabled=ENS_CFG_TEMPLATE.correlation_hedge_enabled,
        correlation_hedge_threshold=ENS_CFG_TEMPLATE.correlation_hedge_threshold,
        correlation_hedge_max=ENS_CFG_TEMPLATE.correlation_hedge_max,
        safe_haven_tickers=ENS_CFG_TEMPLATE.safe_haven_tickers,
        vol_regime_scaling=ENS_CFG_TEMPLATE.vol_regime_scaling,
        vol_elevated_threshold=ENS_CFG_TEMPLATE.vol_elevated_threshold,
        vol_crisis_threshold=ENS_CFG_TEMPLATE.vol_crisis_threshold,
        leverage_elevated=ENS_CFG_TEMPLATE.leverage_elevated,
        leverage_crisis=ENS_CFG_TEMPLATE.leverage_crisis,
    )
    return EnsembleStrategy(strategies, cfg)


def run_ensemble(ensemble, prices, regime):
    w = ensemble.backtest_weights(prices, regime)
    res = backtest(prices, w, BT_CFG)
    m = res["metrics"]
    return m.get("sharpe", 0.0), m.get("cagr", 0.0), m.get("max_drawdown", 0.0)


def compute_correlation(strat, base_strats, prices, regime):
    """Compute avg pairwise correlation of new strat's returns vs baseline ensemble."""
    w_new = strat.backtest_weights(prices, regime)
    ret_new = backtest(prices, w_new, BT_CFG)["returns"]

    corrs = []
    for s in base_strats:
        try:
            w_s = s.backtest_weights(prices, regime)
            ret_s = backtest(prices, w_s, BT_CFG)["returns"]
            both = pd.concat([ret_new, ret_s], axis=1).dropna()
            if len(both) > 50:
                corrs.append(both.iloc[:, 0].corr(both.iloc[:, 1]))
        except Exception:
            pass
    return np.mean(corrs) if corrs else 0.0


def metrics_for_year(ensemble, prices, regime, year):
    """Backtest ensemble on a single calendar year."""
    mask = prices.index.year == year
    if mask.sum() < 30:
        return 0.0
    w = ensemble.backtest_weights(prices, regime)
    p_y = prices.loc[mask]
    w_y = w.loc[mask]
    res = backtest(p_y, w_y, BT_CFG)
    return res["metrics"].get("sharpe", 0.0)


# =========================================================================
def main():
    print("=" * 80)
    print("INITIATIVE 7 -- R-STRATEGY ENSEMBLE A/B TEST")
    print("=" * 80)

    # -- Load data --
    print("\nLoading data ...", end=" ", flush=True)
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER]
    except Exception:
        vix = None
    regime = detect_regime(prices, vix=vix)

    full_mask = (prices.index >= FULL_START)
    p_full = prices.loc[full_mask]
    r_full = regime.loc[full_mask]
    print(f"done. {p_full.shape[0]} days x {p_full.shape[1]} tickers")

    # -- Candidates --
    candidates = {
        "R1-BearMarketAlpha":   (BearMarketAlpha(),      0.84),   # standalone IS Sharpe proxy
        "R3-DefensiveRotation": (DefensiveRotationR3(),  0.90),
        "R4-AdaptiveRiskBudget":(AdaptiveRiskBudget(),   0.85),
        "R5-RatesTightening":   (RatesTighteningAlpha(), 0.87),
        "R6-BondEquityHedge":   (BondEquityHedge(),      0.70),
        "R7-VolExplosionAlpha": (VolExplosionAlpha(),     0.90),
        "R8-VolRegimeSwitcher": (VolRegimeSwitcher(),     0.81),
    }

    # -- Baseline --
    print("\n[1/4] Running BASELINE (v9 - 26 members) ...")
    base_strats = build_baseline_members()
    ens_base = make_ensemble(base_strats, _BASELINE_SHARPES)
    s_base, c_base, dd_base = run_ensemble(ens_base, p_full, r_full)
    s2015_base = metrics_for_year(ens_base, p_full, r_full, 2015)
    s2018_base = metrics_for_year(ens_base, p_full, r_full, 2018)
    s2022_base = metrics_for_year(ens_base, p_full, r_full, 2022)
    print(f"   Baseline: Sharpe={s_base:.4f}  CAGR={c_base:.2%}  MaxDD={dd_base:.2%}")
    print(f"   Weak years: 2015={s2015_base:.2f}  2018={s2018_base:.2f}  2022={s2022_base:.2f}")

    # -- Individual candidates --
    print("\n[2/4] Testing candidates individually ...")
    rows = []
    for name, (strat, is_sharpe) in candidates.items():
        print(f"   Testing {name} ...", end=" ", flush=True)

        # Correlation check
        avg_corr = compute_correlation(strat, base_strats[:8], p_full, r_full)

        # Ensemble test: +1 strategy
        new_strats = base_strats + [strat]
        new_sharpes = _BASELINE_SHARPES + [is_sharpe]
        ens_new = make_ensemble(new_strats, new_sharpes)
        s_new, c_new, dd_new = run_ensemble(ens_new, p_full, r_full)
        d_sharpe = s_new - s_base
        d_dd = dd_new - dd_base
        s2015 = metrics_for_year(ens_new, p_full, r_full, 2015)
        s2018 = metrics_for_year(ens_new, p_full, r_full, 2018)
        s2022 = metrics_for_year(ens_new, p_full, r_full, 2022)

        verdict = "ADD" if (d_sharpe > 0.005 or d_dd > 0.005) and avg_corr < 0.7 else "skip"
        print(f"dSharpe={d_sharpe:+.4f}  dMaxDD={d_dd:+.4f}  AvgCorr={avg_corr:.2f}  -> {verdict}")

        rows.append({
            "Strategy": name,
            "AvgCorr": round(avg_corr, 3),
            "dSharpe": round(d_sharpe, 4),
            "dMaxDD": round(d_dd, 4),
            "Full_Sharpe": round(s_new, 4),
            "Full_CAGR": f"{c_new:.2%}",
            "Full_MaxDD": f"{dd_new:.2%}",
            "2015": round(s2015, 2),
            "2018": round(s2018, 2),
            "2022": round(s2022, 2),
            "Verdict": verdict,
        })

    # -- Best combo --
    print("\n[3/4] Testing best combo (top improvers) ...")
    adds = [r for r in rows if r["Verdict"] == "ADD"]
    if adds:
        add_names = [r["Strategy"] for r in adds]
        print(f"   Building combo with: {add_names}")
        combo_strats = base_strats.copy()
        combo_sharpes = _BASELINE_SHARPES.copy()
        for name, (strat, is_sharpe) in candidates.items():
            if name in add_names:
                combo_strats.append(strat)
                combo_sharpes.append(is_sharpe)
        ens_combo = make_ensemble(combo_strats, combo_sharpes)
        s_combo, c_combo, dd_combo = run_ensemble(ens_combo, p_full, r_full)
        s2015_c = metrics_for_year(ens_combo, p_full, r_full, 2015)
        s2018_c = metrics_for_year(ens_combo, p_full, r_full, 2018)
        s2022_c = metrics_for_year(ens_combo, p_full, r_full, 2022)
        print(f"   COMBO ({len(combo_strats)} members): "
              f"Sharpe={s_combo:.4f}  CAGR={c_combo:.2%}  MaxDD={dd_combo:.2%}")
        print(f"   Weak years: 2015={s2015_c:.2f}  2018={s2018_c:.2f}  2022={s2022_c:.2f}")
        combo_delta = s_combo - s_base
        print(f"   vs Baseline: dSharpe={combo_delta:+.4f}")
    else:
        print("   No candidates improved the ensemble individually. Combo skipped.")
        s_combo, c_combo, dd_combo = s_base, c_base, dd_base
        combo_delta = 0.0
        add_names = []

    # -- Summary table --
    print("\n[4/4] RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Strategy':<26} {'Corr':>6} {'dSharpe':>9} {'dMaxDD':>8} "
          f"{'Full.S':>7} {'Full.CAGR':>10} {'MaxDD':>7} "
          f"{'2015S':>6} {'2018S':>6} {'2022S':>6}  {'Verdict'}")
    print("-" * 100)
    print(f"{'v9-Baseline':<26} {'':>6} {'':>9} {'':>8} "
          f"{s_base:>7.4f} {c_base:>9.2%} {dd_base:>7.2%} "
          f"{s2015_base:>6.2f} {s2018_base:>6.2f} {s2022_base:>6.2f}")
    for r in rows:
        print(f"{r['Strategy']:<26} {r['AvgCorr']:>6.3f} {r['dSharpe']:>+9.4f} "
              f"{r['dMaxDD']:>+8.4f} {r['Full_Sharpe']:>7.4f} "
              f"{r['Full_CAGR']:>10} {r['Full_MaxDD']:>7}  "
              f"{r['2015']:>6.2f} {r['2018']:>6.2f} {r['2022']:>6.2f}  {r['Verdict']}")
    if adds:
        print("-" * 100)
        print(f"{'COMBO ('+str(len(combo_strats))+' members)':<26} {'':>6} "
              f"{combo_delta:>+9.4f} {'':>8} {s_combo:>7.4f} {c_combo:>9.2%} "
              f"{dd_combo:>7.2%}  {s2015_c:>6.2f} {s2018_c:>6.2f} {s2022_c:>6.2f}")
    print("=" * 100)

    # -- Decision --
    best_single_row = max(rows, key=lambda r: r["dSharpe"])
    print(f"\nBest single addition: {best_single_row['Strategy']} (dSharpe {best_single_row['dSharpe']:+.4f})")
    if adds and combo_delta > best_single_row["dSharpe"]:
        print(f"Best overall: COMBO of {add_names} (dSharpe {combo_delta:+.4f})")
        print("DECISION: Add combo to ensemble -> v10\n")
    elif best_single_row["dSharpe"] > 0.005:
        print(f"DECISION: Add {best_single_row['Strategy']} to ensemble -> v10\n")
    else:
        print("DECISION: No R-strategy improves ensemble. Keep v9.\n")


if __name__ == "__main__":
    main()
