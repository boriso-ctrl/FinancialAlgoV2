"""Evaluate DL-1, DL-2, DL-3 strategies and A/B test into ensemble v10."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
warnings.filterwarnings("ignore")

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime

# DL strategies
from financial_algo.strategies.dl_strategies import (
    TemporalCNNAlpha,
    LSTMRegimeDetector,
    AttentionCrossSectionalRanker,
)

# Current ensemble members (v9 = 26 members)
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
from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
from financial_algo.fundamental.strategies import (
    SentimentCrisisAlpha, FearGreedContrarian, SentimentDivergence,
)

from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy

# =====================================================================
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

BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=-0.25,
    drawdown_recovery_rate=0.10,
)

BT_CONFIG_ENSEMBLE = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=None,
    strategy_dd_scale_start=-0.15,
    strategy_dd_scale_end=-0.25,
)


def run_backtest(strat, prices, regime, label):
    """Run single strategy backtest and return metrics dict."""
    try:
        w = strat.backtest_weights(prices, regime)
        res = backtest(prices, w, BT_CONFIG)
        m = res["metrics"]
        returns = res["returns"]
        return {
            "name": label,
            "cagr": m.get("cagr", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "sortino": m.get("sortino", 0.0),
            "max_dd": m.get("max_drawdown", 0.0),
            "calmar": m.get("calmar", 0.0),
            "returns": returns,
        }
    except Exception as e:
        print(f"  ERROR backtesting {label}: {e}")
        return {
            "name": label,
            "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_dd": 0.0, "calmar": 0.0, "returns": None,
        }


def main():
    print("=" * 80)
    print("DL STRATEGY EVALUATION + ENSEMBLE v10 A/B TEST")
    print("=" * 80)
    print()

    # 1. Load data
    print("[1/5] Loading price data ...")
    prices_raw = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
    try:
        vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
        vix = vix_df["^VIX"]
    except Exception:
        vix = None
    regime = detect_regime(prices_raw, vix=vix)

    mask = (prices_raw.index >= "2010-01-01") & (prices_raw.index <= "2025-12-31")
    prices = prices_raw.loc[mask]
    reg = regime.loc[mask]
    print(f"  {prices.shape[0]} days x {prices.shape[1]} tickers")
    print()

    # 2. Backtest DL strategies
    print("[2/5] Backtesting DL strategies (this may take a few minutes) ...")
    dl_strats = [
        (TemporalCNNAlpha(), "DL1-TemporalCNNAlpha"),
        (LSTMRegimeDetector(), "DL2-LSTMRegimeDetector"),
        (AttentionCrossSectionalRanker(), "DL3-AttentionRanker"),
    ]

    dl_results = []
    for strat, label in dl_strats:
        print(f"  Running {label} ...", end=" ", flush=True)
        r = run_backtest(strat, prices, reg, label)
        dl_results.append(r)
        print(f"Sharpe={r['sharpe']:.2f}  CAGR={r['cagr']:.1%}  MaxDD={r['max_dd']:.1%}")

    print()
    print("  DL Strategy Results:")
    print("  " + "-" * 70)
    print(f"  {'Strategy':<30} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
    print("  " + "-" * 70)
    for r in dl_results:
        print(f"  {r['name']:<30} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} "
              f"{r['sortino']:>8.2f} {r['max_dd']:>7.1%}")
    print()

    # 3. Check correlation with existing ensemble members
    print("[3/5] Computing DL correlation with existing strategies ...")
    # Run a subset of existing strategies for correlation check
    existing = [
        (CrashHedgeQQQ(), "D2-CrashHedgeQQQ"),
        (VolCarry(), "D3-VolCarry"),
        (VolRiskPremium(), "L1-VolRiskPremium"),
        (MomentumCrashFilter(), "Q3-MomentumCrashFilter"),
        (MultiAssetCTATrend(), "R9-MultiAssetCTATrend"),
        (MonthlyMacroRegime(), "MF2-MonthlyMacroRegime"),
    ]

    # Collect returns for correlation
    all_returns = {}
    for r in dl_results:
        if r["returns"] is not None:
            all_returns[r["name"]] = r["returns"]

    for strat, label in existing:
        try:
            w = strat.backtest_weights(prices, reg)
            res = backtest(prices, w, BT_CONFIG)
            all_returns[label] = res["returns"]
        except Exception:
            pass

    if all_returns:
        ret_df = pd.DataFrame(all_returns).dropna()
        corr_matrix = ret_df.corr()
        print()
        dl_names = [r["name"] for r in dl_results]
        existing_names = [n for n in all_returns if n not in dl_names]
        for dl_name in dl_names:
            if dl_name in corr_matrix.columns:
                corrs_with_existing = corr_matrix.loc[dl_name, existing_names]
                avg_corr = corrs_with_existing.mean()
                max_corr = corrs_with_existing.max()
                print(f"  {dl_name}: avg_corr={avg_corr:.3f}  max_corr={max_corr:.3f}")
    print()

    # 4. Filter: Only DL strategies with Sharpe > 0.3 get A/B tested
    candidates = [r for r in dl_results if r["sharpe"] > 0.3]
    if not candidates:
        print("[4/5] No DL strategies meet Sharpe > 0.3 threshold. Skipping A/B test.")
        print("  RECOMMENDATION: Tune hyperparameters or architecture before retrying.")
        return

    print(f"[4/5] A/B testing {len(candidates)} DL candidate(s) into ensemble v10 ...")

    # Build v9 baseline ensemble (26 members)
    v9_members = [
        FeatureComboSignal(), CrashHedgeQQQ(), CryptoRecoverySurge(),
        VolRiskPremium(), VolOfVolRegime(), TailRiskParity(),
        VolCarry(), CryptoGoldDivergence(), MultiAssetTrend(),
        FearGreedContrarian(), VolTermStructure(), VolSpreadHarvest(),
        LowVolFactor(), MultiFactorComposite(), VolSpikeRecovery(),
        QualityTrend(), MomentumCrashFilter(), ValueFactor(),
        SentimentCrisisAlpha(), SeasonalStrategy(), SentimentDivergence(),
        TailHedgeOverlay(), AdaptiveThreshold(),
        WeeklyMomentumRotation(), MonthlyMacroRegime(), MultiAssetCTATrend(),
    ]
    v9_sharpes = [
        0.97, 0.94, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89,
        0.88, 0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84,
        0.83, 0.81, 0.80, 0.79, 0.78, 0.50, 0.75,
        0.78, 0.93, 1.14,
    ]

    # v9 baseline
    v9_prior = [s ** 2 for s in v9_sharpes]
    v9_cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=v9_prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    v9_ensemble = EnsembleStrategy(v9_members, v9_cfg)
    v9_w = v9_ensemble.backtest_weights(prices, reg)
    v9_res = backtest(prices, v9_w, BT_CONFIG_ENSEMBLE)
    v9_m = v9_res["metrics"]
    v9_sharpe = v9_m.get("sharpe", 0.0)
    v9_cagr = v9_m.get("cagr", 0.0)
    v9_maxdd = v9_m.get("max_drawdown", 0.0)

    print(f"  v9-Baseline ({len(v9_members)} members): "
          f"Sharpe={v9_sharpe:.2f}  CAGR={v9_cagr:.1%}  MaxDD={v9_maxdd:.1%}")
    print()

    # Test each DL candidate as v10 variant
    dl_strat_map = {
        "DL1-TemporalCNNAlpha": TemporalCNNAlpha(),
        "DL2-LSTMRegimeDetector": LSTMRegimeDetector(),
        "DL3-AttentionRanker": AttentionCrossSectionalRanker(),
    }

    best_variant = None
    best_delta = -999.0

    for cand in candidates:
        name = cand["name"]
        strat_obj = dl_strat_map.get(name)
        if strat_obj is None:
            continue

        v10_members = v9_members + [strat_obj]
        v10_sharpes = v9_sharpes + [cand["sharpe"]]
        v10_prior = [s ** 2 for s in v10_sharpes]

        v10_cfg = EnsembleConfig(
            use_inverse_vol=False,
            max_gross_leverage=2.5,
            max_single_weight=0.20,
            dd_scale_start=-0.12,
            dd_scale_end=-0.22,
            prior_weights=v10_prior,
            correlation_hedge_enabled=True,
            correlation_hedge_threshold=0.65,
            correlation_hedge_max=0.25,
            vol_regime_scaling=True,
            vol_elevated_threshold=0.20,
            vol_crisis_threshold=0.30,
            leverage_elevated=1.8,
            leverage_crisis=1.2,
        )
        v10_ensemble = EnsembleStrategy(v10_members, v10_cfg)
        v10_w = v10_ensemble.backtest_weights(prices, reg)
        v10_res = backtest(prices, v10_w, BT_CONFIG_ENSEMBLE)
        v10_m = v10_res["metrics"]

        v10_sharpe = v10_m.get("sharpe", 0.0)
        v10_cagr = v10_m.get("cagr", 0.0)
        v10_maxdd = v10_m.get("max_drawdown", 0.0)
        delta = v10_sharpe - v9_sharpe

        flag = "IMPROVEMENT" if delta > 0.01 else ("REGRESSION" if delta < -0.01 else "NEUTRAL")
        print(f"  v10+{name} ({len(v10_members)} members): "
              f"Sharpe={v10_sharpe:.2f}  CAGR={v10_cagr:.1%}  MaxDD={v10_maxdd:.1%}  "
              f"dSharpe={delta:+.3f}  [{flag}]")

        if delta > best_delta:
            best_delta = delta
            best_variant = name

    print()
    if best_variant and best_delta > 0.005:
        print(f"  WINNER: {best_variant} (dSharpe={best_delta:+.3f})")
        print(f"  RECOMMENDATION: Add {best_variant} to production ensemble v10")
    else:
        print(f"  NO WINNER: Best delta={best_delta:+.3f}. DL strategies don't improve ensemble.")
        print("  RECOMMENDATION: Keep v9 ensemble. Tune DL hyperparameters in next sprint.")

    print()
    print("[5/5] Done.")


if __name__ == "__main__":
    main()
