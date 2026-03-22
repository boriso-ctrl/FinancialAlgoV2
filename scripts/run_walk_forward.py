"""Walk-Forward Validation — Initiative 2.

Runs walk-forward analysis on all 22 ensemble members plus the
full ensemble.  Compares in-sample vs out-of-sample metrics to
detect overfitting.

Usage:
    .venv\\Scripts\\python.exe scripts/run_walk_forward.py
"""

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
from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy
from financial_algo.walk_forward import (
    WalkForwardConfig,
    generate_folds,
    walk_forward_strategy,
    walk_forward_ensemble,
)

# --- Strategy imports (same as run_crisis_backtest.py) ---
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
    FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
# Configuration
# =========================================================================

DATA_START = "2009-01-01"
DATA_END = "2025-12-31"

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP", "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "HYG", "LQD", "BTC-USD",
]))

VIX_TICKER = "^VIX"

# Individual strategy backtest config (with DD controls)
BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=-0.25,
    drawdown_recovery_rate=0.10,
)

# Ensemble backtest config
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

# Walk-forward: 3-year minimum training, 1-year test, 1-year step
WF_CONFIG = WalkForwardConfig(
    min_train_years=3,
    test_years=1,
    step_years=1,
    warmup_days=252,
    bt_config=BT_CONFIG,
)

WF_CONFIG_ENSEMBLE = WalkForwardConfig(
    min_train_years=3,
    test_years=1,
    step_years=1,
    warmup_days=252,
    bt_config=BT_CONFIG_ENSEMBLE,
)


def build_ensemble_members():
    """Return (members_list, names_list, in_sample_sharpes)."""
    members = [
        ("P1-FeatureComboSignal",    FeatureComboSignal(),    False),
        ("D2-CrashHedgeQQQ",        CrashHedgeQQQ(),         False),
        ("F2-CryptoRecoverySurge",   CryptoRecoverySurge(),   True),
        ("L1-VolRiskPremium",        VolRiskPremium(),        False),
        ("L4-VolOfVolRegime",        VolOfVolRegime(),        False),
        ("O1-TailRiskParity",        TailRiskParity(),        False),
        ("D3-VolCarry",              VolCarry(),              False),
        ("F3-CryptoGoldDivergence",  CryptoGoldDivergence(),  True),
        ("Q2-MultiAssetTrend",       MultiAssetTrend(),       False),
        ("G2-FearGreedContrarian",   FearGreedContrarian(),   True),
        ("L3-VolTermStructure",      VolTermStructure(),      False),
        ("L2-VolSpreadHarvest",      VolSpreadHarvest(),      False),
        ("K1-LowVolFactor",          LowVolFactor(),          False),
        ("K2-MultiFactorComposite",  MultiFactorComposite(),  False),
        ("L5-VolSpikeRecovery",      VolSpikeRecovery(),      False),
        ("Q1-QualityTrend",          QualityTrend(),          False),
        ("Q3-MomentumCrashFilter",   MomentumCrashFilter(),   False),
        ("K4-ValueFactor",           ValueFactor(),           False),
        ("G1-SentimentCrisisAlpha",  SentimentCrisisAlpha(),  True),
        ("N1-SeasonalStrategy",      SeasonalStrategy(),      False),
        ("G3-SentimentDivergence",   SentimentDivergence(),   True),
        ("O6-TailHedgeOverlay",      TailHedgeOverlay(),      True),
        ("P4-AdaptiveThreshold",     AdaptiveThreshold(),     False),
        ("MF1-WeeklyMomRotation",    WeeklyMomentumRotation(), False),
        ("MF2-MonthlyMacroRegime",   MonthlyMacroRegime(),    False),
        ("R9-MultiAssetCTATrend",    MultiAssetCTATrend(),    False),
    ]
    return members


def main() -> None:
    print("=" * 80)
    print("WALK-FORWARD VALIDATION — INITIATIVE 2")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/4] Loading price data ...")
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    print(f"       {prices.shape[0]} days x {prices.shape[1]} tickers")

    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER]
    except Exception:
        vix = None

    regime = detect_regime(prices, vix=vix)
    print()

    # ------------------------------------------------------------------
    # 2. Show fold structure
    # ------------------------------------------------------------------
    folds = generate_folds(prices, WF_CONFIG)
    print(f"[2/4] Walk-forward: {len(folds)} folds "
          f"({WF_CONFIG.min_train_years}yr min train, "
          f"{WF_CONFIG.test_years}yr test, "
          f"{WF_CONFIG.step_years}yr step)")
    for f in folds:
        print(f"       Fold {f['fold']}: "
              f"Train {f['train_start'].date()} -> {f['train_end'].date()} | "
              f"Test {f['test_start'].date()} -> {f['test_end'].date()}")
    print()

    # ------------------------------------------------------------------
    # 3. Walk-forward each ensemble member
    # ------------------------------------------------------------------
    members = build_ensemble_members()
    print(f"[3/4] Running walk-forward for {len(members)} strategies ...")
    print()

    # Also need full-period in-sample metrics for comparison
    full_mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
    p_full = prices.loc[full_mask]
    r_full = regime.loc[full_mask]

    results_rows = []

    for name, strat, needs_regime in members:
        print(f"  Running {name} ...", end=" ", flush=True)

        # In-sample full-period backtest
        try:
            if needs_regime:
                w_is = strat.backtest_weights(p_full, r_full)
            else:
                w_is = strat.backtest_weights(p_full)
            is_result = backtest(p_full, w_is, BT_CONFIG)
            is_m = is_result["metrics"]
        except Exception as e:
            print(f"IS ERR: {e}")
            is_m = {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

        # Walk-forward OOS
        try:
            wf = walk_forward_strategy(
                strat, prices, regime, WF_CONFIG,
                needs_regime=needs_regime, vix=vix,
            )
            oos_m = wf["oos_metrics"]
            per_fold = wf["per_fold"]
        except Exception as e:
            print(f"WF ERR: {e}")
            oos_m = {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
            per_fold = []

        is_sharpe = is_m.get("sharpe", 0.0)
        oos_sharpe = oos_m.get("sharpe", 0.0)
        degradation = is_sharpe - oos_sharpe if is_sharpe else 0.0

        # Per-fold test Sharpes for stability
        fold_sharpes = [f["test_sharpe"] for f in per_fold]
        sharpe_std = np.std(fold_sharpes) if fold_sharpes else 0.0
        pct_positive = (
            sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes) * 100
            if fold_sharpes else 0
        )

        results_rows.append({
            "Strategy": name,
            "IS_Sharpe": round(is_sharpe, 2),
            "WF_Sharpe": round(oos_sharpe, 2),
            "Degradation": round(degradation, 2),
            "IS_CAGR": f"{is_m.get('cagr', 0):.1%}",
            "WF_CAGR": f"{oos_m.get('cagr', 0):.1%}",
            "WF_MaxDD": f"{oos_m.get('max_drawdown', 0):.1%}",
            "Fold_Std": round(sharpe_std, 2),
            "Pct_Pos": f"{pct_positive:.0f}%",
        })

        flag = ""
        if degradation > 0.5:
            flag = " ** OVERFIT WARNING **"
        elif degradation > 0.3:
            flag = " * WATCH *"
        print(f"IS={is_sharpe:.2f} WF={oos_sharpe:.2f} "
              f"deg={degradation:+.2f}{flag}")

    # ------------------------------------------------------------------
    # 4. Walk-forward the ensemble
    # ------------------------------------------------------------------
    print()
    print("[4/4] Running walk-forward for ENSEMBLE (training-derived weights) ...")
    strats_only = [s for _, s, _ in members]

    ensemble_cfg_template = EnsembleConfig(
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

    try:
        wf_ens = walk_forward_ensemble(
            strats_only, prices, regime, WF_CONFIG_ENSEMBLE,
            ensemble_cfg_template=ensemble_cfg_template, vix=vix,
        )
        ens_oos = wf_ens["oos_metrics"]
        ens_folds = wf_ens["per_fold"]
    except Exception as e:
        print(f"  Ensemble WF ERR: {e}")
        ens_oos = {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
        ens_folds = []

    # In-sample ensemble (full-period with fixed Sharpe weights)
    # Order: P1,D2,F2,L1,L4,O1,D3,F3,Q2,G2,L3,L2,K1,K2,L5,Q1,Q3,K4,G1,N1,G3,O6,P4,MF1,MF2,R9
    sharpe_scores = [0.97, 0.94, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89,
                     0.88, 0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84,
                     0.83, 0.81, 0.80, 0.79, 0.78, 0.50, 0.75, 0.78, 0.93, 1.14]
    is_prior = [s ** 2 for s in sharpe_scores]
    is_ens_cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=is_prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    is_ensemble = EnsembleStrategy(strats_only, is_ens_cfg)
    is_ens_w = is_ensemble.backtest_weights(p_full, r_full)
    is_ens_result = backtest(p_full, is_ens_w, BT_CONFIG_ENSEMBLE)
    is_ens_m = is_ens_result["metrics"]

    is_ens_sharpe = is_ens_m.get("sharpe", 0.0)
    wf_ens_sharpe = ens_oos.get("sharpe", 0.0)
    ens_deg = is_ens_sharpe - wf_ens_sharpe

    results_rows.append({
        "Strategy": "** ENSEMBLE **",
        "IS_Sharpe": round(is_ens_sharpe, 2),
        "WF_Sharpe": round(wf_ens_sharpe, 2),
        "Degradation": round(ens_deg, 2),
        "IS_CAGR": f"{is_ens_m.get('cagr', 0):.1%}",
        "WF_CAGR": f"{ens_oos.get('cagr', 0):.1%}",
        "WF_MaxDD": f"{ens_oos.get('max_drawdown', 0):.1%}",
        "Fold_Std": round(
            np.std([f["test_sharpe"] for f in ens_folds]) if ens_folds else 0, 2
        ),
        "Pct_Pos": f"{sum(1 for f in ens_folds if f['test_sharpe'] > 0) / max(len(ens_folds), 1) * 100:.0f}%",
    })

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("WALK-FORWARD RESULTS: IN-SAMPLE vs OUT-OF-SAMPLE")
    print("=" * 110)

    df = pd.DataFrame(results_rows)
    print(df.to_string(index=False))
    print()

    # Classification
    print("=" * 110)
    print("OVERFIT CLASSIFICATION")
    print("=" * 110)
    for row in results_rows:
        deg = row["Degradation"]
        name = row["Strategy"]
        if deg > 0.5:
            print(f"  [OVERFIT]  {name}: IS={row['IS_Sharpe']:.2f} -> WF={row['WF_Sharpe']:.2f} (deg={deg:+.2f})")
        elif deg > 0.3:
            print(f"  [WATCH]    {name}: IS={row['IS_Sharpe']:.2f} -> WF={row['WF_Sharpe']:.2f} (deg={deg:+.2f})")
        elif deg < -0.1:
            print(f"  [BETTER OOS] {name}: IS={row['IS_Sharpe']:.2f} -> WF={row['WF_Sharpe']:.2f} (deg={deg:+.2f})")
    print()

    # Ensemble fold detail
    print("=" * 110)
    print("ENSEMBLE WALK-FORWARD — PER-FOLD DETAIL")
    print("=" * 110)
    for f in ens_folds:
        flag = " <<< NEGATIVE" if f["test_sharpe"] < 0 else ""
        print(f"  Fold {f['fold']}: {f['test']} | "
              f"Train Sharpe={f['train_sharpe']:.2f} -> "
              f"Test Sharpe={f['test_sharpe']:.2f} | "
              f"Test CAGR={f['test_cagr']:.1%} | "
              f"Test MaxDD={f['test_maxdd']:.1%}{flag}")
    print()

    # Summary
    print("=" * 110)
    print("EXECUTIVE SUMMARY")
    print("=" * 110)
    print(f"  Ensemble In-Sample Sharpe:       {is_ens_sharpe:.2f}")
    print(f"  Ensemble Walk-Forward Sharpe:     {wf_ens_sharpe:.2f}")
    print(f"  Degradation:                      {ens_deg:+.2f}")
    wf_target = 0.8
    if wf_ens_sharpe >= wf_target:
        print(f"  STATUS: PASS (WF Sharpe {wf_ens_sharpe:.2f} >= {wf_target} target)")
    else:
        print(f"  STATUS: BELOW TARGET (WF Sharpe {wf_ens_sharpe:.2f} < {wf_target} target)")
    print()

    overfit_count = sum(1 for r in results_rows if r["Degradation"] > 0.5)
    watch_count = sum(1 for r in results_rows if 0.3 < r["Degradation"] <= 0.5)
    robust_count = sum(1 for r in results_rows if r["Degradation"] <= 0.3)
    print(f"  Strategies Overfit (deg > 0.5):   {overfit_count}")
    print(f"  Strategies Watch (deg 0.3-0.5):   {watch_count}")
    print(f"  Strategies Robust (deg <= 0.3):    {robust_count}")
    print()

    # Save
    out_path = _root / "results" / "walk_forward_results.csv"
    df.to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
