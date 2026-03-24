"""
Sprint 12: Walk-forward optimization benchmark.

Compares original walk_forward_ensemble vs walk_forward_ensemble_fast with caching.
Shows timing improvement for 5-fold ensemble validation.
"""

import sys
import time
import pandas as pd

sys.path.insert(0, "src")

from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.walk_forward import WalkForwardConfig, walk_forward_ensemble_fast
from financial_algo.strategies.crash_hedge import VolCarry, CrashHedgeQQQ
from financial_algo.strategies.macro import DollarCarry
from financial_algo.strategies.signal_combo import FeatureComboSignal, Alpha158Ranker
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig
from financial_algo.strategies.factor import FormulaicAlphaMomentum, MultiFactorComposite, LowVolFactor, ValueFactor
from financial_algo.strategies.volatility_strats import VolRiskPremium, VolSpreadHarvest, VolTermStructure, VolOfVolRegime, VolSpikeRecovery
from financial_algo.strategies.tail_risk import TailRiskParity, TailHedgeOverlay
from financial_algo.strategies.quality_trend import MultiAssetTrend, MomentumCrashFilter, QualityTrend
from financial_algo.strategies.ml_strategies import AdaptiveThreshold
from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.multi_freq import WeeklyMomentumRotation, MonthlyMacroRegime
from financial_algo.strategies.crypto_crisis import CryptoRecoverySurge, CryptoGoldDivergence
from financial_algo.backtest import BacktestConfig


# ---- Build v10 ensemble (sample members) ----
def build_v10_members():
    """Sample 10-member ensemble for benchmarking walk-forward optimization."""
    return [
        FeatureComboSignal(),
        CrashHedgeQQQ(),
        CryptoRecoverySurge(),
        MonthlyMacroRegime(),
        VolRiskPremium(),
        VolOfVolRegime(),
        TailRiskParity(),
        VolCarry(),
        CryptoGoldDivergence(),
        MultiAssetTrend(),
    ]


def main():
    print("=" * 90)
    print("  SPRINT 12: Walk-Forward Optimization Benchmark")
    print("=" * 90)
    print()
    
    # ---- Load data ----
    print("[1/4] Loading price data...")
    DATA_START = "2009-01-01"
    DATA_END = "2025-12-31"
    VIX_TICKER = "^VIX"
    
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
        "BTC-USD",
    ]))
    
    prices_full = load_prices(TICKERS + [VIX_TICKER], start=DATA_START, end=DATA_END)
    vix = prices_full.get(VIX_TICKER)
    prices = prices_full.drop(columns=[VIX_TICKER], errors="ignore")
    print(f"       {prices.shape[0]} days x {prices.shape[1]} tickers")
    
    print()
    
    # ---- Regime ----
    print("[2/4] Detecting regimes...")
    regime = detect_regime(prices, vix=vix)
    print()
    
    # ---- Slice to 2010-2025 ----
    fp_mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
    fp_prices = prices.loc[fp_mask].copy()
    fp_regime = regime.loc[fp_mask].copy()
    print(f"[3/4] Sliced to 2010-2025: {fp_prices.shape[0]} days x {fp_prices.shape[1]} tickers")
    print()
    
    # ---- Build strategies ----
    strategies = build_v10_members()
    print(f"[4/4] Loaded {len(strategies)} v10 strategies; running WF tests...\n")
    
    bt_cfg = BacktestConfig(tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
                            initial_capital=1_000_000.0, vol_target=0.20, max_drawdown_trigger=None,
                            strategy_dd_scale_start=-0.15, strategy_dd_scale_end=-0.25)
    
    # ---- BENCHMARK 1: Original walk_forward_ensemble (14-fold) ----
    print("=" * 90)
    print("  TEST 1: walk_forward_ensemble (original, 14-fold, NO cache)")
    print("=" * 90)
    wf_cfg_orig = WalkForwardConfig(
        min_train_years=3, test_years=1, step_years=1, warmup_days=252,
        bt_config=bt_cfg,
        use_cache=False,
        max_folds=None,
    )
    
    t1_start = time.time()
    try:
        result_orig = walk_forward_ensemble(strategies, fp_prices, fp_regime, wf_cfg_orig)
        t1 = time.time() - t1_start
        metric_sharpe_orig = result_orig["oos_metrics"].get("sharpe", 0)
        print(f"\n  OOS Sharpe: {metric_sharpe_orig:.4f}")
        print(f"  Time: {t1:.1f}s ({t1/60:.1f}min)")
    except Exception as e:
        t1 = time.time() - t1_start
        print(f"\n  ERROR: {e}")
        print(f"  Time before error: {t1:.1f}s")
        metric_sharpe_orig = None
    
    print()
    
    # ---- BENCHMARK 2: Optimized walk_forward_ensemble_fast (14-fold WITH cache) ----
    print("=" * 90)
    print("  TEST 2: walk_forward_ensemble_fast (optimized, 14-fold, WITH cache)")
    print("=" * 90)
    wf_cfg_fast = WalkForwardConfig(
        min_train_years=3, test_years=1, step_years=1, warmup_days=252,
        bt_config=bt_cfg,
        use_cache=True,
        max_folds=None,
    )
    
    t2_start = time.time()
    try:
        result_fast = walk_forward_ensemble_fast(strategies, fp_prices, fp_regime, wf_cfg_fast)
        t2 = result_fast.get("timing_sec", time.time() - t2_start)
        metric_sharpe_fast = result_fast["oos_metrics"].get("sharpe", 0)
        print(f"\n  OOS Sharpe: {metric_sharpe_fast:.4f}")
        print(f"  Time: {t2:.1f}s ({t2/60:.1f}min)")
    except Exception as e:
        t2 = time.time() - t2_start
        print(f"\n  ERROR: {e}")
        print(f"  Time before error: {t2:.1f}s")
        metric_sharpe_fast = None
    
    print()
    
    # ---- BENCHMARK 3: Optimized walk_forward_ensemble_fast (5-fold WITH cache) ----
    print("=" * 90)
    print("  TEST 3: walk_forward_ensemble_fast (optimized, 5-fold, WITH cache)")
    print("=" * 90)
    wf_cfg_fast_5fold = WalkForwardConfig(
        min_train_years=3, test_years=1, step_years=1, warmup_days=252,
        bt_config=bt_cfg,
        use_cache=True,
        max_folds=5,
    )
    
    t3_start = time.time()
    try:
        result_fast_5 = walk_forward_ensemble_fast(strategies, fp_prices, fp_regime, wf_cfg_fast_5fold)
        t3 = result_fast_5.get("timing_sec", time.time() - t3_start)
        metric_sharpe_fast_5 = result_fast_5["oos_metrics"].get("sharpe", 0)
        print(f"\n  OOS Sharpe: {metric_sharpe_fast_5:.4f}")
        print(f"  Time: {t3:.1f}s ({t3/60:.1f}min)")
    except Exception as e:
        t3 = time.time() - t3_start
        print(f"\n  ERROR: {e}")
        print(f"  Time before error: {t3:.1f}s")
        metric_sharpe_fast_5 = None
    
    print()
    
    # ---- Summary ----
    print("=" * 90)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print()
    print(f"  TEST 1 (original 14-fold, no cache):  {t1:7.1f}s ({t1/60:5.1f}min)")
    if metric_sharpe_orig is not None:
        print(f"    OOS Sharpe: {metric_sharpe_orig:.4f}")
    print()
    print(f"  TEST 2 (fast 14-fold, cached):        {t2:7.1f}s ({t2/60:5.1f}min)  [Speedup: {t1/t2:.1f}x]")
    if metric_sharpe_fast is not None:
        print(f"    OOS Sharpe: {metric_sharpe_fast:.4f}")
    print()
    print(f"  TEST 3 (fast 5-fold, cached):         {t3:7.1f}s ({t3/60:5.1f}min)  [Speedup: {t1/t3:.1f}x]")
    if metric_sharpe_fast_5 is not None:
        print(f"    OOS Sharpe: {metric_sharpe_fast_5:.4f}")
    print()
    
    # ---- Recommendation ----
    print("=" * 90)
    print("  RECOMMENDATION")
    print("=" * 90)
    print(f"  For Sprint 12 A/B tests: Use walk_forward_ensemble_fast with max_folds=5")
    print(f"  Expected runtime: {t3/60:.1f}min (was {t1/60:.1f}min with original code)")
    print(f"  Decision quality: ~{100*(metric_sharpe_fast_5/metric_sharpe_orig if metric_sharpe_orig else 0):.0f}% of full 14-fold test")
    print()


if __name__ == "__main__":
    main()
