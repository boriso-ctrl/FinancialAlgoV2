"""Quick ensemble DD test — Full Period only."""
from __future__ import annotations
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig

from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium, VolOfVolRegime, VolTermStructure, VolSpreadHarvest, VolSpikeRecovery,
)
from financial_algo.strategies.tail_risk import TailRiskParity, TailHedgeOverlay
from financial_algo.strategies.crypto_crisis import CryptoRecoverySurge, CryptoGoldDivergence
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.quality_trend import QualityTrend, MultiAssetTrend, MomentumCrashFilter
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite, ValueFactor
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.fundamental.strategies import (
    FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence,
)

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP", "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "HYG", "LQD", "BTC-USD",
]))

print("Loading data...")
prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
vix = vix_df["^VIX"]
regime = detect_regime(prices, vix=vix)

mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
p = prices.loc[mask]
r = regime.loc[mask]

ensemble_members = [
    FeatureComboSignal(), CrashHedgeQQQ(), CryptoRecoverySurge(),
    VolRiskPremium(), VolOfVolRegime(), TailRiskParity(), VolCarry(),
    CryptoGoldDivergence(), MultiAssetTrend(), FearGreedContrarian(),
    VolTermStructure(), VolSpreadHarvest(), LowVolFactor(), MultiFactorComposite(),
    VolSpikeRecovery(), QualityTrend(), MomentumCrashFilter(), ValueFactor(),
    SentimentCrisisAlpha(), SeasonalStrategy(), SentimentDivergence(),
    TailHedgeOverlay(),
]
sharpe_scores = [
    0.97, 0.94, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89, 0.88, 0.87,
    0.86, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.81, 0.80, 0.79, 0.78,
    0.50,
]
prior_weights = [s ** 2 for s in sharpe_scores]

bt_cfg = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20,
    max_drawdown_trigger=None,
    strategy_dd_scale_start=-0.15,
    strategy_dd_scale_end=-0.25,
)

configs = [
    ("v4-baseline (no DD controls)", EnsembleConfig(
        use_inverse_vol=False, max_gross_leverage=3.0, max_single_weight=0.20,
        dd_scale_start=-0.50, dd_scale_end=-0.80, prior_weights=prior_weights,
    ), BacktestConfig(
        tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
        initial_capital=1_000_000.0, vol_target=0.20, max_drawdown_trigger=None,
    )),
    ("Init1: all DD controls", EnsembleConfig(
        use_inverse_vol=False, max_gross_leverage=2.5, max_single_weight=0.20,
        dd_scale_start=-0.12, dd_scale_end=-0.22, prior_weights=prior_weights,
        correlation_hedge_enabled=True, correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True, vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30, leverage_elevated=1.8, leverage_crisis=1.2,
    ), bt_cfg),
    ("Init1-tight: dd -10/-18", EnsembleConfig(
        use_inverse_vol=False, max_gross_leverage=2.5, max_single_weight=0.20,
        dd_scale_start=-0.10, dd_scale_end=-0.18, prior_weights=prior_weights,
        correlation_hedge_enabled=True, correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True, vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30, leverage_elevated=1.8, leverage_crisis=1.2,
    ), bt_cfg),
    ("Init1-2x: max lev 2.0", EnsembleConfig(
        use_inverse_vol=False, max_gross_leverage=2.0, max_single_weight=0.20,
        dd_scale_start=-0.12, dd_scale_end=-0.22, prior_weights=prior_weights,
        correlation_hedge_enabled=True, correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True, vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30, leverage_elevated=1.5, leverage_crisis=1.0,
    ), bt_cfg),
]

print(f"{'Config':<35s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>8s} {'Calmar':>7s} {'TotRet':>10s}")
print("-" * 80)

for name, ens_cfg, bt_config in configs:
    ens = EnsembleStrategy(ensemble_members, ens_cfg)
    w = ens.backtest_weights(p, r)
    result = backtest(p, w, bt_config)
    m = result["metrics"]
    print(f"{name:<35s} {m['cagr']:>7.2%} {m['sharpe']:>7.2f} {m['max_drawdown']:>8.2%} "
          f"{m['calmar']:>7.2f} {m['total_return']:>9.2%}")

print("\nDone.")
