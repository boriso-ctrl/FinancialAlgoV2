"""Quick A/B test: ensemble v7a (L6 only) vs v7b (L6+F4) vs v6 (baseline)."""
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.backtest import backtest, BacktestConfig
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig

# Ensemble members
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.crypto_crisis import CryptoRecoverySurge, CryptoGoldDivergence, CryptoContagionHedge
from financial_algo.strategies.multi_freq import MonthlyMacroRegime, WeeklyMomentumRotation
from financial_algo.strategies.volatility_strats import VolRiskPremium, VolOfVolRegime, VolTermStructure, VolSpreadHarvest, VolSpikeRecovery, CrossAssetVolSignal
from financial_algo.strategies.tail_risk import TailRiskParity, TailHedgeOverlay
from financial_algo.strategies.quality_trend import QualityTrend, MultiAssetTrend, MomentumCrashFilter
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite, ValueFactor
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.macro import DollarCarry
from financial_algo.fundamental.strategies import FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence

BT_ENS = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20,
    max_drawdown_trigger=None,
    strategy_dd_scale_start=-0.15, strategy_dd_scale_end=-0.25,
)

TICKERS = sorted(set([
    "SPY","QQQ","IWM","EFA","EEM","GLD","SLV","TLT","IEF","SHY","UUP",
    "XLE","USO","XOP","ITA","LMT","RTX",
    "XLK","XLF","XLI","XLB","XLP","XLU","XLY","XLV","XBI","XLC","XLRE",
    "DBC","DBA","HYG","LQD","TIP","AGG","EMB",
    "FXI","VGK","EWJ","INDA","VNQ","BTC-USD","ETH-USD",
]))

print("Loading data...")
prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
vix = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")["^VIX"]
regime = detect_regime(prices, vix=vix)

mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
p = prices.loc[mask].copy()
r = regime.loc[mask].copy()

# v6 baseline members (24 members)
base_members = [
    FeatureComboSignal(),    # P1
    CrashHedgeQQQ(),         # D2
    CryptoRecoverySurge(),   # F2
    MonthlyMacroRegime(),    # MF2
    VolRiskPremium(),        # L1
    VolOfVolRegime(),        # L4
    TailRiskParity(),        # O1
    VolCarry(),              # D3
    CryptoGoldDivergence(),  # F3
    MultiAssetTrend(),       # Q2
    FearGreedContrarian(),   # G2
    VolTermStructure(),      # L3
    VolSpreadHarvest(),      # L2
    LowVolFactor(),          # K1
    MultiFactorComposite(),  # K2
    VolSpikeRecovery(),      # L5
    QualityTrend(),          # Q1
    MomentumCrashFilter(),   # Q3
    ValueFactor(),           # K4
    SentimentCrisisAlpha(),  # G1
    SeasonalStrategy(),      # N1
    WeeklyMomentumRotation(), # MF1
    SentimentDivergence(),   # G3
    TailHedgeOverlay(),      # O6
]
base_scores = [0.97, 0.94, 0.93, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89, 0.88,
               0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.81, 0.80,
               0.79, 0.78, 0.78, 0.50]

def make_cfg(scores):
    return EnsembleConfig(
        use_inverse_vol=False, max_gross_leverage=2.5, max_single_weight=0.20,
        dd_scale_start=-0.12, dd_scale_end=-0.22,
        prior_weights=[s ** 2 for s in scores],
        correlation_hedge_enabled=True, correlation_hedge_threshold=0.65, correlation_hedge_max=0.25,
        vol_regime_scaling=True, vol_elevated_threshold=0.20, vol_crisis_threshold=0.30,
        leverage_elevated=1.8, leverage_crisis=1.2,
    )

def run_ens(label, members, scores):
    cfg = make_cfg(scores)
    ens = EnsembleStrategy(members, cfg)
    w = ens.backtest_weights(p, r)
    res = backtest(p, w, BT_ENS)
    m = res["metrics"]
    print(f"  {label:<25} Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_drawdown']:.2%}  Calmar={m['calmar']:.2f}  Sortino={m['sortino']:.2f}")
    return m

print("\nEnsemble Variants:")
# v6: baseline
v6 = run_ens("v6-Baseline(24)", base_members, base_scores)

# v7a: +L6 only (25 members)
v7a_members = [CrossAssetVolSignal()] + base_members
v7a_scores = [1.01] + base_scores
v7a = run_ens("v7a-+L6(25)", v7a_members, v7a_scores)

# v7b: +L6 +F4 (26 members)
v7b_members = [CrossAssetVolSignal()] + base_members[:24] + [CryptoContagionHedge()] + [base_members[-1]]
v7b_scores = [1.01] + base_scores[:24] + [0.60] + [base_scores[-1]]
# Fix: just build it cleanly
v7b_members = [CrossAssetVolSignal()] + base_members[:-1] + [CryptoContagionHedge(), TailHedgeOverlay()]
v7b_scores = [1.01] + base_scores[:-1] + [0.60, 0.50]
v7b = run_ens("v7b-+L6+F4(26)", v7b_members, v7b_scores)

print("\n--- Delta vs v6 ---")
for label, m in [("v7a-+L6", v7a), ("v7b-+L6+F4", v7b)]:
    print(f"  {label}: dSharpe={m['sharpe']-v6['sharpe']:+.3f}  dCAGR={m['cagr']-v6['cagr']:+.2%}  dMaxDD={m['max_drawdown']-v6['max_drawdown']:+.2%}")
