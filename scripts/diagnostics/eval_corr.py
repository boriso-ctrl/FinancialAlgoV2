"""Check correlations between new Init 4 strategy returns and existing ensemble members."""
import warnings, sys
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.backtest import backtest, BacktestConfig
from financial_algo.fundamental import build_synthetic_sentiment

# New candidates
from financial_algo.strategies.volatility_strats import CrossAssetVolSignal
from financial_algo.strategies.tail_risk import PreciousMetalsCrisisHedge
from financial_algo.strategies.crypto_crisis import CryptoContagionHedge
from financial_algo.fundamental.strategies import CryptoSentimentDivergence

# Existing top ensemble members
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.multi_freq import MonthlyMacroRegime
from financial_algo.strategies.volatility_strats import VolRiskPremium, VolOfVolRegime, VolTermStructure
from financial_algo.strategies.tail_risk import TailRiskParity
from financial_algo.strategies.quality_trend import MultiAssetTrend, QualityTrend
from financial_algo.strategies.factor import LowVolFactor

BT = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20,
    max_drawdown_trigger=-0.25, drawdown_recovery_rate=0.10,
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
vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
vix = vix_df["^VIX"]
regime = detect_regime(prices, vix=vix)

mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
p = prices.loc[mask].copy()
r = regime.loc[mask].copy()
vix_win = vix.reindex(p.index).ffill()
sent_df = build_synthetic_sentiment(p, vix=vix_win)

def get_returns(strat, needs_regime=False, needs_sent=False):
    if needs_sent:
        w = strat.generate_weights(p, r, sent_df)
        w = w.shift(1).fillna(0.0)
    elif needs_regime:
        w = strat.backtest_weights(p, r)
    else:
        w = strat.backtest_weights(p)
    res = backtest(p, w, BT)
    return res["equity"].pct_change().fillna(0)

# Build return series for all
all_strats = {
    # New candidates
    "L6-CrossAssetVol": (CrossAssetVolSignal(), False, False),
    "O7-PreciousMetals": (PreciousMetalsCrisisHedge(), False, False),
    "G5-CryptoSentDiv": (CryptoSentimentDivergence(), True, True),
    "F4-CryptoContagion": (CryptoContagionHedge(), False, False),
    # Existing top ensemble members
    "P1-FeatureCombo": (FeatureComboSignal(), False, False),
    "D2-CrashHedgeQQQ": (CrashHedgeQQQ(), False, False),
    "MF2-MonthlyMacro": (MonthlyMacroRegime(), False, False),
    "L1-VolRiskPrem": (VolRiskPremium(), False, False),
    "L4-VolOfVol": (VolOfVolRegime(), False, False),
    "O1-TailRiskParity": (TailRiskParity(), False, False),
    "D3-VolCarry": (VolCarry(), False, False),
    "Q2-MultiAssetTrend": (MultiAssetTrend(), False, False),
    "L3-VolTermStruct": (VolTermStructure(), False, False),
    "K1-LowVolFactor": (LowVolFactor(), False, False),
    "Q1-QualityTrend": (QualityTrend(), False, False),
}

returns_df = pd.DataFrame()
for name, (strat, nr, ns) in all_strats.items():
    try:
        returns_df[name] = get_returns(strat, nr, ns)
        print(f"  Computed: {name}")
    except Exception as e:
        print(f"  ERROR computing {name}: {e}")

# Correlation matrix
new_names = ["L6-CrossAssetVol", "O7-PreciousMetals", "G5-CryptoSentDiv", "F4-CryptoContagion"]
existing_names = [n for n in returns_df.columns if n not in new_names]

print("\n=== Correlation of NEW strategies with EXISTING ensemble members ===")
corr = returns_df.corr()
for new in new_names:
    if new not in corr.columns:
        continue
    print(f"\n{new}:")
    for ex in existing_names:
        if ex not in corr.columns:
            continue
        c = corr.loc[new, ex]
        flag = " ***HIGH***" if abs(c) > 0.5 else ""
        print(f"  vs {ex:<22} corr = {c:+.3f}{flag}")
    avg = corr.loc[new, existing_names].abs().mean()
    print(f"  Average |corr| = {avg:.3f}")

print("\n=== Correlations between NEW strategies ===")
for i, n1 in enumerate(new_names):
    for n2 in new_names[i+1:]:
        if n1 in corr.columns and n2 in corr.columns:
            print(f"  {n1} vs {n2}: {corr.loc[n1, n2]:+.3f}")
