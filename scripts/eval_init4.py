"""Quick evaluation of new Init 4 strategies against the full period."""
import sys, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.backtest import backtest, BacktestConfig

# New Init 4 strategies
from financial_algo.strategies.momentum import GlobalMomentumRotation
from financial_algo.strategies.mean_reversion import GlobalMeanReversion
from financial_algo.strategies.factor import RealAssetsFactor
from financial_algo.strategies.macro import InflationBreakevenTrade, GlobalRotation, CommodityMacroSignal
from financial_algo.strategies.tail_risk import PreciousMetalsCrisisHedge
from financial_algo.strategies.crypto_crisis import CryptoContagionHedge
from financial_algo.strategies.volatility_strats import CrossAssetVolSignal
from financial_algo.fundamental import build_synthetic_sentiment
from financial_algo.fundamental.strategies import CryptoSentimentDivergence

# Also eval existing strategies for comparison
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ

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
    "FXI","VGK","EWJ","INDA","VNQ",
    "BTC-USD","ETH-USD",
]))

print(f"Loading {len(TICKERS)} tickers...")
prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
print(f"Loaded: {prices.shape}")

vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
vix = vix_df["^VIX"]
regime = detect_regime(prices, vix=vix)

# Slice to full period
mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
p = prices.loc[mask].copy()
r = regime.loc[mask].copy()
vix_win = vix.reindex(p.index).ffill()
sent_df = build_synthetic_sentiment(p, vix=vix_win)

# All strategies to evaluate
strats = [
    ("I5-GlobalMomRotation", GlobalMomentumRotation(), False, False),
    ("J5-GlobalMeanReversion", GlobalMeanReversion(), False, False),
    ("K5-RealAssetsFactor", RealAssetsFactor(), False, False),
    ("M6-InflationBreakeven", InflationBreakevenTrade(), False, False),
    ("M7-GlobalRotation", GlobalRotation(), False, False),
    ("M8-CommodityMacro", CommodityMacroSignal(), False, False),
    ("O7-PreciousMetalsCH", PreciousMetalsCrisisHedge(), False, False),
    ("F4-CryptoContagion", CryptoContagionHedge(), False, False),
    ("L6-CrossAssetVol", CrossAssetVolSignal(), False, False),
    ("G5-CryptoSentDiv", CryptoSentimentDivergence(), True, True),
    # Reference
    ("P1-FeatureCombo(ref)", FeatureComboSignal(), False, False),
    ("D2-CrashHedgeQQQ(ref)", CrashHedgeQQQ(), False, False),
]

# Benchmark
w_spy = pd.DataFrame(0.0, index=p.index, columns=p.columns)
w_spy["SPY"] = 1.0
spy_res = backtest(p, w_spy.shift(1).fillna(0), BT)
spy_m = spy_res["metrics"]

print(f"\n{'Strategy':<28} {'CAGR':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'AnnVol':>8} {'WinRate':>8}")
print("-" * 102)
print(f"{'SPY-BuyHold':<28} {spy_m['cagr']:>8.2%} {spy_m['sharpe']:>7.2f} {spy_m['sortino']:>8.2f} {spy_m['max_drawdown']:>8.2%} {spy_m['calmar']:>7.2f} {spy_m['annual_vol']:>8.2%} {spy_m['win_rate']:>8.2%}")
print("-" * 102)

results = []
for name, strat, needs_regime, needs_sent in strats:
    try:
        if needs_sent:
            w = strat.generate_weights(p, r, sent_df)
            w = w.shift(1).fillna(0.0)
        elif needs_regime:
            w = strat.backtest_weights(p, r)
        else:
            w = strat.backtest_weights(p)
        res = backtest(p, w, BT)
        m = res["metrics"]
        print(f"{name:<28} {m['cagr']:>8.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['max_drawdown']:>8.2%} {m['calmar']:>7.2f} {m['annual_vol']:>8.2%} {m['win_rate']:>8.2%}")
        results.append((name, m))
    except Exception as e:
        print(f"{name:<28} ERROR: {e}")
        results.append((name, None))

print("\n--- New strategies sorted by Sharpe (candidates for ensemble) ---")
new_only = [(n, m) for n, m in results if "(ref)" not in n and m is not None]
new_only.sort(key=lambda x: x[1]["sharpe"], reverse=True)
for name, m in new_only:
    status = "ENSEMBLE-READY" if m["sharpe"] > 0.5 else ("MARGINAL" if m["sharpe"] > 0.3 else "KILL")
    print(f"  {name:<28} Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_drawdown']:.2%}  -> {status}")
