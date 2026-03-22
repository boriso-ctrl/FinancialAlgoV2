"""Quick ensemble parameter optimization - part 2."""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, FourStateTactical, VolCarry
from financial_algo.strategies.volatility_strats import VolRiskPremium, VolOfVolRegime
from financial_algo.strategies.tail_risk import TailRiskParity, BlackSwanInsurance
from financial_algo.fundamental.strategies import FearGreedContrarian
from financial_algo.strategies.factor import LowVolFactor, SizeFactor
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.macro import DollarCarry, GoldDollarInverse

warnings.filterwarnings("ignore")

print("Loading data...")
prices = load_prices(
    sorted(set([
        "SPY","QQQ","IWM","EFA","EEM","GLD","TLT","IEF","UUP",
        "XLE","USO","XOP","ITA","LMT","RTX",
        "XLK","XLF","XLI","XLB","XLP","XLU","XLY","XLV",
        "HYG","LQD","BTC-USD",
    ])),
    start="2009-01-01", end="2025-12-31",
)
try:
    vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
    vix = vix_df["^VIX"]
except Exception:
    vix = None

mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
p = prices.loc[mask].copy()
regime = detect_regime(prices, vix=vix).loc[mask]

bt_cfg = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20,
    max_drawdown_trigger=-0.25, drawdown_recovery_rate=0.10,
)

def test_ensemble(name, members, sharpes, dd_start, dd_end):
    priors = [s**2 for s in sharpes]
    cfg = EnsembleConfig(
        max_gross_leverage=5.0, max_single_weight=0.25,
        dd_scale_start=dd_start, dd_scale_end=dd_end,
        prior_weights=priors,
    )
    ens = EnsembleStrategy(members, cfg)
    ens_w = ens.backtest_weights(p, regime)
    result = backtest(p, ens_w, bt_cfg)
    m = result["metrics"]
    print(f"{name:<42} {m['sharpe']:>7.2f} {m['cagr']:>7.2%} {m['max_drawdown']:>7.2%} {m['calmar']:>8.2f} {m['sortino']:>8.2f}")

members_12 = [
    CrashHedgeQQQ(), VolRiskPremium(), TailRiskParity(), FearGreedContrarian(),
    VolCarry(), LowVolFactor(), SeasonalStrategy(), DollarCarry(),
    GoldDollarInverse(), FourStateTactical(), VolOfVolRegime(), SizeFactor(),
]
sharpes_12 = [0.94, 0.92, 0.90, 0.87, 0.81, 0.81, 0.79, 0.70, 0.68, 0.63, 0.63, 0.56]

# Without D3 (VolCarry - worst DD)
members_noD3 = [m for m, s in zip(members_12, sharpes_12) if not isinstance(m, VolCarry)]
sharpes_noD3 = [s for m, s in zip(members_12, sharpes_12) if not isinstance(m, VolCarry)]

# Without D3 and D1 (redundant with D2)
members_noD3D1 = [m for m, s in zip(members_12, sharpes_12) if not isinstance(m, (VolCarry, FourStateTactical))]
sharpes_noD3D1 = [s for m, s in zip(members_12, sharpes_12) if not isinstance(m, (VolCarry, FourStateTactical))]

# With O4 added
members_withO4 = members_12 + [BlackSwanInsurance()]
sharpes_withO4 = sharpes_12 + [0.28]

# Top 8 only (Sharpe > 0.7)
members_top8 = members_12[:8]
sharpes_top8 = sharpes_12[:8]

# Top 6 only (Sharpe > 0.8)
members_top6 = members_12[:6]
sharpes_top6 = sharpes_12[:6]

print(f"\n{'Config':<42} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Sortino':>8}")
print("-" * 85)

test_ensemble("Base 12 (relaxed -15/-35)",    members_12,    sharpes_12,    -0.15, -0.35)
test_ensemble("No D3 (relaxed -15/-35)",       members_noD3,  sharpes_noD3,  -0.15, -0.35)
test_ensemble("No D3/D1 (relaxed -15/-35)",    members_noD3D1,sharpes_noD3D1,-0.15, -0.35)
test_ensemble("+O4 (relaxed -15/-35)",         members_withO4,sharpes_withO4,-0.15, -0.35)
test_ensemble("Top 8 only (relaxed -15/-35)",  members_top8,  sharpes_top8,  -0.15, -0.35)
test_ensemble("Top 6 only (relaxed -15/-35)",  members_top6,  sharpes_top6,  -0.15, -0.35)
test_ensemble("Top 8 (medium -12/-30)",        members_top8,  sharpes_top8,  -0.12, -0.30)
test_ensemble("Top 6 (medium -12/-30)",        members_top6,  sharpes_top6,  -0.12, -0.30)

print("\nBenchmark SPY: Sharpe 0.90, CAGR 16.41%, MaxDD -32.99%, Calmar 0.50")
