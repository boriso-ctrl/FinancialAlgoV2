"""Quick ensemble parameter optimization on Full Period only."""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
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

# Build ensemble members
members = [
    CrashHedgeQQQ(),         # D2 0.94
    VolRiskPremium(),        # L1 0.92
    TailRiskParity(),        # O1 0.90
    FearGreedContrarian(),   # G2 0.87
    VolCarry(),              # D3 0.81
    LowVolFactor(),          # K1 0.81
    SeasonalStrategy(),      # N1 0.79
    DollarCarry(),           # M1 0.70
    GoldDollarInverse(),     # M2 0.68
    FourStateTactical(),     # D1 0.63
    VolOfVolRegime(),        # L4 0.63
    SizeFactor(),            # K3 0.56
]
sharpe_scores = [0.94, 0.92, 0.90, 0.87, 0.81, 0.81, 0.79, 0.70, 0.68, 0.63, 0.63, 0.56]
prior_weights = [s ** 2 for s in sharpe_scores]

# Test configurations
configs = [
    ("Relaxed CB -15/-35", -0.15, -0.35, -0.25),
    ("Medium CB -12/-30",  -0.12, -0.30, -0.25),
    ("Tight CB -10/-25",   -0.10, -0.25, -0.25),
    ("Tight CB -10/-25 + BT-20", -0.10, -0.25, -0.20),
    ("Tight CB -08/-22 + BT-20", -0.08, -0.22, -0.20),
    ("No ens CB + BT-20",  -0.50, -0.90, -0.20),  # effectively disable ensemble CB
]

print(f"\n{'Config':<35} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Sortino':>8}")
print("-" * 80)

for name, dd_start, dd_end, bt_dd in configs:
    cfg = EnsembleConfig(
        max_gross_leverage=5.0,
        max_single_weight=0.25,
        dd_scale_start=dd_start,
        dd_scale_end=dd_end,
        prior_weights=prior_weights,
    )
    bt_cfg = BacktestConfig(
        tx_cost_bps=5.0,
        leverage_cost_annual=0.015,
        short_cost_annual=0.005,
        initial_capital=1_000_000.0,
        vol_target=0.20,
        max_drawdown_trigger=bt_dd,
        drawdown_recovery_rate=0.10,
    )
    try:
        ens = EnsembleStrategy(members, cfg)
        ens_w = ens.backtest_weights(p, regime)
        result = backtest(p, ens_w, bt_cfg)
        m = result["metrics"]
        print(f"{name:<35} {m['sharpe']:>7.2f} {m['cagr']:>7.2%} {m['max_drawdown']:>7.2%} {m['calmar']:>8.2f} {m['sortino']:>8.2f}")
    except Exception as e:
        print(f"{name:<35} ERROR: {e}")

# Also test with BlackSwanInsurance added
print("\n--- With BlackSwanInsurance (O4) added ---")
members_v2 = members + [BlackSwanInsurance()]
sharpe_v2 = sharpe_scores + [0.28]
priors_v2 = [s**2 for s in sharpe_v2]

for name, dd_start, dd_end, bt_dd in [("Medium CB -12/-30", -0.12, -0.30, -0.25), ("Tight CB -10/-25", -0.10, -0.25, -0.25)]:
    cfg = EnsembleConfig(
        max_gross_leverage=5.0,
        max_single_weight=0.25,
        dd_scale_start=dd_start,
        dd_scale_end=dd_end,
        prior_weights=priors_v2,
    )
    bt_cfg = BacktestConfig(
        tx_cost_bps=5.0,
        leverage_cost_annual=0.015,
        short_cost_annual=0.005,
        initial_capital=1_000_000.0,
        vol_target=0.20,
        max_drawdown_trigger=bt_dd,
        drawdown_recovery_rate=0.10,
    )
    try:
        ens = EnsembleStrategy(members_v2, cfg)
        ens_w = ens.backtest_weights(p, regime)
        result = backtest(p, ens_w, bt_cfg)
        m = result["metrics"]
        print(f"+O4 {name:<30} {m['sharpe']:>7.2f} {m['cagr']:>7.2%} {m['max_drawdown']:>7.2%} {m['calmar']:>8.2f} {m['sortino']:>8.2f}")
    except Exception as e:
        print(f"+O4 {name:<30} ERROR: {e}")

# Test without D3-VolCarry (worst DD member)
print("\n--- Without D3-VolCarry (worst DD) ---")
members_v3 = [m for m in members if not isinstance(m, VolCarry)]
sharpe_v3 = [s for s, m in zip(sharpe_scores, members) if not isinstance(m, VolCarry)]
priors_v3 = [s**2 for s in sharpe_v3]

for name, dd_start, dd_end, bt_dd in [("Medium CB -12/-30", -0.12, -0.30, -0.25), ("Tight CB -10/-25", -0.10, -0.25, -0.25)]:
    cfg = EnsembleConfig(
        max_gross_leverage=5.0,
        max_single_weight=0.25,
        dd_scale_start=dd_start,
        dd_scale_end=dd_end,
        prior_weights=priors_v3,
    )
    bt_cfg = BacktestConfig(
        tx_cost_bps=5.0,
        leverage_cost_annual=0.015,
        short_cost_annual=0.005,
        initial_capital=1_000_000.0,
        vol_target=0.20,
        max_drawdown_trigger=bt_dd,
        drawdown_recovery_rate=0.10,
    )
    try:
        ens = EnsembleStrategy(members_v3, cfg)
        ens_w = ens.backtest_weights(p, regime)
        result = backtest(p, ens_w, bt_cfg)
        m = result["metrics"]
        print(f"-D3 {name:<30} {m['sharpe']:>7.2f} {m['cagr']:>7.2%} {m['max_drawdown']:>7.2%} {m['calmar']:>8.2f} {m['sortino']:>8.2f}")
    except Exception as e:
        print(f"-D3 {name:<30} ERROR: {e}")

print("\nBenchmark SPY: Sharpe 0.90, CAGR 16.41%, MaxDD -32.99%, Calmar 0.50")
