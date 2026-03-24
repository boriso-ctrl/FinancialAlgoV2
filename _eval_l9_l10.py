"""Quick eval of L9, L10, L1b strategies."""
from __future__ import annotations
import sys
sys.path.insert(0, "src")

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.volatility_strats import (
    VIXAdaptiveCarry,
    DynamicVolRegimeSwitch,
    VolRiskPremiumAdaptive,
    VolRiskPremium,
)
import pandas as pd

TICKERS = [
    "SPY", "QQQ", "TLT", "GLD", "SLV", "EFA", "EEM", "IWM",
    "XLE", "XLF", "HYG", "LQD", "SHY", "IEF", "DBC", "UUP", "^VIX",
]

prices = load_prices(TICKERS, start="2010-01-01")
print(f"Loaded {len(prices)} days, {len(prices.columns)} tickers")
regime = detect_regime(prices)
cfg = BacktestConfig(tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005)

strats = [
    ("L9-VIXAdaptiveCarry", VIXAdaptiveCarry()),
    ("L10-DynVolRegimeSwitch", DynamicVolRegimeSwitch()),
    ("L1b-VolRiskPremAdaptive", VolRiskPremiumAdaptive()),
    ("L1-VolRiskPremium(orig)", VolRiskPremium()),
]

ret_dict = {}
print(f"\n{'Strategy':<30} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'WinRate':>8} {'AnnVol':>8}")
print("-" * 100)
for name, s in strats:
    w = s.backtest_weights(prices, regime)
    res = backtest(prices, w, cfg)
    m = res["metrics"]
    ret_dict[name] = res["returns"]
    print(f"{name:<30} {m['cagr']:>8.4f} {m['sharpe']:>8.4f} {m['sortino']:>8.4f} {m['max_drawdown']:>8.4f} {m['calmar']:>8.4f} {m['win_rate']:>8.4f} {m['annual_vol']:>8.4f}")

print("\nReturn Correlations:")
corr = pd.DataFrame(ret_dict).corr()
for col in corr.columns:
    vals = [f"{corr.loc[col, c]:.3f}" for c in corr.columns]
    print(f"  {col:<30} {' '.join(vals)}")
