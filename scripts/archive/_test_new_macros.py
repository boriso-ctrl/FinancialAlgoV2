"""Quick test of new M6-M8 strategies and M3 fix."""
from __future__ import annotations
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import pandas as pd

# Build synthetic prices
rng = np.random.RandomState(42)
n = 400
dates = pd.bdate_range("2020-01-01", periods=n)
tickers = [
    "SPY", "QQQ", "IWM", "TLT", "GLD", "IEF", "UUP",
    "XLE", "USO", "XOP", "XLK", "XLF", "XLI", "XLB",
    "XLP", "XLU", "XLY", "XLV", "XLRE", "XLC",
    "ITA", "LMT", "RTX",
    "EFA", "EEM", "HYG", "LQD",
    "BTC-USD", "ETH-USD",
    "SLV", "DBC", "DBA", "SHY", "TIP", "EMB", "AGG",
    "VNQ", "FXI", "VGK", "EWJ", "INDA", "XBI",
]
data = {}
for t in tickers:
    ret = rng.normal(0.0003, 0.015, n)
    data[t] = 100 * np.exp(np.cumsum(ret))
prices = pd.DataFrame(data, index=dates)

print("Testing M3 fix (EMRiskPremium)...")
from financial_algo.strategies.macro import EMRiskPremium
strat = EMRiskPremium()
w = strat.generate_weights(prices)
assert not w.isna().any().any(), "M3 has NaN!"
assert not np.isinf(w.values).any(), "M3 has inf!"
print(f"  PASS: shape={w.shape}, non-zero days={int((w.abs().sum(axis=1) > 0).sum())}")

print("Testing M6 (InflationBreakevenTrade)...")
from financial_algo.strategies.macro import InflationBreakevenTrade
strat = InflationBreakevenTrade()
w = strat.generate_weights(prices)
assert not w.isna().any().any(), "M6 has NaN!"
assert not np.isinf(w.values).any(), "M6 has inf!"
print(f"  PASS: shape={w.shape}, non-zero days={int((w.abs().sum(axis=1) > 0).sum())}")

print("Testing M7 (GlobalRotation)...")
from financial_algo.strategies.macro import GlobalRotation
strat = GlobalRotation()
w = strat.generate_weights(prices)
assert not w.isna().any().any(), "M7 has NaN!"
assert not np.isinf(w.values).any(), "M7 has inf!"
assert (w >= 0).all().all(), "M7 has negative weights!"
print(f"  PASS: shape={w.shape}, non-zero days={int((w.abs().sum(axis=1) > 0).sum())}")

print("Testing M8 (CommodityMacroSignal)...")
from financial_algo.strategies.macro import CommodityMacroSignal
strat = CommodityMacroSignal()
w = strat.generate_weights(prices)
assert not w.isna().any().any(), "M8 has NaN!"
assert not np.isinf(w.values).any(), "M8 has inf!"
assert (w >= 0).all().all(), "M8 has negative weights!"
print(f"  PASS: shape={w.shape}, non-zero days={int((w.abs().sum(axis=1) > 0).sum())}")

print("\nAll new macro strategies PASS basic checks!")
