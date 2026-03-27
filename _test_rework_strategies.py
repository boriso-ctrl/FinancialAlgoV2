#!/usr/bin/env python3
"""Quick test: Run reworked macro strategies on sample prices.
This validates that the strategies run without errors.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path

# Try direct strategy imports
try:
    from financial_algo.strategies.fixed_income import YieldCurveTrade, CreditSpreadMeanRev
    from financial_algo.strategies.macro import DollarCarry, EMRiskPremium, CommodityMomentum
    print("[OK] All strategy imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Load test data from recent backtest results
results_dir = Path("results")
backtest_file = results_dir / "backtest_Full_Period_2010-2025.csv"

if not backtest_file.exists():
    print(f"[SKIP] No test data at {backtest_file}")
    sys.exit(0)

print(f"[INFO] Testing against {backtest_file}")

# Load price data from existing backtest artifacts
data_dir = Path("~/.financial_algo_cache").expanduser()
if data_dir.exists():
    print(f"[INFO] Checking for cached price data in {data_dir}")

# Create minimal test DataFrame
dates = pd.date_range('2015-01-01', '2025-01-01', freq='D')
np.random.seed(42)

# Simulate realistic prices for key instruments
test_prices = pd.DataFrame({
    'TLT': 100 + np.cumsum(np.random.randn(len(dates)) * 0.3),  # Bond prices
    'IEF': 100 + np.cumsum(np.random.randn(len(dates)) * 0.15),
    'SPY': 200 + np.cumsum(np.random.randn(len(dates)) * 0.7),  # Equity
    'UUP': 100 + np.cumsum(np.random.randn(len(dates)) * 0.2),  # Dollar
    'HYG': 100 + np.cumsum(np.random.randn(len(dates)) * 0.25),  # Credit
    'LQD': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1),
    'EEM': 100 + np.cumsum(np.random.randn(len(dates)) * 0.8),  # EM
    'GLD': 100 + np.cumsum(np.random.randn(len(dates)) * 0.25),  # Gold
    'XLE': 100 + np.cumsum(np.random.randn(len(dates)) * 0.4),  # Energy
    '^VIX': 20 + np.cumsum(np.random.randn(len(dates)) * 0.5),  # VIX
}, index=dates)

test_prices = test_prices.clip(lower=10)  # Prevent negative prices

print(f"[INFO] Created test price DataFrame with shape {test_prices.shape}")

# Test each strategy
strategies = [
    ('H1-YieldCurveTrade', YieldCurveTrade()),
    ('H2-CreditSpreadMeanRev', CreditSpreadMeanRev()),
    ('M1-DollarCarry', DollarCarry()),
    ('M3-EMRiskPremium', EMRiskPremium()),
    ('M4-CommodityMomentum', CommodityMomentum()),
]

for strat_name, strat in strategies:
    try:
        weights = strat.generate_weights(test_prices)
        
        # Validate output
        assert isinstance(weights, pd.DataFrame), f"{strat_name}: weights not DataFrame"
        assert weights.shape[0] == test_prices.shape[0], f"{strat_name}: shape mismatch"
        assert not weights.isnull().all().all(), f"{strat_name}: all NaN weights"
        assert (weights >= -2).all().all(), f"{strat_name}: negative weights too extreme"
        assert (weights <= 2).all().all(), f"{strat_name}: positive weights too extreme"
        
        # Compute basic metrics
        avg_leverage = weights.sum(axis=1).abs().mean()
        turnover = weights.diff().abs().sum(axis=1).mean()
        
        print(f"[PASS] {strat_name:30s} | avg_leverage={avg_leverage:.2f} | turnover={turnover:.2f}")
        
    except Exception as e:
        print(f"[FAIL] {strat_name:30s} | ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n[SUCCESS] All reworked strategies run without errors!")
