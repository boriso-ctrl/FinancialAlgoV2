#!/usr/bin/env python3
"""Quick test of reworked strategies on crisis windows.

Tests C2, B1, H1, H4 before/after reworks in crisis windows.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# Avoid importing torch-dependent modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_algo.strategies.war_crisis import SafeHavenFlight
from financial_algo.strategies.oil_crisis import OilMomentumSurge
from financial_algo.strategies.crisis_spike import CommodityShockRider, MultiAssetCrisisLong

from financial_algo.backtester import run_backtest
from financial_algo.data_fetcher import fetch_data_all_windows
from financial_algo.regimes import compute_regime_series


def test_window(window_name: str) -> dict:
    """Test reworked strategies in a single crisis window."""
    print(f"\n{'='*60}")
    print(f"Testing window: {window_name}")
    print(f"{'='*60}")
    
    # Fetch data for this window
    data_all = fetch_data_all_windows()
    
    if window_name not in data_all:
        print(f"Warning: {window_name} not found in data")
        return {}
    
    prices = data_all[window_name]
    regimes = compute_regime_series(prices)
    
    strategies_to_test = {
        "C2-SafeHavenFlight": SafeHavenFlight(),
        "B1-OilMomentumSurge": OilMomentumSurge(),
        "H1-CommodityShockRider": CommodityShockRider(),
        "H4-MultiAssetCrisisLong": MultiAssetCrisisLong(),
    }
    
    results = {}
    for strat_name, strat in strategies_to_test.items():
        try:
            weights = strat.generate_weights(prices, regimes)
            
            # Quick backtest
            ret, metrics = run_backtest(
                weights=weights,
                prices=prices,
                strategy_name=strat_name,
            )
            
            sharpe = metrics.get("sharpe", 0.0)
            cagr = metrics.get("cagr", 0.0)
            max_dd = metrics.get("max_dd", 0.0)
            
            results[strat_name] = {
                "sharpe": sharpe,
                "cagr": cagr,
                "max_dd": max_dd,
            }
            
            print(f"\n{strat_name}:")
            print(f"  Sharpe: {sharpe:.3f}")
            print(f"  CAGR: {cagr*100:.2f}%")
            print(f"  Max DD: {max_dd*100:.2f}%")
            
        except Exception as e:
            print(f"\nError testing {strat_name}: {e}")
            results[strat_name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    windows = [
        "Oil Crash 2014-2016",
        "Russia-Ukraine + Inflation 2022",
        "Volmageddon + Fed 2018",
    ]
    
    all_results = {}
    for window in windows:
        try:
            res = test_window(window)
            all_results[window] = res
        except Exception as e:
            print(f"Failed to test window {window}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for window, res in all_results.items():
        print(f"\n{window}:")
        for strat, metrics in res.items():
            if "error" not in metrics:
                print(f"  {strat}: Sharpe={metrics['sharpe']:.3f}")
