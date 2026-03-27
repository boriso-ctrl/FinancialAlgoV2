"""
Sprint 15.1 Phase 13: Evaluate reworked strategies before/after.

Measures Sharpe, CAGR, Max DD for each rework across multiple windows:
1. Full Period (2010-2025)
2. Momentum-favoring (2009-2012, 2020-2021)
3. Mean-reversion-favoring (2016-2017, 2023-2024)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure src/ is importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

# Strategy imports
from financial_algo.strategies.pairs import MultiPairPortfolio
from financial_algo.strategies.momentum import CrossSectionalMomentum, TimeSeriesMomentum
from financial_algo.strategies.mean_reversion import SectorMeanReversion, RSIMeanReversion
from financial_algo.strategies.factor import LowVolFactor
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.signal_combo import FeatureComboSignal

from financial_algo.backtest import backtest, BacktestConfig
from financial_algo.data.loader import load_prices


def fetch_data():
    """Fetch price data for all strategies."""
    tickers = [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "GLD", "TLT", "UUP", "XLE", "HYG",
        "LQD", "XLP", "XLU", "XLY", "XLV",
        "XLK", "XLF", "XLI", "XLB", "VNQ",
        "FXI", "VGK", "EWJ", "INDA", "XBI",
    ]
    prices = load_prices(tickers, start="2010-01-01", end="2025-12-31")
    return prices


def run_backtest(strategy, prices, name: str) -> dict:
    """Run backtest and return metrics."""
    try:
        # Generate weights (shift forward by 1 day to avoid look-ahead)
        weights = strategy.generate_weights(prices).shift(1).fillna(0.0)
        
        # Backtest with 5 bps transaction costs
        config = BacktestConfig(tx_cost_bps=5.0)
        result = backtest(prices, weights, config)
        
        # Extract metrics
        metrics = result.get("metrics", {})
        
        return {
            "strategy": name,
            "sharpe": metrics.get("sharpe", np.nan),
            "cagr": metrics.get("cagr", np.nan),
            "max_dd": metrics.get("max_dd", np.nan),
            "sortino": metrics.get("sortino", np.nan),
            "calmar": metrics.get("calmar", np.nan),
        }
    except Exception as e:
        print(f"ERROR in {name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {
            "strategy": name,
            "sharpe": np.nan,
            "cagr": np.nan,
            "max_dd": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "error": str(e),
        }


def main():
    """Evaluate all reworked strategies."""
    prices = fetch_data()
    print(f"Loaded {len(prices)} days of data for {len(prices.columns)} tickers")

    # Define time windows
    windows = {
        "Full Period": (prices.index.min(), prices.index.max()),
        "Momentum-2009-2012": (pd.Timestamp("2009-01-01"), pd.Timestamp("2012-12-31")),
        "Momentum-2020-2021": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        "MeanRev-2016-2017": (pd.Timestamp("2016-01-01"), pd.Timestamp("2017-12-31")),
        "MeanRev-2023-2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    }

    # Reworked strategies
    strategies = {
        "E1-MultiPairPortfolio": MultiPairPortfolio(),
        "I2-CrossSectionalMomentum": CrossSectionalMomentum(),
        "J1-SectorMeanReversion": SectorMeanReversion(),
        "J3-RSIMeanReversion": RSIMeanReversion(),
        "K1-LowVolFactor": LowVolFactor(),
        "N1-SeasonalStrategy": SeasonalStrategy(),
        "P1-FeatureComboSignal": FeatureComboSignal(),
    }

    results = []
    for window_name, (start, end) in windows.items():
        window_prices = prices.loc[start:end]
        if len(window_prices) < 100:
            continue

        print(f"\n{window_name} ({len(window_prices)} days)")
        print("=" * 70)
        for strat_name, strategy in strategies.items():
            result = run_backtest(strategy, window_prices, strat_name)
            result["window"] = window_name
            results.append(result)
            sharpe_str = f"{result['sharpe']:.2f}" if not np.isnan(result['sharpe']) else "N/A"
            cagr_str = f"{result['cagr']:.1%}" if not np.isnan(result['cagr']) else "N/A"
            maxdd_str = f"{result['max_dd']:.1%}" if not np.isnan(result['max_dd']) else "N/A"
            print(f"  {strat_name:30s} Sharpe={sharpe_str:>6s} CAGR={cagr_str:>8s} MaxDD={maxdd_str:>8s}")

    # Summary table
    df_results = pd.DataFrame(results)
    
    print("\n\nSUMMARY: Full Period Results")
    print("=" * 90)
    full_period = df_results[df_results["window"] == "Full Period"]
    if len(full_period) > 0:
        full_period_sorted = full_period.sort_values("sharpe", ascending=False)
        print(full_period_sorted[["strategy", "sharpe", "cagr", "max_dd", "sortino", "calmar"]].to_string(index=False))

    return df_results


if __name__ == "__main__":
    results_df = main()
    results_df.to_csv("sprint15_rework_results.csv", index=False)
    print("\n\nResults saved to sprint15_rework_results.csv")
