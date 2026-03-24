"""Backtest OP9, OP10, OP11 — Real Options Strategies (Alpha Sprint Day 1-8).

Tests three new options strategies with real Alpaca chain data (fallback to
proxy signals for historical backtest periods without live options data).

Crisis windows:
  2011 Q3-Q4  — EU Sovereign Debt Crisis
  2014-2016   — Oil Price Crash
  2018 Q1/Q4  — Volmageddon + Fed tightening
  2020 Q1-Q2  — COVID-19 pandemic
  2022 Q1-Q3  — Russia-Ukraine War + inflation

Usage:
    python scripts/backtest_op9_op10_op11.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure src/ is importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.strategies.options import (
    SkewCarryPremium,
    TermStructureRegime,
    VolDispersionArbitrage,
)


def compute_window_metrics(nav_series: pd.Series) -> dict:
    """Compute backtest metrics from NAV series."""
    returns = nav_series.pct_change().ffill().fillna(0.0)
    if len(returns) < 2:
        return {}
    
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    trading_days = len(returns[returns != 0])  # days with returns
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / max(years, 0.25)) - 1 if years > 0 else 0
    
    cumulative_max = nav_series.expanding().max()
    drawdown = nav_series / cumulative_max - 1
    max_dd = drawdown.min()
    
    volatility = returns.std() * np.sqrt(252)
    sharpe = returns.mean() * 252 / volatility if volatility > 0 else 0
    
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "volatility": volatility,
        "sharpe": sharpe,
        "days": len(returns),
    }


def run_strategy_window(
    strategy_name: str,
    strategy,
    prices: pd.DataFrame,
) -> dict | None:
    """Run a single strategy window and return NAV + metrics."""
    try:
        weights = strategy.backtest_weights(prices)
        result = backtest(prices, weights, BacktestConfig())
        nav = result.get("equity", result.get("equity_curve", pd.Series()))
        metrics = result.get("metrics", {})
        
        if nav.empty or not metrics:
            return None
        
        return {
            "nav": nav,
            "metrics": metrics,
        }
    except Exception as e:
        print(f"    [WARN] {strategy_name}: {e}")
        return None


def main():
    """Run backtests for OP9, OP10, OP11 across crisis windows."""

    # Crisis windows
    crisis_windows = {
        "Full Period (2010-2025)": ("2010-01-01", "2025-12-31"),
        "EU Debt Crisis 2011": ("2011-07-01", "2011-12-31"),
        "Oil Crash 2014-16": ("2014-06-01", "2016-02-29"),
        "Volmageddon 2018": ("2018-01-01", "2018-12-31"),
        "COVID-19 2020": ("2020-02-15", "2020-06-30"),
        "War & Inflation 2022": ("2022-01-01", "2022-12-31"),
    }

    # Universe
    tickers = ["SPY", "QQQ", "IWM", "SHY", "TLT", "GLD", "^VIX"]

    print("=" * 80)
    print("BACKTEST: OP9, OP10, OP11 — Real Options Strategies")
    print("=" * 80)
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: {tickers}")
    print(f"Transaction cost: 5.0 bps")
    print()

    # Load prices once
    print("Loading price data...")
    try:
        prices = load_prices(
            tickers=tickers,
            start="2010-01-01",
            end="2025-12-31",
        )
    except Exception as e:
        print(f"ERROR loading prices: {e}")
        return

    print(f"Loaded {len(prices)} trading days, {len(prices.columns)} tickers\n")

    if prices.empty:
        print("ERROR: No price data loaded. Exiting.")
        return

    # Initialize strategies
    strategies = {
        "OP9-SkewCarryPremium": SkewCarryPremium(),
        "OP10-TermStructureRegime": TermStructureRegime(),
        "OP11-VolDispersionArbitrage": VolDispersionArbitrage(),
    }

    results = {}

    # Run backtests
    for window_name, (start_date, end_date) in crisis_windows.items():
        print(f"\n{'='*80}")
        print(f"Window: {window_name} ({start_date} to {end_date})")
        print(f"{'='*80}")

        # Slice data to window
        try:
            window_prices = prices.loc[start_date:end_date]
        except Exception:
            window_prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]

        if window_prices.empty or len(window_prices) < 20:
            print(f"  [SKIP] No/insufficient data for this window")
            continue

        print(f"  Data points: {len(window_prices)}")

        window_results = {}

        for strat_name, strat in strategies.items():
            print(f"  Running {strat_name}...")
            result = run_strategy_window(strat_name, strat, window_prices)

            if result is None:
                window_results[strat_name] = None
                continue

            window_results[strat_name] = result

            # Print summary
            metrics = result["metrics"]
            print(f"    Sharpe:   {metrics.get('sharpe', np.nan):7.2f}")
            print(f"    CAGR:     {metrics.get('cagr', np.nan):7.2%}")
            print(f"    MaxDD:    {metrics.get('max_drawdown', np.nan):7.2%}")
            print(f"    Calmar:   {metrics.get('calmar', np.nan):7.2f}")

        results[window_name] = window_results

    # ===== Summary Report =====
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")

    for window_name in crisis_windows.keys():
        if window_name not in results:
            continue

        print(f"\n{window_name}:")
        print(f"{'Strategy':<30} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10} {'Volatility':>10}")
        print("-" * 70)

        for strat_name, res in results[window_name].items():
            if res is None:
                print(f"{strat_name:<30} {'ERROR':>10}")
            else:
                metrics = res["metrics"]
                sharpe = metrics.get("sharpe", np.nan)
                cagr = metrics.get("cagr", np.nan)
                max_dd = metrics.get("max_drawdown", np.nan)
                vol = metrics.get("annual_vol", np.nan)

                print(
                    f"{strat_name:<30} {sharpe:10.2f} {cagr:10.1%} {max_dd:10.1%} {vol:10.1%}"
                )

    # ===== Correlation Analysis =====
    print(f"\n\n{'='*80}")
    print("CORRELATION ANALYSIS (Full Period)")
    print(f"{'='*80}\n")

    if "Full Period (2010-2025)" in results:
        full_results = results["Full Period (2010-2025)"]
        navs = {}
        for strat_name, res in full_results.items():
            if res is not None and "nav" in res:
                navs[strat_name] = res["nav"]

        if len(navs) >= 2:
            nav_df = pd.DataFrame(navs)
            returns_df = nav_df.pct_change().fillna(0.0)
            corr_matrix = returns_df.corr()

            print("Correlation Matrix (pairwise):")
            print(corr_matrix.to_string())

            # Add SPY benchmark
            if "SPY" in prices.columns:
                spy_returns = prices["SPY"].pct_change().loc[returns_df.index].fillna(0.0)
                print(f"\nCorrelations with SPY benchmark:")
                for col in returns_df.columns:
                    corr_with_spy = spy_returns.corr(returns_df[col])
                    print(f"  {col:30s}: {corr_with_spy:7.3f}")

    print(f"\n\nBacktest complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
