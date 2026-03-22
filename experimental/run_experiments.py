"""Experimental Strategy Lab -- Backtest Runner.

Runs all experimental (X/Y/Z/W) strategies through the same crisis
windows and cost model used by production, then reports:
  - Full-period and per-window metrics table
  - Pairwise correlation with production ensemble members
  - Promotion candidates (Sharpe > 0.5, low correlation)

Usage:
    .venv\\Scripts\\python.exe experimental/run_experiments.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ and repo root are importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import Regime, detect_regime

# Experimental strategy imports
from experimental.strategies import ALL_EXPERIMENTAL

# Production ensemble members (for correlation check)
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.volatility_strats import VolRiskPremium, VolSpikeRecovery
from financial_algo.strategies.tail_risk import TailRiskParity
from financial_algo.strategies.quality_trend import QualityTrend, MultiAssetTrend, MomentumCrashFilter

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
# Configuration
# =========================================================================

DATA_START = "2009-01-01"
DATA_END = "2025-12-31"

CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "Full Period (2010-2025)":              ("2010-01-01", "2025-12-31"),
    "EU Debt Crisis (2011)":                ("2011-01-01", "2012-01-01"),
    "Oil Crash (2014-2016)":                ("2014-06-01", "2016-06-01"),
    "Volmageddon + Fed (2018)":             ("2018-01-01", "2019-01-01"),
    "COVID-19 (2020)":                      ("2020-01-01", "2021-01-01"),
    "Russia-Ukraine + Inflation (2022)":    ("2022-01-01", "2023-01-01"),
    "Recovery & Recent (2023-2025)":        ("2023-01-01", "2025-12-31"),
}

BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=-0.25,
    drawdown_recovery_rate=0.10,
)

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP",
    "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "HYG", "LQD",
    "BTC-USD",
]))

VIX_TICKER = "^VIX"

PROMOTION_SHARPE = 0.5
MAX_CORR_WITH_PRODUCTION = 0.4


# =========================================================================
# Helpers
# =========================================================================

def run_one(strategy, prices, regime, config):
    """Backtest a single strategy; return metrics dict or None."""
    try:
        weights = strategy.backtest_weights(prices)
        result = backtest(prices, weights, config)
        return result
    except Exception as e:
        print(f"  [WARN] {strategy.name}: {e}")
        return None


def fmt(val, kind="pct"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "  N/A"
    if kind == "pct":
        return f"{val:>7.2%}"
    if kind == "f2":
        return f"{val:>7.2f}"
    return f"{val:>7}"


# =========================================================================
# Correlation analysis
# =========================================================================

def compute_daily_returns(strategy, prices, regime):
    """Return the daily equity curve returns for a strategy."""
    try:
        weights = strategy.backtest_weights(prices)
        result = backtest(prices, weights, BT_CONFIG)
        eq = result["equity"]
        return eq.pct_change().fillna(0.0)
    except Exception:
        return None


def correlation_with_production(exp_strategies, prod_strategies, prices, regime):
    """Compute pairwise correlation between experimental and production strats."""
    print("\n" + "=" * 80)
    print("  CORRELATION WITH PRODUCTION ENSEMBLE MEMBERS")
    print("=" * 80)

    prod_returns = {}
    for s in prod_strategies:
        r = compute_daily_returns(s, prices, regime)
        if r is not None:
            prod_returns[s.name] = r

    exp_returns = {}
    for s in exp_strategies:
        r = compute_daily_returns(s, prices, regime)
        if r is not None:
            exp_returns[s.name] = r

    if not prod_returns or not exp_returns:
        print("  Could not compute correlation (missing returns)")
        return {}

    results = {}
    header = f"  {'Experimental':<30s}"
    for pn in prod_returns:
        header += f" {pn[:12]:>12s}"
    header += "    AvgCorr"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for en, er in exp_returns.items():
        row = f"  {en:<30s}"
        corrs = []
        for pn, pr in prod_returns.items():
            # Align indices
            common = er.index.intersection(pr.index)
            if len(common) < 60:
                row += "         N/A"
                continue
            c = er.loc[common].corr(pr.loc[common])
            corrs.append(c)
            row += f" {c:>12.3f}"
        avg = np.mean(corrs) if corrs else float("nan")
        row += f"    {avg:>7.3f}"
        results[en] = avg
        print(row)

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 80)
    print("  EXPERIMENTAL ALPHA LAB -- BACKTEST RUNNER")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"[1/4] Loading {len(TICKERS)} tickers ({DATA_START} -> {DATA_END}) ...")
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    print(f"       {prices.shape[0]} days x {prices.shape[1]} tickers")
    print()

    print("       Loading VIX ...")
    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER]
    except Exception:
        vix = None
    print()

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------
    print("[2/4] Detecting regimes ...")
    regime = detect_regime(prices, vix=vix)
    print()

    # ------------------------------------------------------------------
    # Run experimental strategies
    # ------------------------------------------------------------------
    strategies = ALL_EXPERIMENTAL
    print(f"[3/4] Running {len(strategies)} experimental strategies "
          f"across {len(CRISIS_WINDOWS)} windows ...")
    print()

    all_results: dict[str, dict[str, dict | None]] = {}

    for window_name, (ws, we) in CRISIS_WINDOWS.items():
        print("-" * 80)
        print(f"  WINDOW: {window_name}")
        print("-" * 80)

        mask = (prices.index >= ws) & (prices.index <= we)
        p_win = prices.loc[mask].copy()
        r_win = regime.loc[mask].copy()

        if len(p_win) < 30:
            print("  [SKIP] Not enough data")
            continue

        print(f"  {p_win.index[0].date()} -> {p_win.index[-1].date()} ({len(p_win)} days)")

        # Table header
        h = (f"  {'Strategy':<30s} {'CAGR':>7s} {'Sharpe':>7s} "
             f"{'Sortino':>7s} {'MaxDD':>7s} {'Calmar':>7s} {'WinRate':>7s}")
        print(h)
        print("  " + "-" * (len(h) - 2))

        for strat in strategies:
            result = run_one(strat, p_win, r_win, BT_CONFIG)
            if result is None:
                print(f"  {strat.name:<30s}   ERROR")
                continue

            m = result["metrics"]
            line = (
                f"  {strat.name:<30s}"
                f" {fmt(m['cagr'])}"
                f" {fmt(m['sharpe'], 'f2')}"
                f" {fmt(m['sortino'], 'f2')}"
                f" {fmt(m['max_drawdown'])}"
                f" {fmt(m['calmar'], 'f2')}"
                f" {fmt(m['win_rate'])}"
            )
            print(line)

            # Store full-period results for promotion analysis
            if window_name.startswith("Full Period"):
                all_results[strat.name] = m

        print()

    # ------------------------------------------------------------------
    # Promotion analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PROMOTION CANDIDATES (Sharpe > %.2f)" % PROMOTION_SHARPE)
    print("=" * 80)

    candidates = []
    for name, m in sorted(all_results.items(), key=lambda x: x[1].get("sharpe", 0), reverse=True):
        sharpe = m.get("sharpe", 0)
        if sharpe >= PROMOTION_SHARPE:
            candidates.append(name)
            print(f"  [PROMOTE] {name:<30s} Sharpe={sharpe:.2f}  "
                  f"CAGR={m['cagr']:.2%}  MaxDD={m['max_drawdown']:.2%}")

    if not candidates:
        print("  No strategies meet promotion criteria yet.")
        print("  Best performers:")
        for name, m in sorted(all_results.items(), key=lambda x: x[1].get("sharpe", 0), reverse=True)[:5]:
            print(f"    {name:<30s} Sharpe={m.get('sharpe', 0):.2f}")

    # ------------------------------------------------------------------
    # Correlation with production
    # ------------------------------------------------------------------
    prod_strategies = [
        FeatureComboSignal(),
        CrashHedgeQQQ(),
        VolRiskPremium(),
        TailRiskParity(),
        VolCarry(),
        MultiAssetTrend(),
        QualityTrend(),
        MomentumCrashFilter(),
        VolSpikeRecovery(),
    ]

    # Use full-period data for correlation
    mask_full = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
    p_full = prices.loc[mask_full]
    r_full = regime.loc[mask_full]
    corr_map = correlation_with_production(strategies, prod_strategies, p_full, r_full)

    # Final promotion verdict
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)
    for name in candidates:
        avg_corr = corr_map.get(name, float("nan"))
        if np.isfinite(avg_corr) and avg_corr < MAX_CORR_WITH_PRODUCTION:
            print(f"  ** READY TO PROMOTE ** {name}  (avg corr={avg_corr:.3f})")
        elif np.isfinite(avg_corr):
            print(f"  [TOO CORRELATED] {name}  (avg corr={avg_corr:.3f} > {MAX_CORR_WITH_PRODUCTION})")
        else:
            print(f"  [NEEDS REVIEW] {name}  (correlation could not be computed)")

    print("\n  Done.")


if __name__ == "__main__":
    main()
