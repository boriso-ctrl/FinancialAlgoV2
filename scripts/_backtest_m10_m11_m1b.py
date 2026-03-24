"""Quick standalone backtest for M10, M11, M1b strategies."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from financial_algo.strategies.macro import (
    MacroSignalScoreboard,
    AdaptiveMacroBlend,
    DollarCarryScoreboard,
    DollarCarry,
    GoldDollarInverse,
    EMRiskPremium,
    CommodityMomentum,
    RatesRegimeTrade,
    GlobalRotation,
    CommodityMacroSignal,
    YieldCurveRegime,
)

TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF",
    "UUP", "XLE", "LQD", "HYG", "^VIX",
]

START = "2010-01-01"
END = "2025-12-31"
CACHE_DIR = os.path.expanduser("~/.financial_algo_cache")


def load_prices() -> pd.DataFrame:
    """Load prices from cache or download."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "macro_backtest_prices.csv")

    if os.path.exists(cache_file):
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(prices) > 3000:
            print(f"Loaded cached prices: {prices.shape}")
            return prices

    print("Downloading prices from yfinance...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.ffill().dropna(how="all")
    prices.to_csv(cache_file)
    print(f"Downloaded and cached: {prices.shape}")
    return prices


def compute_metrics(returns: pd.Series, name: str) -> dict:
    """Compute CAGR, Sharpe, Sortino, MaxDD, Calmar, Win Rate."""
    r = returns.dropna()
    if len(r) < 252:
        return {"name": name, "CAGR": 0, "Sharpe": 0, "Sortino": 0,
                "MaxDD": 0, "Calmar": 0, "WinRate": 0}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    neg_vol = r[r < 0].std() * np.sqrt(252)
    sortino = ann_ret / neg_vol if neg_vol > 0 else 0.0

    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max.replace(0, np.nan)
    max_dd = drawdown.min()

    # CAGR
    n_years = len(r) / 252
    total_ret = cum.iloc[-1] / cum.iloc[0] if cum.iloc[0] > 0 else 1.0
    cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0.0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (r > 0).mean()

    return {
        "name": name,
        "CAGR": f"{cagr:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "MaxDD": f"{max_dd:.1%}",
        "Calmar": f"{calmar:.2f}",
        "WinRate": f"{win_rate:.1%}",
    }


def backtest_strategy(strat, prices: pd.DataFrame) -> pd.Series:
    """Run backtest: generate weights, shift +1, compute returns."""
    weights = strat.generate_weights(prices)
    weights = weights.shift(1).fillna(0.0)  # avoid look-ahead
    daily_ret = prices.pct_change().fillna(0.0)
    # Portfolio return = sum of weight * asset return
    port_ret = (weights * daily_ret).sum(axis=1)
    return port_ret


def main() -> None:
    prices = load_prices()
    print(f"\nPrices: {prices.shape[0]} days, {prices.shape[1]} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Tickers: {list(prices.columns)}\n")

    # New strategies
    new_strats = [
        MacroSignalScoreboard(),
        AdaptiveMacroBlend(),
        DollarCarryScoreboard(),
    ]

    # Existing M-series for correlation comparison
    existing_strats = [
        DollarCarry(),
        GoldDollarInverse(),
        EMRiskPremium(),
        CommodityMomentum(),
        RatesRegimeTrade(),
        GlobalRotation(),
        CommodityMacroSignal(),
        YieldCurveRegime(),
    ]

    all_returns = {}

    # Backtest new strategies
    print("=" * 70)
    print("NEW STRATEGY PERFORMANCE (Full Period)")
    print("=" * 70)
    results = []
    for strat in new_strats:
        ret = backtest_strategy(strat, prices)
        all_returns[strat.name] = ret
        metrics = compute_metrics(ret, strat.name)
        results.append(metrics)
        print(f"{metrics['name']:30s} | CAGR {metrics['CAGR']:>7s} | Sharpe {metrics['Sharpe']:>6s} | "
              f"Sortino {metrics['Sortino']:>6s} | MaxDD {metrics['MaxDD']:>7s} | "
              f"Calmar {metrics['Calmar']:>6s} | WinRate {metrics['WinRate']:>6s}")

    # Backtest existing for correlation
    print("\n" + "=" * 70)
    print("EXISTING M-SERIES (for correlation reference)")
    print("=" * 70)
    for strat in existing_strats:
        ret = backtest_strategy(strat, prices)
        all_returns[strat.name] = ret
        metrics = compute_metrics(ret, strat.name)
        print(f"{metrics['name']:30s} | CAGR {metrics['CAGR']:>7s} | Sharpe {metrics['Sharpe']:>6s} | "
              f"MaxDD {metrics['MaxDD']:>7s}")

    # SPY buy-and-hold for correlation
    spy_ret = prices["SPY"].pct_change().fillna(0.0)
    all_returns["SPY-BuyHold"] = spy_ret

    # Correlation matrix
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX (new vs existing M-series)")
    print("=" * 70)
    ret_df = pd.DataFrame(all_returns)
    corr = ret_df.corr()

    new_names = [s.name for s in new_strats]
    existing_names = [s.name for s in existing_strats] + ["SPY-BuyHold"]

    for new in new_names:
        print(f"\n{new}:")
        for old in existing_names:
            if old in corr.columns:
                print(f"  vs {old:30s}: {corr.loc[new, old]:+.3f}")


if __name__ == "__main__":
    main()
