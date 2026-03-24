"""Quick evaluation of P7-FundamentalMomentumSignal."""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import yfinance as yf

from financial_algo.strategies.signal_combo import FundamentalMomentumSignal


def fetch_prices() -> pd.DataFrame:
    tickers = [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "SLV", "TLT",
        "SHY", "USO", "DBA", "DBC", "HYG", "AGG", "EMB", "TIP",
        "VNQ", "XBI", "FXI", "VGK", "EWJ", "INDA",
        "XLE", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY",
        "XLV", "XLRE", "XLC",
    ]
    print(f"Fetching {len(tickers)} tickers from yfinance...")
    data = yf.download(tickers, start="2010-01-01", end="2025-12-31",
                       auto_adjust=True, progress=False)
    prices = data["Close"].dropna(how="all")
    print(f"Got {len(prices)} days, {prices.shape[1]} tickers")
    return prices


def compute_metrics(equity: pd.Series) -> dict:
    """Compute standard backtest metrics from an equity curve."""
    returns = equity.pct_change().dropna()
    n_years = len(returns) / 252
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252 / downside) if downside > 0 else 0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    win_rate = (returns > 0).mean()

    # Worst single day
    worst_day = returns.min()

    # Correlation with SPY
    spy_eq = equity  # placeholder
    corr_spy = 0.0

    return {
        "CAGR": f"{cagr:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max DD": f"{max_dd:.1%}",
        "Calmar": f"{calmar:.2f}",
        "Win Rate": f"{win_rate:.1%}",
        "Worst Day": f"{worst_day:.1%}",
        "Ann Vol": f"{ann_vol:.1%}",
    }


def main():
    prices = fetch_prices()

    strat = FundamentalMomentumSignal()
    print(f"\nRunning {strat.name}...")
    weights = strat.generate_weights(prices)

    # Compute equity curve
    returns = prices.pct_change().fillna(0.0)
    strat_returns = (weights.shift(1) * returns).sum(axis=1)
    equity = (1 + strat_returns).cumprod()
    equity.iloc[0] = 1.0

    metrics = compute_metrics(equity)
    print(f"\n{'='*60}")
    print(f"  {strat.name} -- Full Period Backtest")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v}")

    # SPY buy-and-hold benchmark
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().fillna(0.0)
        spy_eq = (1 + spy_ret).cumprod()
        spy_eq.iloc[0] = 1.0
        spy_metrics = compute_metrics(spy_eq)
        print(f"\n  SPY Buy-Hold Benchmark:")
        for k, v in spy_metrics.items():
            print(f"  {k:>12}: {v}")

        # Correlation
        common = strat_returns.index.intersection(spy_ret.index)
        corr = strat_returns.loc[common].corr(spy_ret.loc[common])
        print(f"\n  Corr w/ SPY: {corr:.2f}")

    # Position diagnostics
    active_days = (weights.sum(axis=1) > 0).sum()
    avg_held = (weights > 0).sum(axis=1).mean()
    print(f"\n  Position diagnostics:")
    print(f"    Active days: {active_days}/{len(weights)}")
    print(f"    Avg assets held: {avg_held:.1f}")
    print(f"    Max gross exposure: {weights.abs().sum(axis=1).max():.2f}")


if __name__ == "__main__":
    main()
