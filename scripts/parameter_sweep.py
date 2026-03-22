"""Parameter sweep using vectorbt for strategy hyperparameter optimization.

Sweeps a configurable grid of strategy parameters and produces performance
heatmaps.  Uses vectorbt's Portfolio simulation engine for fast vectorized
backtesting across parameter combinations.

Usage:
    .venv\\Scripts\\python.exe scripts/parameter_sweep.py

Requirements:
    uv pip install vectorbt
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.data.loader import load_prices


def sweep_rsi_parameters(
    prices: pd.DataFrame,
    ticker: str = "SPY",
    rsi_periods: list[int] | None = None,
    entry_thresholds: list[int] | None = None,
    exit_thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Sweep RSI strategy parameters and return a results DataFrame.

    Parameters
    ----------
    prices:
        Daily adjusted close prices (Date x Ticker).
    ticker:
        Which ticker to run the RSI sweep on.
    rsi_periods:
        List of RSI lookback periods to test.
    entry_thresholds:
        RSI levels below which we enter long (oversold).
    exit_thresholds:
        RSI levels above which we exit (overbought).

    Returns
    -------
    pd.DataFrame
        Columns: rsi_period, entry, exit, sharpe, cagr, max_dd, total_return
    """
    try:
        import vectorbt as vbt  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "vectorbt is not installed. Run: uv pip install vectorbt"
        ) from exc

    if rsi_periods is None:
        rsi_periods = [5, 10, 14, 20, 30]
    if entry_thresholds is None:
        entry_thresholds = [20, 25, 30, 35]
    if exit_thresholds is None:
        exit_thresholds = [65, 70, 75, 80]

    close = prices[ticker].dropna()

    results = []
    for rsi_period, entry, exit_thresh in product(
        rsi_periods, entry_thresholds, exit_thresholds
    ):
        if entry >= exit_thresh:
            continue  # nonsensical combo

        rsi = vbt.RSI.run(close, window=rsi_period)
        entries = rsi.rsi_crossed_below(entry)
        exits = rsi.rsi_crossed_above(exit_thresh)

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=1_000_000,
            fees=0.0005,  # 5 bps one-way
            freq="1D",
        )

        stats = pf.stats()
        results.append({
            "rsi_period": rsi_period,
            "entry": entry,
            "exit": exit_thresh,
            "sharpe": stats.get("Sharpe Ratio", np.nan),
            "cagr": stats.get("Annualized Return", np.nan),
            "max_dd": stats.get("Max Drawdown", np.nan),
            "total_return": stats.get("Total Return", np.nan),
        })

    return pd.DataFrame(results)


def sweep_momentum_lookback(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    lookbacks: list[int] | None = None,
    holding_periods: list[int] | None = None,
) -> pd.DataFrame:
    """Sweep momentum lookback and holding periods.

    Parameters
    ----------
    prices:
        Daily adjusted close prices.
    tickers:
        Which tickers to include in a cross-sectional momentum strategy.
    lookbacks:
        List of lookback windows in days (e.g. [21, 63, 126, 252]).
    holding_periods:
        List of holding/rebalance periods in days.

    Returns
    -------
    pd.DataFrame
        Columns: lookback, holding, sharpe, cagr, max_dd
    """
    try:
        import vectorbt as vbt  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "vectorbt is not installed. Run: uv pip install vectorbt"
        ) from exc

    if tickers is None:
        tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT"]
    if lookbacks is None:
        lookbacks = [21, 42, 63, 126, 252]
    if holding_periods is None:
        holding_periods = [5, 10, 21, 42]

    close = prices[tickers].dropna()

    results = []
    for lookback, hold in product(lookbacks, holding_periods):
        # Compute momentum: return over lookback period
        mom = close.pct_change(lookback)
        # Rank: buy top 3 tickers each rebalance
        rank = mom.rank(axis=1, ascending=False)
        entries = rank <= 3  # top 3
        # Rebalance every 'hold' days
        entries_rebal = entries.copy()
        entries_rebal.iloc[::hold] = entries.iloc[::hold]

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries_rebal,
            exits=~entries_rebal,
            init_cash=1_000_000,
            fees=0.0005,
            freq="1D",
        )

        stats = pf.stats()
        results.append({
            "lookback": lookback,
            "holding": hold,
            "sharpe": stats.get("Sharpe Ratio", np.nan),
            "cagr": stats.get("Annualized Return", np.nan),
            "max_dd": stats.get("Max Drawdown", np.nan),
        })

    return pd.DataFrame(results)


def plot_heatmap(
    results: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str = "sharpe",
    title: str = "Parameter Sweep Heatmap",
    output_path: str | Path = "results/param_sweep_heatmap.html",
) -> Path:
    """Pivot results into a heatmap and save as HTML via plotly.

    Parameters
    ----------
    results:
        DataFrame from one of the sweep functions.
    x_col, y_col:
        Column names for the two swept parameters.
    value_col:
        Metric to display (default: sharpe).
    title:
        Chart title.
    output_path:
        Where to save the HTML file.

    Returns
    -------
    Path
        Absolute path to the generated HTML file.
    """
    import plotly.express as px  # already in our deps

    pivot = results.pivot_table(
        index=y_col, columns=x_col, values=value_col, aggfunc="mean"
    )

    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title=title,
        labels={"x": x_col, "y": y_col, "color": value_col},
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    return out.resolve()


# =====================================================================
# CLI entry point
# =====================================================================

def main() -> None:
    print("=" * 70)
    print("PARAMETER SWEEP - RSI Strategy on SPY")
    print("=" * 70)

    print("Loading prices ...")
    prices = load_prices(
        ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT"],
        start="2010-01-01",
        end="2025-12-31",
    )
    print(f"  {prices.shape[0]} days x {prices.shape[1]} tickers")
    print()

    # --- RSI sweep ---
    print("Running RSI parameter sweep ...")
    rsi_results = sweep_rsi_parameters(prices, ticker="SPY")
    best = rsi_results.loc[rsi_results["sharpe"].idxmax()]
    print(f"  Best: RSI({int(best['rsi_period'])}), "
          f"entry<{int(best['entry'])}, exit>{int(best['exit'])}")
    print(f"  Sharpe={best['sharpe']:.2f}, "
          f"CAGR={best['cagr']:.1%}, "
          f"MaxDD={best['max_dd']:.1%}")
    print()

    out1 = plot_heatmap(
        rsi_results,
        x_col="rsi_period",
        y_col="entry",
        value_col="sharpe",
        title="RSI Param Sweep: Sharpe by Period x Entry Threshold",
        output_path="results/rsi_sweep_heatmap.html",
    )
    print(f"  Heatmap saved: {out1}")

    # --- Momentum sweep ---
    print()
    print("Running Momentum lookback sweep ...")
    mom_results = sweep_momentum_lookback(prices)
    best_m = mom_results.loc[mom_results["sharpe"].idxmax()]
    print(f"  Best: lookback={int(best_m['lookback'])}d, "
          f"hold={int(best_m['holding'])}d")
    print(f"  Sharpe={best_m['sharpe']:.2f}, "
          f"CAGR={best_m['cagr']:.1%}, "
          f"MaxDD={best_m['max_dd']:.1%}")

    out2 = plot_heatmap(
        mom_results,
        x_col="lookback",
        y_col="holding",
        value_col="sharpe",
        title="Momentum Sweep: Sharpe by Lookback x Holding Period",
        output_path="results/momentum_sweep_heatmap.html",
    )
    print(f"  Heatmap saved: {out2}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
