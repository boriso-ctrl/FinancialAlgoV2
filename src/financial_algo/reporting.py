"""QuantStats-based HTML tear sheet generation.

Wraps quantstats to produce rich HTML reports from backtest results.
Designed to slot into the existing backtest pipeline with zero friction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_tearsheet(
    backtest_result: dict[str, Any],
    benchmark_prices: pd.Series | None = None,
    output_path: str | Path = "results/tearsheet.html",
    title: str = "Strategy Tear Sheet",
) -> Path:
    """Generate an HTML tear sheet from a backtest() result dict.

    Parameters
    ----------
    backtest_result:
        Dict returned by ``backtest()``. Must contain ``'returns'``
        (pd.Series of daily net returns).
    benchmark_prices:
        Daily adjusted close prices for the benchmark (e.g. SPY).
        If provided, QuantStats computes alpha, beta, and relative
        metrics against it.  Pass ``prices['SPY']`` from the backtest.
    output_path:
        File path for the HTML report.
    title:
        Report title shown at the top of the tear sheet.

    Returns
    -------
    Path
        Absolute path to the generated HTML file.
    """
    try:
        import quantstats as qs  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "quantstats is not installed. "
            "Run: uv pip install quantstats"
        ) from exc

    returns = backtest_result["returns"]
    if not isinstance(returns, pd.Series):
        raise TypeError(
            f"Expected returns as pd.Series, got {type(returns).__name__}"
        )

    # Ensure returns index is DatetimeIndex (QuantStats requirement)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    # Prepare benchmark returns from prices (not from returns series)
    benchmark_returns = None
    if benchmark_prices is not None:
        benchmark_returns = benchmark_prices.pct_change().fillna(0.0)
        benchmark_returns = benchmark_returns.reindex(returns.index).fillna(0.0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # QuantStats extends pandas; calling .html() on the returns Series
    qs.reports.html(
        returns,
        benchmark=benchmark_returns,
        output=str(out),
        title=title,
        download_filename=str(out.stem),
    )

    return out.resolve()


def generate_metrics_snapshot(
    backtest_result: dict[str, Any],
    benchmark_prices: pd.Series | None = None,
) -> dict[str, float]:
    """Return extended metrics dict using QuantStats.

    Supplements our built-in compute_metrics() with metrics that are
    expensive or complex to implement from scratch: Kelly Criterion,
    VaR, CVaR, tail ratio, common sense ratio, etc.

    Parameters
    ----------
    backtest_result:
        Dict returned by ``backtest()``.
    benchmark_prices:
        Daily adjusted close for the benchmark (e.g. SPY).

    Returns
    -------
    dict
        Extended metrics keyed by metric name.
    """
    try:
        import quantstats as qs  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "quantstats is not installed. "
            "Run: uv pip install quantstats"
        ) from exc

    returns = backtest_result["returns"]
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    benchmark_returns = None
    if benchmark_prices is not None:
        benchmark_returns = benchmark_prices.pct_change().fillna(0.0)
        benchmark_returns = benchmark_returns.reindex(returns.index).fillna(0.0)

    # Collect QuantStats scalar metrics
    metrics = {}
    metrics["kelly_criterion"] = qs.stats.kelly_criterion(returns)
    metrics["value_at_risk"] = qs.stats.value_at_risk(returns)
    metrics["cvar"] = qs.stats.cvar(returns)
    metrics["tail_ratio"] = qs.stats.tail_ratio(returns)
    metrics["common_sense_ratio"] = qs.stats.common_sense_ratio(returns)
    metrics["payoff_ratio"] = qs.stats.payoff_ratio(returns)
    metrics["profit_factor"] = qs.stats.profit_factor(returns)
    metrics["outlier_win_ratio"] = qs.stats.outlier_win_ratio(returns)
    metrics["outlier_loss_ratio"] = qs.stats.outlier_loss_ratio(returns)
    metrics["avg_win"] = qs.stats.avg_win(returns)
    metrics["avg_loss"] = qs.stats.avg_loss(returns)
    metrics["best_day"] = qs.stats.best(returns)
    metrics["worst_day"] = qs.stats.worst(returns)
    metrics["skew"] = qs.stats.skew(returns)
    metrics["kurtosis"] = qs.stats.kurtosis(returns)
    metrics["gain_to_pain_ratio"] = qs.stats.gain_to_pain_ratio(returns)

    if benchmark_returns is not None:
        metrics["information_ratio"] = qs.stats.information_ratio(
            returns, benchmark_returns
        )
        metrics["greeks_beta"] = qs.stats.greeks(returns, benchmark_returns).get(
            "beta", None
        )
        metrics["greeks_alpha"] = qs.stats.greeks(returns, benchmark_returns).get(
            "alpha", None
        )

    return metrics
