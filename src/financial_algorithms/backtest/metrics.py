"""Portfolio metric helpers used by the backtest engine."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    turnover: Optional[pd.Series] = None,
    gross_exposure: Optional[pd.Series] = None,
) -> dict:
    total_return = (equity.iloc[-1] / initial_capital) - 1
    num_years = len(returns) / 252 if len(returns) else 0
    cagr = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = excess.mean() / downside_std * np.sqrt(252) if downside_std and downside_std > 0 else 0

    calmar = (cagr / abs(max_drawdown)) if max_drawdown < 0 else 0

    winning_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    win_rate = winning_days / total_days if total_days else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_turnover = turnover.mean() if turnover is not None else 0
    avg_gross = gross_exposure.mean() if gross_exposure is not None else 0

    return {
        "Total Return": round(total_return, 6),
        "CAGR": round(cagr, 6),
        "Sharpe Ratio": round(sharpe, 4),
        "Sortino Ratio": round(sortino, 4),
        "Calmar Ratio": round(calmar, 4),
        "Max Drawdown": round(max_drawdown, 6),
        "Win Rate": round(win_rate, 4),
        "Avg Win": round(wins.mean() if len(wins) else 0.0, 6),
        "Avg Loss": round(losses.mean() if len(losses) else 0.0, 6),
        "Avg Turnover": round(avg_turnover, 4),
        "Avg Gross Leverage": round(avg_gross, 4),
        "Total Days": int(total_days),
        "Winning Days": int(winning_days),
        "Final Equity": round(equity.iloc[-1], 2),
    }
