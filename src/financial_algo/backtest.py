"""Vectorised backtest engine with cost model and performance metrics.

Ported and refined from Financial-Algorithms v18.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import drawdown as _drawdown


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Cost model and risk overlay parameters."""

    # Transaction costs
    tx_cost_bps: float = 5.0          # one-way cost per trade in basis points
    leverage_cost_annual: float = 0.015  # annualised borrowing cost (1.5 %)
    short_cost_annual: float = 0.005   # annual short-borrow cost (0.5 %)

    # Risk overlays
    vol_target: float | None = None    # target annualised vol (e.g. 0.15)
    vol_lookback: int = 20
    max_drawdown_trigger: float | None = None  # e.g. -0.15 -> cut exposure at -15 %
    drawdown_recovery_rate: float = 0.10       # resume at this DD improvement

    # Gradual per-strategy DD scale-down (complements the binary trigger above)
    strategy_dd_scale_start: float | None = None  # e.g. -0.10: start scaling at -10%
    strategy_dd_scale_end: float | None = None     # e.g. -0.20: fully flat at -20%

    initial_capital: float = 1_000_000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict:
    """Run a vectorised daily backtest.

    Parameters
    ----------
    prices:
        Daily adjusted close prices (Date × Ticker).
    weights:
        Target portfolio weights (Date × Ticker). Weights should already
        be **shifted forward by 1 day** (e.g. via ``backtest_weights()``).
        The engine applies them directly to same-day asset returns.
        Weights can sum to > 1 (leveraged) or include negative values
        (short).
    config:
        Cost and risk overlay parameters. Defaults are used when ``None``.

    Returns
    -------
    dict
        Keys: ``equity``, ``returns``, ``weights``, ``metrics``.
    """
    if config is None:
        config = BacktestConfig()

    # Align weights to prices index, fill gaps with 0
    tickers = [c for c in weights.columns if c in prices.columns]
    prices = prices[tickers].copy()
    weights = weights[tickers].reindex(prices.index).fillna(0.0)

    # --- Daily returns per asset (simple) ---
    asset_returns = prices.pct_change().fillna(0.0)

    # --- Risk overlays ------------------------------------------------
    if config.vol_target is not None:
        weights = _vol_target_overlay(weights, asset_returns, config.vol_target, config.vol_lookback)

    # --- Portfolio return before costs ---
    # Weights are already shifted +1 day by backtest_weights(), so no
    # additional shift is needed here.
    port_return = (weights * asset_returns).sum(axis=1)

    # --- Cost deductions -----------------------------------------------
    # Turnover
    turnover = weights.diff().abs().sum(axis=1)
    tx_cost = turnover * config.tx_cost_bps / 10_000

    # Leverage cost: paid on gross exposure > 1
    gross = weights.abs().sum(axis=1)
    leverage_cost = (gross - 1).clip(lower=0) * config.leverage_cost_annual / 252

    # Short cost: paid on short notional
    short_notional = weights.clip(upper=0).abs().sum(axis=1)
    short_cost = short_notional * config.short_cost_annual / 252

    net_return = port_return - tx_cost - leverage_cost - short_cost

    # --- Gradual per-strategy DD scale-down ---------------------------
    if config.strategy_dd_scale_start is not None:
        net_return = _gradual_dd_scale(
            net_return,
            config.strategy_dd_scale_start,
            config.strategy_dd_scale_end or (config.strategy_dd_scale_start * 2),
        )

    # --- Drawdown control overlay -------------------------------------
    if config.max_drawdown_trigger is not None:
        net_return = _drawdown_control(net_return, config.max_drawdown_trigger, config.drawdown_recovery_rate)

    # --- Equity curve --------------------------------------------------
    equity = (1 + net_return).cumprod() * config.initial_capital

    metrics = compute_metrics(net_return, equity)

    # --- Turnover / trading frequency ---------------------------------
    # A "trade" is any day with meaningful weight change (> 1% turnover)
    trade_days = (turnover > 0.01).sum()
    months = max(len(turnover) / 21, 1)  # ~21 trading days per month
    metrics["avg_trades_per_month"] = round(trade_days / months, 1)
    metrics["annual_turnover"] = round(float(turnover.sum() / max(len(turnover) / 252, 1)), 4)

    return {
        "equity": equity,
        "returns": net_return,
        "weights": weights,
        "metrics": metrics,
    }


def compute_metrics(returns: pd.Series, equity: pd.Series | None = None) -> dict:
    """Compute standard performance metrics.

    Parameters
    ----------
    returns:
        Daily net returns.
    equity:
        Equity curve. If ``None``, computed from *returns*.

    Returns
    -------
    dict
        Sharpe, Sortino, CAGR, max_drawdown, Calmar, win_rate, etc.
    """
    if equity is None:
        equity = (1 + returns).cumprod()

    total_days = len(returns)
    if total_days < 2:
        return {}

    ann_factor = 252
    mean_r = returns.mean()
    std_r = returns.std()

    # CAGR
    years = total_days / ann_factor
    total_return = equity.iloc[-1] / equity.iloc[0]
    cagr = total_return ** (1 / years) - 1 if years > 0 else 0.0

    # Sharpe
    sharpe = (mean_r / std_r * np.sqrt(ann_factor)) if std_r > 0 else 0.0

    # Sortino (downside deviation from zero, over ALL observations)
    downside = np.sqrt((np.minimum(returns, 0) ** 2).mean())
    sortino = (mean_r / downside * np.sqrt(ann_factor)) if downside > 0 else 0.0

    # Max drawdown
    dd = _drawdown(equity)
    max_dd = dd.min()

    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # Win rate
    winning = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning / total_trades if total_trades > 0 else 0.0

    return {
        "cagr": round(cagr, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "annual_vol": round(std_r * np.sqrt(ann_factor), 6),
        "total_return": round(total_return - 1, 6),
    }


# ---------------------------------------------------------------------------
# Risk overlays
# ---------------------------------------------------------------------------

def _vol_target_overlay(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_vol: float,
    lookback: int,
) -> pd.DataFrame:
    """Scale weights so portfolio realised vol targets *target_vol*."""
    # Weights are already shifted +1, use them directly for return calc
    port_return = (weights * asset_returns).sum(axis=1)
    rvol = port_return.rolling(lookback).std() * np.sqrt(252)
    # Scale factor: target / realised (capped at 3x to avoid blowup)
    scale = (target_vol / rvol).clip(upper=3.0).fillna(1.0)
    return weights.multiply(scale, axis=0)


def _gradual_dd_scale(
    returns: pd.Series,
    dd_start: float,
    dd_end: float,
) -> pd.Series:
    """Linearly scale returns from 1.0 to 0.0 as drawdown deepens.

    Parameters
    ----------
    returns:
        Daily net returns.
    dd_start:
        Drawdown level where scaling begins (e.g. -0.10).
    dd_end:
        Drawdown level where returns are fully zeroed (e.g. -0.20).
        Must be more negative than *dd_start*.
    """
    equity = (1 + returns).cumprod()
    dd = _drawdown(equity)

    # Linear interpolation: 1.0 at dd_start, 0.0 at dd_end
    dd_range = dd_end - dd_start  # negative value
    if dd_range >= 0:
        return returns  # misconfigured, no-op
    raw_scale = (dd - dd_start) / dd_range  # 0 at dd_start, 1 at dd_end
    scale = (1.0 - raw_scale).clip(0.0, 1.0)

    return returns * scale


def _drawdown_control(
    returns: pd.Series,
    max_dd_trigger: float,
    recovery_rate: float,
) -> pd.Series:
    """Flatten returns when drawdown exceeds trigger; re-enter on recovery."""
    equity = (1 + returns).cumprod()
    dd = _drawdown(equity)

    dd_vals = dd.values
    ret_vals = returns.values.copy()
    n = len(dd_vals)
    flat = False
    trigger_dd = 0.0

    for i in range(n):
        if not flat:
            if dd_vals[i] <= max_dd_trigger:
                flat = True
                trigger_dd = dd_vals[i]
                ret_vals[i] = 0.0
        else:
            if dd_vals[i] >= trigger_dd + recovery_rate:
                flat = False
            else:
                ret_vals[i] = 0.0

    return pd.Series(ret_vals, index=returns.index)
