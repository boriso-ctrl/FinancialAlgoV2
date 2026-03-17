"""Vectorized long/short backtest with turnover frictions."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def _validate_price_data(prices: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    if prices.index.duplicated().any():
        if strict:
            raise ValueError("Price index has duplicates; strict mode enabled.")
        prices = prices[~prices.index.duplicated(keep="first")]

    if prices.isna().values.any():
        prices = prices.ffill().bfill()
        if strict and prices.isna().values.any():
            raise ValueError("Price data contains NaNs after fill; strict mode enabled.")

    return prices


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    cost_bps: float = 1.0,
    slippage_bps: float = 5.0,
    allow_short: bool = False,
    risk_free_rate: float = 0.0,
    max_gross_leverage: float = 1.0,
    max_position_weight: float = 1.0,
    max_signal_abs: float = 5.0,
    strict_data: bool = False,
) -> Dict[str, pd.DataFrame | pd.Series | dict]:
    prices = _validate_price_data(prices, strict=strict_data)

    signals = signals.reindex(prices.index).fillna(0)
    common = prices.columns.intersection(signals.columns)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between prices and signals.")
    prices = prices[common]
    signals = signals[common].clip(lower=-max_signal_abs, upper=max_signal_abs)

    returns = prices.pct_change()

    shifted_signals = signals.shift(1).fillna(0)
    if not allow_short:
        shifted_signals = shifted_signals.clip(lower=0)

    abs_sum = shifted_signals.abs().sum(axis=1)
    weights = shifted_signals.div(abs_sum, axis=0).fillna(0)
    weights = weights.clip(lower=-max_position_weight, upper=max_position_weight)

    gross = weights.abs().sum(axis=1)
    scale = (max_gross_leverage / gross).where(gross > max_gross_leverage, other=1.0)
    weights = weights.mul(scale, axis=0)

    strategy_returns = returns * weights
    portfolio_gross = strategy_returns.sum(axis=1)

    prev_weights = weights.shift().fillna(0)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    cost_rate = (cost_bps + slippage_bps) / 10_000.0
    trading_cost = turnover * cost_rate
    portfolio_returns = (portfolio_gross - trading_cost).fillna(0)

    equity_curve = initial_capital * (1 + portfolio_returns).cumprod()
    equity_curve.name = "Equity"

    metrics = compute_metrics(
        portfolio_returns,
        equity_curve,
        initial_capital,
        risk_free_rate=risk_free_rate,
        turnover=turnover,
        gross_exposure=gross,
    )

    return {
        "equity_curve": equity_curve,
        "portfolio_returns": portfolio_returns,
        "strategy_returns": strategy_returns,
        "num_positions": abs_sum,
        "weights": weights,
        "turnover": turnover,
        "gross_exposure": gross,
        "metrics": metrics,
    }
