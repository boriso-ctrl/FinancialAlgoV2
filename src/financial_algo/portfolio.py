"""Portfolio management utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from financial_algo.indicators import drawdown as _drawdown
from financial_algo.indicators import realized_vol


class Portfolio:
    """A simple long-only portfolio tracker.

    Parameters
    ----------
    cash:
        Initial cash balance in account currency. Defaults to 10 000.
    """

    def __init__(self, cash: float = 10_000.0) -> None:
        if cash < 0:
            raise ValueError("Initial cash cannot be negative")
        self._cash = cash
        self._positions: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        """Current holdings as ``{ticker: shares}``."""
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Trading operations
    # ------------------------------------------------------------------

    def buy(self, ticker: str, shares: float, price: float) -> None:
        """Buy *shares* of *ticker* at *price*.

        Parameters
        ----------
        ticker:
            Instrument identifier.
        shares:
            Number of shares to buy (must be positive).
        price:
            Price per share (must be positive).

        Raises
        ------
        ValueError
            If *shares* or *price* are not positive, or if there is
            insufficient cash.
        """
        if shares <= 0:
            raise ValueError("shares must be positive")
        if price <= 0:
            raise ValueError("price must be positive")
        cost = shares * price
        if cost > self._cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self._cash:.2f}"
            )
        self._cash -= cost
        self._positions[ticker] = self._positions.get(ticker, 0.0) + shares

    def sell(self, ticker: str, shares: float, price: float) -> None:
        """Sell *shares* of *ticker* at *price*.

        Parameters
        ----------
        ticker:
            Instrument identifier.
        shares:
            Number of shares to sell (must be positive and <= held shares).
        price:
            Price per share (must be positive).

        Raises
        ------
        ValueError
            If *shares* or *price* are not positive, or if the position
            is insufficient.
        """
        if shares <= 0:
            raise ValueError("shares must be positive")
        if price <= 0:
            raise ValueError("price must be positive")
        held = self._positions.get(ticker, 0.0)
        if shares > held:
            raise ValueError(
                f"Insufficient position in {ticker}: need {shares}, have {held}"
            )
        self._cash += shares * price
        remaining = held - shares
        if remaining == 0.0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = remaining

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def market_value(self, prices: dict[str, float]) -> float:
        """Return the total market value (cash + positions).

        Parameters
        ----------
        prices:
            Mapping of ``{ticker: current_price}``.
        """
        equity = sum(
            self._positions[t] * prices.get(t, 0.0) for t in self._positions
        )
        return self._cash + equity

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Portfolio(cash={self._cash:.2f}, positions={self._positions})"
        )


# =========================================================================
# Leverage & risk overlay utilities (operate on weight DataFrames)
# =========================================================================


def apply_leverage(
    weights: pd.DataFrame,
    leverage: float,
) -> pd.DataFrame:
    """Scale all weights by a constant leverage factor.

    Parameters
    ----------
    weights:
        Target weights (Date × Ticker).
    leverage:
        Multiplier (e.g. 2.0 for 2× leverage).
    """
    return weights * leverage


def apply_vol_target(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    lookback: int = 20,
) -> pd.DataFrame:
    """Scale weights so the portfolio's realised vol tracks *target_vol*.

    Parameters
    ----------
    weights:
        Target weights (Date × Ticker).
    prices:
        Adjusted close prices.
    target_vol:
        Annualised target volatility (e.g. 0.15 = 15 %).
    lookback:
        Rolling window for realised vol.
    """
    asset_ret = prices.pct_change().fillna(0.0)
    port_ret = (weights.shift(1).fillna(0) * asset_ret).sum(axis=1)
    rvol = port_ret.rolling(lookback).std() * np.sqrt(252)
    scale = (target_vol / rvol).clip(upper=3.0).fillna(1.0)
    return weights.multiply(scale, axis=0)


def apply_drawdown_control(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    max_dd_trigger: float = -0.15,
    recovery_rate: float = 0.05,
) -> pd.DataFrame:
    """Zero out weights when portfolio drawdown exceeds a trigger.

    Re-enters when drawdown improves by *recovery_rate* from the trigger
    point.

    Parameters
    ----------
    weights:
        Target weights (Date × Ticker).
    prices:
        Adjusted close prices.
    max_dd_trigger:
        Drawdown threshold (e.g. -0.15 = −15 %).
    recovery_rate:
        DD improvement needed to re-enter (positive number).
    """
    asset_ret = prices.pct_change().fillna(0.0)
    port_ret = (weights.shift(1).fillna(0) * asset_ret).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    dd = _drawdown(equity)

    dd_vals = dd.values
    n = len(dd_vals)
    flat_mask = np.zeros(n, dtype=bool)
    flat = False
    trigger_dd = 0.0

    for i in range(n):
        if not flat:
            if dd_vals[i] <= max_dd_trigger:
                flat = True
                trigger_dd = dd_vals[i]
                flat_mask[i] = True
        else:
            if dd_vals[i] >= trigger_dd + recovery_rate:
                flat = False
            else:
                flat_mask[i] = True

    controlled = weights.copy()
    controlled.loc[flat_mask] = 0.0
    return controlled
