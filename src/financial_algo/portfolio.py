"""Portfolio management utilities."""

from __future__ import annotations

from typing import Iterable


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
