"""Abstract base class for all strategy bots."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for every strategy bot.

    Subclasses must implement :meth:`generate_weights`, which returns a
    ``pd.DataFrame`` of target portfolio weights (Date × Ticker).

    Weights are **not** shifted — the backtest engine expects the
    caller to :meth:`shift` them by 1 day to avoid look-ahead bias.
    """

    name: str = "BaseStrategy"

    @abstractmethod
    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Produce daily target weights.

        Parameters
        ----------
        prices:
            Adjusted close prices (Date × Ticker).
        regime:
            Optional regime Series from :func:`detect_regime`.

        Returns
        -------
        pd.DataFrame
            Target weights.  Columns are ticker symbols.  Values can be
            negative (short) or > 1 (leveraged).
        """

    def backtest_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Generate weights shifted +1 day for backtest (no look-ahead).

        Convenience wrapper: calls :meth:`generate_weights` and shifts
        the result forward by one bar.
        """
        return self.generate_weights(prices, regime).shift(1).fillna(0.0)
