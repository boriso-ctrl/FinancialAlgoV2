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

    Intraday features (Phase 2)
    ---------------------------
    Strategies that can consume Alpaca-derived intraday features should
    check ``self._intraday_features`` inside ``generate_weights``.  The
    feature DataFrame is injected via :meth:`set_intraday_features` before
    calling ``generate_weights``/``backtest_weights``.
    """

    name: str = "BaseStrategy"

    # Class-level default — instances shadow this when set_intraday_features
    # is called.  Subclasses do NOT need to call super().__init__().
    _intraday_features: "pd.DataFrame | None" = None

    def __init__(self) -> None:
        # Initialize instance-level attribute so each instance has its own slot.
        # Subclasses that define __init__ and DON'T call super().__init__() will
        # fall back to the class-level None above, which is safe.
        self._intraday_features = None

    def set_intraday_features(self, features: pd.DataFrame | None) -> None:
        """Inject precomputed intraday feature DataFrame.

        Parameters
        ----------
        features:
            Daily date-indexed DataFrame with columns
            ``"{ticker}_{feature_name}"`` as produced by
            ``FeatureStore.load()``.  Pass ``None`` to clear.
        """
        self._intraday_features = features

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

