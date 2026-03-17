"""Simple SMA crossover signal for quick experiments/registry defaults."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma_signal(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    signal = np.sign(sma_fast - sma_slow)
    return signal.fillna(0)
