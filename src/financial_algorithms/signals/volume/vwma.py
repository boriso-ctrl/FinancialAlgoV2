"""Volume-Weighted Moving Average (VWAP-style)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def volume_weighted_moving_average(close: pd.Series, volume: pd.Series, n: int = 20, fillna: bool = False) -> pd.Series:
    vwma = (close * volume).rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum()
    if fillna:
        vwma = vwma.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vwma, name='VWAP')


def vwma_signal(close: pd.Series, volume: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_vwma = volume_weighted_moving_average(close, volume, n=fast)
    slow_vwma = volume_weighted_moving_average(close, volume, n=slow)
    spread = fast_vwma - slow_vwma
    raw = np.sign(spread)
    return scale_signal(raw, spread)
