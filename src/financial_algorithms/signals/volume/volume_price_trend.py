"""Volume-Price Trend (VPT)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def volume_price_trend(close: pd.Series, volume: pd.Series, fillna: bool = False) -> pd.Series:
    ret = close.pct_change()
    vpt = (volume * ret).cumsum()
    if fillna:
        vpt = vpt.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vpt, name='vpt')


def volume_price_trend_signal(close: pd.Series, volume: pd.Series) -> pd.Series:
    vpt = volume_price_trend(close, volume, fillna=False)
    raw = np.sign(close.pct_change().fillna(0))
    return scale_signal(raw, vpt)
