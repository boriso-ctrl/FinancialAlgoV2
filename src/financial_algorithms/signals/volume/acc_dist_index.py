"""Accumulation/Distribution Index (ADI)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def acc_dist_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fillna: bool = False) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0)
    ad = (clv * volume).cumsum()
    if fillna:
        ad = ad.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ad, name='adi')


def acc_dist_signal(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    adi = acc_dist_index(high, low, close, volume, fillna=False)
    raw = np.sign(close.diff().fillna(0))
    return scale_signal(raw, adi)
