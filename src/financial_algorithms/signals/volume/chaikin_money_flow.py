"""Chaikin Money Flow (CMF)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20, fillna: bool = False) -> pd.Series:
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum()
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cmf, name='cmf')


def chaikin_money_flow_signal(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    cmf = chaikin_money_flow(high, low, close, volume, n=n, fillna=False)
    raw = np.sign(cmf)
    return scale_signal(raw, cmf)
