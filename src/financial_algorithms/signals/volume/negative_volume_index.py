"""Negative Volume Index (NVI)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def negative_volume_index(close: pd.Series, volume: pd.Series, fillna: bool = False) -> pd.Series:
    price_change = close.pct_change()
    vol_decrease = (volume.shift(1) > volume)

    nvi = pd.Series(data=np.nan, index=close.index, dtype='float64', name='nvi')
    nvi.iloc[0] = 1000
    for i in range(1, len(nvi)):
        if vol_decrease.iloc[i]:
            nvi.iloc[i] = nvi.iloc[i - 1] * (1.0 + price_change.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]

    if fillna:
        nvi = nvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

    return pd.Series(nvi, name='nvi')


def negative_volume_index_signal(close: pd.Series, volume: pd.Series) -> pd.Series:
    nvi = negative_volume_index(close, volume, fillna=False)
    raw = np.sign(close.pct_change().fillna(0))
    return scale_signal(raw, nvi)
