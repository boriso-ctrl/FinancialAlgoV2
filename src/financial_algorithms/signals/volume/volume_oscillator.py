"""Volume Oscillator."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def volume_oscillator(volume: pd.Series, s: int = 9, l: int = 26, fillna: bool = False) -> pd.Series:
    emas = volume.ewm(span=s, adjust=False, min_periods=s).mean()
    emal = volume.ewm(span=l, adjust=False, min_periods=l).mean()
    vo = (emas - emal) * 100 / emal
    if fillna:
        vo = vo.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vo, name='VO')


def volume_oscillator_signal(volume: pd.Series, s: int = 9, l: int = 26) -> pd.Series:
    vo = volume_oscillator(volume, s=s, l=l, fillna=False)
    raw = np.sign(vo)
    return scale_signal(raw, vo)
