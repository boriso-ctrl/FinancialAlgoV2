"""Force Index (FI)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def force_index(close: pd.Series, volume: pd.Series, n: int = 2, fillna: bool = False) -> pd.Series:
    fi = close.diff(n) * volume.diff(n)
    if fillna:
        fi = fi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(fi, name='fi_' + str(n))


def force_index_signal(close: pd.Series, volume: pd.Series, n: int = 2) -> pd.Series:
    fi = force_index(close, volume, n=n, fillna=False)
    raw = np.sign(fi)
    return scale_signal(raw, fi)
