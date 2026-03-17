"""MACD signal with graded output.

Returns both MACD line and a graded signal based on MACD - signal line.
"""

from __future__ import annotations

import pandas as pd

from .utils import scale_signal


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    raw = hist
    graded = scale_signal(raw, hist)
    graded.name = 'macd_signal'
    return graded
