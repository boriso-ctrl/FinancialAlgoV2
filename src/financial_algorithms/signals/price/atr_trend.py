"""ATR-smoothed trend filter producing graded signal based on distance from ATR band."""

from __future__ import annotations

import pandas as pd

from .utils import scale_signal


def atr_trend_signal(close: pd.Series, atr_window: int = 14, k: float = 1.5) -> pd.Series:
    ret = close.diff()
    tr = (close - close.shift(1)).abs()
    atr = tr.rolling(window=atr_window).mean()
    upper = close.rolling(window=atr_window).mean() + k * atr
    lower = close.rolling(window=atr_window).mean() - k * atr
    # Positive when above upper band, negative when below lower band; graded by distance
    raw = pd.Series(0.0, index=close.index)
    raw = raw.where(close.between(lower, upper), (close - upper) / (atr + 1e-9))
    raw = raw.where(close.between(lower, upper), (close - lower) / (atr + 1e-9))
    graded = scale_signal(raw.fillna(0), raw)
    graded.name = 'atr_trend_signal'
    return graded
