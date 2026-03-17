"""Utility helpers for signal scaling."""

import numpy as np
import pandas as pd


def scale_signal(
    raw_signal: pd.Series,
    driver: pd.Series,
    max_abs: float = 5.0,
    min_abs: float = 0.5,
    window: int = 50,
) -> pd.Series:
    """Scale a raw {-1,0,1} style signal by conviction from a driver series.

    - Conviction is based on rolling z-score magnitude of the driver.
    - Magnitudes below ``min_abs`` are zeroed out to avoid noise.
    - Output is clipped to ``[-max_abs, max_abs]``.
    """
    if not isinstance(raw_signal, pd.Series):
        raw_signal = pd.Series(raw_signal, index=driver.index)

    sign = np.sign(raw_signal).fillna(0)

    mean = driver.rolling(window=window, min_periods=window).mean()
    std = driver.rolling(window=window, min_periods=window).std()
    z = (driver - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)

    magnitude = z.abs().clip(upper=max_abs)
    magnitude = magnitude.where(magnitude >= min_abs, other=0)

    # Fallback: if magnitude is NaN but signal exists, use min_abs
    magnitude = magnitude.where(magnitude.notna(), other=min_abs)

    scaled = sign * magnitude
    scaled = scaled.where(sign != 0, 0)
    return scaled.clip(lower=-max_abs, upper=max_abs).fillna(0)
