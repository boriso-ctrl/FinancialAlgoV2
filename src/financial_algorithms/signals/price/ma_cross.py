"""Moving-average crossover strategy with graded conviction."""

import numpy as np
import pandas as pd


def ma_cross_strategy(df: pd.DataFrame, sl: float, n1: int = 20, n2: int = 50) -> pd.DataFrame:
    """Annotate DataFrame with MA signals in `MA_signal` column.

    Conviction is proportional to the z-score of the MA spread, clipped to ±5.
    Shorts enabled when fast MA is sufficiently below slow MA.
    """
    df = df.copy()
    df['MA_fast'] = df['Close'].rolling(window=n1, min_periods=n1).mean()
    df['MA_slow'] = df['Close'].rolling(window=n2, min_periods=n2).mean()
    spread = df['MA_fast'] - df['MA_slow']

    # Use rolling std of spread to size conviction; fall back to 1 to avoid div-by-zero
    spread_std = spread.rolling(window=max(n1, n2), min_periods=max(n1, n2)).std().replace(0, np.nan)
    z = spread / spread_std
    z = z.replace([np.inf, -np.inf], np.nan)

    # Clip conviction to [-5, 5]
    signal = z.clip(lower=-5, upper=5)

    # Require a minimum absolute z before entering
    signal = signal.where(signal.abs() >= 0.5, other=0)

    # No dependence on Trend column; purely price-based
    df['MA_signal'] = signal.fillna(0)
    return df
