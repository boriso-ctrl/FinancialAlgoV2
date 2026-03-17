"""Williams %R strategy (self-contained)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, lbp: int = 14) -> pd.Series:
    highest_high = high.rolling(window=lbp).max()
    lowest_low = low.rolling(window=lbp).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / denom


def wr_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with WR signals in `WR_signal`."""
    df = df.copy()
    df['WR'] = _williams_r(df['High'], df['Low'], df['Close'], lbp=14)

    wr_signal = [0] * len(df)
    wr = list(df['WR'])
    mwr = list(df['WR'].pct_change().rolling(window=4).mean())
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if wr[i] > -50 and wr[i - 1] < -50 and mwr[i] > 0:
            wr_signal[i] = 1
            count = i + 1
            if count < len(df):
                cur_close = close[i]
                new_close = close[count]
                pend_ret = (new_close - cur_close) / cur_close
                while sl < pend_ret and wr[count] < -20 and count < len(df) - 1:
                    wr_signal[count] = 1
                    count += 1
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                if count == len(df) - 1 and wr[count] < -20 and sl < pend_ret:
                    wr_signal[count] = 1
            i = count
        elif wr[i] < -50 and wr[i - 1] > -50 and mwr[i] < 0:
            wr_signal[i] = -1
            count = i + 1
            if count < len(df):
                cur_close = close[i]
                new_close = close[count]
                pend_ret = (-1) * (new_close - cur_close) / cur_close
                while sl < pend_ret and wr[count] > -80 and count < len(df) - 1:
                    wr_signal[count] = -1
                    count += 1
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                if count == len(df) - 1 and wr[count] > -80 and sl < pend_ret:
                    wr_signal[count] = -1
            i = count
        else:
            i += 1

    df['WR_signal'] = scale_signal(pd.Series(wr_signal, index=df.index), df['WR'])
    return df
