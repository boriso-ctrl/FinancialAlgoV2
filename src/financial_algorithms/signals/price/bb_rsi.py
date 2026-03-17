"""Bollinger Bands + RSI strategy (self-contained implementation)."""

import numpy as np
import pandas as pd

from .rsi import _rsi
from .utils import scale_signal


def _bollinger_bands(series: pd.Series, window: int = 20, ndev: float = 2.0):
    ma = series.rolling(window=window).mean()
    sd = series.rolling(window=window).std()
    upper = ma + ndev * sd
    lower = ma - ndev * sd
    return upper, lower


def bb_rsi_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with BBRSI signals in `BBRSI_signal`."""
    df = df.copy()
    df['BBHigh'], df['BBLow'] = _bollinger_bands(df['Close'], window=20, ndev=2)
    df['%B'] = (df['Close'] - df['BBLow']) / ((df['BBHigh'] - df['BBLow']))
    df['RSI'] = _rsi(df['Close'], period=14)

    rsi = list(df['RSI'])
    B = list(df['%B'])
    bbrsi_signal = [0] * len(df)
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if B[i] > 0 and B[i - 1] < 0:
            bbrsi_signal[i] = 1
            count = i + 1
            if count < len(df):
                cur_close = close[count - 1]
                new_close = close[count]
                pend_ret = (new_close - cur_close) / cur_close
                while sl < pend_ret and B[count] < 1 and count < len(df) - 1:
                    bbrsi_signal[count] = 1
                    count += 1
                    cur_close = close[count - 1]
                    new_close = close[count]
                    x = (new_close - cur_close) / cur_close
                    pend_ret += x
                if count == len(df) - 1 and sl < pend_ret:
                    bbrsi_signal[count] = 1
            i = count
        elif B[i] < 0.2:
            if rsi[i] <= 50:
                bbrsi_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while sl < pend_ret and B[count] < 0.8 and count < len(df) - 1:
                        bbrsi_signal[count] = 1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (new_close - cur_close) / cur_close
                        pend_ret += x
                    if count == len(df) - 1 and sl < pend_ret:
                        bbrsi_signal[count] = 1
                i = count
            else:
                i += 1
        elif B[i] > 0.8:
            if rsi[i] >= 50:
                bbrsi_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and B[count] > 0 and count < len(df) - 1:
                        bbrsi_signal[count] = -1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (-1) * (new_close - cur_close) / cur_close
                        pend_ret += x
                    if count == len(df) - 1 and sl < pend_ret:
                        bbrsi_signal[count] = -1
                i = count
            else:
                i += 1
        else:
            i += 1

    df['BBRSI_signal'] = scale_signal(pd.Series(bbrsi_signal, index=df.index), df['%B'])
    return df
