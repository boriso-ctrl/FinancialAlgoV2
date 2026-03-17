"""RSI range strategy with local RSI implementation (no ta dependency)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with RSI signals in `RSI_signal`."""
    df = df.copy()
    df['RSI'] = _rsi(df['Close'], period=14)

    rsi_signal = [0] * len(df)
    rsi = list(df['RSI'])
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if df['Trend'].iloc[i] == 'Range':
            if rsi[i] < 30:
                rsi_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while sl < pend_ret and rsi[count] < 70 and count < len(df) - 1:
                        x = (new_close - cur_close) / cur_close
                        rsi_signal[count] = 1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        pend_ret += x
                    if count == len(df) - 1 and rsi[count] < 70 and sl < pend_ret:
                        rsi_signal[count] = 1
                i = count
            elif rsi[i] > 70:
                rsi_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and rsi[count] > 30 and count < len(df) - 1:
                        x = (-1) * (new_close - cur_close) / cur_close
                        rsi_signal[count] = -1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        pend_ret += x
                    if count == len(df) - 1 and rsi[count] > 30 and sl < pend_ret:
                        rsi_signal[count] = -1
                i = count
            else:
                i += 1
        else:
            i += 1

    df['RSI_signal'] = scale_signal(pd.Series(rsi_signal, index=df.index), df['RSI'])
    return df
