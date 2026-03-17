"""RSI + OBV + Bollinger Bands strategy (self-contained)."""

import numpy as np
import pandas as pd

from .rsi import _rsi
from .bb_rsi import _bollinger_bands
from .utils import scale_signal


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (volume * direction).cumsum()


def rsi_obv_bb_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with ROB signals in `ROB_signal`."""
    df = df.copy()
    df['RSI'] = _rsi(df['Close'], period=14)
    df['MRSI'] = df['RSI'].rolling(window=4).mean()
    df['OBV'] = _obv(df['Close'], df['Volume'])
    df['BBHigh'], df['BBLow'] = _bollinger_bands(df['Close'], window=20, ndev=2)
    df['%B'] = (df['Close'] - df['BBLow']) / ((df['BBHigh'] - df['BBLow']))

    rob_signal = [0] * len(df)
    obv = list(df['OBV'])
    rsi = list(df['RSI'])
    mrsi = list(df['MRSI'])
    B = list(df['%B'])
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if df['Trend'].iloc[i] == 'Uptrend':
            if B[i] > 0.5:
                if rsi[i] >= 50 and mrsi[i] > mrsi[i - 1] and mrsi[i - 1] > mrsi[i - 2]:
                    if (obv[i] - obv[i - 1]) / obv[i - 1] > 5e-3:
                        rob_signal[i] = 1
                        count = i + 1
                        if count < len(df):
                            cur_close = close[i]
                            new_close = close[count]
                            pend_ret = (new_close - cur_close) / cur_close
                            while sl < pend_ret and B[count] < 1 and count < len(df) - 1:
                                rob_signal[count] = 1
                                count += 1
                                new_close = close[count]
                                pend_ret = (new_close - cur_close) / cur_close
                            if count == len(df) - 1 and B[count] < 1 and sl < pend_ret:
                                rob_signal[count] = 1
                        i = count
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        elif df['Trend'].iloc[i] == 'Downtrend':
            if B[i] < 0.5:
                if rsi[i] <= 50 and mrsi[i] < mrsi[i - 1] and mrsi[i - 1] < mrsi[i - 2]:
                    if (obv[i] - obv[i - 1]) / obv[i - 1] < -5e-3:
                        rob_signal[i] = -1
                        count = i + 1
                        if count < len(df):
                            cur_close = close[i]
                            new_close = close[count]
                            pend_ret = (-1) * (new_close - cur_close) / cur_close
                            while sl < pend_ret and B[count] > 0 and count < len(df) - 1:
                                rob_signal[count] = -1
                                count += 1
                                new_close = close[count]
                                pend_ret = (-1) * (new_close - cur_close) / cur_close
                            if count == len(df) - 1 and B[count] > 0 and sl < pend_ret:
                                rob_signal[count] = -1
                        i = count
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    df['ROB_signal'] = scale_signal(pd.Series(rob_signal, index=df.index), df['RSI'])
    return df
