"""CCI + ADX combined strategy (self-contained)."""

import numpy as np
import pandas as pd

from .adx import _true_range
from .utils import scale_signal


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, c: float = 0.015) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=window).mean()
    mean_dev = (tp - sma_tp).abs().rolling(window=window).mean()
    denom = c * mean_dev
    return (tp - sma_tp) / denom.replace(0, np.nan)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = pd.Series(tr).rolling(window=window).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=window).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(window=window).mean() / tr_smooth
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    return dx.rolling(window=window).mean()


def cci_adx_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with CDX signals in `CDX_signal`."""
    df = df.copy()
    df['CCI'] = _cci(df['High'], df['Low'], df['Close'], window=20, c=0.015)
    df['ADX'] = _adx(df['High'], df['Low'], df['Close'], window=14)

    cdx_signal = [0] * len(df)
    cci = list(df['CCI'])
    adx = list(df['ADX'])
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if adx[i] < 25:
            if df['Trend'][i] != 'Downtrend':
                if cci[i] < 100 and cci[i - 1] > 100:
                    cdx_signal[i] = 1
                    count = i + 1
                    if count < len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        pend_ret = (new_close - cur_close) / cur_close
                        while sl < pend_ret and cci[count] > -100 and count < len(df) - 1:
                            cdx_signal[count] = 1
                            count += 1
                            new_close = close[count]
                            pend_ret = (new_close - cur_close) / cur_close
                        if count == len(df) - 1 and cci[count] > -100 and sl < pend_ret:
                            cdx_signal[count] = 1
                    i = count
                else:
                    i += 1
            else:
                if cci[i] > -100 and cci[i - 1] < -100:
                    cdx_signal[i] = -1
                    count = i + 1
                    if count < len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        pend_ret = (-1) * (new_close - cur_close) / cur_close
                        while sl < pend_ret and cci[count] < 100 and count < len(df) - 1:
                            cdx_signal[count] = -1
                            count += 1
                            new_close = close[count]
                            pend_ret = (-1) * (new_close - cur_close) / cur_close
                        if count == len(df) - 1 and cci[count] < 100 and sl < pend_ret:
                            cdx_signal[count] = -1
                    i = count
                else:
                    i += 1
        else:
            i += 1

    df['CDX_signal'] = scale_signal(pd.Series(cdx_signal, index=df.index), df['CCI'])
    return df
