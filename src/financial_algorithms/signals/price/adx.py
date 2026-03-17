"""ADX-based trend strategy (self-contained)."""

import numpy as np
import pandas as pd

from .utils import scale_signal


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


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


def adx_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with ADX signals in `ADX_signal`."""
    df = df.copy()
    df['ADX'] = _adx(df['High'], df['Low'], df['Close'], window=14)

    adx_signal = [0] * len(df)
    adx = list(df['ADX'])
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if df['Trend'].iloc[i] == 'Uptrend':
            if adx[i] > 25:
                adx_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while sl < pend_ret and adx[count] > 20 and count < len(df) - 1:
                        adx_signal[count] = 1
                        count += 1
                        new_close = close[count]
                        pend_ret = (new_close - cur_close) / cur_close
                    if count == len(df) - 1 and adx[count] > 20 and sl < pend_ret:
                        adx_signal[count] = 1
                i = count
            else:
                i += 1
        elif df['Trend'].iloc[i] == 'Downtrend':
            if adx[i] > 25:
                adx_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and adx[count] > 25 and count < len(df) - 1:
                        adx_signal[count] = -1
                        count += 1
                        new_close = close[count]
                        pend_ret = (-1) * (new_close - cur_close) / cur_close
                    if count == len(df) - 1 and adx[count] > 25 and sl < pend_ret:
                        adx_signal[count] = -1
                i = count
            else:
                i += 1
        else:
            i += 1

    df['ADX_signal'] = scale_signal(pd.Series(adx_signal, index=df.index), df['ADX'])
    return df
