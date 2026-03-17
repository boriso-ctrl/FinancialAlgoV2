"""Stochastic + MACD crossover strategy."""

import pandas as pd
import ta


def stoch_macd_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with SMACD signals in `SMACD_signal`."""
    df = df.copy()
    df['Stoch%K'] = ta.momentum.stoch(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stoch%D'] = ta.momentum.stoch_signal(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['KminusD'] = df['Stoch%K'] - df['Stoch%D']

    df['MACD'] = ta.trend.macd(pd.Series(df['Close']), n_fast=12, n_slow=26)
    df['MACD_signal_line'] = ta.trend.macd_signal(pd.Series(df['Close']), n_fast=12, n_slow=26, n_sign=9)

    smacd_signal = [0] * len(df)
    macds = list(df['MACD'] - df['MACD_signal_line'])
    kminusd = list(df['KminusD'])
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if kminusd[i] > 0:
            if macds[i] > 0:
                smacd_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while sl < pend_ret and macds[count] > 0 and count < len(df) - 1:
                        smacd_signal[count] = 1
                        count += 1
                        new_close = close[count]
                        pend_ret = (new_close - cur_close) / cur_close
                    if count == len(df) - 1 and macds[count] > 0 and sl < pend_ret:
                        smacd_signal[count] = 1
                i = count
            else:
                i += 1
        elif kminusd[i] < 0:
            if macds[i] < 0:
                smacd_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and macds[count] < 0 and count < len(df) - 1:
                        smacd_signal[count] = -1
                        count += 1
                        new_close = close[count]
                        pend_ret = (-1) * (new_close - cur_close) / cur_close
                    if count == len(df) - 1 and macds[count] < 0 and sl < pend_ret:
                        smacd_signal[count] = -1
                i = count
            else:
                i += 1
        else:
            i += 1

    df['SMACD_signal'] = smacd_signal
    return df
