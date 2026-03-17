"""Parabolic SAR + Stochastic oscillator strategy with local PSAR fallback."""

import pandas as pd


def _parabolic_sar(high: pd.Series, low: pd.Series, af: float = 0.02, amax: float = 0.2) -> pd.Series:
    """Compute Parabolic SAR without external deps.

    Simplified standard implementation; returns a Series aligned to high/low.
    """
    if len(high) == 0:
        return pd.Series(dtype=float)

    af_step = af
    trend_up = True
    ep = high.iloc[0]
    sar = low.iloc[0]
    out = [sar]

    for i in range(1, len(high)):
        prev_sar = sar
        if trend_up:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, low.iloc[i - 1], low.iloc[i - 2] if i > 1 else low.iloc[i - 1])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, amax)
            if low.iloc[i] < sar:
                trend_up = False
                sar = ep
                ep = low.iloc[i]
                af = af_step
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, high.iloc[i - 1], high.iloc[i - 2] if i > 1 else high.iloc[i - 1])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, amax)
            if high.iloc[i] > sar:
                trend_up = True
                sar = ep
                ep = high.iloc[i]
                af = af_step
        out.append(sar)

    return pd.Series(out, index=high.index, name='SAR')


def sar_stoch_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate DataFrame with combined SAR/Stochastic signals in `SS_signal`."""
    df = df.copy()
    df['SAR'] = _parabolic_sar(df['High'], df['Low'], af=0.02, amax=0.2)
    # Simple Stoch %K/%D
    low_roll = df['Low'].rolling(window=14)
    high_roll = df['High'].rolling(window=14)
    lowest_low = low_roll.min()
    highest_high = high_roll.max()
    denom = (highest_high - lowest_low).replace(0, pd.NA)
    df['Stoch%K'] = (df['Close'] - lowest_low) / denom * 100
    df['Stoch%D'] = df['Stoch%K'].rolling(window=3).mean()
    df['KminusD'] = df['Stoch%K'] - df['Stoch%D']

    n = len(df)
    if n == 0:
        df['SS_signal'] = []
        return df

    sar = df['SAR'].to_numpy()
    stoch = df['Stoch%K'].to_numpy()
    kminusd = df['KminusD'].to_numpy()
    close = df['Close'].to_numpy()
    trend = df['Trend'].to_numpy()
    ss_signal = [0] * n

    if n <= 52:
        df['SS_signal'] = ss_signal
        return df

    i = 51  # skip warmup rows required by SAR/Stoch windows
    while i < n:
        tr = trend[i]

        if tr == 'Uptrend':
            if sar[i] < close[i]:
                long_trigger = (
                    (stoch[i] >= 20 and stoch[i - 1] <= 20)
                    or (kminusd[i] > 0 and kminusd[i - 1] < 0)
                )
                if long_trigger:
                    ss_signal[i] = 1
                    cur_close = close[i]
                    count = i + 1
                    while count < n - 1:
                        if sar[count] >= close[count]:
                            break
                        slsar = (sar[count] - cur_close) / cur_close
                        pend_ret = (close[count] - cur_close) / cur_close
                        if slsar >= pend_ret:
                            break
                        ss_signal[count] = 1
                        count += 1
                    if count == n - 1 and sar[count] < close[count]:
                        slsar_last = (sar[count] - cur_close) / cur_close
                        pend_last = (close[count] - cur_close) / cur_close
                        if slsar_last < pend_last:
                            ss_signal[count] = 1
                    i = max(count, i + 1)
                else:
                    i += 1
            else:
                i += 1
        elif tr == 'Downtrend':
            if sar[i] > close[i]:
                short_trigger = (
                    (stoch[i] <= 80 and stoch[i - 1] >= 80)
                    or (kminusd[i] < 0 and kminusd[i - 1] > 0)
                )
                if short_trigger:
                    ss_signal[i] = -1
                    cur_close = close[i]
                    count = i + 1
                    while count < n - 1:
                        if sar[count] <= close[count]:
                            break
                        slsar = -1 * (sar[count] - cur_close) / cur_close
                        pend_ret = -1 * (close[count] - cur_close) / cur_close
                        if slsar >= pend_ret:
                            break
                        ss_signal[count] = -1
                        count += 1
                    if count == n - 1 and sar[count] > close[count]:
                        slsar_last = -1 * (sar[count] - cur_close) / cur_close
                        pend_last = -1 * (close[count] - cur_close) / cur_close
                        if slsar_last < pend_last:
                            ss_signal[count] = -1
                    i = max(count, i + 1)
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    df['SS_signal'] = ss_signal
    return df
