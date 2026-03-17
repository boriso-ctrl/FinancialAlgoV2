"""Composite strategy blending BBRSI, SAR-Stoch, VWSMA, Williams %R, and volume signals."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from financial_algorithms.backtest import blend_signals
from financial_algorithms.signals.price.bb_rsi import bb_rsi_strategy
from financial_algorithms.signals.price.sar_stoch import sar_stoch_strategy
from financial_algorithms.signals.price.vwsma import vwsma_strategy
from financial_algorithms.signals.price.williams_r import wr_strategy
from financial_algorithms.signals.price.macd import macd_signal
from financial_algorithms.signals.price.atr_trend import atr_trend_signal
from financial_algorithms.signals.volume.volume_oscillator import volume_oscillator_signal
from financial_algorithms.signals.volume.put_call_ratio import put_call_ratio
from financial_algorithms.signals.volume.on_balance_volume import on_balance_volume
from financial_algorithms.signals.volume.volume_price_trend import volume_price_trend_signal


DEFAULT_WEIGHTS: Dict[str, float] = {
    'bbrsi': 2.0,
    'sar_stoch': 1.5,
    'vwsma': 1.0,
    'williams_r': 1.0,
    'macd': 1.0,
    'atr_trend': 1.0,
    'volume_osc': 1.0,
    'put_call_ratio': 0.5,  # placeholder, will be zero until real data wired
    'obv': 1.0,
    'vpt': 1.0,
}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'Close' not in out.columns:
        out['Close'] = out.iloc[:, 0]
    # If High/Low missing, derive from Close with tiny spread to satisfy indicators
    if 'High' not in out.columns:
        out['High'] = out['Close'] * 1.001
    if 'Low' not in out.columns:
        out['Low'] = out['Close'] * 0.999
    if 'Volume' not in out.columns:
        out['Volume'] = 1_000_000
    if 'Trend' not in out.columns:
        out['Trend'] = 'Uptrend'
    return out


def _to_df(series: pd.Series, name: str) -> pd.DataFrame:
    s = series.rename(name)
    df = s.to_frame()
    df.columns = ['signal']
    return df


def multi_factor_combo(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    max_signal_abs: float = 5.0,
) -> pd.DataFrame:
    """Blend multiple indicator signals into a single conviction score.

    Args:
        df: OHLCV DataFrame with columns Close, High, Low, Volume (Trend optional).
        weights: optional overrides per component.
        max_signal_abs: clip final conviction to this absolute value.

    Returns:
        DataFrame with column `combo_signal` aligned to df.index.
    """
    tickers = [c for c in df.columns if c not in {'Open', 'Close', 'High', 'Low', 'Volume', 'Trend'}]
    wts = {**DEFAULT_WEIGHTS, **(weights or {})}
    signals = {}

    price_df = _ensure_ohlcv(df)

    # Price-based components
    try:
        bbrsi = bb_rsi_strategy(price_df.copy(), sl=0.0)['BBRSI_signal']
        signals['bbrsi'] = _to_df(bbrsi, 'bbrsi')
    except Exception:
        pass

    try:
        sar = sar_stoch_strategy(price_df.copy())['SS_signal']
        signals['sar_stoch'] = _to_df(sar, 'sar_stoch')
    except Exception:
        pass

    try:
        vwsma = vwsma_strategy(price_df.copy(), sl=0.0)['VWSMA_signal']
        signals['vwsma'] = _to_df(vwsma, 'vwsma')
    except Exception:
        pass

    try:
        wr = wr_strategy(price_df.copy(), sl=0.0)['WR_signal']
        signals['williams_r'] = _to_df(wr, 'williams_r')
    except Exception:
        pass

    try:
        macd = macd_signal(price_df['Close'])
        signals['macd'] = _to_df(macd, 'macd')
    except Exception:
        pass

    try:
        atrt = atr_trend_signal(price_df['Close'])
        signals['atr_trend'] = _to_df(atrt, 'atr_trend')
    except Exception:
        pass

    # Volume-based components (require Volume column)
    volume = price_df['Volume']
    close = price_df['Close']
    try:
        vo = volume_oscillator_signal(volume)
        signals['volume_osc'] = _to_df(vo, 'volume_osc')
    except Exception:
        pass

    # Placeholder PCR: currently returns None; skip if not implemented
    try:
        pcr_val = put_call_ratio()
        if pcr_val is not None:
            signals['put_call_ratio'] = _to_df(pcr_val, 'put_call_ratio')
    except Exception:
        pass

    try:
        obv = on_balance_volume(close, volume)
        signals['obv'] = _to_df(obv, 'obv')
    except Exception:
        pass

    try:
        vpt = volume_price_trend_signal(close, volume)
        signals['vpt'] = _to_df(vpt, 'vpt')
    except Exception:
        pass

    if not signals:
        raise ValueError("No signals could be computed; check input columns.")

    used_weights = {name: wts.get(name, 1.0) for name in signals.keys()}
    blended = blend_signals(signals, used_weights, max_signal_abs=max_signal_abs)

    # Broadcast to tickers if provided; otherwise keep single column
    if tickers:
        blended = pd.concat([blended.iloc[:, 0]] * len(tickers), axis=1)
        blended.columns = tickers
    else:
        blended.columns = ['combo_signal']

    return blended
