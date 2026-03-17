"""Strategy that blends every available indicator into one signal."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from financial_algorithms.backtest import blend_signals
from financial_algorithms.signals.price.bb_rsi import bb_rsi_strategy
from financial_algorithms.signals.price.sar_stoch import sar_stoch_strategy
from financial_algorithms.signals.price.vwsma import vwsma_strategy
from financial_algorithms.signals.price.williams_r import wr_strategy
from financial_algorithms.signals.price.ma_cross import ma_cross_strategy
from financial_algorithms.signals.price.rsi import rsi_strategy
from financial_algorithms.signals.price.stoch_macd import stoch_macd_strategy
from financial_algorithms.signals.price.rsi_obv_bb import rsi_obv_bb_strategy
from financial_algorithms.signals.price.adx import adx_strategy
from financial_algorithms.signals.price.cci_adx import cci_adx_strategy
from financial_algorithms.signals.price.macd import macd_signal
from financial_algorithms.signals.price.atr_trend import atr_trend_signal
from financial_algorithms.signals.volume.volume_oscillator import volume_oscillator_signal
from financial_algorithms.signals.volume.put_call_ratio import put_call_ratio
from financial_algorithms.signals.volume.on_balance_volume import on_balance_volume
from financial_algorithms.signals.volume.volume_price_trend import volume_price_trend_signal
from financial_algorithms.signals.volume.acc_dist_index import acc_dist_signal
from financial_algorithms.signals.volume.chaikin_money_flow import chaikin_money_flow_signal
from financial_algorithms.signals.volume.force_index import force_index_signal
from financial_algorithms.signals.volume.ease_of_movement import ease_of_movement_signal
from financial_algorithms.signals.volume.vwma import vwma_signal
from financial_algorithms.signals.volume.negative_volume_index import negative_volume_index_signal
DEFAULT_ALL_WEIGHTS: Dict[str, float] = {
    "ma_cross": 1.0,
    "sar_stoch": 1.0,
    "stoch_macd": 1.0,
    "rsi": 1.0,
    "bb_rsi": 1.0,
    "rsi_obv_bb": 1.0,
    "adx": 1.0,
    "cci_adx": 1.0,
    "williams_r": 1.0,
    "vwsma": 1.0,
    "macd": 1.0,
    "atr_trend": 1.0,
    "volume_osc": 1.0,
    "obv": 1.0,
    "vpt": 1.0,
    "acc_dist": 1.0,
    "cmf": 1.0,
    "force_index": 1.0,
    "eom": 1.0,
    "vwma": 1.0,
    "nvi": 1.0,
    "put_call_ratio": 0.0,  # keep neutral until real data is wired
}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close" not in out.columns:
        out["Close"] = out.iloc[:, 0]
    if "High" not in out.columns:
        out["High"] = out["Close"] * 1.001
    if "Low" not in out.columns:
        out["Low"] = out["Close"] * 0.999
    if "Volume" not in out.columns:
        out["Volume"] = 1_000_000
    if "Trend" not in out.columns:
        out["Trend"] = "Uptrend"
    return out


def _make_signal_df(series: pd.Series, name: str, tickers: List[str]) -> pd.DataFrame:
    s = pd.Series(series, index=series.index).rename(name).fillna(0)
    if tickers:
        df = pd.concat([s] * len(tickers), axis=1)
        df.columns = tickers
        return df
    return s.to_frame(name="signal")


def all_indicator_combo(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    max_signal_abs: float = 5.0,
) -> pd.DataFrame:
    """Blend every available indicator signal into one conviction DataFrame.

    Args:
        df: OHLCV-like DataFrame. If only prices are supplied, Close/High/Low/Volume/Trend
            columns are fabricated so indicators can run.
        weights: Optional per-component overrides; any omitted key falls back to defaults.
        max_signal_abs: Clip final conviction magnitude.

    Returns:
        DataFrame of blended signals aligned to the input index and broadcast across tickers.
    """

    tickers = [c for c in df.columns if c not in {"Open", "Close", "High", "Low", "Volume", "Trend"}]
    price_df = _ensure_ohlcv(df)
    close = price_df["Close"]
    high = price_df["High"]
    low = price_df["Low"]
    volume = price_df["Volume"]

    signals: Dict[str, pd.DataFrame] = {}

    def add_signal(name: str, series: Optional[pd.Series]):
        if series is None:
            return
        signals[name] = _make_signal_df(series, name, tickers)

    # Price-based strategies
    try:
        add_signal("ma_cross", ma_cross_strategy(price_df.copy(), sl=0.0)["MA_signal"])
    except Exception:
        pass
    try:
        add_signal("sar_stoch", sar_stoch_strategy(price_df.copy())["SS_signal"])
    except Exception:
        pass
    try:
        add_signal("stoch_macd", stoch_macd_strategy(price_df.copy(), sl=0.0)["SMACD_signal"])
    except Exception:
        pass
    try:
        add_signal("rsi", rsi_strategy(price_df.copy(), sl=0.0)["RSI_signal"])
    except Exception:
        pass
    try:
        add_signal("bb_rsi", bb_rsi_strategy(price_df.copy(), sl=0.0)["BBRSI_signal"])
    except Exception:
        pass
    try:
        add_signal("rsi_obv_bb", rsi_obv_bb_strategy(price_df.copy(), sl=0.0)["ROB_signal"])
    except Exception:
        pass
    try:
        add_signal("adx", adx_strategy(price_df.copy(), sl=0.0)["ADX_signal"])
    except Exception:
        pass
    try:
        add_signal("cci_adx", cci_adx_strategy(price_df.copy(), sl=0.0)["CDX_signal"])
    except Exception:
        pass
    try:
        add_signal("williams_r", wr_strategy(price_df.copy(), sl=0.0)["WR_signal"])
    except Exception:
        pass
    try:
        add_signal("vwsma", vwsma_strategy(price_df.copy(), sl=0.0)["VWSMA_signal"])
    except Exception:
        pass
    try:
        add_signal("macd", macd_signal(close))
    except Exception:
        pass
    try:
        add_signal("atr_trend", atr_trend_signal(close))
    except Exception:
        pass

    # Volume-based strategies
    try:
        add_signal("volume_osc", volume_oscillator_signal(volume))
    except Exception:
        pass
    try:
        add_signal("obv", on_balance_volume(close, volume))
    except Exception:
        pass
    try:
        add_signal("vpt", volume_price_trend_signal(close, volume))
    except Exception:
        pass
    try:
        add_signal("acc_dist", acc_dist_signal(high, low, close, volume))
    except Exception:
        pass
    try:
        add_signal("cmf", chaikin_money_flow_signal(high, low, close, volume))
    except Exception:
        pass
    try:
        add_signal("force_index", force_index_signal(close, volume))
    except Exception:
        pass
    try:
        add_signal("eom", ease_of_movement_signal(high, low, close, volume))
    except Exception:
        pass
    try:
        add_signal("vwma", vwma_signal(close, volume))
    except Exception:
        pass
    try:
        add_signal("nvi", negative_volume_index_signal(close, volume))
    except Exception:
        pass
    try:
        pcr = put_call_ratio()
        if pcr is not None:
            add_signal("put_call_ratio", pcr)
    except Exception:
        pass

    if not signals:
        raise ValueError("No signals could be computed; check input data.")

    base_weights = {**DEFAULT_ALL_WEIGHTS, **(weights or {})}
    used_weights = {name: base_weights.get(name, 1.0) for name in signals.keys()}
    blended = blend_signals(signals, used_weights, max_signal_abs=max_signal_abs)

    if tickers:
        blended.columns = tickers
    else:
        blended.columns = ["all_indicator_combo"]

    return blended
