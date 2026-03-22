"""Alpha158 feature generator inspired by Microsoft Qlib.

Computes 30 battle-tested features from OHLCV data:
  - 9 KBAR features (intraday price relationships)
  - 21 rolling features (7 types x 3 windows: 5, 20, 60)

All features use `.shift(1)` to avoid look-ahead bias -- features
on day T are computed from data up to day T-1.

Usage
-----
>>> from financial_algo.technical.alpha158 import compute_alpha158_features
>>> features = compute_alpha158_features(price_data)  # MultiIndex DataFrame

Or from close-only prices:
>>> from financial_algo.technical.alpha158 import compute_alpha158_from_close
>>> features = compute_alpha158_from_close(close_prices)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Windows used for rolling features
_WINDOWS = (5, 20, 60)


def compute_alpha158_features(
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute Alpha158 features for all tickers from OHLCV data.

    Parameters
    ----------
    price_data:
        dict mapping ticker -> DataFrame with columns
        [Open, High, Low, Close, Volume].

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker) with 30 feature columns.
        Features are shifted +1 day (no look-ahead).
        NaN rows from the warmup period (first 60 rows) are forward-filled
        then filled with 0.
    """
    if not price_data:
        return pd.DataFrame()

    frames = []
    for ticker, ohlcv in price_data.items():
        feat = _compute_single_ticker(ohlcv, ticker)
        if feat is not None:
            frames.append(feat)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames)
    result.sort_index(inplace=True)
    return result


def compute_alpha158_from_close(
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Alpha158 features using only Close prices.

    KBAR features that require Open/High/Low/Volume are approximated:
      - Open  = previous Close (Close.shift(1))
      - High  = rolling(2).max() of Close
      - Low   = rolling(2).min() of Close
      - Volume = 1.0 (constant, so volume-interaction features are neutral)

    This is a convenience wrapper for strategies that only receive
    close prices in ``generate_weights``.

    Parameters
    ----------
    close_prices:
        DataFrame indexed by date, one column per ticker.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker) with 30 feature columns.
    """
    if close_prices.empty:
        return pd.DataFrame()

    price_data: dict[str, pd.DataFrame] = {}
    for ticker in close_prices.columns:
        c = close_prices[ticker]
        ohlcv = pd.DataFrame({
            "Open": c.shift(1).fillna(c.iloc[0]),
            "High": c.rolling(2, min_periods=1).max(),
            "Low": c.rolling(2, min_periods=1).min(),
            "Close": c,
            "Volume": 1.0,
        }, index=c.index)
        price_data[ticker] = ohlcv

    return compute_alpha158_features(price_data)


# ---------------------------------------------------------------------------
# Internal: single-ticker feature computation
# ---------------------------------------------------------------------------

def _compute_single_ticker(
    ohlcv: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame | None:
    """Compute 30 Alpha158 features for one ticker.

    Returns a DataFrame with MultiIndex (date, ticker) and 30 columns,
    or None if the input is too short.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(ohlcv.columns):
        return None
    if len(ohlcv) < 2:
        return None

    o = ohlcv["Open"].astype(float)
    h = ohlcv["High"].astype(float)
    lo = ohlcv["Low"].astype(float)
    c = ohlcv["Close"].astype(float)
    v = ohlcv["Volume"].astype(float).clip(lower=0)

    features: dict[str, pd.Series] = {}

    # =================================================================
    # KBAR features (9) -- intraday price relationships
    # =================================================================
    hl_range = h - lo  # reused

    features["KBAR_1"] = (c - o) / (o + 1e-8)
    features["KBAR_2"] = hl_range / (o + 1e-8)
    features["KBAR_3"] = (c - lo) / (o + 1e-8)
    features["KBAR_4"] = (h - c) / (o + 1e-8)
    features["KBAR_5"] = hl_range / (c + 1e-8)
    features["KBAR_6"] = (c - o) / (hl_range + 1e-8)
    features["KBAR_7"] = (h - o) / (o + 1e-8)
    features["KBAR_8"] = (lo - o) / (o + 1e-8)
    features["KBAR_9"] = hl_range / (o + 1e-8) * v

    # =================================================================
    # Rolling features (7 types x 3 windows = 21)
    # =================================================================
    ret = c.pct_change().fillna(0.0)

    for n in _WINDOWS:
        # ROC_N: rate of change
        roc = c / c.shift(n) - 1
        features[f"ROC_{n}"] = roc.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # MA_RATIO_N: price to moving average ratio
        ma = c.rolling(n, min_periods=max(1, n // 2)).mean()
        ratio = c / (ma + 1e-8)
        features[f"MA_RATIO_{n}"] = ratio.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)

        # STD_N: rolling volatility of returns
        features[f"STD_{n}"] = ret.rolling(
            n, min_periods=max(2, n // 2)
        ).std().fillna(0.0)

        # MAX_N: distance from rolling max
        rmax = c.rolling(n, min_periods=1).max()
        features[f"MAX_{n}"] = (rmax / (c + 1e-8) - 1).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        # MIN_N: distance from rolling min
        rmin = c.rolling(n, min_periods=1).min()
        features[f"MIN_{n}"] = (c / (rmin + 1e-8) - 1).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        # RANK_N: time-series percentile rank
        features[f"RANK_{n}"] = c.rolling(
            n, min_periods=max(2, n // 2)
        ).rank(pct=True).fillna(0.5)

        # RSV_N: stochastic (relative strength value)
        denom = rmax - rmin + 1e-8
        features[f"RSV_{n}"] = ((c - rmin) / denom).clip(0, 1).fillna(0.5)

    # =================================================================
    # Assemble, shift +1 to prevent look-ahead, sanitize
    # =================================================================
    feat_df = pd.DataFrame(features, index=ohlcv.index)

    # Shift all features by 1 day to avoid look-ahead bias
    feat_df = feat_df.shift(1)

    # Replace any remaining inf/nan after warmup
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Add ticker level to index -> MultiIndex (date, ticker)
    feat_df["ticker"] = ticker
    feat_df = feat_df.set_index("ticker", append=True)
    feat_df.index.names = ["date", "ticker"]

    return feat_df


# ---------------------------------------------------------------------------
# Utility: extract single feature as cross-sectional DataFrame
# ---------------------------------------------------------------------------

def pivot_feature(
    features: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    """Pivot a single feature from the MultiIndex format to (date x ticker).

    Parameters
    ----------
    features:
        Output of ``compute_alpha158_features`` with MultiIndex (date, ticker).
    feature_name:
        Column name, e.g. ``"ROC_20"`` or ``"KBAR_6"``.

    Returns
    -------
    pd.DataFrame
        Date-indexed, one column per ticker.
    """
    if feature_name not in features.columns:
        raise KeyError(f"Feature '{feature_name}' not found. "
                        f"Available: {list(features.columns)}")
    return features[feature_name].unstack("ticker")
