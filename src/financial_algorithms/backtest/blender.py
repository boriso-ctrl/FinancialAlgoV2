"""Weighted blending of component signals."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def blend_signals(
    signal_map: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    *,
    max_signal_abs: float = 5.0,
) -> pd.DataFrame:
    if not signal_map:
        raise ValueError("signal_map is empty")

    indices = [df.index for df in signal_map.values()]
    cols = [df.columns for df in signal_map.values()]
    common_index = indices[0]
    common_cols = cols[0]
    for idx in indices[1:]:
        common_index = common_index.intersection(idx)
    for c in cols[1:]:
        common_cols = common_cols.intersection(c)

    if len(common_index) == 0 or len(common_cols) == 0:
        raise ValueError("No common dates or tickers across signals")

    aligned = {
        name: df.reindex(index=common_index, columns=common_cols).fillna(0)
        for name, df in signal_map.items()
    }

    total = pd.DataFrame(0.0, index=common_index, columns=common_cols)
    for name, df in aligned.items():
        w = weights.get(name, 1.0)
        total += df * w

    denom = sum(abs(w) for w in weights.values()) or 1.0
    blended = total / denom

    return blended.clip(lower=-max_signal_abs, upper=max_signal_abs)
