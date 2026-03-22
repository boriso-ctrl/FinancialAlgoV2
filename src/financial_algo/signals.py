"""Trading signal generation utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from financial_algo.indicators import zscore as _zscore


def crossover_signal(
    fast: Sequence[float],
    slow: Sequence[float],
) -> list[int]:
    """Generate buy/sell signals from a moving-average crossover.

    A **+1** signal is produced when *fast* crosses above *slow*
    (golden cross / buy).  A **-1** signal is produced when *fast*
    crosses below *slow* (death cross / sell).  A **0** indicates no
    crossover at that bar.

    Parameters
    ----------
    fast:
        Fast moving-average series (e.g. 10-period SMA).
    slow:
        Slow moving-average series (e.g. 30-period SMA).

    Returns
    -------
    list[int]
        Signal values ``{-1, 0, +1}``.

    Raises
    ------
    ValueError
        If *fast* and *slow* have different lengths, or if either is
        shorter than 2 elements.
    """
    import math

    fast = list(fast)
    slow = list(slow)

    if len(fast) != len(slow):
        raise ValueError(
            f"fast and slow must have the same length "
            f"(got {len(fast)} vs {len(slow)})"
        )
    if len(fast) < 2:
        raise ValueError("Series must have at least 2 elements")

    signals: list[int] = [0]
    for i in range(1, len(fast)):
        f_prev, f_curr = fast[i - 1], fast[i]
        s_prev, s_curr = slow[i - 1], slow[i]

        if any(math.isnan(v) for v in (f_prev, f_curr, s_prev, s_curr)):
            signals.append(0)
            continue

        if f_prev < s_prev and f_curr > s_curr:
            signals.append(1)
        elif f_prev > s_prev and f_curr < s_curr:
            signals.append(-1)
        else:
            signals.append(0)

    return signals


# ---------------------------------------------------------------------------
# Pandas-based signals (used by strategy layer)
# ---------------------------------------------------------------------------


def momentum_score(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Cross-sectional momentum score for each ticker.

    For each window the return is computed, then ranked cross-sectionally
    (``0`` = worst, ``1`` = best).  The final score is the equal-weighted
    average across windows.

    Parameters
    ----------
    prices:
        DataFrame of prices, one column per ticker.
    windows:
        Look-back windows in trading days.  Default ``[21, 63, 126]``
        (≈ 1 / 3 / 6 months).

    Returns
    -------
    pd.DataFrame
        Scores in ``[0, 1]``, same shape as *prices*.
    """
    if windows is None:
        windows = [21, 63, 126]

    ranks = []
    for w in windows:
        ret = prices.pct_change(w)
        # Rank cross-sectionally; normalise to [0, 1]
        r = ret.rank(axis=1, pct=True)
        ranks.append(r)

    return pd.concat(ranks).groupby(level=0).mean().reindex(prices.index)


def regime_signal(
    regime: pd.Series,
    mapping: dict[str, float],
) -> pd.Series:
    """Map regime labels to target exposure levels.

    Parameters
    ----------
    regime:
        Series of regime labels (e.g. from :func:`detect_regime`).
    mapping:
        ``{regime_label: target_exposure}``, e.g.
        ``{"NORMAL": 1.0, "CRISIS": -0.5}``.

    Returns
    -------
    pd.Series
        Numeric exposure, same index as *regime*.
    """
    return regime.map(mapping).fillna(0.0).astype(float)


def pair_zscore_signal(
    leg_a: pd.Series,
    leg_b: pd.Series,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Z-score mean-reversion signal for a pair trade.

    Returns ``+1`` (long A / short B) when z-score drops below
    ``-entry_z``, ``-1`` (short A / long B) when above ``+entry_z``,
    and ``0`` when z-score reverts inside ``±exit_z``.

    Parameters
    ----------
    leg_a, leg_b:
        Price series for the two legs.
    window:
        Z-score look-back.
    entry_z:
        Entry threshold (absolute).
    exit_z:
        Exit threshold (absolute).

    Returns
    -------
    pd.Series
        Signal in ``{-1, 0, +1}``.
    """
    spread = np.log(leg_a / leg_b)
    z = _zscore(spread, window)

    z_vals = z.values
    n = len(z_vals)
    sig_vals = np.zeros(n, dtype=int)
    pos = 0
    for i in range(n):
        val = z_vals[i]
        if np.isnan(val):
            sig_vals[i] = 0
            continue
        if pos == 0:
            if val <= -entry_z:
                pos = 1   # long A / short B
            elif val >= entry_z:
                pos = -1  # short A / long B
        elif pos == 1 and val >= -exit_z:
            pos = 0
        elif pos == -1 and val <= exit_z:
            pos = 0
        sig_vals[i] = pos

    return pd.Series(sig_vals, index=z.index)
