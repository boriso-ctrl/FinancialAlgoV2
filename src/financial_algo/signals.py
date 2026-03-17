"""Trading signal generation utilities."""

from __future__ import annotations

from typing import Sequence


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
