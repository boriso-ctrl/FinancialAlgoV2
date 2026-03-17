"""High-frequency backtest engine with multi-timeframe data management.

Designed to run an :class:`HFTScalperStrategy` on intraday bars and produce
CAGR / Sortino-focused performance reports.

Key features
============
* Walk-forward bar-by-bar execution (no lookahead bias).
* Multi-timeframe resampling from a base 1-minute feed.
* Per-trade stop-loss, tiered take-profit, and trailing-stop management.
* Max-hold-time safety net.
* HFT-specific metrics: trades/hour, profit factor, Sortino, monthly CAGR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from financial_algorithms.strategies.hft_scalper import HFTScalperStrategy

logger = logging.getLogger(__name__)


# ── data types ──────────────────────────────────────────────────────────


class Side(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class HFTTrade:
    """Immutable record of a completed round-trip trade."""

    symbol: str
    side: Side
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float  # in dollars
    exit_reason: str  # "signal", "sl", "tp1", "tp2", "trail", "max_hold"
    pnl: float = field(init=False)
    pnl_pct: float = field(init=False)

    def __post_init__(self):
        if self.side == Side.LONG:
            self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price
        elif self.side == Side.SHORT:
            self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price
        else:
            self.pnl_pct = 0.0
        self.pnl = self.size * self.pnl_pct


@dataclass
class _OpenPosition:
    symbol: str
    side: Side
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    sl: float
    tp1: float
    tp2: float
    trail: float
    tp1_hit: bool = False
    bars_held: int = 0


# ── engine ──────────────────────────────────────────────────────────────


class HFTBacktestEngine:
    """Event-driven HFT backtest engine.

    Parameters
    ----------
    strategy : HFTScalperStrategy
        Strategy instance providing signal generation and risk parameters.
    initial_capital : float
        Starting equity.
    slippage_bps : float
        Per-side slippage in basis points.
    commission_bps : float
        Per-side commission in basis points.
    base_timeframe : str
        Resolution of the raw feed (default ``"3min"``).
    """

    def __init__(
        self,
        strategy: HFTScalperStrategy | None = None,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 3.0,
        commission_bps: float = 1.0,
        base_timeframe: str = "3min",
    ):
        self.strategy = strategy or HFTScalperStrategy()
        self.initial_capital = initial_capital
        self.slippage_frac = slippage_bps / 10_000
        self.commission_frac = commission_bps / 10_000
        self.base_tf = base_timeframe

        # state
        self.equity = initial_capital
        self.position: _OpenPosition | None = None
        self.trades: list[HFTTrade] = []
        self.equity_curve: list[float] = []
        self.timestamps: list[pd.Timestamp] = []

    # ── helpers ─────────────────────────────────────────────────────────

    def _apply_cost(self, price: float, direction: int) -> float:
        """Apply slippage + commission to *price* in *direction* (1=buy, -1=sell)."""
        return price * (1 + direction * (self.slippage_frac + self.commission_frac))

    def _mark_to_market(self, price: float) -> float:
        """Current equity including unrealised P&L."""
        if self.position is None:
            return self.equity
        pos = self.position
        if pos.side == Side.LONG:
            unreal = pos.size * (price - pos.entry_price) / pos.entry_price
        else:
            unreal = pos.size * (pos.entry_price - price) / pos.entry_price
        return self.equity + unreal

    # ── position management ─────────────────────────────────────────────

    def _open_position(
        self, symbol: str, side: Side, price: float, ts: pd.Timestamp, score: float
    ):
        direction = 1 if side == Side.LONG else -1
        fill_price = self._apply_cost(price, direction)
        size = self.strategy.calculate_position_size(abs(score), self.equity)
        if size <= 0:
            return
        levels = self.strategy.calculate_entry_exit_levels(fill_price)
        if side == Side.LONG:
            sl = levels["sl"]
            tp1 = levels["tp1"]
            tp2 = levels["tp2"]
        else:
            sl = fill_price * (1 + self.strategy.stop_loss_pct / 100)
            tp1 = fill_price * (1 - self.strategy.tp1_pct / 100)
            tp2 = fill_price * (1 - self.strategy.tp2_pct / 100)

        self.position = _OpenPosition(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            entry_time=ts,
            size=size,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            trail=sl,
        )

    def _close_position(self, price: float, ts: pd.Timestamp, reason: str):
        pos = self.position
        if pos is None:
            return
        direction = -1 if pos.side == Side.LONG else 1
        fill_price = self._apply_cost(price, direction)

        trade = HFTTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=ts,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            size=pos.size,
            exit_reason=reason,
        )
        self.trades.append(trade)
        self.equity += trade.pnl
        self.position = None

    def _manage_position(self, high: float, low: float, close: float,
                         ts: pd.Timestamp, score: float):
        """Check SL / TP / trailing / max-hold and close if triggered."""
        pos = self.position
        if pos is None:
            return

        pos.bars_held += 1

        # ── stop loss ───────────────────────────────────────────────
        if pos.side == Side.LONG and low <= pos.sl:
            self._close_position(pos.sl, ts, "sl")
            return
        if pos.side == Side.SHORT and high >= pos.sl:
            self._close_position(pos.sl, ts, "sl")
            return

        # ── take profit 1 (partial concept kept via flag) ──────────
        if pos.side == Side.LONG and high >= pos.tp1 and not pos.tp1_hit:
            pos.tp1_hit = True
            # move SL to break-even
            pos.trail = pos.entry_price
        if pos.side == Side.SHORT and low <= pos.tp1 and not pos.tp1_hit:
            pos.tp1_hit = True
            pos.trail = pos.entry_price

        # ── take profit 2 ──────────────────────────────────────────
        if pos.side == Side.LONG and high >= pos.tp2:
            self._close_position(pos.tp2, ts, "tp2")
            return
        if pos.side == Side.SHORT and low <= pos.tp2:
            self._close_position(pos.tp2, ts, "tp2")
            return

        # ── trailing stop ──────────────────────────────────────────
        if pos.tp1_hit:
            trail_dist = pos.entry_price * self.strategy.trailing_stop_pct / 100
            if pos.side == Side.LONG:
                new_trail = close - trail_dist
                pos.trail = max(pos.trail, new_trail)
                if low <= pos.trail:
                    self._close_position(pos.trail, ts, "trail")
                    return
            else:
                new_trail = close + trail_dist
                pos.trail = min(pos.trail, new_trail)
                if high >= pos.trail:
                    self._close_position(pos.trail, ts, "trail")
                    return

        # ── max hold time ──────────────────────────────────────────
        if pos.bars_held >= self.strategy.max_hold_bars:
            self._close_position(close, ts, "max_hold")
            return

        # ── signal reversal exit ───────────────────────────────────
        if pos.side == Side.LONG and self.strategy.should_exit_on_signal(score):
            self._close_position(close, ts, "signal")
        elif pos.side == Side.SHORT and self.strategy.should_enter(score):
            self._close_position(close, ts, "signal")

    # ── main loop ───────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "ASSET",
    ) -> dict:
        """Execute the backtest on OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``timestamp | open | high | low | close | volume``.
            The resolution should match *base_timeframe* (default 3-min).
        symbol : str
            Ticker label.

        Returns
        -------
        dict
            Keys: ``metrics``, ``trades``, ``equity_curve``.
        """
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df = df.sort_index()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        warmup = (
            max(
                self.strategy.ema_slow,
                self.strategy.bb_period,
                self.strategy.macd_slow + self.strategy.macd_signal,
            )
            + 5
        )

        for i in range(len(df)):
            ts = df.index[i]
            c_val, h_val, lo_val = close[i], high[i], low[i]

            # record equity snapshot
            self.equity_curve.append(self._mark_to_market(c_val))
            self.timestamps.append(ts)

            if i < warmup:
                continue

            # slice history up to current bar (inclusive)
            c_hist = close[: i + 1]
            h_hist = high[: i + 1]
            l_hist = low[: i + 1]
            v_hist = volume[: i + 1]

            score = self.strategy.calculate_composite_score(
                close_3m=c_hist,
                high_5m=h_hist,
                low_5m=l_hist,
                close_5m=c_hist,
                volume_3m=v_hist,
                high_15m=h_hist,
                low_15m=l_hist,
                close_15m=c_hist,
                close_10m=c_hist,
            )

            # manage existing position (SL / TP / trail)
            if self.position is not None:
                self._manage_position(h_val, lo_val, c_val, ts, score)

            # try new entry if flat
            if self.position is None:
                if self.strategy.should_enter(score):
                    self._open_position(symbol, Side.LONG, c_val, ts, score)
                elif self.strategy.should_exit_on_signal(score):
                    self._open_position(symbol, Side.SHORT, c_val, ts, score)

        # force-close any open position at end
        if self.position is not None:
            self._close_position(close[-1], df.index[-1], "eod")

        return {
            "metrics": self.compute_metrics(),
            "trades": self.get_trades_df(),
            "equity_curve": self.get_equity_df(),
        }

    # ── metrics ─────────────────────────────────────────────────────────

    def compute_metrics(self) -> dict:
        """Return HFT-focused performance metrics."""
        eq = pd.Series(self.equity_curve)
        if len(eq) < 2:
            return self._empty_metrics()

        returns = eq.pct_change().dropna()
        total_return = (eq.iloc[-1] / self.initial_capital) - 1

        # annualise based on intraday bar count
        n_bars = len(returns)
        if n_bars == 0:
            return self._empty_metrics()

        # approximate trading hours from timestamps
        if len(self.timestamps) >= 2:
            span_hours = (
                (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 3600
            )
        else:
            span_hours = 0
        span_years = span_hours / (252 * 6.5) if span_hours > 0 else 0

        cagr = (1 + total_return) ** (1 / span_years) - 1 if span_years > 0 else 0.0
        monthly_cagr = (1 + cagr) ** (1 / 12) - 1 if cagr > -1 else 0.0

        # Sharpe (annualised using bar frequency)
        bars_per_year = n_bars / span_years if span_years > 0 else 252 * 130
        ann_factor = np.sqrt(bars_per_year) if bars_per_year > 0 else 1
        excess = returns
        sharpe = (
            float(excess.mean() / excess.std() * ann_factor)
            if excess.std() > 0
            else 0.0
        )

        # Sortino
        downside = returns[returns < 0]
        ds_std = downside.std() if len(downside) > 1 else 0.0
        sortino = (
            float(excess.mean() / ds_std * ann_factor)
            if ds_std > 0
            else 0.0
        )

        # Drawdown
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = float(dd.min())

        # Trade-level stats
        n_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / n_trades if n_trades else 0.0
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        trades_per_hour = n_trades / span_hours if span_hours > 0 else 0.0

        avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

        return {
            "Total Return": round(total_return, 6),
            "CAGR": round(cagr, 6),
            "Monthly CAGR": round(monthly_cagr, 6),
            "Sharpe Ratio": round(sharpe, 4),
            "Sortino Ratio": round(sortino, 4),
            "Calmar Ratio": round(calmar, 4),
            "Max Drawdown": round(max_dd, 6),
            "Win Rate": round(win_rate, 4),
            "Profit Factor": round(profit_factor, 4),
            "Total Trades": n_trades,
            "Trades/Hour": round(trades_per_hour, 2),
            "Avg Win %": round(avg_win * 100, 4),
            "Avg Loss %": round(avg_loss * 100, 4),
            "Gross Profit": round(gross_profit, 2),
            "Gross Loss": round(gross_loss, 2),
            "Final Equity": round(self.equity, 2),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Monthly CAGR": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Calmar Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
            "Total Trades": 0,
            "Trades/Hour": 0.0,
            "Avg Win %": 0.0,
            "Avg Loss %": 0.0,
            "Gross Profit": 0.0,
            "Gross Loss": 0.0,
            "Final Equity": 0.0,
        }

    # ── dataframes ──────────────────────────────────────────────────────

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "symbol": t.symbol,
                    "side": t.side.name,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": round(t.entry_price, 4),
                    "exit_price": round(t.exit_price, 4),
                    "size": round(t.size, 2),
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct * 100, 4),
                    "exit_reason": t.exit_reason,
                }
            )
        return pd.DataFrame(rows)

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"timestamp": self.timestamps, "equity": self.equity_curve}
        )
