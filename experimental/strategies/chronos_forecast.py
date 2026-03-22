"""Category Q: Chronos-based Probabilistic Forecast Strategies.

Q1 -- ChronosForecast: Zero-shot daily return forecasting using Amazon
      Chronos-Bolt-Small (9M params, CPU/GPU).  Uses quantile spread
      (p75 - p25) for conviction-weighted position sizing.

Academic basis:
  Ansari et al. (2024) "Chronos: Learning the Language of Time Series"
  arXiv:2403.07815.  Pre-trained on 27 benchmark datasets; zero-shot
  forecasts competitive with task-specific supervised models.

Depends: pip install chronos-forecasting torch
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from financial_algo.strategies.base import Strategy


@dataclass
class ChronosForecastConfig:
    """Configuration for Q1-ChronosForecast."""

    # Model selection: chronos-bolt-small = 9M params, CPU-friendly
    # For GPU: "amazon/chronos-bolt-base" (85M) or "amazon/chronos-t5-small"
    model_id: str = "amazon/chronos-bolt-small"

    # Universe -- matches our 42-ticker setup (subset that trades well)
    tickers: tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "GLD", "TLT", "IEF", "UUP",
        "XLE", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
        "HYG", "LQD", "SLV", "VNQ", "DBC",
    )

    # Forecast parameters
    context_length: int = 512     # lookback window (trading days) fed to model
    prediction_length: int = 5    # forecast horizon (5 trading days = 1 week)
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)  # quantile levels

    # Position sizing
    top_n: int = 5                # long top-N predicted gainers
    bottom_n: int = 3             # short bottom-N predicted losers
    max_gross_leverage: float = 1.5
    max_single_weight: float = 0.30

    # Conviction scaling: narrow quantile spread = high conviction = larger pos
    conviction_exponent: float = 1.5  # amplify conviction differences

    # Rebalance / warm-up
    rebalance_freq: int = 5       # weekly rebalance (match prediction horizon)
    min_history: int = 252        # 1 year warm-up before first forecast

    # Safety
    sma_trend_window: int = 200   # 200-day SMA trend filter on SPY

    # Device: "cuda" if available, else "cpu"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ChronosForecast(Strategy):
    """Zero-shot probabilistic return forecasting via Chronos-Bolt-Small.

    Thesis
    ------
    Pre-trained foundation models for time series (Chronos) capture
    universal temporal patterns across thousands of domains.  Applied
    zero-shot to financial returns, they can identify short-term
    distributional shifts that traditional factors miss.  The quantile
    spread (p75-p25) provides a built-in uncertainty measure: narrow
    spread = model is confident = size up; wide spread = uncertain =
    size down.  This is a structural advantage over point-estimate
    models that must bolt on separate uncertainty quantification.

    Expected characteristics:
      - Low correlation with momentum/value factors (different signal source)
      - Short holding period (weekly rebalance)
      - Moderate Sharpe (0.4-0.7 expected; zero-shot has limits)
      - Should perform well in regime transitions (model sees distributional shift)

    Architecture
    ------------
    1. Feed 512 days of daily log-returns per ticker into Chronos
    2. Get quantile forecasts at p25, p50, p75
    3. Median (p50) = expected return direction
    4. IQR (p75-p25) = uncertainty -> conviction = 1/IQR
    5. Rank tickers by conviction-weighted median forecast
    6. Long top-N, short bottom-N, conviction-weighted sizing
    7. 200-day SMA on SPY as macro trend guard (reduce shorts in uptrend)
    """

    name = "Q1-ChronosForecast"

    def __init__(self, config: ChronosForecastConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or ChronosForecastConfig()
        self._pipeline = None  # lazy-loaded

    def _load_model(self) -> None:
        """Lazy-load Chronos pipeline (heavy import, slow first call)."""
        if self._pipeline is not None:
            return
        from chronos import ChronosPipeline  # type: ignore[import-untyped]

        self._pipeline = ChronosPipeline.from_pretrained(
            self.cfg.model_id,
            device_map=self.cfg.device,
            torch_dtype=torch.float32,
        )

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        avail = [t for t in c.tickers if t in prices.columns]
        if len(avail) < c.top_n + c.bottom_n:
            return weights

        # Lazy-load model on first call
        self._load_model()

        p = prices[avail]
        n_days = len(p)

        # Log returns for Chronos input (more stationary than raw prices)
        price_ratio = (p / p.shift(1)).clip(lower=1e-10)
        log_ret = np.log(price_ratio).fillna(0.0)

        # 200-day SMA trend filter on SPY (if available)
        spy_above_sma = pd.Series(True, index=prices.index)
        if "SPY" in prices.columns:
            spy_sma = prices["SPY"].rolling(c.sma_trend_window).mean()
            spy_above_sma = prices["SPY"] >= spy_sma
            spy_above_sma = spy_above_sma.fillna(True)

        # Identify rebalance days
        rebal_days = list(range(c.min_history, n_days, c.rebalance_freq))
        current_weights: dict[str, float] = {}

        for day_idx in range(c.min_history, n_days):
            if day_idx in rebal_days:
                current_weights = self._forecast_and_allocate(
                    log_ret, avail, day_idx, spy_above_sma.iloc[day_idx], c,
                )

            for t, w in current_weights.items():
                col_idx = weights.columns.get_loc(t)
                weights.iloc[day_idx, col_idx] = w

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _forecast_and_allocate(
        self,
        log_ret: pd.DataFrame,
        avail: list[str],
        day_idx: int,
        spy_uptrend: bool,
        c: ChronosForecastConfig,
    ) -> dict[str, float]:
        """Run Chronos forecast for all tickers and compute allocations."""
        ctx_start = max(0, day_idx - c.context_length)
        ctx_data = log_ret.iloc[ctx_start:day_idx]

        # Batch predict: build list of context tensors
        contexts = []
        valid_tickers = []
        for t in avail:
            series = ctx_data[t].values.astype(np.float32)
            # Skip if all zeros (no real data)
            if np.all(series == 0):
                continue
            contexts.append(torch.tensor(series))
            valid_tickers.append(t)

        if len(valid_tickers) < c.top_n + c.bottom_n:
            return {}

        # Chronos batch forecast -> quantiles
        # Output shape: (n_tickers, n_quantiles, prediction_length)
        quantile_forecasts = self._pipeline.predict_quantiles(
            context=contexts,
            prediction_length=c.prediction_length,
            quantile_levels=list(c.quantiles),
        )
        # quantile_forecasts: tensor (n_tickers, prediction_length, n_quantiles)
        # We need cumulative forecast over the prediction horizon
        qf = quantile_forecasts.numpy()

        # Sum log-returns over prediction horizon to get total forecast
        # qf shape: (n_tickers, prediction_length, n_quantiles)
        # After summing over prediction_length: (n_tickers, n_quantiles)
        cum_forecast = qf.sum(axis=1)

        # Extract quantiles: p25, p50 (median), p75
        q_idx = {q: i for i, q in enumerate(c.quantiles)}
        p25 = cum_forecast[:, q_idx[0.25]]
        p50 = cum_forecast[:, q_idx[0.50]]  # median = expected direction
        p75 = cum_forecast[:, q_idx[0.75]]

        # IQR = uncertainty measure
        iqr = np.abs(p75 - p25)
        iqr = np.where(iqr < 1e-8, 1e-8, iqr)  # floor to avoid div-by-zero

        # Conviction = inverse IQR, raised to exponent for amplification
        conviction = (1.0 / iqr) ** c.conviction_exponent

        # Conviction-weighted expected return
        signal = p50 * conviction

        # Build ranked signal Series
        signal_series = pd.Series(signal, index=valid_tickers)
        signal_series = signal_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Rank and select top-N long, bottom-N short
        ranked = signal_series.sort_values(ascending=False)
        long_tickers = ranked.head(c.top_n).index.tolist()
        short_tickers = ranked.tail(c.bottom_n).index.tolist()

        # Only allow shorts if SPY below 200d SMA (defensive macro filter)
        if spy_uptrend:
            short_tickers = []

        # Conviction-weighted sizing within each leg
        alloc: dict[str, float] = {}

        # Long leg
        long_signals = signal_series[long_tickers].abs()
        long_total = long_signals.sum()
        if long_total > 0:
            long_frac = long_signals / long_total
        else:
            long_frac = pd.Series(1.0 / len(long_tickers), index=long_tickers)

        long_budget = c.max_gross_leverage if not short_tickers else c.max_gross_leverage * 0.6
        for t in long_tickers:
            w = float(long_frac.get(t, 0.0)) * long_budget
            alloc[t] = min(w, c.max_single_weight)

        # Short leg (only in downtrend)
        if short_tickers:
            short_signals = signal_series[short_tickers].abs()
            short_total = short_signals.sum()
            if short_total > 0:
                short_frac = short_signals / short_total
            else:
                short_frac = pd.Series(1.0 / len(short_tickers), index=short_tickers)

            short_budget = c.max_gross_leverage * 0.4
            for t in short_tickers:
                w = float(short_frac.get(t, 0.0)) * short_budget
                alloc[t] = -min(w, c.max_single_weight)

        # Enforce max gross leverage
        gross = sum(abs(v) for v in alloc.values())
        if gross > c.max_gross_leverage:
            scale = c.max_gross_leverage / gross
            alloc = {t: w * scale for t, w in alloc.items()}

        return alloc
