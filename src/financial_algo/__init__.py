"""FinancialAlgoV2 -- crisis-thriving leveraged trading bot system.

Architecture
------------
technical/
    Price & volume-based analysis: indicators, signals, regimes, strategies.
    (indicators.py, signals.py, regimes.py, strategies/)

fundamental/
    Sentiment & macro-based analysis: news feeds, fear/greed, divergence.
    (fundamental/indicators.py, fundamental/signals.py, fundamental/strategies/)

Both layers feed into the shared backtest engine and portfolio management.
"""

# --- Technical: Core indicators (original + new) -------------------------
from financial_algo.indicators import (
    atr,
    bollinger_bands,
    breadth_count,
    drawdown,
    ema,
    moving_average,
    realized_vol,
    rsi,
    zscore,
)

# --- Technical: Portfolio -------------------------------------------------
from financial_algo.portfolio import (
    Portfolio,
    apply_drawdown_control,
    apply_leverage,
    apply_vol_target,
)

# --- Technical: Signals ---------------------------------------------------
from financial_algo.signals import (
    crossover_signal,
    momentum_score,
    pair_zscore_signal,
    regime_signal,
)

# --- Technical: Regime detection ------------------------------------------
from financial_algo.regimes import Regime, RegimeConfig, detect_regime, is_crisis

# --- Backtest engine (shared) ---------------------------------------------
from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.logging_utils import get_logger, set_trace_id, setup_logging, trace_context

# --- Data pipeline (shared) -----------------------------------------------
from financial_algo.data import load_prices

# --- Technical: Strategies ------------------------------------------------
from financial_algo.strategies import Strategy

# --- Fundamental: Sentiment & macro analysis ------------------------------
from financial_algo.fundamental import (
    # Data
    build_synthetic_sentiment,
    # Indicators
    fear_greed_composite,
    sentiment_zscore,
    sentiment_momentum,
    # Signals
    crisis_onset_signal,
    fear_greed_signal,
    recovery_signal,
    # Strategies
    SentimentCrisisAlpha,
    FearGreedContrarian,
    SentimentDivergence,
    SentimentEnhancedRegime,
)

__all__ = [
    # Technical indicators
    "moving_average", "rsi", "bollinger_bands",
    "realized_vol", "ema", "zscore", "atr", "breadth_count", "drawdown",
    # Portfolio
    "Portfolio", "apply_leverage", "apply_vol_target", "apply_drawdown_control",
    # Technical signals
    "crossover_signal", "momentum_score", "regime_signal", "pair_zscore_signal",
    # Regimes
    "Regime", "RegimeConfig", "detect_regime", "is_crisis",
    # Backtest
    "BacktestConfig", "backtest", "compute_metrics",
    # Logging
    "setup_logging", "get_logger", "set_trace_id", "trace_context",
    # Data
    "load_prices",
    # Technical strategies
    "Strategy",
    # Fundamental
    "build_synthetic_sentiment",
    "fear_greed_composite", "sentiment_zscore", "sentiment_momentum",
    "crisis_onset_signal", "fear_greed_signal", "recovery_signal",
    "SentimentCrisisAlpha", "FearGreedContrarian",
    "SentimentDivergence", "SentimentEnhancedRegime",
]
