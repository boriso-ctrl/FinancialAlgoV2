"""Strategy registry mapping names to indicator callables and default params."""

from typing import Callable, Dict, Any

from financial_algorithms.signals import sma_signal
from financial_algorithms.signals.price import (
    adx_strategy,
    bb_rsi_strategy,
    cci_adx_strategy,
    ma_cross_strategy,
    rsi_obv_bb_strategy,
    rsi_strategy,
    vwsma_strategy,
    wr_strategy,
)
from financial_algorithms.strategies.price import (
    all_indicator_combo,
    multi_factor_combo,
)
from financial_algorithms.strategies.hft_scalper import HFTScalperStrategy


def _hft_scalper_signal(df, **kwargs):
    """Thin wrapper that returns an HFT scalper signal series."""
    strat = HFTScalperStrategy(
        min_buy_score=kwargs.get("min_buy_score", 3.0),
        max_sell_score=kwargs.get("max_sell_score", -3.0),
    )
    return strat.generate_signals(df)


class StrategyRegistry:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(
            "sma",
            func=lambda prices, **kwargs: sma_signal(
                prices,
                fast=kwargs.get("fast", 20),
                slow=kwargs.get("slow", 50),
            ),
            params={"fast": 20, "slow": 50},
            description="Simple SMA crossover on close prices",
        )
        self.register(
            "ma_cross",
            func=lambda df, **kwargs: ma_cross_strategy(
                df.copy(), sl=kwargs.get("sl", 0.0), n1=kwargs.get("n1", 20), n2=kwargs.get("n2", 50)
            ),
            params={"sl": 0.0, "n1": 20, "n2": 50},
            description="Legacy MA20/MA50 cross with stop-loss threshold",
        )
        self.register(
            "rsi",
            func=lambda df, **kwargs: rsi_strategy(df.copy(), sl=kwargs.get("sl", 0.0)),
            params={"sl": 0.0},
            description="RSI range strategy",
        )
        self.register(
            "bb_rsi",
            func=lambda df, **kwargs: bb_rsi_strategy(df.copy(), sl=kwargs.get("sl", 0.0)),
            params={"sl": 0.0},
            description="Bollinger Bands + RSI",
        )
        self.register(
            "rsi_obv_bb",
            func=lambda df, **kwargs: rsi_obv_bb_strategy(df.copy(), sl=kwargs.get("sl", 0.0)),
            params={"sl": 0.0},
            description="RSI + OBV + Bollinger Bands",
        )
        self.register(
            "adx",
            func=lambda df, **kwargs: adx_strategy(df.copy(), sl=kwargs.get("sl", 0.0)),
            params={"sl": 0.0},
            description="ADX trend strategy",
        )
        self.register(
            "cci_adx",
            func=lambda df, **kwargs: cci_adx_strategy(df.copy(), sl=kwargs.get("sl", 0.0)),
            params={"sl": 0.0},
            description="CCI + ADX combo",
        )
        self.register(
            "wr",
            func=lambda df, **kwargs: wr_strategy(df.copy(), sl=kwargs.get("sl", 0.0)),
            params={"sl": 0.0},
            description="Williams %R strategy",
        )
        self.register(
            "vwsma",
            func=lambda df, **kwargs: vwsma_strategy(df.copy(), sl=kwargs.get("sl", 0.0)),
            params={"sl": 0.0},
            description="VWAP-SMA mean reversion",
        )
        self.register(
            "multi_factor_combo",
            func=lambda df, **kwargs: multi_factor_combo(
                df.copy(),
                weights=kwargs.get("weights"),
                max_signal_abs=kwargs.get("max_signal_abs", 5.0),
            ),
            params={"max_signal_abs": 5.0},
            description="Composite: BBRSI + SAR-Stoch + VWSMA + WR + volume (VO/OBV/VPT/PCR placeholder)",
        )
        self.register(
            "all_indicator_combo",
            func=lambda df, **kwargs: all_indicator_combo(
                df.copy(),
                weights=kwargs.get("weights"),
                max_signal_abs=kwargs.get("max_signal_abs", 5.0),
            ),
            params={"max_signal_abs": 5.0},
            description="Composite of every available price/volume indicator (graded blend)",
        )
        self.register(
            "hft_scalper",
            func=lambda df, **kwargs: _hft_scalper_signal(df, **kwargs),
            params={"min_buy_score": 3.0, "max_sell_score": -3.0},
            description="HFT scalper: 7-indicator multi-timeframe strategy (~1 trade/hour)",
        )

    def register(self, name: str, func: Callable, params: Dict[str, Any], description: str = ""):
        self._registry[name] = {
            "func": func,
            "params": params,
            "description": description,
        }

    def get(self, name: str):
        return self._registry.get(name)

    def list(self):
        return {k: {"params": v["params"], "description": v["description"]} for k, v in self._registry.items()}


registry = StrategyRegistry()
