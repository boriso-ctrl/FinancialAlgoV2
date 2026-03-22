"""Ticker universe definitions for the crisis-thriving trading system."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sub-universes
# ---------------------------------------------------------------------------

ENERGY: list[str] = ["XLE", "USO", "XOP", "OIH", "CVX", "XOM"]
"""Energy sector — oil crisis plays."""

DEFENSE: list[str] = ["ITA", "LMT", "RTX", "NOC", "GD"]
"""Aerospace & Defense — war/geopolitical crisis plays."""

SAFE_HAVEN: list[str] = ["GLD", "TLT", "IEF", "UUP"]
"""Traditional flight-to-safety instruments."""

BROAD: list[str] = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
"""Broad equity indices (US large, tech, small, intl/dev, intl/EM)."""

CRYPTO: list[str] = ["BTC-USD"]
"""Crypto assets (yfinance tickers)."""

SECTORS: list[str] = [
    "XLE",   # Energy
    "XLF",   # Financials
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]
"""SPDR sector ETFs for breadth & rotation analysis."""

VOLATILITY: list[str] = ["^VIX"]
"""Volatility indices (non-tradeable, used for regime detection only)."""

# ---------------------------------------------------------------------------
# Credit proxy (fear gauge)
# ---------------------------------------------------------------------------

CREDIT_PROXY: list[str] = ["HYG", "LQD"]
"""High-yield vs investment-grade for credit spread proxy."""

# ---------------------------------------------------------------------------
# Combined universe — every ticker the system may touch
# ---------------------------------------------------------------------------

CRISIS_UNIVERSE: list[str] = sorted(
    set(ENERGY + DEFENSE + SAFE_HAVEN + BROAD + CRYPTO + SECTORS + VOLATILITY + CREDIT_PROXY)
)
"""Full ticker universe across all categories (deduplicated, sorted)."""
