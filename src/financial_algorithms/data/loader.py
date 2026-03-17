"""Data loading utilities built on SimFin.

The previous stack baked the API key and data directory directly into
source.  This module centralizes configuration using two environment
variables:

- ``SIMFIN_API_KEY``
- ``SIMFIN_DATA_DIR`` (defaults to ``~/.simfin`` if unset)

Example::

    export SIMFIN_API_KEY="..."
    export SIMFIN_DATA_DIR="C:/Users/boris/.simfin"

    from financial_algorithms.data import load_daily_prices

    prices = load_daily_prices(["AAPL", "MSFT"])
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import simfin as sf
from dotenv import load_dotenv

# Load .env file (if it exists) at module import time
load_dotenv()

SIMFIN_API_KEY_ENV = "SIMFIN_API_KEY"
SIMFIN_DATA_DIR_ENV = "SIMFIN_DATA_DIR"
DEFAULT_MARKET = "us"


@dataclass(frozen=True)
class SimFinConfig:
    api_key: str
    data_dir: str
    market: str = DEFAULT_MARKET


def _resolve_simfin_config() -> SimFinConfig:
    api_key = os.getenv(SIMFIN_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"SimFin API key missing. Set {SIMFIN_API_KEY_ENV} to continue."
        )

    data_dir = os.getenv(SIMFIN_DATA_DIR_ENV)
    if not data_dir:
        data_dir = str(Path.home().joinpath(".simfin"))
    return SimFinConfig(api_key=api_key, data_dir=data_dir)


def _configure_simfin_client(config: SimFinConfig) -> None:
    sf.set_api_key(config.api_key)
    sf.set_data_dir(config.data_dir)


def load_daily_prices(tickers: Iterable[str], market: str = DEFAULT_MARKET) -> pd.DataFrame:
    """Return a price matrix (index=Date, columns=tickers, values=Close).
    
    Falls back to cached data if API refresh fails (useful for demo/testing with invalid keys).
    """

    config = _resolve_simfin_config()
    _configure_simfin_client(config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*date_parser.*", category=FutureWarning)
        try:
            # Try to load with optional refresh (refresh_days=0 means use cache only)
            df = sf.load_shareprices(variant="daily", market=market, refresh_days=99999)
        except Exception as e:
            # If even cached data load fails, try without any refresh logic
            print(f"  [Note] Load failed, attempting direct cache read...")
            import csv
            cache_path = Path(config.data_dir) / f"us-shareprices-daily.csv"
            if cache_path.exists():
                df = pd.read_csv(cache_path, sep=";", index_col=None)
                df.columns = df.columns.str.strip()
            else:
                raise RuntimeError(f"No cached data at {cache_path} and API key invalid. Cannot load prices.")

    df = df.loc[list(tickers)]
    df = (
        df[["Close"]]
        .reset_index()
        .drop_duplicates(subset=["Ticker", "Date"])
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
    )
    return df


def load_daily_returns(tickers: Iterable[str], market: str = DEFAULT_MARKET) -> pd.DataFrame:
    prices = load_daily_prices(tickers, market=market)
    return prices.pct_change().dropna()
