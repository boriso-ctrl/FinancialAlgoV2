"""Binance crypto data loader for 1-minute and intraday bars (free, no authentication required)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Optional
import logging

import pandas as pd
import ccxt

logger = logging.getLogger(__name__)


def load_crypto_bars(
    symbols: List[str],
    timeframe: str = "1m",
    limit: int = 1000,
    days_back: int = 7,
) -> pd.DataFrame:
    """Load cryptocurrency bars from Binance (free, no API key needed).
    
    Args:
        symbols: List of crypto pairs (e.g., ["BTC/USDT", "ETH/USDT"])
        timeframe: Bar duration ("1m", "5m", "15m", "1h", "1d")
        limit: Max bars per symbol (max 1000 per request)
        days_back: Approximate days of history to load
    
    Returns:
        DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
    
    Example:
        >>> df = load_crypto_bars(["BTC/USDT", "ETH/USDT"], timeframe="1m", days_back=1)
        >>> print(df.head())
    """
    logger.info(f"Loading crypto bars from Binance: {symbols}, timeframe={timeframe}")
    
    # Initialize Binance exchange (no API key needed for public data)
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    all_data = []
    
    for symbol in symbols:
        logger.info(f"  Fetching {symbol}...")
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to dataframe
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            all_data.append(df)
            
            # Rate limiting - be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            continue
    
    if not all_data:
        logger.warning("No data retrieved")
        return pd.DataFrame()
    
    # Combine all data
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"Successfully loaded {len(result)} bars")
    return result


def load_crypto_ohlcv_matrix(
    symbols: List[str],
    timeframe: str = "1m",
    days_back: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load crypto bars and return separate OHLCV dataframes (timestamp × symbol).
    
    Returns:
        Tuple of (open_df, high_df, low_df, close_df, volume_df)
    """
    df = load_crypto_bars(symbols, timeframe=timeframe, days_back=days_back)
    
    if df.empty:
        return None, None, None, None, None
    
    open_df = df.pivot_table(index='timestamp', columns='symbol', values='open').sort_index()
    high_df = df.pivot_table(index='timestamp', columns='symbol', values='high').sort_index()
    low_df = df.pivot_table(index='timestamp', columns='symbol', values='low').sort_index()
    close_df = df.pivot_table(index='timestamp', columns='symbol', values='close').sort_index()
    volume_df = df.pivot_table(index='timestamp', columns='symbol', values='volume').sort_index()
    
    return open_df, high_df, low_df, close_df, volume_df


def get_top_cryptocurrencies() -> List[str]:
    """Return top liquid cryptocurrencies on Binance (USDT pairs)."""
    return [
        "BTC/USDT",   # Bitcoin
        "ETH/USDT",   # Ethereum
        "BNB/USDT",   # Binance Coin
        "SOL/USDT",   # Solana
        "XRP/USDT",   # Ripple
        "ADA/USDT",   # Cardano
        "DOGE/USDT",  # Dogecoin
        "LINK/USDT",  # Chainlink
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Testing Binance data loader (crypto)...")
        
        # Load 1-minute bars for top 3 cryptos, last 7 days
        df = load_crypto_bars(
            ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            timeframe="1m",
            days_back=7
        )
        
        print(f"\nLoaded {len(df)} records")
        print(f"\nSample data:")
        print(df.head(10))
        
        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Symbols: {df['symbol'].unique()}")
        
    except Exception as e:
        print(f"Error: {e}")
