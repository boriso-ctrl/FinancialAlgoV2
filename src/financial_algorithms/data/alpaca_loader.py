"""Alpaca Markets data loader for 1-minute and intraday bars."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional
import logging

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


def _get_alpaca_credentials():
    """Load Alpaca credentials from environment."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca credentials not found. "
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
        )
    
    return api_key, secret_key


def load_intraday_bars(
    tickers: List[str],
    timeframe: str = "1min",
    days_back: int = 7,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load intraday bars from Alpaca Markets.
    
    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "SPY", "QQQ"])
        timeframe: Bar duration ("1min", "5min", "15min", "1h", "1d")
        days_back: If start_date not provided, use last N days
        start_date: Start date for bars (default: N days ago)
        end_date: End date for bars (default: now)
    
    Returns:
        DataFrame with MultiIndex (timestamp, ticker) and OHLCV columns
    
    Example:
        >>> df = load_intraday_bars(["AAPL", "SPY"], timeframe="1min", days_back=1)
        >>> print(df.head())
    """
    api_key, secret_key = _get_alpaca_credentials()
    
    # Map timeframe strings to Alpaca TimeFrame enum
    timeframe_map = {
        "1min": TimeFrame.Minute,
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use {list(timeframe_map.keys())}")
    
    alpaca_tf = timeframe_map[timeframe]
    
    # Default date range
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=days_back)
    
    logger.info(
        f"Loading {timeframe} bars for {tickers} "
        f"from {start_date.date()} to {end_date.date()}"
    )
    
    # Initialize client
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # Build request
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=alpaca_tf,
        start=start_date,
        end=end_date,
    )
    
    # Fetch bars
    try:
        bars = client.get_stock_bars(request)
    except Exception as e:
        logger.error(f"Failed to fetch bars from Alpaca: {e}")
        raise
    
    # Convert to DataFrame
    if not bars.df.empty:
        # Flatten MultiIndex columns
        df = bars.df.reset_index()
        df.columns = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        
        # Sort by ticker and timestamp
        df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Successfully loaded {len(df)} bars")
        return df
    else:
        logger.warning(f"No data returned for {tickers}")
        return pd.DataFrame()


def load_ohlcv_matrix(
    tickers: List[str],
    timeframe: str = "1min",
    days_back: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load intraday bars and return separate OHLCV dataframes.
    
    Args:
        tickers: List of stock tickers
        timeframe: Bar duration
        days_back: Days of history to load
    
    Returns:
        Tuple of (open_df, high_df, low_df, close_df, volume_df)
        Each indexed by timestamp with columns for each ticker
    
    Example:
        >>> o, h, l, c, v = load_ohlcv_matrix(["AAPL", "SPY"], timeframe="1min", days_back=1)
        >>> print(c.head())  # Close prices
    """
    df = load_intraday_bars(tickers, timeframe=timeframe, days_back=days_back)
    
    if df.empty:
        return None, None, None, None, None
    
    # Pivot each OHLCV component
    open_df = df.pivot_table(
        index='timestamp', columns='ticker', values='open'
    ).sort_index()
    
    high_df = df.pivot_table(
        index='timestamp', columns='ticker', values='high'
    ).sort_index()
    
    low_df = df.pivot_table(
        index='timestamp', columns='ticker', values='low'
    ).sort_index()
    
    close_df = df.pivot_table(
        index='timestamp', columns='ticker', values='close'
    ).sort_index()
    
    volume_df = df.pivot_table(
        index='timestamp', columns='ticker', values='volume'
    ).sort_index()
    
    return open_df, high_df, low_df, close_df, volume_df


def load_minute_bars(
    tickers: List[str],
    days_back: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load 1-minute bars as OHLCV matrices."""
    return load_ohlcv_matrix(tickers, timeframe="1min", days_back=days_back)


def load_5min_bars(
    tickers: List[str],
    days_back: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load 5-minute bars as OHLCV matrices."""
    return load_ohlcv_matrix(tickers, timeframe="5min", days_back=days_back)


def load_15min_bars(
    tickers: List[str],
    days_back: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load 15-minute bars as OHLCV matrices."""
    return load_ohlcv_matrix(tickers, timeframe="15min", days_back=days_back)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Testing Alpaca data loader...")
        
        # Load 1-minute bars for 60 days back (historical data)
        df = load_intraday_bars(
            ["AAPL", "SPY", "QQQ"],
            timeframe="1min",
            days_back=60
        )
        
        print(f"\nLoaded {len(df)} records")
        print(f"\nSample data:")
        print(df.head(10))
        
        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Tickers: {df['ticker'].unique()}")
        
    except Exception as e:
        print(f"Error: {e}")
