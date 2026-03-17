"""Yahoo Finance stock data loader for 1-minute and multi-minute bars."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional, Union
import yfinance as yf


def load_stock_bars(
    tickers: Union[str, List[str]],
    interval: str = "1m",
    period: str = "7d",
    progress: bool = False,
) -> pd.DataFrame:
    """Load 1-minute (or other interval) stock bars from Yahoo Finance.
    
    Args:
        tickers: Single ticker or list (e.g., 'AAPL' or ['AAPL', 'MSFT'])
        interval: Timeframe ('1m', '5m', '15m', '1h', '1d', etc.)
        period: Data range ('1d', '7d', '1mo', '1y', etc.)
        progress: Show progress bar
    
    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume, symbol]
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_data = []
    
    # Fetch data per ticker individually (more reliable)
    for ticker in tickers:
        try:
            history = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=progress,
                prepost=False,
            )
            
            if history.empty:
                continue
            
            df = history.reset_index()
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['symbol'] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {ticker}: {e}")
            continue
    
    # Combine and sort
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = result.dropna(subset=['close'])
        result = result.sort_values('timestamp').reset_index(drop=True)
        return result
    
    return pd.DataFrame()


def load_stock_ohlcv_matrix(
    tickers: Union[str, List[str]],
    interval: str = "1m",
    period: str = "7d",
    field: str = "close",
) -> pd.DataFrame:
    """Load stock data as wide matrix (timestamp × ticker).
    
    Args:
        tickers: Single or list of tickers
        interval: Timeframe
        period: Data range
        field: Which OHLCV field ('close', 'volume', etc.)
    
    Returns:
        DataFrame with timestamp as index, tickers as columns
    """
    df = load_stock_bars(tickers, interval=interval, period=period)
    
    # Pivot to matrix
    matrix = df.pivot_table(
        index='timestamp',
        columns='symbol',
        values=field,
        aggfunc='first'
    )
    
    return matrix


def get_sp500_tickers() -> List[str]:
    """Get list of S&P 500 tickers (top 50 by market cap)."""
    # Top 50 most liquid S&P 500 components
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'TESLA', 'META', 'BRK.B', 'JNJ', 'V',
        'WMT', 'JPM', 'PG', 'COST', 'HD',
        'AXP', 'MS', 'IBM', 'GE', 'BA',
        'MMM', 'INTC', 'AMD', 'CSCO', 'QCOM',
        'ABBV', 'XOM', 'CVX', 'MRK', 'KLAC',
        'CRM', 'ACN', 'AVGO', 'ASML', 'ARM',
        'MSTR', 'LRCX', 'MU', 'SNPS', 'CDNS',
        'ADBE', 'NFLX', 'TSM', 'UBER', 'SNOW',
        'COIN', 'CRWD', 'DDOG', 'MELI', 'SPOT',
    ]


def get_tech_tickers() -> List[str]:
    """Get list of major tech stocks for testing."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TESLA']


def load_market_data_snapshot(
    tickers: List[str] = None,
    interval: str = "1m",
    period: str = "1d",
) -> pd.DataFrame:
    """Quick snapshot of multiple stocks at a specific interval."""
    if tickers is None:
        tickers = get_tech_tickers()
    
    return load_stock_bars(tickers, interval=interval, period=period)


if __name__ == "__main__":
    print("Testing Yahoo Finance stock loader...")
    
    # Load single ticker
    print("\n1. Loading single ticker (AAPL, 1m, last 7d)...")
    try:
        df_single = load_stock_bars('AAPL', interval='1m', period='7d', progress=False)
        print(f"   ✓ Loaded {len(df_single)} bars")
        print(f"   Date range: {df_single['timestamp'].min()} to {df_single['timestamp'].max()}")
        print(f"   Sample:\n{df_single.head()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Load multiple tickers
    print("\n2. Loading multiple tickers (AAPL, MSFT, 5m, 1d)...")
    try:
        df_multi = load_stock_bars(['AAPL', 'MSFT'], interval='5m', period='1d', progress=False)
        print(f"   ✓ Loaded {len(df_multi)} bars")
        print(f"   Symbols: {df_multi['symbol'].unique()}")
        print(f"   Sample:\n{df_multi.head()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Load as matrix
    print("\n3. Loading as close price matrix (AAPL, MSFT, GOOGL, 1h, 1d)...")
    try:
        matrix = load_stock_ohlcv_matrix(['AAPL', 'MSFT', 'GOOGL'], interval='1h', period='1d')
        print(f"   ✓ Matrix shape: {matrix.shape}")
        print(f"   Sample:\n{matrix.head()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✓ yfinance stock loader ready for production")
