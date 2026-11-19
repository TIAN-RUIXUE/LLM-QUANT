# data/fetcher.py

import yfinance as yf
import pandas as pd
from datetime import datetime
from data.config import TICKERS, START_DATE,END_DATE

def fetch_price_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    '''
    fetch the history price of the single ticker 
    
    returns:
        dataframe with columns: open, high, low, close, volume, adj close
    '''

    start = start or START_DATE
    end = end or END_DATE

    print(f"Fetching price data for {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol.")

    print(f"    -> Got {len(data)} rows")
    return data

def fetch_multiple(tickers: dict = None, start: str = None, end: str = None) -> dict:
    '''
    fetch multiple tickers data

    args: tickers: dict of {name: ticker_symbol}

    returns:
        dict of {name: DataFrame}
    '''
    tickers = tickers or TICKERS
    start = start or START_DATE
    end = end or END_DATE

    results = {}
    for name, symbol in tickers.items():
        try:
            df  = fetch_price_data(symbol, start, end)
            results[name] = df
        except Exception as e:
            print(f"  ✗ Failed to fetch {name} ({symbol}): {e}")

    return results

def fetch_sp500_proxy(start: str = None, end: str = None) -> pd.DataFrame:
    start = start or START_DATE
    end = end or END_DATE

    print('Fetching S&P 500 benchmark...')
    data = yf.download('^GSPC', start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

if __name__ == '__main__':
    df  = fetch_price_data('D05.SI')
    print(df.tail())