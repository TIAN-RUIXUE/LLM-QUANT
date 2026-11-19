"""
data/indicators.py
Compute technical indicators from OHLCV data.
Output is a dict suitable for JSON serialization and LLM consumption.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns: Open, High, Low, Close, Volume
    (case-insensitive, handles MultiIndex from yfinance)
    Returns DataFrame with 15 technical indicators added.
    """
    # Normalize column names
    df = df.copy()
    df.columns = [str(c).lower().replace(' ', '') for c in df.columns]

    # Ensure we have the right columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Got: {list(df.columns)}")

    close = df['close']
    volume = df['volume']

    # --- Return features ---
    df['return_1d'] = close.pct_change(1)
    df['return_5d'] = close.pct_change(5)
    df['return_20d'] = close.pct_change(20)

    # --- Moving averages ---
    df['ma_5'] = close.rolling(5).mean()
    df['ma_20'] = close.rolling(20).mean()
    df['ma_60'] = close.rolling(60).mean()
    df['ma_cross_signal'] = np.where(df['ma_5'] > df['ma_20'], 1,
                              np.where(df['ma_5'] < df['ma_20'], -1, 0))

    # --- RSI (14-day) ---
    delta = close.diff(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    # No losses in the window -> RSI is 100 (not NaN); flat window stays NaN
    df.loc[(loss == 0) & (gain > 0), 'rsi'] = 100.0
    df['rsi_signal'] = np.where(df['rsi'] < 30, 1,
                         np.where(df['rsi'] > 70, -1, 0))

    # --- MACD ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal_line']
    df['macd_cross_signal'] = np.where(df['macd_histogram'] > 0, 1, -1)

    # --- Volatility & Volume ---
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    df['volume_ma_20'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_ma_20'].replace(0, np.nan)

    return df


def get_latest_indicators_json(df: pd.DataFrame, date: Optional[str] = None) -> Dict:
    """
    Extract the latest row of indicators as a JSON-serializable dict.
    This is what gets passed to the LLM.
    """
    if date:
        row = df.loc[date]
    else:
        row = df.iloc[-1]

    indicator_keys = [
        'close', 'return_1d', 'return_5d', 'return_20d',
        'ma_5', 'ma_20', 'ma_60', 'ma_cross_signal',
        'rsi', 'rsi_signal',
        'macd', 'macd_signal_line', 'macd_histogram', 'macd_cross_signal',
        'volatility_20d', 'volume_ma_20', 'volume_ratio'
    ]

    result = {}
    for key in indicator_keys:
        if key in row.index:
            val = row[key]
            if pd.isna(val):
                result[key] = None
            elif isinstance(val, (np.integer, int)):
                result[key] = int(val)
            elif isinstance(val, (np.floating, float)):
                result[key] = round(float(val), 6)
            else:
                result[key] = val
        else:
            result[key] = None

    result['date'] = str(row.name) if hasattr(row, 'name') else str(date)

    return result


def get_indicator_summary(df: pd.DataFrame, date: Optional[str] = None) -> Dict:
    """
    Return a human-readable summary of key signals for the rule engine.
    """
    indicators = get_latest_indicators_json(df, date)

    def _scaled(key: str, factor: float = 1.0):
        # `is not None` so that a legitimate 0.0 is not turned into None
        val = indicators.get(key)
        return round(val * factor, 2) if val is not None else None

    summary = {
        'date': indicators.get('date'),
        'price': indicators.get('close'),
        'signals': {
            'ma_cross': 'BULLISH' if indicators.get('ma_cross_signal') == 1 else
                        'BEARISH' if indicators.get('ma_cross_signal') == -1 else 'NEUTRAL',
            'rsi': _scaled('rsi'),
            'rsi_state': 'OVERSOLD' if indicators.get('rsi_signal') == 1 else
                         'OVERBOUGHT' if indicators.get('rsi_signal') == -1 else 'NORMAL',
            'macd': 'POSITIVE' if indicators.get('macd_cross_signal') == 1 else 'NEGATIVE',
            'momentum_5d': _scaled('return_5d', 100),
            'volatility': _scaled('volatility_20d', 100),
        },
        'raw_indicators': indicators
    }

    return summary


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append('.')
    from data.config import TICKERS
    from data.fetcher import fetch_price_data

    if isinstance(TICKERS, dict):
        symbol = list(TICKERS.values())[0]
    else:
        symbol = TICKERS[0]

    df = fetch_price_data(symbol)
    df = compute_technical_features(df)
    summary = get_indicator_summary(df)
    print(f"Date: {summary['date']}")
    print(f"Price: {summary['price']}")
    print(f"MA Cross: {summary['signals']['ma_cross']}")
    print(f"RSI: {summary['signals']['rsi']} ({summary['signals']['rsi_state']})")
    print(f"MACD: {summary['signals']['macd']}")
    print(f"5d Momentum: {summary['signals']['momentum_5d']}%")