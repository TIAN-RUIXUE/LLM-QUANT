# strategy/quant_model.py

import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators (keep existing logic unchanged)"""
    df = df.copy()
    
    # Handle column names (yfinance may return MultiIndex or capitalized column names)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [str(c).lower().replace(' ', '') for c in df.columns]

    required = ['close', 'high', 'low', 'open', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Returns
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    
    # Moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()
    df['ma_cross_signal'] = np.where(df['ma_5'] > df['ma_20'], 1, -1)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_signal'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal_line'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal_line']
    df['macd_cross_signal'] = np.where(df['macd_histogram'] > 0, 1, -1)
    
    # Volatility and volume
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    # Replace inf from division by zero with NaN to avoid StandardScaler errors
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based signals from moving average crossovers (used by evaluation/backtest.py and strategy/signals.py).
    MA5 crosses above MA20 (golden cross) -> BUY; MA5 crosses below MA20 (death cross) -> SELL; otherwise -> HOLD.
    """
    df = df.copy()
    above = df['ma_5'] > df['ma_20']
    prev_above = above.shift(1)
    valid = df['ma_5'].notna() & df['ma_20'].notna() & df['ma_20'].shift(1).notna()

    df['signal'] = 'HOLD'
    df.loc[valid & above & (prev_above == False), 'signal'] = 'BUY'
    df.loc[valid & ~above & (prev_above == True), 'signal'] = 'SELL'
    return df


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature matrix for machine learning.
    Select columns related to technical indicators, drop intermediate calculation columns.
    """
    feature_cols = [
        'return_1d', 'return_5d', 'return_20d',
        'ma_5', 'ma_20', 'ma_60', 'ma_cross_signal',
        'rsi', 'rsi_signal',
        'macd', 'macd_signal_line', 'macd_histogram', 'macd_cross_signal',
        'volatility_20d', 'volume_ma_20', 'volume_ratio'
    ]
    
    available = [c for c in feature_cols if c in df.columns]
    return df[available].copy()


def create_target(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    # Last `horizon` rows have no future price, must remain NaN (otherwise wrongly labeled as 0=down)
    target = (future_return > threshold).astype(float)
    target[future_return.isna()] = np.nan
    return target


def train_xgb_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    save_path: str = "models/xgb_model.json"
) -> tuple:
    """
    Train XGBoost classifier.
    
    Returns:
        model: trained model
        X_train, X_test, y_train, y_test: split data
        scaler: fitted StandardScaler
    """
    # Prepare features and target
    X = prepare_ml_features(df)
    y = create_target(df)
    
    # Drop rows with NaN
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx].astype(int)
    
    # Split dataset (chronological order, no shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    joblib.dump(scaler, save_path.replace('.json', '_scaler.pkl'))
    
    print(f"Model training complete, saved to {save_path}")
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def generate_ml_signals(
    df: pd.DataFrame,
    model: XGBClassifier,
    scaler: StandardScaler,
    buy_threshold: float = 0.8,
    sell_threshold: float = 0.20
) -> pd.DataFrame:
    """
    Generate trading signals using the trained XGBoost model.
    
    Parameters:
        buy_threshold: BUY when predicted upward probability exceeds this value
        sell_threshold: SELL when predicted upward probability falls below this value
    """
    df = df.copy()
    X = prepare_ml_features(df)
    
    # Find valid rows
    valid_idx = X.dropna().index
    X_valid = X.loc[valid_idx]
    
    # Predict probabilities
    X_scaled = scaler.transform(X_valid)
    prob_up = model.predict_proba(X_scaled)[:, 1]
    
    # Generate signals
    df['pred_prob'] = np.nan
    df.loc[valid_idx, 'pred_prob'] = prob_up
    
    df['signal'] = 'HOLD'
    df.loc[df['pred_prob'] >= buy_threshold, 'signal'] = 'BUY'
    df.loc[df['pred_prob'] <= sell_threshold, 'signal'] = 'SELL'
    
    # Signal strength: distance from threshold
    df['signal_strength'] = 0.0
    df.loc[df['signal'] == 'BUY', 'signal_strength'] = df['pred_prob'] - buy_threshold
    df.loc[df['signal'] == 'SELL', 'signal_strength'] = sell_threshold - df['pred_prob']
    
    return df


if __name__ == "__main__":
    print("Testing XGBoost quant model on DBS...")
    
    # Fetch data
    raw = yf.download('D05.SI', start='2024-09-24', end='2026-09-24', progress=False)
    print(f"  -> Fetched {len(raw)} rows")
    
    # Compute technical indicators
    featured = compute_technical_features(raw)
    
    # Train model
    model, X_train, X_test, y_train, y_test, scaler = train_xgb_model(featured)
    
    # Generate signals
    signaled = generate_ml_signals(featured, model, scaler)
    
    # View signal distribution
    print(f"\nSignal Distribution:")
    print(signaled['signal'].value_counts())
    
    # View last few rows of prediction probabilities and signals
    print(f"\nLast 5 rows:")
    print(signaled[['close', 'pred_prob', 'signal']].tail(5))
    import matplotlib.pyplot as plt
    probs = signaled['pred_prob'].dropna()
    print(f"\nPrediction Probability Distribution:")
    print(f"  Min: {probs.min():.4f}")
    print(f"  25th percentile: {probs.quantile(0.25):.4f}")
    print(f"  Median: {probs.median():.4f}")
    print(f"  75th percentile: {probs.quantile(0.75):.4f}")
    print(f"  Max: {probs.max():.4f}")

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")