# strategy/signals.py

import pandas as pd


def extract_technical_state(row: pd.Series) -> dict:
    """
    Extract structured technical state from a single row of data.
    Used by downstream decision engine or LLM explanation.
    """
    ma_short = row["ma_5"]
    ma_long = row["ma_60"]
    rsi = row["rsi"]
    macd_hist = row["macd_histogram"]

    ma_trend = "up" if ma_short > ma_long else "down"

    if rsi >= 70:
        rsi_state = "overbought"
    elif rsi <= 30:
        rsi_state = "oversold"
    else:
        rsi_state = "neutral"

    macd_state = "positive" if macd_hist > 0 else "negative"

    return {
        "date": str(row.name.date()),
        "close": float(row["close"]),
        "ma_short": float(ma_short),
        "ma_long": float(ma_long),
        "ma_trend": ma_trend,
        "rsi": float(rsi),
        "rsi_state": rsi_state,
        "macd_hist": float(macd_hist),
        "macd_state": macd_state,
        "original_signal": row["signal"]
    }


if __name__ == "__main__":
    import yfinance as yf
    from strategy.quant_model_old import compute_technical_features, generate_signals

    raw = yf.download('D05.SI', start='2024-09-24', end='2026-09-24', progress=False)
    featured = compute_technical_features(raw)
    signaled = generate_signals(featured)

    state = extract_technical_state(signaled.iloc[-1])
    print("Technical state on the last day:")
    for k, v in state.items():
        print(f"  {k}: {v}")