# evaluation/backtest.py

import pandas as pd
import numpy as np
import yfinance as yf

from strategy.quant_model_old import compute_technical_features, generate_signals
from evaluation.metrics import total_return, win_rate, sharpe_ratio, annualized_return, max_drawdown, profit_factor
from strategy.decision_engine import DecisionEngine


def compute_portfolio_returns(signals_df: pd.DataFrame, initial_capital: float = 100000.0) -> pd.DataFrame:
    df = signals_df.copy()
    df['trade_signal'] = df['signal'].shift(1)
    
    cash = initial_capital
    shares = 0
    portfolio = []
    decision_log = []

    for idx, row in df.iterrows():
        signal_today = row['trade_signal'] if isinstance(row['trade_signal'], str) else 'HOLD'
        price_today = row['close']
        action = None  # actual executed action (signal may not always execute)

        if signal_today == 'BUY' and shares == 0:
            invest_amount = cash * 0.8
            shares = invest_amount // price_today
            cash = cash - shares * price_today
            action = 'BUY'
        elif signal_today == 'SELL' and shares != 0:
            cash = cash + shares * price_today
            shares = 0
            action = 'SELL'

        total = cash + shares * price_today

        decision_log.append({
            "date": idx,
            "price": price_today,
            "signal": signal_today,
            "action": action,
            "ma_5": row["ma_5"],
            "ma_20": row["ma_20"],
            "shares": shares,
            "cash": cash,
            "total": total
        })
        
        portfolio.append(total)

    df['portfolio'] = portfolio
    df['daily_return'] = df['portfolio'].pct_change()
    return df, decision_log


if __name__ == "__main__":
    raw = yf.download('D05.SI', start='2024-09-24', end='2026-09-24', progress=False)
    featured = compute_technical_features(raw)
    signaled = generate_signals(featured)
    
    result, log = compute_portfolio_returns(signaled)
    
    print(result[['close', 'signal', 'trade_signal', 'portfolio', 'daily_return']].tail(10))
    print(f"\nFinal assets: ${result['portfolio'].iloc[-1]:.2f}")
    print(f"Total return: {(result['portfolio'].iloc[-1] / 100000 - 1) * 100:.2f}%")
    
    tr = total_return(result['portfolio'])
    ar = annualized_return(result['portfolio'])
    md = max_drawdown(result['portfolio'])
    sr = sharpe_ratio(result['daily_return'].dropna())
    
    print(f"\n=== Performance Metrics ===")
    print(f"Total Return: {tr:.4f} ({tr*100:.2f}%)")
    print(f"Annualized Return: {ar:.4f} ({ar*100:.2f}%)")
    print(f"Max Drawdown: {md:.4f} ({md*100:.2f}%)")
    print(f"Sharpe Ratio: {sr:.4f}")
    
    # Print last few trades
    trades = [e for e in log if e['action'] is not None]
    print(f"\n=== {len(trades)} trades executed ===")
    for t in trades[-5:]:
        print(f"{t['date'].date()} | {t['signal']:5s} | Price:{t['price']:.2f} | MA5:{t['ma_5']:.2f} MA20:{t['ma_20']:.2f} | Shares:{t['shares']:.0f} | Cash:${t['cash']:.0f} | Total:${t['total']:.0f}")
    
    engine = DecisionEngine(result, log)

    print("\n=== Trade Summary ===")
    summary = engine.summary()
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Last Trade Details ===")
    print(engine.explain_last_trade())

    print("\n=== All Trades Overview ===")
    for i, t in enumerate(engine.trades):
        status = "✅" if t.profit > 0 else "❌"
        print(f"{i+1:2d}. {status} {t.entry_date.date()} -> {t.exit_date.date()} "
            f"| Buy ${t.entry_price:.2f} -> Sell ${t.exit_price:.2f} "
            f"| PnL:{t.return_pct:+.2f}%")