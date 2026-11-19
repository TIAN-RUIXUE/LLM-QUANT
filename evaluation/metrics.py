# evaluation/metrics.py

import pandas as pd
import numpy as np
import math

def total_return(portfolio_values: pd.Series) -> float:
    return (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]

def annualized_return(portfolio_values: pd.Series, periods_per_year: int =252) -> float:
    total_ret = total_return(portfolio_values)
    n_period = len(portfolio_values) -1
    return (1+total_ret)**(periods_per_year / n_period) -1

def max_drawdown(portfolio_values: pd.Series) -> float:
    rolling_max = portfolio_values.cummax()
    drawdowns = portfolio_values / rolling_max - 1
    return drawdowns.min()

def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    excess_returns = daily_returns - risk_free_rate/252
    sharpe = (excess_returns.mean() / excess_returns.std()) * math.sqrt(252)
    return sharpe

def win_rate(trades: list) -> float:
    if len(trades) == 0:
        return 0.0
    winning_trades = sum(1 for t in trades if t > 0)
    return winning_trades / len(trades)

def profit_factor(trades: list) -> float:
    gains = sum(t for t in trades if t > 0)
    losses = abs(sum(t for t in trades if t < 0))
    if losses ==0:
        return float('inf')
    return gains / losses