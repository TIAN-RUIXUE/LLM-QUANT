# strategy/decision_engine.py

import pandas as pd
from typing import List, Dict


class TradeRecord:
    """Record of a single trade"""
    def __init__(self, entry_idx, exit_idx, entry_price, exit_price,
                 shares, entry_cash, exit_cash, entry_total, exit_total,
                 entry_ma=None, exit_ma=None):
        self.entry_ma = entry_ma or {}   # {"ma_5": .., "ma_20": ..} moving averages at entry
        self.exit_ma = exit_ma or {}     # moving averages at exit
        self.entry_date = entry_idx
        self.exit_date = exit_idx
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.shares = shares
        self.entry_cash = entry_cash
        self.exit_cash = exit_cash
        self.entry_total = entry_total
        self.exit_total = exit_total
        self.profit = exit_total - entry_total
        self.return_pct = (exit_total - entry_total) / entry_total * 100


class DecisionEngine:
    def __init__(self, result_df: pd.DataFrame, decision_log: List[Dict]):
        self.result = result_df
        self.log = decision_log
        self.trades = self._parse_trades()
    
    def _parse_trades(self) -> List[TradeRecord]:
        """Parse complete buy-sell pairs from the decision log"""
        trades = []
        entry = None
        
        for entry_dict in self.log:
            # Use actually executed actions; signals that didn't result in a trade
            # (e.g., BUY while already holding) should not change the entry point
            action = entry_dict.get('action', entry_dict.get('signal'))
            if action == 'BUY' and entry is None:
                entry = entry_dict
            elif action == 'SELL' and entry is not None:
                trades.append(TradeRecord(
                    entry_idx=entry['date'],
                    exit_idx=entry_dict['date'],
                    entry_price=entry['price'],
                    exit_price=entry_dict['price'],
                    shares=entry['shares'],
                    entry_cash=entry['cash'],
                    exit_cash=entry_dict['cash'],
                    entry_total=entry['total'],
                    exit_total=entry_dict['total'],
                    entry_ma={k: entry.get(k) for k in ('ma_5', 'ma_20')},
                    exit_ma={k: entry_dict.get(k) for k in ('ma_5', 'ma_20')}
                ))
                entry = None
        
        return trades
    
    def summary(self) -> Dict:
        """Return backtest summary"""
        if not self.trades:
            return {"message": "No trades executed"}
        
        wins = [t for t in self.trades if t.profit > 0]
        losses = [t for t in self.trades if t.profit <= 0]
        initial = self.result['portfolio'].iloc[0]

        return {
            "total_trades": len(self.trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100,
            "avg_profit_pct": sum(t.return_pct for t in self.trades) / len(self.trades),
            "total_profit": sum(t.profit for t in self.trades),
            "final_portfolio": self.result['portfolio'].iloc[-1],
            "total_return": (self.result['portfolio'].iloc[-1] / initial - 1) * 100
        }
    
    def explain_last_trade(self) -> str:
        """Generate explanation text for the last trade"""
        if not self.trades:
            return "No trades yet."
        
        t = self.trades[-1]

        def _ma_desc(ma: Dict, cross: str) -> str:
            m5, m20 = ma.get('ma_5'), ma.get('ma_20')
            if m5 is None or m20 is None:
                return cross
            return f"{cross}(MA5={m5:.2f}, MA20={m20:.2f})"

        return (
            f"Trade #{len(self.trades)}\n"
            f"Entry Date: {t.entry_date.date()}, Price: ${t.entry_price:.2f}\n"
            f"Exit Date: {t.exit_date.date()}, Price: ${t.exit_price:.2f}\n"
            f"Position: {t.shares} shares\n"
            f"PnL: ${t.profit:.2f} ({t.return_pct:.2f}%)\n"
            f"Entry Reason: {_ma_desc(t.entry_ma, 'Golden Cross signal: MA5 crosses above MA20')}\n"
            f"Exit Reason: {_ma_desc(t.exit_ma, 'Death Cross signal: MA5 crosses below MA20')}"
        )
    
    def build_prompt_context(self, trade_index: int = -1) -> Dict:
        """Build context for LLM"""
        t = self.trades[trade_index]
        return {
            "trade": {
                "entry_date": str(t.entry_date.date()),
                "exit_date": str(t.exit_date.date()),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "profit_pct": round(t.return_pct, 2),
                "shares": t.shares
            },
            "portfolio_summary": self.summary()
        }