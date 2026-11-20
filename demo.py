# demo.py

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import pandas as pd
import numpy as np
import json

from strategy.quant_model_old import (
    compute_technical_features, prepare_ml_features,
    train_xgb_model, generate_ml_signals
)
from strategy.shap_explainer import (
    compute_shap_values, explain_prediction, get_feature_importance, get_base_value
)
from strategy.decision_engine import DecisionEngine
from llm.client import MockLLMClient, RealLLMClient
from llm.analyzer import LLMConfig
from rag.retriever import RAGRetriever
from data.config import TICKERS


def run_demo(ticker: str = "D05.SI", use_llm: bool = True, llm_config: LLMConfig = None):
    """
    use_llm: When True, calls real LLM (default Ollama qwen2.5:32b), falls back to MockLLMClient on failure.
    """
    print("=" * 60)
    print("Quant-LLM Explainable Quantitative Investment System Demo")
    print("=" * 60)
    
    # [1] Fetch data
    print(f"\n[1/6] Fetching data...")
    raw = yf.download(ticker, start='2024-09-24', end='2026-09-24', progress=False)
    print(f"  -> Fetched {len(raw)} records")
    
    # [2] Compute technical indicators
    print(f"\n[2/6] Computing technical indicators and training XGBoost model...")
    featured = compute_technical_features(raw)
    model, X_train, X_test, y_train, y_test, scaler = train_xgb_model(featured)
    
    # [3] Generate signals
    print(f"\n[3/6] Generating trading signals...")
    signaled = generate_ml_signals(featured, model, scaler, buy_threshold=0.85, sell_threshold=0.15)
    signal_dist = signaled['signal'].value_counts().to_dict()
    print(f"  -> Signal distribution: {signal_dist}")
    
    # [4] Backtest
    print(f"\n[4/6] Running backtest...")
    initial_capital = 100000.0
    capital = initial_capital
    position = 0
    trades = []

    for i in range(len(signaled)):
        date = signaled.index[i]
        row = signaled.iloc[i]
        price = row['close']
        signal = row['signal']
        
        if signal == 'BUY' and position == 0:
            # Buy: use 80% of current capital
            invest_amount = capital * 0.8
            position = invest_amount / price
            cash_reserve = capital * 0.2
            
            trades.append({
                'entry_date': date,
                'entry_price': price,
                'position': position,
                'cash_before': capital,
                'invested': invest_amount,
                'cash_reserve': cash_reserve
            })
            
        elif signal == 'SELL' and position > 0:
            trade = trades[-1]
            trade['exit_date'] = date
            trade['exit_price'] = price
            
            # Total assets after sell = position market value + cash reserve
            new_capital = position * price + trade['cash_reserve']
            trade['pnl'] = new_capital - trade['cash_before']
            trade['return_pct'] = (new_capital - trade['cash_before']) / trade['cash_before'] * 100
            
            capital = new_capital
            position = 0

    # Handle case where position is still held at the end
    if position > 0:
        last_price = signaled.iloc[-1]['close']
        capital = position * last_price + trades[-1]['cash_reserve']

    final_asset = capital
    total_return = (final_asset - initial_capital) / initial_capital * 100

    print(f"  -> Initial capital: ${initial_capital:,.2f}")
    print(f"  -> Final assets: ${final_asset:,.2f}")
    print(f"  -> Total return: {total_return:.2f}%")
    
    # [5] Trade parsing
    print(f"\n[5/6] Parsing trade records...")
    completed_trades = [t for t in trades if 'exit_date' in t]
    wins = [t for t in completed_trades if t.get('return_pct', 0) > 0]
    win_rate = len(wins) / len(completed_trades) * 100 if completed_trades else 0

    print(f"  -> {len(completed_trades)} completed trades")
    print(f"  -> Win rate: {win_rate:.1f}%")

    # Display the last trade
    if completed_trades:
        last = completed_trades[-1]
        print(f"\n  Last trade:")
        print(f"    Entry: {last['entry_date'].strftime('%Y-%m-%d')} @ ${last['entry_price']:.2f}")
        print(f"    Exit: {last['exit_date'].strftime('%Y-%m-%d')} @ ${last['exit_price']:.2f}")
        ret = last.get('return_pct', 0)
        print(f"    Return: {ret:.2f}%")
    
    # [6] LLM analysis of the last trade (with SHAP explanation)
    print(f"\n[6/6] SHAP explanation + LLM investment memo...")
    
    # Prepare SHAP explanation
    feature_names = prepare_ml_features(featured).columns.tolist()
    all_shap_values = compute_shap_values(model, scaler.transform(prepare_ml_features(featured).fillna(0)), feature_names)
    
    # Find the row corresponding to the last trade
    if completed_trades:
        last_trade = completed_trades[-1]
        exit_idx = signaled.index.get_loc(last_trade['exit_date'])
        shap_for_exit = all_shap_values[exit_idx]
        X_row = prepare_ml_features(featured).iloc[exit_idx]
        
        explanation = explain_prediction(shap_for_exit, feature_names, X_row, top_k=5,
                                         base_value=get_base_value(model))

        # RAG: retrieve explanations for SHAP top features, and retrieve stock news
        retriever = RAGRetriever()
        top_names = [f['name'] for f in explanation['top_features']]
        feature_docs = retriever.lookup_features(top_names)
        aliases = [name for name, sym in TICKERS.items() if sym.upper() == ticker.upper()]
        news_docs = [d for d in retriever.retrieve("", ticker=ticker, aliases=aliases)
                     if d['relevance'] >= 0.5]  # only keep docs for this specific stock
        missing = [n for n in top_names if n not in {d['feature'] for d in feature_docs}]
        print(f"  -> RAG: {len(feature_docs)} feature explanations, {len(news_docs)} stock news items"
              + (f" (knowledge base missing: {', '.join(missing)})" if missing else ""))

        # Build LLM context
        context = {
            "ticker": f"{aliases[0]} ({ticker})" if aliases else ticker,
            "date": str(last_trade['exit_date'].date()),
            "prediction": explanation['prediction'],
            "base_value": explanation['base_value'],
            "action": signaled.iloc[exit_idx]['signal'],
            "shap_top_features": explanation['top_features'],
            "feature_docs": feature_docs,
            "news_docs": news_docs,
            "portfolio_summary": {
                "total_trades": len(completed_trades),
                "win_rate": win_rate,
                "total_return": total_return
            }
        }

        # Call LLM (falls back to Mock on failure)
        llm_client = RealLLMClient(llm_config) if use_llm else MockLLMClient()
        result = llm_client.analyze(context)

        print(f"\n  📊 SHAP Explanation (predicted probability {explanation['prediction']:.2%}, base {explanation['base_value']:.2%}):")
        for feat in explanation['top_features']:
            arrow = "↑" if feat['impact'] == 'positive' else "↓"
            print(f"     {feat['name']}: {feat['value']:.4f} -> SHAP={feat['shap']:.4f} {arrow}")

        print(f"\n  📚 RAG Reference Sources:")
        for i, d in enumerate(feature_docs + news_docs, 1):
            print(f"     [source{i}] {d['title']} ({d['source']})")

        print(f"\n  📋 LLM Investment Memo ({result.get('model', '')}):")
        print(f"     Action: {result['action']}")
        print(f"     Reason: {result['reason']}")
        print(f"     Risk Level: {result['risk_level']}")
        print(f"     Confidence: {result['confidence']:.0%}")
        print(f"     Citations: {', '.join(result.get('citations', [])) or 'None'}")
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()