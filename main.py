"""
main.py
End-to-end pipeline.
Demonstrates: fetch → indicators → LLM (+RAG, mock fallback) → rules → report
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path (works regardless of the current working directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.config import TICKERS, START_DATE, END_DATE
from data.fetcher import fetch_price_data
from data.indicators import compute_technical_features, get_indicator_summary
from rules.engine import SymbolicEngine
from llm.mock import mock_llm_analyzer  # re-exported for backward compatibility


def run_pipeline(ticker_symbol: str, ticker_name: str = None):
    """
    Run the full pipeline for a single ticker.
    """
    name = ticker_name or ticker_symbol
    print(f"\n{'#' * 70}")
    print(f"# PIPELINE RUN: {name} ({ticker_symbol})")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")

    # ── Step 1: Fetch data ────────────────────────────────────
    print("─" * 50)
    print("STEP 1: Fetching price data...")
    print("─" * 50)
    df = fetch_price_data(ticker_symbol, START_DATE, END_DATE)
    print(f"  ✓ {len(df)} rows loaded\n")

    # ── Step 2: Compute indicators ────────────────────────────
    print("─" * 50)
    print("STEP 2: Computing technical indicators...")
    print("─" * 50)
    df = compute_technical_features(df)
    summary = get_indicator_summary(df)
    summary['ticker'] = ticker_symbol   # used by RAG to filter documents
    summary['name'] = name              # alias, e.g. "DBS" for "D05.SI"
    print(f"  Latest date: {summary['date']}")
    print(f"  Price:       {summary['price']}")
    print(f"  MA Cross:    {summary['signals']['ma_cross']}")
    print(f"  RSI:         {summary['signals']['rsi']} ({summary['signals']['rsi_state']})")
    print(f"  MACD:        {summary['signals']['macd']}")
    print(f"  5d Momentum: {summary['signals']['momentum_5d']}%\n")

    # ── Step 3: LLM analysis ─────────────────────────────────────
    print("─" * 50)
    print("STEP 3: LLM Analysis (Ollama + Qwen3-8B)...")
    print("─" * 50)

    from llm.analyzer import FinancialAnalyzer, LLMConfig

    config = LLMConfig(provider="ollama", model="qwen3:8b")

    try:
        analyzer = FinancialAnalyzer(config)
        llm_output = analyzer.analyze(summary)
        print(f"  Recommendation: {llm_output['recommendation']}")
        print(f"  Confidence:     {llm_output['confidence']:.2%}")
        print(f"  Sentiment:      {llm_output['sentiment']} ({llm_output['sentiment_score']})")
        print(f"  Risk flags:     {llm_output['risk_flags']}")
        print(f"  Reasoning:      {llm_output['reasoning'][:300]}...")
    except Exception as e:
        print(f"  ✗ LLM failed ({e}), falling back to mock")
        llm_output = mock_llm_analyzer(summary)
        print(f"  Recommendation: {llm_output['recommendation']}")
        print(f"  Confidence:     {llm_output['confidence']:.2%}")
        print(f"  Sentiment:      {llm_output['sentiment']} ({llm_output['sentiment_score']})")
        print(f"  Risk flags:     {llm_output['risk_flags']}")
        print(f"  Reasoning:      {llm_output['reasoning']}")

    print()

    # ── Step 4: Symbolic rule validation ──────────────────────
    print("─" * 50)
    print("STEP 4: Symbolic Rule Engine validation...")
    print("─" * 50)
    engine = SymbolicEngine()
    log = engine.validate(llm_output, summary)
    print(engine.format_validation_report(log))

    # ── Step 5: Final output (mock memo) ──────────────────────
    print("\n" + "─" * 50)
    print("STEP 5: Final Investment Memo (template)")
    print("─" * 50)
    print(f"""
    INVESTMENT MEMO — {name} ({ticker_symbol})
    Date: {summary['date']}
    Price: {summary['price']}

    RECOMMENDATION: {log.final_recommendation}
    CONFIDENCE: {log.final_confidence:.2%}
    UNCERTAINTY: {log.uncertainty:.2%}

    Technical Snapshot:
      • MA Cross: {summary['signals']['ma_cross']}
      • RSI: {summary['signals']['rsi']} ({summary['signals']['rsi_state']})
      • MACD: {summary['signals']['macd']}
      • 5d Return: {summary['signals']['momentum_5d']}%

    LLM Reasoning: {llm_output['reasoning']}

    Rule Validation: {"PASSED (no conflicts)" if not log.conflicts else f"{len(log.conflicts)} conflict(s) resolved"}
    {"  Conflicts: " + "; ".join(log.conflicts) if log.conflicts else ""}

    Risk Flags: {', '.join(llm_output['risk_flags']) if llm_output['risk_flags'] else 'None'}
    """)

    return {
        'ticker': ticker_symbol,
        'date': summary['date'],
        'price': summary['price'],
        'llm_recommendation': llm_output['recommendation'],
        'final_recommendation': log.final_recommendation,
        'final_confidence': log.final_confidence,
        'uncertainty': log.uncertainty,
        'conflicts': log.conflicts,
        'indicators': summary['signals']
    }


if __name__ == "__main__":
    # Pick ticker from config
    if isinstance(TICKERS, dict):
        first_name = list(TICKERS.keys())[0]
        first_symbol = list(TICKERS.values())[0]
    else:
        first_name = first_symbol = TICKERS[0]

    result = run_pipeline(first_symbol, first_name)

    # Save result as JSON for later use
    output_path = 'pipeline_output.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✓ Pipeline output saved to {output_path}")