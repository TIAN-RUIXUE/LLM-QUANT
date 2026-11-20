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


def run_pipeline(ticker_symbol: str, ticker_name: str = None, model: str = "qwen2.5:32b",
                 online: bool = True):
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
    print(f"  Price:       {summary['price']:.2f}")
    print(f"  MA Cross:    {summary['signals']['ma_cross']}")
    print(f"  RSI:         {summary['signals']['rsi']} ({summary['signals']['rsi_state']})")
    print(f"  MACD:        {summary['signals']['macd']}")
    print(f"  5d Momentum: {summary['signals']['momentum_5d']}%")
    sig = summary['signals']
    print(f"  Volatility:  {sig['volatility']}% daily ≈ {sig['volatility_annualized_pct']}% annualised "
          f"({sig['volatility_state']})\n")

    # ── Step 3: LLM analysis ─────────────────────────────────────
    print("─" * 50)
    print(f"STEP 3: LLM Analysis (Ollama + {model})...")
    print("─" * 50)

    from llm.analyzer import FinancialAnalyzer, LLMConfig

    config = LLMConfig(provider="ollama", model=model)

    try:
        analyzer = FinancialAnalyzer(config, online=online)
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

    # Explain in the memo why the final call differs from the LLM's (its reasoning argues for its own call)
    override_note = ""
    if log.final_recommendation != llm_output['recommendation']:
        reasons = [r.description for r in log.rules_fired if r.action.name.startswith("OVERRIDE")]
        override_note = (f"\n    NOTE: LLM proposed {llm_output['recommendation']} @ {llm_output['confidence']:.0%}; "
                         f"overridden to {log.final_recommendation} by the rule engine"
                         + (f" ({'; '.join(reasons)})" if reasons else "") + ".\n"
                         f"    The LLM reasoning below argues for its original call.")

    # ── Step 5: Final output (mock memo) ──────────────────────
    print("\n" + "─" * 50)
    print("STEP 5: Final Investment Memo (template)")
    print("─" * 50)
    print(f"""
    INVESTMENT MEMO — {name} ({ticker_symbol})
    Date: {summary['date']}
    Price: {summary['price']:.2f}

    RECOMMENDATION: {log.final_recommendation}
    CONFIDENCE: {log.final_confidence:.2%}
    UNCERTAINTY: {log.uncertainty:.2%}{override_note}

    Technical Snapshot:
      • MA Cross: {summary['signals']['ma_cross']}
      • RSI: {summary['signals']['rsi']} ({summary['signals']['rsi_state']})
      • MACD: {summary['signals']['macd']}
      • 5d Return: {summary['signals']['momentum_5d']}%
      • Volatility: {summary['signals']['volatility_annualized_pct']}% annualised ({summary['signals']['volatility_state']})

    LLM Reasoning: {llm_output['reasoning']}

    LLM said: {llm_output['recommendation']} @ {llm_output['confidence']:.2%}
    Rule Validation: {"PASSED (no conflicts)" if not log.conflicts else f"{len(log.conflicts)} conflict(s) detected"}{f", {len(log.rules_fired)} rule(s) fired" if log.rules_fired else ""}
    {"  Conflicts: " + "; ".join(log.conflicts) if log.conflicts else ""}

    Risk Flags: {', '.join(log.effective_risk_flags) if log.effective_risk_flags else 'None'}
    """)

    return {
        'ticker': ticker_symbol,
        'date': summary['date'],
        'price': summary['price'],
        'llm_recommendation': llm_output['recommendation'],
        'final_recommendation': log.final_recommendation,
        'final_confidence': log.final_confidence,
        'uncertainty': log.uncertainty,
        'llm_confidence': llm_output['confidence'],
        'conflicts': log.conflicts,
        'rules_fired': [r.rule_name for r in log.rules_fired],
        'risk_flags': log.effective_risk_flags,
        'indicators': summary['signals']
    }


if __name__ == "__main__":
    import argparse

    names = list(TICKERS.keys())
    ap = argparse.ArgumentParser(description="Run the LLM-Quant pipeline")
    ap.add_argument("--ticker", nargs="+", default=[names[0]],
                    help=f"ticker names or symbols (default: {names[0]}); choices: {', '.join(names)}")
    ap.add_argument("--all", action="store_true", help="run every ticker in data/config.py")
    ap.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    ap.add_argument("--output", default="pipeline_output.json")
    ap.add_argument("--no-online", action="store_true",
                    help="don't search the web; use only storage/news_cache.json")
    args = ap.parse_args()

    symbol_to_name = {v: k for k, v in TICKERS.items()}
    selected = names if args.all else args.ticker
    results = []
    for t in selected:
        if t in TICKERS:
            name, symbol = t, TICKERS[t]
        else:
            name, symbol = symbol_to_name.get(t, t), t
        try:
            results.append(run_pipeline(symbol, name, model=args.model, online=not args.no_online))
        except Exception as e:
            print(f"  ✗ Pipeline failed for {name} ({symbol}): {e}")

    # Single ticker keeps the original flat JSON shape; multiple tickers -> list
    with open(args.output, 'w') as f:
        json.dump(results[0] if len(results) == 1 else results, f, indent=2, default=str)
    print(f"\n✓ Pipeline output saved to {args.output}")
