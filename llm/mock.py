"""
llm/mock.py
Heuristic stand-in for the LLM, used when the real LLM is unavailable.
"""


def mock_llm_analyzer(indicators_summary: dict) -> dict:
    """
    MOCK LLM output, simulated from indicator signals.
    Returns the same schema as llm.analyzer.FinancialAnalyzer.analyze().
    """
    signals = indicators_summary.get('signals', {})

    bullish_count = sum([
        signals.get('ma_cross') == 'BULLISH',
        signals.get('macd') == 'POSITIVE',
        signals.get('rsi_state') == 'OVERSOLD',
    ])
    bearish_count = sum([
        signals.get('ma_cross') == 'BEARISH',
        signals.get('macd') == 'NEGATIVE',
        signals.get('rsi_state') == 'OVERBOUGHT',
    ])

    if bullish_count > bearish_count:
        rec = 'BUY'
        conf = 0.70 + 0.05 * (bullish_count - bearish_count)
    elif bearish_count > bullish_count:
        rec = 'SELL'
        conf = 0.70 + 0.05 * (bearish_count - bullish_count)
    else:
        rec = 'HOLD'
        conf = 0.50

    sentiment_score = 0.2 if bullish_count > bearish_count else (-0.2 if bearish_count > bullish_count else 0.0)

    risk_flags = []
    if signals.get('rsi_state') == 'OVERBOUGHT':
        risk_flags.append('overbought_condition')
    volatility = signals.get('volatility')
    if isinstance(volatility, (int, float)) and volatility > 3.0:
        risk_flags.append('high_volatility')

    return {
        'recommendation': rec,
        'confidence': min(conf, 0.95),
        'sentiment_score': sentiment_score,
        'sentiment': 'positive' if sentiment_score > 0 else ('negative' if sentiment_score < 0 else 'neutral'),
        'events': [],
        'risk_flags': risk_flags,
        'reasoning': f"Based on technical analysis: MA={signals.get('ma_cross')}, "
                     f"RSI={signals.get('rsi')} ({signals.get('rsi_state')}), "
                     f"MACD={signals.get('macd')}. "
                     f"Bullish signals: {bullish_count}, Bearish: {bearish_count}.",
        'evidence_citations': [],
        'is_mock': True,
    }
