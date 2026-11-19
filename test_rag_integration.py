# test_rag_integration.py
from llm.analyzer import FinancialAnalyzer, LLMConfig
import json

config = LLMConfig(provider="ollama", model="qwen3:8b")
analyzer = FinancialAnalyzer(config=config, use_rag=True)


indicators = {
    "ticker": "DBS",
    "price": 228.50,
    "date": "2025-10-15",
    "signals": {
        "rsi": 45.2,
        "macd": -0.32,
        "sma_20": 225.10,
        "sma_50": 220.80,
        "volume_ratio": 0.85
    }
}

result = analyzer.analyze(indicators)

print(json.dumps(result, indent=2))