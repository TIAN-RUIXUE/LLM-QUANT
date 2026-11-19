"""
llm/analyzer.py
Real LLM integration for financial analysis.
Supports Ollama (local), DeepSeek, OpenAI backends.
"""

import json
import os
import re
from typing import Dict, Optional
from dataclasses import dataclass

OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"
PROVIDER_DEFAULT_URLS = {
    "ollama": OLLAMA_DEFAULT_URL,
    "deepseek": "https://api.deepseek.com",
    "openai": None,  # OpenAI SDK default
}
PROVIDER_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def create_openai_client(config: "LLMConfig"):
    """Build an OpenAI-compatible client for Ollama / DeepSeek / OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    provider = config.provider
    if provider not in PROVIDER_DEFAULT_URLS:
        raise ValueError(f"Unknown provider: {provider}")
    base_url = config.base_url or PROVIDER_DEFAULT_URLS[provider]

    if provider == "ollama":
        client = OpenAI(base_url=base_url, api_key="ollama")
    else:
        env_name = PROVIDER_KEY_ENV[provider]
        api_key = config.api_key or os.environ.get(env_name)
        if not api_key:
            raise ValueError(f"Set the {env_name} environment variable.")
        client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"  ✓ {provider} client initialized (model: {config.model})")
    return client


def extract_json(content: str) -> Dict:
    """
    Parse a JSON object from an LLM reply, tolerating <think> blocks,
    markdown code fences and prose around the JSON.
    """
    if not content or not content.strip():
        raise ValueError("Empty response from LLM")

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", content.strip())

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} block if the model added prose around it
        start, end = content.find("{"), content.rfind("}")
        try:
            if start == -1 or end <= start:
                raise json.JSONDecodeError("no JSON object found", content, 0)
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM output is not valid JSON: {e}\nRaw output:\n{content}")

    if not isinstance(parsed, dict):
        raise ValueError(f"LLM output is not a JSON object: {content}")
    return parsed


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen3:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # None -> provider default (see PROVIDER_DEFAULT_URLS)
    temperature: float = 0.1
    max_tokens: int = 2048


class FinancialAnalyzer:
    """
    Calls LLM to analyze technical indicators.
    Returns structured JSON matching the schema expected by rules/engine.py
    """

    SYSTEM_PROMPT = """You are a financial analysis assistant. Given technical indicators for a stock, output a structured analysis.

    You MUST output ONLY valid JSON in this exact schema:
    {
        "recommendation": "BUY" | "HOLD" | "SELL",
        "confidence": float between 0.0 and 1.0,
        "sentiment_score": float between -1.0 and 1.0,
        "sentiment": "positive" | "negative" | "neutral",
        "events": [{"type": "...", "magnitude": "...", "direction": "..."}],
        "risk_flags": ["flag1", "flag2"],
        "reasoning": "chain of thought linking evidence to recommendation",
        "evidence_citations": []
    }

    Rules:
    1. Base your recommendation on the provided indicators ONLY.
    2. If indicators conflict, prefer HOLD with moderate confidence.
    3. If RSI > 70, do NOT recommend BUY unless there is overwhelming positive news.
    4. If RSI < 30, do NOT recommend SELL.
    5. Be conservative with confidence — if uncertain, output lower confidence.
    6. Do NOT output any text outside the JSON block.
    7. **IMPORTANT**: The "## Retrieved Context" section contains real financial news and filings.
    8. You MUST reference these sources in your `reasoning` field when they are relevant.
    9. Add source citations like "[source1]", "[source2]" in your reasoning text.
    10. If the retrieved context contradicts an indicator signal, mention the contradiction."""

    def __init__(self, config: LLMConfig = None, use_rag: bool = True):
        self.config = config or LLMConfig()
        self._client = None
        self.use_rag = use_rag
        self._retriever = None
        
        if self.use_rag:
            from rag.retriever import RAGRetriever
            self._retriever = RAGRetriever()
        self._init_client()

    def _init_client(self):
        self._client = create_openai_client(self.config)

    def analyze(self, indicators_summary: Dict, context_docs: Optional[str] = None) -> Dict:
        """
        Main entry point. Sends indicators to LLM, parses JSON response.
        """

        rag_context = ""
        if self.use_rag and self._retriever:
            # Extract query keywords from indicators (indicator name + state, e.g. "rsi_state OVERBOUGHT macd NEGATIVE")
            signals = indicators_summary.get('signals', {})
            query_text = " ".join(f"{k} {v}" for k, v in signals.items())
            ticker = indicators_summary.get('ticker', '')
            aliases = [a for a in [indicators_summary.get('name')] if a]

            retrieved_docs = self._retriever.retrieve(query_text, ticker=ticker, aliases=aliases)
            if retrieved_docs:
                rag_context = self._retriever.format_for_prompt(retrieved_docs)
                print(f"  ✓ RAG retrieved {len(retrieved_docs)} relevant documents")
        
        # Merge context_docs (manually passed) and rag_context (auto-retrieved)
        combined_context = context_docs or ""
        if rag_context:
            combined_context = combined_context + "\n" + rag_context if combined_context else rag_context
        

        user_prompt = self._build_prompt(indicators_summary, combined_context)

        try:
            kwargs = dict(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Ollama-specific tweaks
            if self.config.provider == "ollama":
                kwargs["extra_body"] = {"think": False}
                # DO NOT pass response_format to Ollama
            else:
                kwargs["response_format"] = {"type": "json_object"}

            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            if not content or not content.strip():
                raise ValueError("Empty response from LLM")

            return self._parse_response(content)

        except Exception as e:
            print(f"  ✗ LLM API call failed ({self.config.provider}): {e}")
            print("  -> Falling back to mock analyzer")
            from llm.mock import mock_llm_analyzer
            return mock_llm_analyzer(indicators_summary)

    def _build_prompt(self, indicators_summary: Dict, context_docs: Optional[str]) -> str:
        prompt_parts = []

        prompt_parts.append("## Technical Indicators")
        prompt_parts.append(json.dumps(indicators_summary.get('signals', {}), indent=2))
        prompt_parts.append(f"\nPrice: {indicators_summary.get('price')}")
        prompt_parts.append(f"Date: {indicators_summary.get('date')}")

        if context_docs:
            prompt_parts.append("\n## Retrieved Context (News / Filings)")
            prompt_parts.append(context_docs)

        prompt_parts.append("\n## Task")
        prompt_parts.append("Analyze the above and output your recommendation as JSON per the schema.")

        full_prompt = "\n".join(prompt_parts)
    
        # === Debug: print the actual prompt sent to LLM ===
        print("\n" + "="*60)
        print("FULL PROMPT SENT TO LLM:")
        print("="*60)
        print(full_prompt[:2000])  # Print only first 2000 chars to avoid flooding
        print("="*60 + "\n")
        
        return full_prompt

    def _parse_response(self, content: str) -> Dict:
        if not content:
            raise ValueError("Empty response from LLM")

        parsed = extract_json(content)

        # Fill every schema field (downstream code indexes risk_flags, sentiment, ...)
        for field in ('recommendation', 'confidence', 'sentiment_score', 'sentiment',
                      'events', 'risk_flags', 'reasoning', 'evidence_citations'):
            if parsed.get(field) is None:
                parsed[field] = self._default_value(field)

        # Normalize
        parsed['recommendation'] = str(parsed['recommendation']).strip().upper()
        if parsed['recommendation'] not in ('BUY', 'SELL', 'HOLD'):
            parsed['recommendation'] = 'HOLD'

        parsed['confidence'] = self._clamp(parsed['confidence'], 0.0, 1.0, 0.5)
        parsed['sentiment_score'] = self._clamp(parsed['sentiment_score'], -1.0, 1.0, 0.0)
        if not isinstance(parsed['risk_flags'], list):
            parsed['risk_flags'] = [str(parsed['risk_flags'])]

        return parsed

    @staticmethod
    def _clamp(value, lo: float, hi: float, default: float) -> float:
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return default

    def _default_value(self, field: str):
        defaults = {
            'recommendation': 'HOLD',
            'confidence': 0.5,
            'sentiment_score': 0.0,
            'sentiment': 'neutral',
            'events': [],
            'risk_flags': [],
            'reasoning': 'Insufficient data for analysis',
            'evidence_citations': []
        }
        return defaults.get(field, None)