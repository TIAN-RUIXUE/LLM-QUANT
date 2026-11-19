# llm/client.py

import json
from typing import Dict, List, Optional


def collect_docs(context: Dict) -> List[Dict]:
    """
    Unified document retrieval order: feature explanations (feature_docs) first,
    then stock-specific news (news_docs).
    [sourceN] numbering follows this order (N starts from 1).
    """
    return list(context.get("feature_docs") or []) + list(context.get("news_docs") or [])


def _first_sentence(text: str) -> str:
    idx = text.find(". ")
    return text if idx == -1 else text[:idx + 1]


class MockLLMClient:
    """
    Mock LLM client.
    Generates structured investment explanations based on SHAP attributions
    and RAG-retrieved feature explanations.
    """

    def __init__(self):
        self.model_name = "mock-llm-shap-v1"

    def analyze(self, context: Dict) -> Dict:
        """
        Generate investment recommendation and explanation based on SHAP attributions.

        context should contain:
            - prediction: prediction probability
            - action: BUY/SELL/HOLD
            - shap_top_features: SHAP top-k feature list
            - feature_docs: RAG-retrieved feature explanations (optional)
            - news_docs: RAG-retrieved stock news (optional)
            - portfolio_summary: portfolio summary (optional)
        """
        prediction = context.get("prediction", 0.5)
        action = context.get("action", "HOLD")
        features = context.get("shap_top_features", [])
        summary = context.get("portfolio_summary", {})
        docs = collect_docs(context)
        doc_index = {d.get("feature"): i for i, d in enumerate(docs, 1) if d.get("feature")}
        citations = []

        # Build reasoning text
        reason_parts = []

        # 1. Overall judgment
        if action == "BUY":
            reason_parts.append(f"Model predicts an upward probability of {prediction:.0%}, issuing a BUY signal.")
        elif action == "SELL":
            reason_parts.append(f"Model predicts an upward probability of only {prediction:.0%}, issuing a SELL signal.")
        else:
            reason_parts.append(f"Model predicts an upward probability of {prediction:.0%}, recommending HOLD.")

        # 2. SHAP-driven feature explanation
        if features:
            pos_features = [f for f in features if f['impact'] == 'positive']
            neg_features = [f for f in features if f['impact'] == 'negative']

            if pos_features:
                names = [f['name'] for f in pos_features[:3]]
                reason_parts.append(f"Positive drivers: {', '.join(names)} push up the prediction probability.")

            if neg_features:
                names = [f['name'] for f in neg_features[:3]]
                reason_parts.append(f"Negative drivers: {', '.join(names)} pull down the prediction probability.")

            # The single most important feature
            top = features[0]
            direction = "pushes up" if top['impact'] == 'positive' else "pulls down"
            reason_parts.append(
                f"The most critical feature is {top['name']} (current value {top['value']:.4f}), "
                f"with a SHAP value of {top['shap']:.4f}, which {direction} the prediction."
            )

            # 3. RAG: cite knowledge base explanations for these intermediate parameters
            for f in features[:3]:
                n = doc_index.get(f['name'])
                if n is not None:
                    reason_parts.append(f"{f['name']}: {_first_sentence(docs[n - 1]['content'])} [source{n}]")
                    citations.append(f"source{n}")

        # 4. Stock news (if any)
        for i, d in enumerate(docs, 1):
            if not d.get("feature"):
                reason_parts.append(f"Related news: {d.get('title', '')} [source{i}]")
                citations.append(f"source{i}")

        # 5. Portfolio performance (if available)
        if summary:
            wr = summary.get('win_rate', 0)
            if wr >= 60:
                verdict = "strategy overall performance is robust"
            elif wr >= 45:
                verdict = "strategy performance is average"
            else:
                verdict = "strategy performance is weak, proceed with caution"
            reason_parts.append(f"Historical win rate {wr:.0f}%, {verdict}.")

        reason = " ".join(reason_parts)

        # Risk level and confidence
        if action == "HOLD":
            risk = "Medium"
            confidence = 0.5 + abs(prediction - 0.5)
        elif action == "BUY":
            risk = "Low" if prediction > 0.8 else "Medium"
            confidence = prediction
        else:
            risk = "High" if prediction < 0.2 else "Medium"
            confidence = 1 - prediction

        confidence = min(max(confidence, 0), 1)

        return {
            "action": action,
            "reason": reason,
            "risk_level": risk,
            "confidence": round(confidence, 2),
            "citations": citations,
            # Structured factor claims (used by evaluation/faithfulness.py)
            "factors": [
                {"feature": f['name'], "direction": "up" if f['impact'] == 'positive' else "down"}
                for f in features
            ],
            "model": self.model_name,
        }


def _clean_factors(raw) -> List[Dict]:
    """Keep only well-formed {"feature": str, "direction": "up"|"down"} items."""
    if not isinstance(raw, list):
        return []
    factors = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature", "")).strip()
        direction = str(item.get("direction", "")).strip().lower()
        if feature and direction in ("up", "down"):
            factors.append({"feature": feature, "direction": direction})
    return factors


class RealLLMClient:
    """
    Real LLM client (Ollama / DeepSeek / OpenAI, OpenAI-compatible interface).
    Lets LLM explain WHY the quantitative model gave its recommendation,
    based on SHAP attributions + RAG-retrieved feature explanations.
    LLM only explains, does not change the model's trading action.
    Falls back to MockLLMClient on failure.
    """

    SYSTEM_PROMPT = """You are a quantitative investment explanation assistant. An XGBoost model has already provided a trading action. Your task is to explain to investors WHY the model gave this recommendation.

Rules:
1. Do NOT change the model's trading action (action), only explain it.
2. Base your explanation on the provided SHAP attributions: positive SHAP values push up the upward probability, negative SHAP values pull it down.
3. Use the "retrieved reference information" to explain the meaning of each key intermediate parameter, and cite sources like [source2] after relevant sentences.
4. Only cite sources that are actually provided; do not fabricate data, news, or sources.
5. If a feature's SHAP contribution contradicts common financial interpretation, or if the feature is merely a price/volume level, explicitly point this out.
6. In "factors", list every feature you rely on, using the exact feature name from the SHAP list, and whether it pushed the upward probability "up" or "down".
7. Output ONLY JSON, no other content:
{
    "action": "BUY/SELL/HOLD (consistent with model)",
    "reason": "explanation text with [sourceN] citations",
    "factors": [{"feature": "exact_feature_name", "direction": "up/down"}],
    "risk_level": "Low/Medium/High",
    "confidence": number between 0.0-1.0,
    "citations": ["source1", "..."]
}"""

    def __init__(self, config=None, fallback: Optional[MockLLMClient] = None):
        from llm.analyzer import LLMConfig
        self.config = config or LLMConfig()
        self.model_name = f"{self.config.provider}:{self.config.model}"
        self.fallback = fallback or MockLLMClient()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from llm.analyzer import create_openai_client
            self._client = create_openai_client(self.config).with_options(max_retries=1, timeout=180)
        return self._client

    def build_user_prompt(self, context: Dict) -> str:
        from rag.retriever import RAGRetriever

        docs = collect_docs(context)
        lines = [
            f"## Model Output",
            f"- Ticker: {context.get('ticker', 'Unknown')}",
            f"- Date: {context.get('date', 'Unknown')}",
            f"- Trading Action: {context.get('action', 'HOLD')}",
            f"- Predicted Upward Probability: {context.get('prediction', 0.5):.2%}",
            f"- Base Probability (mean of all samples): {context.get('base_value', 0.5):.2%}",
            "",
            "## SHAP Attributions (sorted by |SHAP| descending, value = raw feature value)",
        ]
        for f in context.get("shap_top_features", []):
            lines.append(f"- {f['name']}: value={f['value']:.6g}, SHAP={f['shap']:+.4f} ({f['impact']})")

        summary = context.get("portfolio_summary") or {}
        if summary:
            lines += ["", "## Backtest Performance", json.dumps(summary, ensure_ascii=False, default=float)]

        if docs:
            lines += ["", "## Retrieved Reference Information", RAGRetriever.format_for_prompt(docs)]

        lines += ["", "## Task", "Please explain in JSON format why the model gave this trading action."]
        return "\n".join(lines)

    def analyze(self, context: Dict) -> Dict:
        action = context.get("action", "HOLD")
        try:
            from llm.analyzer import extract_json

            kwargs = dict(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.build_user_prompt(context)},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            if self.config.provider == "ollama":
                kwargs["extra_body"] = {"think": False}
            else:
                kwargs["response_format"] = {"type": "json_object"}

            response = self._get_client().chat.completions.create(**kwargs)
            parsed = extract_json(response.choices[0].message.content)
        except Exception as e:
            print(f"  ✗ LLM call failed ({self.model_name}): {e}")
            print("  -> Falling back to MockLLMClient")
            return self.fallback.analyze(context)

        # LLM only explains, does not change model action
        if str(parsed.get("action", action)).upper() != action:
            print(f"  ! LLM attempted to change action to {parsed.get('action')}; keeping model action {action}")
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        citations = parsed.get("citations") or []

        return {
            "action": action,
            "reason": str(parsed.get("reason", "")),
            "risk_level": parsed.get("risk_level") if parsed.get("risk_level") in ("Low", "Medium", "High") else "Medium",
            "confidence": round(confidence, 2),
            "citations": citations if isinstance(citations, list) else [str(citations)],
            "factors": _clean_factors(parsed.get("factors")),
            "model": self.model_name,
        }


if __name__ == "__main__":
    from rag.retriever import RAGRetriever

    context = {
        "prediction": 0.78,
        "action": "BUY",
        "shap_top_features": [
            {"name": "return_5d", "value": 0.023, "shap": 0.156, "impact": "positive"},
            {"name": "rsi", "value": 62.5, "shap": 0.092, "impact": "positive"},
            {"name": "macd_histogram", "value": -0.134, "shap": -0.078, "impact": "negative"},
            {"name": "volume_ratio", "value": 1.245, "shap": 0.045, "impact": "positive"},
            {"name": "volatility_20d", "value": 0.012, "shap": -0.034, "impact": "negative"}
        ],
        "portfolio_summary": {
            "total_trades": 12,
            "win_rate": 66.7,
            "total_return": 35.2
        }
    }
    context["feature_docs"] = RAGRetriever().lookup_features(
        [f["name"] for f in context["shap_top_features"]]
    )

    result = MockLLMClient().analyze(context)
    print(json.dumps(result, indent=2, ensure_ascii=False))