# rag/retriever.py

import json
import re
from typing import Dict, List, Optional, Sequence
from pathlib import Path

# Knowledge base entries with empty ticker or values below are treated as "general knowledge"
# (e.g., indicator definitions), retrievable for any stock
GENERAL_TICKERS = {"", "GENERAL", "ALL", "*"}


def _tokenize(text: str) -> set:
    """Simple tokenizer: English words (including underscores, e.g., rsi_state), Chinese by character."""
    text = str(text).lower()
    words = set(re.findall(r"[a-z0-9_\.]+", text))
    # Split underscore words so "macd_histogram" also matches "macd"
    for w in list(words):
        if "_" in w:
            words.update(p for p in w.split("_") if p)
    # Pure numbers (e.g., "20", "58.94") do not participate in matching
    words = {w for w in words if not re.fullmatch(r"[\d\.]+", w)}
    words.update(re.findall(r"[一-鿿]", text))
    return {w for w in words if len(w) > 1 or "一" <= w <= "鿿"}


class RAGRetriever:

    def __init__(self, storage_path: Optional[str] = None):
        # Default to storage/ under project root, independent of current working directory
        if storage_path is None:
            storage_path = Path(__file__).resolve().parent.parent / "storage"
        self.storage_path = Path(storage_path)
        self.knowledge_base: List[Dict] = []
        self._init_index()

    def _init_index(self):
        """Initialize retrieval index (local files for now, vector DB later)"""
        kb_path = self.storage_path / "knowledge_base.json"
        if kb_path.exists():
            with open(kb_path, encoding="utf-8") as f:
                self.knowledge_base = json.load(f)
            print(f"  ✓ Loaded knowledge base: {len(self.knowledge_base)} records")
        else:
            self.knowledge_base = []
            print("  ! Knowledge base not found, RAG will return empty")

    def retrieve(self, query: str, ticker: str = "", top_k: int = 3,
                 aliases: Sequence[str] = ()) -> List[Dict]:
        """
        Retrieve documents relevant to the query.

        - ticker/aliases: stock ticker and aliases (e.g., "D05.SI" and "DBS"), any match works.
          When ticker is specified, only returns documents for that stock + general knowledge.
        - Scored by word overlap between query and title/content.
        Always returns list (may be empty).
        """
        if not self.knowledge_base:
            return []

        wanted = {t.upper() for t in (ticker, *aliases) if t}

        candidates = []
        for item in self.knowledge_base:
            item_ticker = str(item.get("ticker", "")).upper()
            is_general = item_ticker in GENERAL_TICKERS
            is_match = item_ticker in wanted
            if wanted and not (is_match or is_general):
                continue
            candidates.append((item, is_match))

        if not candidates:
            return []

        q_tokens = _tokenize(query)
        scored = []
        for item, is_match in candidates:
            title_tokens = _tokenize(item.get("title", ""))
            content_tokens = _tokenize(item.get("content", ""))
            overlap_title = len(q_tokens & title_tokens)
            overlap_content = len(q_tokens & content_tokens)
            score = 0.5 if is_match else 0.0
            if q_tokens:
                score += 0.3 * min(1.0, overlap_title / 2) + 0.2 * min(1.0, overlap_content / 3)
            # General knowledge docs must be relevant to query, avoid injecting unrelated definitions
            if not is_match and overlap_title + overlap_content == 0:
                continue
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "title": item.get("title", ""),
                "content": item["content"],
                "source": item.get("source", ""),
                "relevance": round(score, 2)
            }
            for score, item in scored[:top_k]
        ]

    def lookup_features(self, feature_names: Sequence[str],
                        include_shap_guide: bool = True) -> List[Dict]:
        """
        Exact-match lookup of "intermediate parameter" explanation docs by feature name
        (the "feature" field in knowledge base entries).
        Used to map SHAP top features to explanations in the knowledge base.
        Return order matches feature_names; missing features are skipped.
        """
        by_feature = {
            str(item["feature"]).lower(): item
            for item in self.knowledge_base if item.get("feature")
        }
        names = list(feature_names)
        if include_shap_guide:
            names = ["_shap"] + names

        docs = []
        for name in names:
            item = by_feature.get(str(name).lower())
            if item is None:
                continue
            docs.append({
                "feature": item["feature"],
                "title": item.get("title", ""),
                "content": item["content"],
                "source": item.get("source", ""),
                "relevance": 1.0,
            })
        return docs

    @staticmethod
    def format_for_prompt(retrieved_docs: List[Dict]) -> str:
        """Format retrieval results into LLM-friendly text (citation tags align with [sourceN] in system prompt)"""
        if not retrieved_docs:
            return ""

        parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = f" {doc['title']}" if doc.get("title") else ""
            parts.append(f"[source{i}]{title} ({doc['source']})")
            parts.append(doc['content'])
            parts.append("---")

        return "\n".join(parts)