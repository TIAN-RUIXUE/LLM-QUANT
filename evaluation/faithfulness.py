"""
evaluation/faithfulness.py

Faithfulness / groundedness evaluation of LLM explanations against SHAP attributions.

Two questions are measured separately:
  * Groundedness  - is every factor / number in the explanation traceable to the input?
  * Faithfulness  - does the explanation name the factors that actually drove the
                    XGBoost prediction (per SHAP), with the right direction and weight?

Per-explanation metrics (all in [0, 1], None when not applicable):
  grounding_precision  mentioned model factors / (model factors + out-of-model factors such as P/E, ROE)
  topk_coverage        share of the top-k SHAP features that are mentioned
  group_coverage       same, at feature-group level (trend / momentum / rsi / macd / volatility / volume)
  weighted_coverage    sum |SHAP| of mentioned top-k features / sum |SHAP| of top-k features
  sign_agreement       stated direction (pushes up / pulls down the up-probability) matches the SHAP sign
  sign_agreement_struct  same, using the structured "factors" field returned by the explainer
  rank_agreement       Spearman correlation between order of first mention and |SHAP| rank, rescaled to [0, 1]
  number_grounding     share of numbers in the text that appear in the prompt context
  citation_validity    share of [sourceN] citations that refer to a provided document
  citation_relevance   share of feature-document citations placed next to a mention of that feature
  faithfulness_score   mean of grounding_precision, weighted_coverage, sign_agreement (those available)

Experiments (run_experiment):
  real_shap      explainer gets the true SHAP values + RAG feature explanations
  no_rag         explainer gets the true SHAP values, no retrieved documents
  shuffled_shap  explainer gets SHAP values permuted across features. An explainer that really
                 uses the attribution agrees with the *given* SHAP and disagrees with the *true* one;
                 one that writes generic finance text agrees with neither.
  control_*      hand-made unfaithful explanations; the metrics must score them low:
                   control_pe_roe        "... because of a high P/E ratio and declining ROE"
                   control_generic       vague text that names no model factor
                   control_sign_flipped  mock explanation written from SHAP with every sign flipped

Limitations (state these in the report):
  * Mentions and directions are extracted with a hand-written lexicon and cue phrases; paraphrases
    outside the lexicon are missed. The structured "factors" field is a cross-check.
  * SHAP itself approximates the model, so this measures faithfulness *to SHAP*.
  * SHAP splits credit arbitrarily between correlated features (ma_5 / ma_20 / ma_60 ...);
    group_coverage is more robust than topk_coverage.
  * Cases are taken from the out-of-sample test period only.

Usage:
  python -m evaluation.faithfulness                  # mock explainer only
  python -m evaluation.faithfulness --llm            # + real LLM (Ollama qwen3:8b by default)
  python -m evaluation.faithfulness --llm --provider deepseek --model deepseek-chat
  python -m evaluation.faithfulness --selftest       # sanity checks of the metrics, no data download
"""

import argparse
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Lexicon: feature names, groups, aliases
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "trend": ["ma_5", "ma_20", "ma_60", "ma_cross_signal"],
    "momentum": ["return_1d", "return_5d", "return_20d"],
    "rsi": ["rsi", "rsi_signal"],
    "macd": ["macd", "macd_signal_line", "macd_histogram", "macd_cross_signal"],
    "volatility": ["volatility_20d"],
    "volume": ["volume_ma_20", "volume_ratio"],
}
FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
ALL_FEATURES = list(FEATURE_TO_GROUP)

_B, _E = r"(?<![a-z0-9_])", r"(?![a-z0-9_])"

# (regex, feature) - ordered from specific to generic; earlier matches mask later ones.
FEATURE_ALIASES = [
    (r"momentum_5d", "return_5d"),
    (r"rsi_state", "rsi_signal"),
    (r"ma_cross", "ma_cross_signal"),
    (r"moving[- ]average cross(?:over)?s?", "ma_cross_signal"),
    (r"golden cross", "ma_cross_signal"),
    (r"death cross", "ma_cross_signal"),
    (r"ma ?5 (?:crossing |crossed |cross )?(?:above|below|over|under|vs\.?|versus) ?(?:the )?ma ?20", "ma_cross_signal"),
    (r"\bma cross(?:over)?", "ma_cross_signal"),
    (r"(?:5|five)[- ]day moving average", "ma_5"),
    (r"short[- ]term moving average", "ma_5"),
    (r"(?:20|twenty)[- ]day moving average", "ma_20"),
    (r"(?:60|sixty)[- ]day moving average", "ma_60"),
    (r"long(?:er)?[- ]term moving average", "ma_60"),
    (r"\bma ?5\b", "ma_5"),
    (r"\bma ?20\b", "ma_20"),
    (r"\bma ?60\b", "ma_60"),
    (r"(?:1|one)[- ]day (?:return|price change|change)", "return_1d"),
    (r"daily return", "return_1d"),
    (r"(?:5|five)[- ]day (?:return|momentum|price change|change)", "return_5d"),
    (r"weekly return", "return_5d"),
    (r"short[- ]term momentum", "return_5d"),
    (r"(?:20|twenty)[- ]day (?:return|momentum|price change|change)", "return_20d"),
    (r"(?:1|one)[- ]month return", "return_20d"),
    (r"monthly return", "return_20d"),
    (r"medium[- ]term (?:trend|momentum)", "return_20d"),
    (r"overbought", "rsi_signal"),
    (r"oversold", "rsi_signal"),
    (r"relative strength index", "rsi"),
    (r"\brsi\b", "rsi"),
    (r"macd histogram", "macd_histogram"),
    (r"\bhistogram\b", "macd_histogram"),
    (r"signal line", "macd_signal_line"),
    (r"macd cross(?:over)?", "macd_cross_signal"),
    (r"\bmacd\b", "macd"),
    (r"volatil\w*", "volatility_20d"),
    (r"volume ratio", "volume_ratio"),
    (r"relative volume", "volume_ratio"),
    (r"volume spike", "volume_ratio"),
    (r"(?:above|below)[- ]average volume", "volume_ratio"),
    (r"(?:20[- ]day )?average (?:daily )?(?:trading )?volume", "volume_ma_20"),
]

# Generic mentions: grounded at group level, but no specific feature.
GROUP_ALIASES = [
    (r"moving averages?", "trend"),
    (r"\bmomentum\b", "momentum"),
    (r"\bvolume\b", "volume"),
]

# Financial factors the model does NOT use. Counted as ungrounded unless a retrieved
# document in the prompt contains the term (then it is grounded by retrieval).
OUT_OF_MODEL_TERMS = {
    "p/e": r"(?<![a-z])p/?e(?: ratio)?(?![a-z])|price[- ]to[- ]earnings",
    "roe": r"\broe\b|return on equity",
    "p/b": r"(?<![a-z])p/b(?![a-z])|price[- ]to[- ]book|book value",
    "eps": r"\beps\b|earnings per share",
    "earnings": r"\bearnings\b",
    "revenue": r"\brevenues?\b|\bsales growth\b",
    "dividend": r"\bdividends?\b|\byield\b",
    "valuation": r"\bvaluations?\b|\bundervalued\b|\bovervalued\b",
    "interest rates": r"interest rates?|net interest margin|\bnim\b",
    "macro": r"\bgdp\b|\binflation\b|macro\w*|central bank|\bfed\b",
    "analysts": r"\banalysts?\b|target price|price target|\bupgrade\b|\bdowngrade\b",
    "debt": r"\bdebt\b|\bleverage\b",
    "sentiment": r"investor sentiment|market sentiment",
    "guidance": r"\bguidance\b|\boutlook\b",
}

# Cue phrases: direction of the contribution to the predicted UP-probability.
UP_CUES = [
    r"push(?:es|ed|ing)? (?:\w+ ){0,4}?(?:up|higher)",
    r"(?:rais|increas|boost|lift|elevat)\w*\s+(?:\w+\s+){0,4}?(?:probabilit|prediction|likelihood|chance|odds)",
    r"(?:higher|greater) (?:\w+ ){0,2}?(?:probabilit|likelihood|chance|odds)",
    r"positive (?:shap|driver|contribution|contributor|factor|impact|influence|effect|attribution)",
    r"(?:support|favou?r)\w* (?:\w+ ){0,2}?(?:buy|upward|rise|price increase)",
    r"(?:against|offset\w*|counter\w*) (?:\w+ ){0,2}?(?:sell|downward)",
]
DOWN_CUES = [
    r"(?:pull|push)(?:s|es|ed|ing)? (?:\w+ ){0,4}?(?:down|lower)",
    r"(?:lower|reduc|decreas|drag|cut|depress|suppress)\w*\s+(?:\w+\s+){0,4}?(?:probabilit|prediction|likelihood|chance|odds)",
    r"weigh\w* (?:on|down)",
    r"negative (?:shap|driver|contribution|contributor|factor|impact|influence|effect|attribution)",
    r"(?:support|favou?r)\w* (?:\w+ ){0,2}?(?:sell|downward|decline|price decrease)",
    r"(?:against|offset\w*|counter\w*) (?:\w+ ){0,2}?(?:buy|upward)",
]

_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\s*;\s*|\n+|,?\s+\b(?:while|whereas|but|however|although|though)\b,?\s*"
)
_CITATION = re.compile(r"\[source(\d+)\]", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _patterns():
    exact = [(_B + re.escape(f) + _E, f) for f in sorted(ALL_FEATURES, key=len, reverse=True)]
    return exact + FEATURE_ALIASES


def find_mentions(text: str) -> List[Dict]:
    """
    Find model-feature mentions in text.
    Returns [{"pos": int, "feature": str|None, "group": str}], sorted by position.
    Longer / more specific patterns win; matched spans are masked so that e.g.
    "macd_histogram" does not also count as "macd".
    """
    masked = text.lower()
    found = []
    for pattern, feature in _patterns():
        for m in re.finditer(pattern, masked):
            found.append({"pos": m.start(), "feature": feature, "group": FEATURE_TO_GROUP[feature]})
        masked = re.sub(pattern, lambda m: " " * len(m.group(0)), masked)
    for pattern, group in GROUP_ALIASES:
        for m in re.finditer(pattern, masked):
            found.append({"pos": m.start(), "feature": None, "group": group})
        masked = re.sub(pattern, lambda m: " " * len(m.group(0)), masked)
    return sorted(found, key=lambda x: x["pos"])


def find_out_of_model_terms(text: str, docs_text: str = "") -> Dict[str, bool]:
    """Return {term: grounded_by_retrieval} for out-of-model factors mentioned in text."""
    lower, docs_lower = text.lower(), docs_text.lower()
    hits = {}
    for term, pattern in OUT_OF_MODEL_TERMS.items():
        if re.search(pattern, lower):
            hits[term] = bool(re.search(pattern, docs_lower))
    return hits


def split_clauses(text: str) -> List[str]:
    return [c for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]


def clause_direction(clause: str) -> Optional[str]:
    lower = clause.lower()
    up = any(re.search(p, lower) for p in UP_CUES)
    down = any(re.search(p, lower) for p in DOWN_CUES)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


def direction_claims(text: str) -> List[Dict]:
    """[{"feature", "direction"}] for every feature mentioned in a clause with an unambiguous cue."""
    claims = []
    for clause in split_clauses(text):
        direction = clause_direction(clause)
        if direction is None:
            continue
        for feat in {m["feature"] for m in find_mentions(clause) if m["feature"]}:
            claims.append({"feature": feat, "direction": direction})
    return claims


_NUMBER = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?%?")
_NUMBER_NOISE = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),                                         # dates
    re.compile(r"\[source\d+\]", re.IGNORECASE),                              # citations
    re.compile(r"[a-z_]+\d+[a-z_\d]*", re.IGNORECASE),                        # ma_5, return_20d, ma20
    re.compile(r"\d+[- ](?:trading )?(?:day|days|week|weeks|month|months)\b", re.IGNORECASE),  # windows
]


def extract_numbers(text: str) -> List[str]:
    for pattern in _NUMBER_NOISE:
        text = pattern.sub(" ", text)
    return [m.group(0) for m in _NUMBER.finditer(text)]


def _number_matches(token: str, candidates: Sequence[float]) -> bool:
    raw = token.replace(",", "").replace("−", "-").rstrip("%")
    try:
        value = abs(float(raw))
    except ValueError:
        return True
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    tol_display = 0.5 * 10 ** (-decimals) + 1e-12
    for c in candidates:
        c = abs(c)
        if abs(value - c) <= max(tol_display, 0.005 * c):
            return True
    return False


def context_numbers(context: Dict) -> List[float]:
    """All numbers an explanation may legitimately quote, in raw and percentage form."""
    # Probabilities and feature values may be quoted as percentages (0.1137 -> 11.37%);
    # SHAP values are log-odds, so only their raw form counts.
    pct_values = [context.get("prediction"), context.get("base_value")]
    raw_values = []
    for f in context.get("shap_top_features", []):
        pct_values.append(f.get("value"))
        raw_values.append(f.get("shap"))
    raw_values += list((context.get("portfolio_summary") or {}).values())
    out = []
    for v in pct_values:
        if isinstance(v, (int, float)) and np.isfinite(v):
            out += [float(v), float(v) * 100]
    for v in raw_values:
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.append(float(v))
    docs = list(context.get("feature_docs") or []) + list(context.get("news_docs") or [])
    for d in docs:
        for tok in extract_numbers(d.get("content", "")):
            try:
                out.append(float(tok.replace(",", "").replace("−", "-").rstrip("%")))
            except ValueError:
                pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    rho = pd.Series(a).rank().corr(pd.Series(b).rank())
    return None if pd.isna(rho) else float(rho)


def evaluate_explanation(text: str,
                         shap: Dict[str, float],
                         context: Optional[Dict] = None,
                         factors: Optional[List[Dict]] = None,
                         top_k: int = 5) -> Dict:
    """
    Score one explanation.

    text     explanation text written by the explainer
    shap     {feature: SHAP value} - the reference attribution (usually the TRUE one)
    context  the context given to the explainer (for numbers, documents, citations)
    factors  optional structured [{"feature", "direction"}] returned by the explainer
    """
    context = context or {}
    docs = list(context.get("feature_docs") or []) + list(context.get("news_docs") or [])
    docs_text = " ".join(d.get("content", "") + " " + d.get("title", "") for d in docs)

    ranked = sorted(shap, key=lambda f: abs(shap[f]), reverse=True)
    top = ranked[:top_k]
    top_abs = sum(abs(shap[f]) for f in top) or 1.0

    mentions = find_mentions(text)
    mentioned_features = []
    for m in mentions:
        if m["feature"] and m["feature"] not in mentioned_features:
            mentioned_features.append(m["feature"])
    mentioned_groups = {m["group"] for m in mentions}

    oom = find_out_of_model_terms(text, docs_text)
    ungrounded_terms = sorted(t for t, by_doc in oom.items() if not by_doc)

    # Grounding precision: grounded factors = features + group-only mentions + terms backed by retrieval
    group_only = {m["group"] for m in mentions if m["feature"] is None} - {
        FEATURE_TO_GROUP[f] for f in mentioned_features}
    grounded = len(mentioned_features) + len(group_only) + sum(oom.values())
    total_factors = grounded + len(ungrounded_terms)
    grounding_precision = grounded / total_factors if total_factors else None

    # Coverage
    in_top = [f for f in mentioned_features if f in top]
    topk_coverage = len(in_top) / len(top) if top else None
    top_groups = {FEATURE_TO_GROUP.get(f) for f in top} - {None}
    group_coverage = len(top_groups & mentioned_groups) / len(top_groups) if top_groups else None
    weighted_coverage = sum(abs(shap[f]) for f in in_top) / top_abs if top else None

    # Sign agreement (text cues)
    def _sign_ok(claims):
        checked = [c for c in claims if c["feature"] in shap and shap[c["feature"]] != 0]
        if not checked:
            return None, 0
        ok = sum((c["direction"] == "up") == (shap[c["feature"]] > 0) for c in checked)
        return ok / len(checked), len(checked)

    claims = direction_claims(text)
    sign_agreement, n_claims = _sign_ok(claims)
    sign_agreement_struct, _ = _sign_ok(factors or [])

    # Rank agreement: earlier mention <-> larger |SHAP|
    ranked_mentions = [f for f in mentioned_features if f in shap]
    rho = _spearman(list(range(len(ranked_mentions))),
                    [-abs(shap[f]) for f in ranked_mentions])
    rank_agreement = None if rho is None else (rho + 1) / 2

    # Number grounding
    numbers = extract_numbers(text)
    candidates = context_numbers(context)
    number_grounding = (sum(_number_matches(t, candidates) for t in numbers) / len(numbers)
                        if numbers and context else None)

    # Citations
    citation_validity = citation_relevance = None
    cited = [int(n) for n in _CITATION.findall(text)]
    if cited:
        citation_validity = sum(1 <= n <= len(docs) for n in cited) / len(cited)
        rel_checks = []
        for clause in split_clauses(text):
            clause_feats = {m["feature"] for m in find_mentions(clause)}
            clause_groups = {m["group"] for m in find_mentions(clause)}
            for n in (int(x) for x in _CITATION.findall(clause)):
                if 1 <= n <= len(docs):
                    doc_feat = docs[n - 1].get("feature")
                    if doc_feat and doc_feat in FEATURE_TO_GROUP:
                        rel_checks.append(doc_feat in clause_feats
                                          or FEATURE_TO_GROUP[doc_feat] in clause_groups)
        citation_relevance = sum(rel_checks) / len(rel_checks) if rel_checks else None

    parts = [x for x in (grounding_precision, weighted_coverage, sign_agreement) if x is not None]
    return {
        "grounding_precision": grounding_precision,
        "topk_coverage": topk_coverage,
        "group_coverage": group_coverage,
        "weighted_coverage": weighted_coverage,
        "sign_agreement": sign_agreement,
        "sign_agreement_struct": sign_agreement_struct,
        "rank_agreement": rank_agreement,
        "number_grounding": number_grounding,
        "citation_validity": citation_validity,
        "citation_relevance": citation_relevance,
        "faithfulness_score": float(np.mean(parts)) if parts else None,
        "n_direction_claims": n_claims,
        "mentioned_features": ",".join(mentioned_features),
        "ungrounded_terms": ",".join(ungrounded_terms),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_TEXTS = {
    "control_pe_roe": "The model recommends {action} because {name} has a high P/E ratio and declining ROE.",
    "control_generic": ("Technical indicators are mixed and the market outlook is uncertain, "
                        "so the recommendation reflects overall investor sentiment."),
}

KEY_METRICS = ["faithfulness_score", "grounding_precision", "topk_coverage", "group_coverage",
               "weighted_coverage", "sign_agreement", "sign_agreement_struct", "rank_agreement",
               "number_grounding", "citation_validity", "citation_relevance"]


def build_cases(ticker: str, start: str, end: str, n_cases: int,
                buy_threshold: float, sell_threshold: float) -> List[Dict]:
    """Train the model and pick out-of-sample days to explain (signals first, then most confident)."""
    import yfinance as yf
    from strategy.quant_model_old import (compute_technical_features, prepare_ml_features,
                                          create_target, train_xgb_model, generate_ml_signals)
    from strategy.shap_explainer import compute_shap_values, get_base_value

    raw = yf.download(ticker, start=start, end=end, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    featured = compute_technical_features(raw)
    model, _, _, _, _, scaler = train_xgb_model(featured)

    X = prepare_ml_features(featured)
    y = create_target(featured)
    valid = X.dropna().index.intersection(y.dropna().index)
    test_idx = valid[int(len(valid) * 0.8):]          # same split as train_xgb_model

    signaled = generate_ml_signals(featured, model, scaler, buy_threshold, sell_threshold)
    names = X.columns.tolist()
    shap_test = compute_shap_values(model, scaler.transform(X.loc[test_idx]), names)
    base = get_base_value(model)

    test = signaled.loc[test_idx, ["pred_prob", "signal"]].copy()
    test["row"] = range(len(test_idx))
    test["conf"] = (test["pred_prob"] - 0.5).abs()
    test["is_signal"] = test["signal"] != "HOLD"
    test = test.sort_values(["is_signal", "conf"], ascending=False).head(n_cases)

    cases = []
    for date, r in test.iterrows():
        cases.append({
            "date": str(pd.Timestamp(date).date()),
            "action": r["signal"],
            "names": names,
            "X_row": X.loc[date],
            "shap": np.asarray(shap_test[int(r["row"])], dtype=float),
            "base": base,
        })
    print(f"  -> {len(cases)} out-of-sample cases "
          f"({int(test['is_signal'].sum())} BUY/SELL signals, test period {test_idx[0].date()} → {test_idx[-1].date()})")
    return cases


def build_context(case: Dict, shap_vec: np.ndarray, retriever, ticker_label: str,
                  use_rag: bool = True, top_k: int = 5) -> Dict:
    from strategy.shap_explainer import explain_prediction
    expl = explain_prediction(shap_vec, case["names"], case["X_row"], top_k=top_k, base_value=case["base"])
    feature_docs = retriever.lookup_features([f["name"] for f in expl["top_features"]]) if use_rag else []
    return {
        "ticker": ticker_label,
        "date": case["date"],
        "action": case["action"],
        "prediction": expl["prediction"],
        "base_value": expl["base_value"],
        "shap_top_features": expl["top_features"],
        "feature_docs": feature_docs,
        "news_docs": [],
    }


def run_experiment(ticker: str = "D05.SI", n_cases: int = 10, use_llm: bool = False,
                   llm_config=None, start: str = "2024-09-24", end: str = "2026-09-24",
                   buy_threshold: float = 0.85, sell_threshold: float = 0.15,
                   top_k: int = 5, seed: int = 42, out_dir: Optional[str] = None) -> pd.DataFrame:
    from llm.client import MockLLMClient, RealLLMClient
    from rag.retriever import RAGRetriever
    from data.config import TICKERS

    aliases = [n for n, s in TICKERS.items() if s.upper() == ticker.upper()]
    name = aliases[0] if aliases else ticker
    label = f"{name} ({ticker})" if aliases else ticker

    print(f"[1/3] Building out-of-sample cases for {label} ...")
    cases = build_cases(ticker, start, end, n_cases, buy_threshold, sell_threshold)
    retriever = RAGRetriever()

    explainers = [MockLLMClient()]
    if use_llm:
        explainers.append(RealLLMClient(llm_config))

    print(f"[2/3] Generating and scoring explanations ({len(explainers)} explainer(s)) ...")
    rng = np.random.default_rng(seed)
    rows = []

    def _record(case_id, case, condition, explainer_name, text, true_shap, given_shap, ctx, factors):
        m = evaluate_explanation(text, true_shap, ctx, factors, top_k)
        given = evaluate_explanation(text, given_shap, ctx, factors, top_k)
        rows.append({
            "case": case_id, "date": case["date"], "action": case["action"],
            "prediction": ctx["prediction"], "condition": condition, "explainer": explainer_name,
            **m,
            "sign_agreement_given": given["sign_agreement"],
            "weighted_coverage_given": given["weighted_coverage"],
            "text": text,
        })

    for i, case in enumerate(cases):
        true_shap = dict(zip(case["names"], case["shap"]))
        perm = rng.permutation(len(case["shap"]))
        shuffled_vec = case["shap"][perm]
        shuffled_shap = dict(zip(case["names"], shuffled_vec))

        conditions = {
            "real_shap": (case["shap"], true_shap, True),
            "no_rag": (case["shap"], true_shap, False),
            "shuffled_shap": (shuffled_vec, shuffled_shap, True),
        }
        for condition, (vec, given_shap, use_rag) in conditions.items():
            ctx = build_context(case, vec, retriever, label, use_rag, top_k)
            for ex in explainers:
                result = ex.analyze(ctx)
                # RealLLMClient silently falls back to the mock - record who actually answered
                _record(i, case, condition, result.get("model", ex.model_name), result.get("reason", ""),
                        true_shap, given_shap, ctx, result.get("factors"))

        # Negative controls (scored against the TRUE SHAP)
        ctx = build_context(case, case["shap"], retriever, label, True, top_k)
        for condition, template in CONTROL_TEXTS.items():
            text = template.format(action=case["action"], name=name)
            _record(i, case, condition, "control", text, true_shap, true_shap, ctx, None)
        flipped_ctx = build_context(case, -case["shap"], retriever, label, True, top_k)
        flipped = MockLLMClient().analyze(flipped_ctx)
        _record(i, case, "control_sign_flipped", "control", flipped["reason"],
                true_shap, dict(zip(case["names"], -case["shap"])), flipped_ctx, flipped["factors"])

    df = pd.DataFrame(rows)

    print("[3/3] Summary (mean over cases; metrics vs TRUE SHAP unless marked _given)\n")
    summary = summarize(df)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.2f}".format):
        print(summary)

    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = os.path.join(out_dir, f"faithfulness_{ticker}_{stamp}.csv")
    summary_path = os.path.join(out_dir, f"faithfulness_{ticker}_{stamp}_summary.csv")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"\n✓ Per-explanation results: {detail_path}")
    print(f"✓ Summary:                 {summary_path}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = KEY_METRICS + ["sign_agreement_given", "weighted_coverage_given"]
    order = ["real_shap", "no_rag", "shuffled_shap",
             "control_pe_roe", "control_generic", "control_sign_flipped"]
    summary = df.groupby(["explainer", "condition"])[cols].mean(numeric_only=True)
    summary["n"] = df.groupby(["explainer", "condition"]).size()
    summary = summary.reset_index()
    summary["_o"] = summary["condition"].map({c: i for i, c in enumerate(order)})
    return summary.sort_values(["_o", "explainer"]).drop(columns="_o").set_index(["condition", "explainer"])


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: the metrics must separate faithful from unfaithful explanations
# ─────────────────────────────────────────────────────────────────────────────

def self_test():
    shap = {f: 0.0 for f in ALL_FEATURES}
    shap.update({"volume_ma_20": -0.81, "return_20d": -0.73, "volatility_20d": -0.43,
                 "ma_5": -0.38, "rsi": 0.37})
    ctx = {"prediction": 0.1137, "base_value": 0.548,
           "shap_top_features": [{"name": k, "value": v, "shap": shap[k]} for k, v in
                                 [("volume_ma_20", 4895198.95), ("return_20d", 0.0104),
                                  ("volatility_20d", 0.0063), ("ma_5", 55.44), ("rsi", 58.9)]],
           "feature_docs": [{"feature": "_shap", "content": "SHAP guide", "title": ""},
                            {"feature": "volume_ma_20", "content": "average volume over 20 days", "title": ""}],
           "news_docs": []}

    good = ("The model predicts an up-probability of 11.37%. Average trading volume pulls down the "
            "probability [source2]; the 20-day return, volatility and the 5-day moving average also "
            "lower the probability, while RSI pushes up the probability slightly.")
    pe = "The model recommends SELL because DBS has a high P/E ratio and declining ROE."
    flipped = ("Average trading volume pushes up the probability, and the 20-day return raises the "
               "probability, while RSI pulls down the prediction.")
    numbers_bad = "The model predicts an up-probability of 73%, RSI is 81 and volume_ma_20 pulls down the probability."

    g = evaluate_explanation(good, shap, ctx)
    p = evaluate_explanation(pe, shap, ctx)
    f = evaluate_explanation(flipped, shap, ctx)
    n = evaluate_explanation(numbers_bad, shap, ctx)

    checks = [
        ("faithful: grounding 1", g["grounding_precision"] == 1.0),
        ("faithful: sign agreement 1", g["sign_agreement"] == 1.0),
        ("faithful: top-5 coverage 1", g["topk_coverage"] == 1.0),
        ("faithful: numbers grounded", g["number_grounding"] == 1.0),
        ("faithful: citation relevant", g["citation_relevance"] == 1.0),
        ("P/E-ROE: grounding 0", p["grounding_precision"] == 0.0),
        ("P/E-ROE: coverage 0", p["topk_coverage"] == 0.0),
        ("P/E-ROE: flags p/e and roe", p["ungrounded_terms"] == "p/e,roe"),
        ("flipped: sign agreement 0", f["sign_agreement"] == 0.0),
        ("wrong numbers: grounding < 1", n["number_grounding"] < 1.0),
        ("macd_histogram not double counted",
         [m["feature"] for m in find_mentions("macd_histogram and MACD")] == ["macd_histogram", "macd"]),
        ("out-of-model term grounded by retrieval",
         evaluate_explanation("earnings rose", shap,
                              {"feature_docs": [], "news_docs": [{"content": "earnings rose 5%"}]})
         ["grounding_precision"] == 1.0),
    ]
    width = max(len(c[0]) for c in checks)
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label.ljust(width)}")
    failed = [c for c in checks if not c[1]]
    if failed:
        print(f"\n{len(failed)} check(s) failed")
        for label, res in (("faithful", g), ("pe", p), ("flipped", f), ("numbers", n)):
            print(label, res)
        sys.exit(1)
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Faithfulness evaluation of LLM explanations vs SHAP")
    ap.add_argument("--ticker", default="D05.SI")
    ap.add_argument("--cases", type=int, default=10, help="number of out-of-sample days to explain")
    ap.add_argument("--llm", action="store_true", help="also evaluate the real LLM (RealLLMClient)")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "deepseek", "openai"])
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true", help="run metric sanity checks and exit")
    args = ap.parse_args()

    if args.selftest:
        self_test()
        sys.exit(0)

    config = None
    if args.llm:
        from llm.analyzer import LLMConfig
        config = LLMConfig(provider=args.provider, model=args.model)
    run_experiment(args.ticker, args.cases, args.llm, config, top_k=args.top_k, seed=args.seed)
