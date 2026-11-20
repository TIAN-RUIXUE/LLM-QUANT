# rag/online.py
"""
Online context for RAG: company news and a fundamentals snapshot, fetched at run time
and cached locally.

Sources (free, no API key):
  * yfinance  Ticker.news                 - headline + summary, publisher, publish time
  * Google News RSS search                - headlines from many publishers (date-bounded query)
  * yfinance  Ticker.info / .calendar     - valuation, analyst consensus, earnings / ex-dividend dates

Design rules:
  * No look-ahead: only items published on or before the analysis date are returned.
    Fundamentals describe "now", so they are only fetched when the analysis date is recent;
    older runs can only use snapshots that were cached at the time.
  * Every fetched item is cached in storage/news_cache.json (deduplicated by URL), so runs are
    reproducible offline (online=False) and history accumulates across runs.
  * Web text is untrusted: items are shortened and labelled; the LLM prompt says to treat
    them as data, not instructions.
"""

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "storage" / "news_cache.json"
USER_AGENT = "Mozilla/5.0 (compatible; llm-quant-research)"
MAX_CONTENT_CHARS = 600
# Days within which the analysis date counts as "today" for fetching current fundamentals
FUNDAMENTALS_MAX_AGE_DAYS = 5
# Headlines about events that typically move the price are ranked above general coverage
_MAJOR = re.compile(r"earnings|profit|results|quarter|guidance|upgrade|downgrade|acqui|merger|"
                    r"lawsuit|probe|investigation", re.IGNORECASE)
_MINOR = re.compile(r"dividend|target|rating|outlook|forecast|ceo|chairman|miss|beat|slump|plunge|"
                    r"surge|rally|fine", re.IGNORECASE)
# Titles that are quote / chart pages rather than news
_NOISE_TITLE = re.compile(r"stock price|price and chart|share price|quote|stock chart|historical data",
                          re.IGNORECASE)


def _to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()[:80]


def _clip(text: str, n: int = MAX_CONTENT_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


# ─────────────────────────────────────────────────────────────────────────────
# Fetchers (each returns a list of cache items; failures return [] and print a warning)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance_news(symbol: str) -> List[Dict]:
    import yfinance as yf
    items = []
    for raw in yf.Ticker(symbol).news or []:
        c = raw.get("content", raw)
        title = c.get("title")
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
        pub = c.get("pubDate") or c.get("displayTime")
        if not (title and url and pub):
            continue
        items.append({
            "ticker": symbol,
            "kind": "news",
            "title": title,
            "content": _clip(c.get("summary") or title),
            "source": (c.get("provider") or {}).get("displayName", "Yahoo Finance"),
            "url": url,
            "published": pub[:10],
            "origin": "yfinance",
        })
    return items


def fetch_google_news(symbol: str, query: str, start: date, end: date) -> List[Dict]:
    """Google News RSS search restricted to [start, end] (after:/before: operators)."""
    q = f"{query} after:{start.isoformat()} before:{(end + timedelta(days=1)).isoformat()}"
    url = ("https://news.google.com/rss/search?" +
           urllib.parse.urlencode({"q": q, "hl": "en-SG", "gl": "SG", "ceid": "SG:en"}))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        root = ET.fromstring(resp.read())

    items = []
    for it in root.findall(".//item"):
        title = it.findtext("title") or ""
        link = it.findtext("link") or ""
        pub = it.findtext("pubDate")
        publisher = it.findtext("source") or ""
        # Google appends " - Publisher" to the headline
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        if not (title and link and pub):
            continue
        try:
            published = parsedate_to_datetime(pub).astimezone(timezone.utc).date().isoformat()
        except (TypeError, ValueError):
            continue
        items.append({
            "ticker": symbol,
            "kind": "news",
            "title": title,
            "content": _clip(title),          # RSS gives headlines only
            "source": publisher or "Google News",
            "url": link,
            "published": published,
            "origin": "google_news",
        })
    return items


def fetch_fundamentals(symbol: str, as_of: date) -> List[Dict]:
    """Current valuation / consensus snapshot, stored as a dated document."""
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = t.info or {}

    def fmt(key, spec="{:.2f}", scale=1.0):
        v = info.get(key)
        return spec.format(v * scale) if isinstance(v, (int, float)) else None

    parts = []
    for label, val in [
        ("trailing P/E", fmt("trailingPE")),
        ("forward P/E", fmt("forwardPE")),
        ("price/book", fmt("priceToBook")),
        ("dividend yield %", fmt("dividendYield")),
        ("payout ratio %", fmt("payoutRatio", "{:.0f}", 100)),
        ("52-week low", fmt("fiftyTwoWeekLow")),
        ("52-week high", fmt("fiftyTwoWeekHigh")),
        ("analyst consensus", info.get("recommendationKey")),
        ("mean analyst target price", fmt("targetMeanPrice")),
        ("number of analysts", info.get("numberOfAnalystOpinions")),
    ]:
        if val is not None:
            parts.append(f"{label} {val}")

    try:
        cal = t.calendar or {}
    except Exception:
        cal = {}
    earnings = cal.get("Earnings Date") or []
    if earnings:
        parts.append("earnings date(s) " + ", ".join(str(d) for d in earnings))
    if cal.get("Ex-Dividend Date"):
        parts.append(f"ex-dividend date {cal['Ex-Dividend Date']}")

    if not parts:
        return []
    currency = info.get("currency", "")
    return [{
        "ticker": symbol,
        "kind": "fundamentals",
        "title": f"Fundamentals snapshot for {info.get('longName', symbol)} ({currency})",
        "content": "; ".join(parts) + ".",
        "source": "Yahoo Finance (snapshot)",
        "url": f"fundamentals:{symbol}:{as_of.isoformat()}",
        "published": as_of.isoformat(),
        "origin": "yfinance_info",
    }]


# ─────────────────────────────────────────────────────────────────────────────
# Cache + retrieval
# ─────────────────────────────────────────────────────────────────────────────

class OnlineRetriever:
    """
    get_documents(symbol, aliases, as_of) -> list of prompt-ready docs
    (same shape as RAGRetriever.retrieve, plus "published").
    """

    def __init__(self, cache_path: Optional[str] = None, online: bool = True,
                 lookback_days: int = 14, max_news: int = 5):
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE
        self.online = online
        self.lookback_days = lookback_days
        self.max_news = max_news
        self._cache: Dict[str, Dict] = self._load()

    # --- cache ---------------------------------------------------------------
    def _load(self) -> Dict[str, Dict]:
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                return {item["url"]: item for item in json.load(f)}
        return {}

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(self._cache.values(), key=lambda x: (x["ticker"], x["published"]), reverse=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=1, ensure_ascii=False)

    def _add(self, items: List[Dict]) -> int:
        new = 0
        for item in items:
            if item["url"] not in self._cache:
                new += 1
            self._cache[item["url"]] = item
        return new

    # --- fetching ------------------------------------------------------------
    def refresh(self, symbol: str, query: str, as_of: date):
        """Fetch online sources into the cache. Errors are reported, never raised."""
        start = as_of - timedelta(days=self.lookback_days)
        fetched = []
        recent = (date.today() - as_of).days <= FUNDAMENTALS_MAX_AGE_DAYS

        jobs = [("Google News", lambda: fetch_google_news(symbol, query, start, as_of))]
        if recent:
            # yfinance news / fundamentals only describe the present
            jobs += [("Yahoo news", lambda: fetch_yfinance_news(symbol)),
                     ("fundamentals", lambda: fetch_fundamentals(symbol, as_of))]
        for label, job in jobs:
            try:
                got = job()
                fetched += got
                print(f"  ✓ online: {label}: {len(got)} item(s)")
            except Exception as e:
                print(f"  ! online: {label} failed ({type(e).__name__}: {e})")

        new = self._add(fetched)
        self._save()
        print(f"  ✓ news cache: {new} new, {len(self._cache)} total ({self.cache_path.name})")

    # --- retrieval -----------------------------------------------------------
    def get_documents(self, symbol: str, aliases: Sequence[str], as_of,
                      query: Optional[str] = None) -> List[Dict]:
        as_of = _to_date(as_of)
        names = [a for a in aliases if a] or [symbol]
        if self.online:
            self.refresh(symbol, query or f'"{names[0]}"', as_of)

        start = as_of - timedelta(days=self.lookback_days)
        alias_alt = "|".join(re.escape(a) for a in [symbol.split(".")[0], *names])
        alias_re = re.compile(alias_alt, re.IGNORECASE)
        research_re = re.compile(
            rf"(?:says|said|according to|by|from)\s+(?:{alias_alt})\b[\w ]*|"
            rf"(?:{alias_alt})\s+(?:group\s+|bank\s+)?(?:research|analysts?|economists?|strategists?)\b|"
            # the company acting as a broker: "DBS Bank Downgrades ST Engineering ..."
            rf"(?:{alias_alt})\s+(?:group\s+|bank\s+|securities\s+|vickers\s+)?"
            rf"(?:upgrades|downgrades|initiates|maintains|reiterates|raises|cuts|lowers|keeps)\b[\w ]*",
            re.IGNORECASE)

        news, fundamentals, seen = [], [], set()
        for item in self._cache.values():
            if item["ticker"] != symbol:
                continue
            published = _to_date(item["published"])
            if published > as_of:                       # no look-ahead
                continue
            if item["kind"] == "fundamentals":
                if (as_of - published).days <= FUNDAMENTALS_MAX_AGE_DAYS:
                    fundamentals.append(item)
                continue
            if published < start:
                continue
            # Drop mentions where the company is only the research house quoted
            # ("..., says DBS Research"), then require the company to still be mentioned
            text = research_re.sub(" ", f"{item['title']} {item['content']}")
            if not alias_re.search(text) or _NOISE_TITLE.search(item["title"]):
                continue                                 # off-topic or a quote page
            key = _norm_title(item["title"])
            if key in seen:
                continue
            seen.add(key)
            news.append(item)

        # Rank: recency (0-1) + major event (+1.5) + minor event (+0.5) + has a summary (+0.5)
        def score(x):
            age = (as_of - _to_date(x["published"])).days
            return (1 - age / max(self.lookback_days, 1)
                    + (1.5 if _MAJOR.search(x["title"]) else 0.0)
                    + (0.5 if _MINOR.search(x["title"]) else 0.0)
                    + (0.5 if x["content"] != x["title"] else 0.0))
        news.sort(key=score, reverse=True)

        # Keep one article per (day, major event): several outlets often report the same upgrade
        picked, events = [], set()
        for x in news:
            m = _MAJOR.search(x["title"])
            event = (x["published"], m.group(0).lower()) if m else None
            if event and event in events:
                continue
            if event:
                events.add(event)
            picked.append(x)
            if len(picked) == self.max_news:
                break
        news = sorted(picked, key=lambda x: x["published"], reverse=True)
        fundamentals.sort(key=lambda x: x["published"], reverse=True)
        selected = fundamentals[:1] + news

        return [{
            "title": item["title"],
            "content": item["content"],
            "source": item["source"],
            "published": item["published"],
            "relevance": 1.0,
        } for item in selected]


if __name__ == "__main__":
    r = OnlineRetriever()
    for d in r.get_documents("D05.SI", ["DBS"], date.today(), query='"DBS Group"'):
        print(f"{d['published']} | {d['source']} | {d['title']}")
