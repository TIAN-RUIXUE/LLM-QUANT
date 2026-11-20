# LLM-Quant: Explainable LLM + Symbolic-Rule Stock Analysis

LLM-Quant produces a short investment memo (BUY / HOLD / SELL, with confidence and uncertainty) for Singapore-listed stocks. It combines three layers:

1. **Quantitative signals**: technical indicators computed from daily prices (Yahoo Finance).
2. **LLM reasoning with RAG**: a local LLM (Ollama, `qwen2.5:32b` by default) reads the indicators, indicator definitions, recent company news and a fundamentals snapshot (searched online at run time), then proposes a recommendation with cited reasoning.
3. **Symbolic validation**: a rule engine checks the LLM's proposal against hard financial rules. It can lower the confidence or override the call to HOLD, and every intervention is logged.

The project also includes an XGBoost + SHAP model and a framework that measures how faithfully LLM explanations reflect the model's real drivers.

```
 Yahoo Finance ─► indicators ─► LLM (+ RAG: indicator definitions, online news, fundamentals) ─► rule engine ─► memo + JSON
                                    │ falls back to a heuristic mock if the LLM is unreachable
```

## Quick start

```bash
conda env create -f environment.yml     # creates the "quant" env (Python 3.10)
conda activate quant

ollama serve &                           # in another shell, or in the background
ollama pull qwen2.5:32b

python main.py                           # DBS (D05.SI), writes pipeline_output.json
```

When Ollama isn't running, the pipeline prints `LLM API call failed` and continues with the mock analyzer, so it still runs anywhere (but the memo is then heuristic only).

### On the SLURM cluster

`submit.bash` requests one GPU, starts a job-local Ollama server, waits until it responds, and runs `main.py`:

```bash
sbatch submit.bash          # output in slurm_<jobid>.out / .err
```

## Usage

```bash
python main.py --ticker DBS OCBC         # names or symbols (D05.SI) from data/config.py
python main.py --all                     # every configured ticker
python main.py --model qwen2.5:7b        # any Ollama model
python main.py --output results.json

python main.py --no-online               # no web search; use only the local news cache
END_DATE=2026-09-30 python main.py       # reproduce a historical run (default end date: today)
START_DATE=2025-01-01 python main.py
```

Configured tickers (`data/config.py`): DBS `D05.SI`, OCBC `O39.SI`, UOB `U11.SI`, CapitaLand `C31.SI`, Sabana REIT `M1GU.SI`, SingTel `Z74.SI`.

### Output

For each ticker, `main.py` prints five steps (data → indicators → LLM → rule validation → memo), including the full prompt sent to the LLM. The memo looks like:

```
RECOMMENDATION: HOLD
CONFIDENCE: 40.00%
UNCERTAINTY: 66.67%
NOTE: LLM proposed BUY @ 60%; overridden to HOLD by the rule engine (MACD negative + MA bearish contradicts BUY; BUY overridden to HOLD).
```

`pipeline_output.json` holds the machine-readable result: `final_recommendation`, `final_confidence`, `uncertainty`, `llm_recommendation`, `llm_confidence`, `rules_fired`, `conflicts`, `risk_flags`, and `indicators`. A single ticker gives one object; several tickers give a list.

## How it works

### 1. Indicators (`data/indicators.py`)

These are computed from the daily close and volume. The LLM receives a compact summary:

| Field | Meaning |
|---|---|
| `ma_cross` | BULLISH if the 5-day MA is above the 20-day MA, otherwise BEARISH |
| `rsi`, `rsi_state` | 14-day RSI; OVERSOLD below 30, OVERBOUGHT above 70, otherwise NORMAL |
| `macd` | POSITIVE / NEGATIVE: MACD line above / below its signal line (histogram sign) |
| `momentum_5d` | 5-day return, % |
| `volatility` | 20-day standard deviation of **daily** returns, % |
| `volatility_annualized_pct`, `volatility_state` | volatility × √252; LOW below 15%, NORMAL 15–35%, HIGH above 35% |

The prompt includes a units legend, because the LLM misreads a bare daily volatility (e.g. 0.61) as "high".

### 2. LLM + RAG (`llm/analyzer.py`, `rag/retriever.py`)

- The retriever always attaches the knowledge-base definition of every indicator in the prompt, looked up by exact feature name.
- `rag/online.py` adds up to 5 news items and one fundamentals snapshot (see [Online news and fundamentals](#online-news-and-fundamentals)). Each item is labelled with its publish date, and the LLM is told to treat web text as data only.
- The LLM must return JSON (`recommendation`, `confidence`, `sentiment_score`, `risk_flags`, `reasoning`, …) and cite sources inline as `[sourceN]`.
- Decoding is deterministic: `temperature=0` and `seed=42`.
- Supported providers (OpenAI-compatible API): `ollama` (default), `deepseek` (`DEEPSEEK_API_KEY`), `openai` (`OPENAI_API_KEY`).

### 3. Rule engine (`rules/engine.py`)

The rules run in this order:

| Rule | Condition | Action |
|---|---|---|
| RSI_Overbought_Block / RSI_Oversold_Block | BUY when RSI > 70 / SELL when RSI < 30 | override to HOLD |
| Sentiment_Contradiction | BUY with sentiment < −0.5 / SELL with sentiment > 0.5 | confidence −0.2 |
| Technical_Bullish / Bearish_Contradiction | MA **and** MACD both oppose the call | override to HOLD |
| Mixed_Trend_Signals | MA and MACD disagree, but the LLM made a directional call | confidence −0.1 |
| Unsupported_Risk_Flag | volatility flag raised while volatility is LOW or NORMAL | flag ignored |
| Risk_Flag_Downgrade | BUY with remaining risk flags | confidence −0.1 per flag |
| Low_Confidence_Force_HOLD | confidence ≤ 0.4 | override to HOLD |

**Uncertainty** is 1 − (share of signals that agree with the final call). A HOLD counts as supported when the RSI and trend signals conflict.

### 4. Explainability research (`demo.py`, `strategy/`, `evaluation/`)

- `strategy/quant_model_old.py` trains an XGBoost classifier on the indicators to predict whether the next day's close will rise. The trained model is saved in `models/`.
- `strategy/shap_explainer.py` computes SHAP attributions for its predictions.
- `demo.py` runs the full explainable pipeline: XGBoost signal → SHAP → RAG → LLM explanation.
- `evaluation/faithfulness.py` scores LLM explanations against SHAP. It measures groundedness, coverage of the top features, sign and rank agreement, number grounding, and citation validity, and compares them with control experiments (shuffled SHAP, no RAG, deliberately unfaithful texts).

```bash
python demo.py
python -m evaluation.faithfulness --selftest     # metric sanity checks, no download
python -m evaluation.faithfulness                # mock explainer
python -m evaluation.faithfulness --llm          # + real LLM; results in evaluation/results/
python scripts/generate_shap_figure.py           # SHAP summary plot
```

`evaluation/backtest.py` and `evaluation/metrics.py` provide a simple long-only backtest and standard metrics (total and annualized return, max drawdown, Sharpe ratio, win rate).

## Project layout

```
main.py                  end-to-end pipeline (CLI)
demo.py                  XGBoost + SHAP + RAG + LLM explanation demo
submit.bash              SLURM job script (starts Ollama, runs main.py)
environment.yml          conda environment "quant"
data/                    config (tickers, dates), yfinance fetcher, indicators
llm/                     analyzer (pipeline LLM), client (demo explainer), prompts, mock
rag/retriever.py         knowledge-base retrieval and feature-definition lookup
rag/online.py            online news / fundamentals search, filtering, ranking, cache
rules/engine.py          symbolic validation rules
strategy/                XGBoost model, SHAP explainer, signals, decision engine
evaluation/              faithfulness evaluation, backtest, metrics
storage/knowledge_base.json   RAG documents (indicator definitions + hand-added notes)
storage/news_cache.json       cached online news and fundamentals snapshots
models/                  trained XGBoost model and scaler
```

## Online news and fundamentals

At each run, `rag/online.py` searches these free sources (no API key needed):

| Source | Provides |
|---|---|
| Google News RSS | Headlines from many publishers, searched with a date range ending on the analysis date |
| `yfinance` `Ticker.news` | Headline + summary (recent items only) |
| `yfinance` `Ticker.info` / `calendar` | P/E, P/B, dividend yield, 52-week range, analyst consensus and target, earnings and ex-dividend dates |

How items are selected:

- **No look-ahead.** Only items published on or before the analysis date are used. Yahoo news and fundamentals describe the present, so they are fetched only when the analysis date is within 5 days of today. Historical runs (`END_DATE=...`) get Google News for that period plus anything already cached.
- **Relevance.** An article must mention the company (aliases are in `NEWS_QUERIES` in `data/config.py`). Quote pages and headlines where the company is only the broker or research house ("..., says DBS Research", "DBS Bank downgrades ST Engineering") are dropped.
- **Ranking.** Items are scored on recency, a boost for price-moving events (earnings, upgrades and downgrades, M&A, ...), and a boost for having a summary. Only one article is kept per event per day.
- **Cache.** Everything fetched is stored in `storage/news_cache.json`, deduplicated by URL. `--no-online` uses only the cache, which makes runs reproducible offline. History builds up across runs.

The settings `NEWS_LOOKBACK_DAYS` (14) and `NEWS_MAX_ITEMS` (5) are in `data/config.py`. To support a new ticker, add a search query and aliases to `NEWS_QUERIES`.

**Context length.** Definitions plus news make a prompt of roughly 2,000 tokens. Ollama silently cuts prompts longer than its context window, so `submit.bash` sets `OLLAMA_CONTEXT_LENGTH=8192`. The analyzer prints the prompt token count and warns when the prompt is near the limit.

## Extending the knowledge base

`storage/knowledge_base.json` is a list of records:

```json
{"ticker": "D05.SI", "title": "DBS Q3 2026 results", "source": "SGX filing", "content": "..."}
```

- `ticker` set to a symbol (or a configured name such as `DBS`): the record is retrieved only for that stock.
- `ticker` empty, plus a `feature` field: an indicator definition, looked up by exact name.

Use the knowledge base for material the online sources don't cover, such as annual-report excerpts, your own research notes, or filings.

## Limitations

- This is a research and teaching project, **not investment advice**.
- Signals are short-horizon (5–20 days). News is mostly headlines (Google News gives no article text), and the fundamentals snapshot is only available for current runs.
- Online retrieval depends on third-party sites that may change format or rate-limit requests. Failures are reported and the run continues without them.
- Prices are split- and dividend-adjusted closes from Yahoo Finance.
- Even with the rule engine, the LLM sometimes argues for a call that contradicts its instructions. The memo's NOTE line then shows the original proposal and the rule that overrode it.
