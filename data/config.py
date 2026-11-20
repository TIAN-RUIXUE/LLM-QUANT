# data/config.py

TICKERS = {
    "DBS": "D05.SI",        # DBS Bank
    "OCBC": "O39.SI",       # OCBC Bank
    "UOB": "U11.SI",        # UOB
    "CapitaLand": "C31.SI", # CapitaLand Integrated Commercial Trust
    "Sabana_REIT": "M1GU.SI",
    "SingTel": "Z74.SI",    # SingTel
}

import os
from datetime import date

# Online news search: symbol -> (Google News query, aliases an article must mention)
NEWS_QUERIES = {
    "D05.SI": ('"DBS Group" OR "DBS Bank"', ["DBS"]),
    "O39.SI": ('OCBC', ["OCBC"]),
    "U11.SI": ('"UOB" OR "United Overseas Bank"', ["UOB", "United Overseas Bank"]),
    "C31.SI": ('"CapitaLand Integrated Commercial Trust" OR CICT', ["CapitaLand", "CICT"]),
    "M1GU.SI": ('"Sabana REIT" OR "Sabana Industrial"', ["Sabana"]),
    "Z74.SI": ('Singtel', ["Singtel"]),
}
NEWS_LOOKBACK_DAYS = 14   # news window before the analysis date
NEWS_MAX_ITEMS = 5        # news documents per prompt

# time range
START_DATE = os.environ.get("START_DATE", "2024-01-01")
# Default to today so the memo reflects the latest close; set END_DATE=YYYY-MM-DD
# (exclusive, yfinance convention) to reproduce a fixed historical run.
END_DATE = os.environ.get("END_DATE", date.today().isoformat())
TOP_K_DOCS = 5          # RAG 
RULE_CONFIDENCE_THRESHOLD = 0.4
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
