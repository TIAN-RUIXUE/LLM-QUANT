# data/config.py

TICKERS = {
    "DBS": "D05.SI",        # DBS Bank
    "OCBC": "O39.SI",       # OCBC Bank
    "UOB": "U11.SI",        # UOB
    "CapitaLand": "C31.SI", # CapitaLand Integrated Commercial Trust
    "Sabana_REIT": "M1GU.SI",
    "SingTel": "Z74.SI",    # SingTel
}

# time range
START_DATE = "2024-01-01"
END_DATE = "2026-09-30"
TOP_K_DOCS = 5          # RAG 
RULE_CONFIDENCE_THRESHOLD = 0.4
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9