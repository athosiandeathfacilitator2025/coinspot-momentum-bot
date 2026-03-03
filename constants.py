# constants.py

SYMBOL_TO_CG_ID = {
    "BTC":   "bitcoin",
    "ETH":   "ethereum",
    "SOL":   "solana",
    "XRP":   "ripple",
    "DOGE":  "dogecoin",
    "ADA":   "cardano",
    "LTC":   "litecoin",
    "BNB":   "binancecoin",
    "DOT":   "polkadot",
    "LINK":  "chainlink",
    "MATIC": "matic-network",
    "AVAX":  "avalanche-2",
    "UNI":   "uniswap",
    "ATOM":  "cosmos",
    "XLM":   "stellar",
    "ALGO":  "algorand",
    "VET":   "vechain",
    "SHIB":  "shiba-inu",
    "DENT":  "dent",
    "AAVE":  "aave",
    "DFI":   "defichain",
    "RUNE":  "thorchain",
}

CG_TO_BINANCE_SYMBOL = {v: k for k, v in SYMBOL_TO_CG_ID.items()}
