"""
engines/gauge_engine.py — Global price reference via CryptoCompare
────────────────────────────────────────────────────────────────────
Fetches AUD prices for all tracked symbols from CryptoCompare's free
API. Results are cached for `cache_ttl` seconds (default 60s) so the
engine is safe to call every 30s without rate-limiting issues.

This is the ONLY GaugeEngine. The old lib/gauge_engine.py and
lib/geometric_engine.py were stubs/copies and have been removed.
"""
import time
import logging
import requests

from constants import CG_TO_BINANCE_SYMBOL

log = logging.getLogger("GaugeEngine")


class GaugeEngine:

    def __init__(self, cache_ttl: int = 60):
        self.cache_ttl          = cache_ttl
        self.global_prices_aud  = {}   # symbol → AUD price
        self.aud_usdt_rate      = 0.65  # fallback if forex fetch fails
        self.last_update        = 0.0

    def update(self):
        """Fetch fresh global prices (respects cache_ttl)."""
        now = time.time()
        if now - self.last_update < self.cache_ttl:
            return

        try:
            # 1. Get AUD/USDT conversion rate
            forex = requests.get(
                "https://min-api.cryptocompare.com/data/price?fsym=USDT&tsyms=AUD",
                timeout=5,
            ).json()
            self.aud_usdt_rate = float(forex.get("AUD", self.aud_usdt_rate))

            # 2. Get prices in AUD for all tracked symbols
            symbols = list(CG_TO_BINANCE_SYMBOL.values())
            if not symbols:
                return

            url  = (
                "https://min-api.cryptocompare.com/data/pricemultifull"
                f"?fsyms={','.join(symbols)}&tsyms=AUD"
            )
            resp = requests.get(url, timeout=10).json()
            raw  = resp.get("RAW", {})

            updated = 0
            for sym in symbols:
                entry = raw.get(sym, {}).get("AUD", {})
                if entry.get("PRICE"):
                    self.global_prices_aud[sym] = float(entry["PRICE"])
                    updated += 1

            self.last_update = now
            log.info("Gauge updated: %d prices | AUD/USDT=%.4f", updated, self.aud_usdt_rate)

        except Exception as e:
            log.warning("Gauge update failed (stale data in use): %s", str(e)[:100])

    def get_global_price(self, symbol: str) -> float:
        """Return cached global AUD price for symbol, or 0.0 if unknown."""
        return self.global_prices_aud.get(symbol.upper(), 0.0)
