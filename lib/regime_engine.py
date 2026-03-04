"""
lib/regime_engine.py — Market regime detector.

bot.py calls:
    self.regime_engine = RegimeEngine()
    regime = self.regime_engine.detect(prices)

Returns "BULL", "BEAR", or "NEUTRAL" (not "SIDEWAYS").
bot.py expects "NEUTRAL" as the neutral value.
market_context.py compares against lowercase — but this
is handled in market_context.py via case-insensitive check.
"""
import math
import logging

log = logging.getLogger("RegimeEngine")


class RegimeEngine:

    def detect(self, prices: list) -> str:
        """
        Detect market regime from a list of coin prices.

        Args:
            prices: list of floats (current_price per coin)

        Returns:
            "BULL", "BEAR", or "NEUTRAL"
        """
        try:
            if len(prices) < 2:
                return "NEUTRAL"

            valid = [float(p) for p in prices if p and float(p) > 0]
            if len(valid) < 2:
                return "NEUTRAL"

            returns = [
                math.log(valid[i] / valid[i - 1])
                for i in range(1, len(valid))
                if valid[i - 1] > 0
            ]

            if not returns:
                return "NEUTRAL"

            avg = sum(returns) / len(returns)

            if avg > 0.001:
                return "BULL"
            if avg < -0.001:
                return "BEAR"
            return "NEUTRAL"

        except Exception as e:
            log.warning("RegimeEngine.detect error: %s", str(e)[:80])
            return "NEUTRAL"
