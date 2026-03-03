# lib/regime_engine.py
import math

class RegimeEngine:
    def detect(self, prices):
        if len(prices) < 20:
            return "NEUTRAL"
        returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
        ]
        avg = sum(returns) / len(returns)
        if avg > 0.001:
            return "BULL"
        elif avg < -0.001:
            return "BEAR"
        return "SIDEWAYS"
