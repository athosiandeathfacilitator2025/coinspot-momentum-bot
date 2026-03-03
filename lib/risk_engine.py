# lib/risk_engine.py
from math import sqrt

MAX_PORTFOLIO_HEAT = 0.80
RISK_PER_TRADE     = 0.03
MIN_NOTIONAL       = 5.0


def estimate_volatility(snapshot):
    high  = float(snapshot.get("high_24h")      or 0)
    low   = float(snapshot.get("low_24h")       or 0)
    price = float(snapshot.get("current_price") or snapshot.get("price") or 1)
    if high > low > 0 and price > 0:
        return (high - low) / price
    return 0.03


def calculate_position_size(balance: float, snapshot, open_positions: int) -> float:
    if balance <= 0:
        return 0.0
    vol            = estimate_volatility(snapshot)
    vol            = max(0.01, min(0.15, vol))
    dollar_risk    = balance * RISK_PER_TRADE
    position_value = dollar_risk / vol
    max_allowed    = balance * MAX_PORTFOLIO_HEAT
    position_value = min(position_value, max_allowed)
    if position_value < MIN_NOTIONAL:
        return 0.0
    return round(position_value, 2)
