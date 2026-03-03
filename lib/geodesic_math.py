# lib/geodesic_math.py
import numpy as np

def calculate_geodesic_tp(prices, base_tp):
    if len(prices) < 5:
        return base_tp
    returns    = np.diff(np.log(prices))
    volatility = np.std(returns)
    scale      = 1 + (volatility * 10)
    scale      = min(max(scale, 0.8), 1.8)
    return base_tp * scale
