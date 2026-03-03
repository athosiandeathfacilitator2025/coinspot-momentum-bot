"""
engines/geometric_engine.py — Elliptic Curve Snap-Back Detection
"""
import numpy as np
import logging
from collections import defaultdict
from lib.math_utils import evaluate_elliptic_point

log = logging.getLogger("GeometricEngine")


class GeometricEngine:

    def __init__(self, history_len: int = 30):
        self.history_len   = history_len
        self.price_history = defaultdict(list)
        self.vol_history   = defaultdict(list)

    def check_rationality(self, symbol: str, price: float, volume: float) -> dict:
        if price <= 0 or volume <= 0:
            return {"is_stable": True, "delta": 0.0, "snap_prob": 0.0}

        p_hist = self.price_history[symbol]
        v_hist = self.vol_history[symbol]
        p_hist.append(price)
        v_hist.append(volume)
        if len(p_hist) > self.history_len:
            p_hist.pop(0)
        if len(v_hist) > self.history_len:
            v_hist.pop(0)

        if len(p_hist) < 5:
            return {"is_stable": True, "delta": 0.0, "snap_prob": 0.0}

        p_arr = np.array(p_hist, dtype=float)
        v_arr = np.array(v_hist, dtype=float)
        p_mean, p_std = float(np.mean(p_arr)), float(np.std(p_arr))
        v_mean, v_std = float(np.mean(v_arr)), float(np.std(v_arr))
        p_std = p_std if p_std > 0 else 1e-9
        v_std = v_std if v_std > 0 else 1e-9

        norm_x = (price  - p_mean) / p_std
        norm_y = (volume - v_mean) / v_std

        return evaluate_elliptic_point(norm_x, norm_y)
