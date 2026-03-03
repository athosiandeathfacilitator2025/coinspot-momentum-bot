# lib/math_utils.py
import numpy as np


def calculate_gauge_curvature(local_price: float, global_price_aud: float) -> float:
    if local_price <= 0 or global_price_aud <= 0:
        return 0.0
    return float(np.log(local_price / global_price_aud))


def evaluate_elliptic_point(norm_x: float, norm_y: float, a: float = -1.0, b: float = 1.0) -> dict:
    try:
        lhs       = norm_y ** 2
        rhs       = (norm_x ** 3) + (a * norm_x) + b
        delta     = abs(lhs - rhs)
        snap_prob = float(1.0 / (1.0 + np.exp(-delta + 2)))
        return {
            "is_stable": delta < 1.0,
            "delta":     float(delta),
            "snap_prob": snap_prob,
        }
    except Exception:
        return {"is_stable": True, "delta": 0.0, "snap_prob": 0.0}


def fit_elliptic_curve(prices: np.ndarray, volumes: np.ndarray) -> dict:
    try:
        if len(prices) < 5 or len(volumes) < 5:
            return {"is_structured": False, "A": 0.0, "B": 0.0, "r_squared": 0.0}
        pm, ps = float(np.mean(prices)),  max(float(np.std(prices)),  1e-9)
        vm, vs = float(np.mean(volumes)), max(float(np.std(volumes)), 1e-9)
        x      = (prices  - pm) / ps
        y      = (volumes - vm) / vs
        lhs    = y ** 2 - x ** 3
        X_mat  = np.column_stack([x, np.ones(len(x))])
        result = np.linalg.lstsq(X_mat, lhs, rcond=None)
        A, B   = result[0]
        y_pred = x ** 3 + A * x + B
        ss_res = float(np.sum((lhs - y_pred) ** 2))
        ss_tot = float(np.sum((lhs - np.mean(lhs)) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "is_structured": r2 > 0.5,
            "A":             float(A),
            "B":             float(B),
            "r_squared":     float(r2),
        }
    except Exception:
        return {"is_structured": False, "A": 0.0, "B": 0.0, "r_squared": 0.0}


def gaussian_curvature_K(prices: np.ndarray, volumes: np.ndarray) -> float:
    try:
        if len(prices) < 5 or len(volumes) < 5:
            return 0.0
        pm, ps = float(np.mean(prices)),  max(float(np.std(prices)),  1e-9)
        vm, vs = float(np.mean(volumes)), max(float(np.std(volumes)), 1e-9)
        x   = (prices  - pm) / ps
        y   = (volumes - vm) / vs
        dx  = np.gradient(x)
        dy  = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        num = dx * ddy - dy * ddx
        den = (dx ** 2 + dy ** 2) ** 1.5
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        K   = float(np.mean(num / den))
        return K
    except Exception:
        return 0.0


def geodesic_deviation(prices: np.ndarray, volumes: np.ndarray) -> dict:
    try:
        if len(prices) < 5 or len(volumes) < 5:
            return {"deviation_z": 0.0, "direction": "inline"}
        pm, ps = float(np.mean(prices)),  max(float(np.std(prices)),  1e-9)
        vm, vs = float(np.mean(volumes)), max(float(np.std(volumes)), 1e-9)
        x      = (prices  - pm) / ps
        y      = (volumes - vm) / vs
        n      = len(x)
        t      = np.linspace(0, 1, n)
        geo_x  = x[0] + t * (x[-1] - x[0])
        geo_y  = y[0] + t * (y[-1] - y[0])
        dist   = np.sqrt((x - geo_x) ** 2 + (y - geo_y) ** 2)
        dev_z  = float(np.mean(dist))
        price_dev = float(np.mean(x - geo_x))
        if price_dev > 0.3:
            direction = "above"
        elif price_dev < -0.3:
            direction = "below"
        else:
            direction = "inline"
        return {"deviation_z": dev_z, "direction": direction}
    except Exception:
        return {"deviation_z": 0.0, "direction": "inline"}


def point_at_infinity(prices: np.ndarray, entry_price: float) -> dict:
    no_signal = {
        "has_valid_signal": False,
        "projected_peak":   0.0,
        "dynamic_tp_pct":   0.0,
        "dynamic_tp_price": 0.0,
        "confidence":       0.0,
    }
    try:
        if len(prices) < 8 or entry_price <= 0:
            return no_signal
        x        = np.arange(len(prices), dtype=float)
        y        = prices.astype(float)
        coeffs   = np.polyfit(x, y, 2)
        a, b, c  = coeffs
        if a >= 0:
            return no_signal
        vertex_x = -b / (2 * a)
        vertex_y = float(np.polyval(coeffs, vertex_x))
        if vertex_y <= entry_price:
            return no_signal
        y_pred = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 < 0.7:
            return no_signal
        tp_price = entry_price + 0.95 * (vertex_y - entry_price)
        tp_pct   = (tp_price - entry_price) / entry_price
        if tp_pct <= 0:
            return no_signal
        return {
            "has_valid_signal": True,
            "projected_peak":   vertex_y,
            "dynamic_tp_pct":   float(tp_pct),
            "dynamic_tp_price": float(tp_price),
            "confidence":       float(r2),
        }
    except Exception:
        return no_signal


def symmetry_break_index(prices: np.ndarray, volumes: np.ndarray) -> dict:
    try:
        if len(prices) < 5 or len(volumes) < 5:
            return {"sbi": 0.0}
        x      = np.arange(len(prices), dtype=float)
        pm, ps = float(np.mean(prices)),  max(float(np.std(prices)),  1e-9)
        vm, vs = float(np.mean(volumes)), max(float(np.std(volumes)), 1e-9)
        p_norm   = (prices  - pm) / ps
        v_norm   = (volumes - vm) / vs
        p_slope  = float(np.polyfit(x, p_norm, 1)[0])
        v_slope  = float(np.polyfit(x, v_norm, 1)[0])
        if p_slope > 0 and v_slope < 0:
            sbi = min(1.0, abs(p_slope - v_slope) / 2.0)
        else:
            sbi = 0.0
        return {"sbi": float(sbi)}
    except Exception:
        return {"sbi": 0.0}


def composite_stress_score(
    K:           float,
    snap_prob:   float,
    deviation_z: float,
    sbi:         float,
    gauge_abs:   float,
) -> float:
    try:
        k_norm     = min(abs(K)      / 3.0,  1.0)
        dev_norm   = min(deviation_z / 3.0,  1.0)
        sbi_norm   = min(sbi,                1.0)
        gauge_norm = min(gauge_abs   / 0.05, 1.0)
        secondary  = max(k_norm, dev_norm, sbi_norm, gauge_norm)
        stress     = 0.70 * snap_prob + 0.30 * secondary
        return float(np.clip(stress, 0.0, 1.0))
    except Exception:
        return 0.0
