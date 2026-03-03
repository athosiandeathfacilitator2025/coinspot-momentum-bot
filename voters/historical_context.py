"""
voters/historical_context.py  —  Tier 6: Deep Historical Intelligence
───────────────────────────────────────────────────────────────────────
The core principle: a coin cannot be traded without knowing its full history.
"""

import time
import requests
import numpy as np
from datetime import datetime, timezone

_cache: dict = {}

FULL_HISTORY_TTL = 12 * 3600
DETAIL_TTL       =  6 * 3600
FNG_TTL          =  1 * 3600


def _get_cached(key, ttl, fn):
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < ttl:
            return val
    try:
        val = fn()
    except Exception:
        val = None
    if val is not None:
        _cache[key] = (now, val)
    return val


def _fetch_fng():
    r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
    if r.status_code == 200:
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "label": d["value_classification"]}
    return {"value": 50, "label": "Unknown"}


def get_fear_greed():
    return _get_cached("fng", FNG_TTL, _fetch_fng) or {"value": 50, "label": "Unknown"}


def _fetch_full_history(coin_id, cg):
    data       = cg.get_coin_market_chart_by_id(coin_id, vs_currency="aud", days="max")
    raw_prices = data.get("prices", [])
    raw_vols   = data.get("total_volumes", [])

    if len(raw_prices) < 30:
        return None

    timestamps = np.array([p[0] for p in raw_prices], dtype=float)
    prices     = np.array([p[1] for p in raw_prices], dtype=float)
    volumes    = np.array([v[1] for v in raw_vols if v[1] is not None], dtype=float)
    n          = len(prices)

    returns = np.diff(prices) / np.maximum(prices[:-1], 1e-10)

    peak_val = prices[0]
    peak_idx = 0
    worst_dd = 0.0
    worst_dd_dur = 0
    for i in range(1, n):
        if prices[i] > peak_val:
            peak_val = prices[i]
            peak_idx = i
        dd = (peak_val - prices[i]) / peak_val
        if dd > worst_dd:
            worst_dd     = dd
            worst_dd_dur = i - peak_idx

    major_crashes  = 0
    recoveries     = 0
    recovery_times = []
    in_crash       = False
    crash_start    = 0
    crash_peak_v   = 0.0
    crash_trough_v = 0.0
    for i in range(1, n):
        if not in_crash:
            recent_high = float(np.max(prices[max(0, i - 30):i + 1]))
            if recent_high > 0 and (recent_high - prices[i]) / recent_high > 0.40:
                in_crash       = True
                crash_start    = i
                crash_peak_v   = recent_high
                crash_trough_v = prices[i]
                major_crashes += 1
        else:
            if prices[i] < crash_trough_v:
                crash_trough_v = prices[i]
            if crash_peak_v > 0 and prices[i] >= crash_peak_v * 0.90:
                in_crash = False
                recoveries += 1
                recovery_times.append(i - crash_start)

    avg_rec = float(np.mean(recovery_times)) if recovery_times else 999.0

    atl_all   = float(np.min(prices))
    ath_all   = float(np.max(prices))
    curr      = float(prices[-1])
    rng       = ath_all - atl_all
    cycle_pos = (curr - atl_all) / rng * 100 if rng > 0 else 50.0

    ma10  = float(np.mean(prices[-10:]))  if n >= 10  else curr
    ma50  = float(np.mean(prices[-50:]))  if n >= 50  else curr
    ma200 = float(np.mean(prices[-200:])) if n >= 200 else curr

    if ma10 > ma50 * 1.02:
        st_trend = "up"
    elif ma10 < ma50 * 0.98:
        st_trend = "down"
    else:
        st_trend = "flat"

    if ma50 > ma200 * 1.03:
        lt_trend = "up"
    elif ma50 < ma200 * 0.97:
        lt_trend = "down"
    else:
        lt_trend = "flat"

    vol_trend     = "unknown"
    vol_change_6m = 0.0
    if len(volumes) >= 180:
        ve = float(np.mean(volumes[-180:-90]))
        vr = float(np.mean(volumes[-90:]))
        if ve > 0:
            vol_change_6m = (vr - ve) /
