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
            vol_change_6m = (vr - ve) / ve * 100
            vol_trend = "growing" if vol_change_6m > 20 else (
                        "shrinking" if vol_change_6m < -40 else "stable")

    arr90 = prices[-90:] if n >= 90 else prices
    ret90 = np.diff(arr90) / np.maximum(arr90[:-1], 1e-10)
    p90   = arr90[0]
    dd90  = 0.0
    for p in arr90:
        if p > p90:
            p90 = p
        dd90 = max(dd90, (p90 - p) / p90)

    bear_streak = max_bear = 0
    for r in ret90:
        if r < 0:
            bear_streak += 1
            max_bear = max(max_bear, bear_streak)
        else:
            bear_streak = 0

    low90   = float(np.min(arr90))
    rec90   = (curr - low90) / low90 * 100 if low90 > 0 else 0.0
    vol_ann = float(np.std(returns) * np.sqrt(365) * 100) if len(returns) > 1 else 50.0

    first_dt      = datetime.fromtimestamp(timestamps[0] / 1000, tz=timezone.utc)
    coin_age_days = (datetime.now(tz=timezone.utc) - first_dt).days

    return {
        "coin_age_days":         coin_age_days,
        "days_of_history":       n,
        "all_time_high":         ath_all,
        "all_time_low":          atl_all,
        "current_price":         curr,
        "cycle_position_pct":    round(cycle_pos, 1),
        "worst_all_time_dd":     round(worst_dd * 100, 1),
        "worst_dd_duration":     worst_dd_dur,
        "major_crashes_count":   major_crashes,
        "recoveries_count":      recoveries,
        "avg_recovery_days":     round(avg_rec, 0),
        "short_trend":           st_trend,
        "long_trend":            lt_trend,
        "vol_trend":             vol_trend,
        "vol_change_6m_pct":     round(vol_change_6m, 1),
        "max_drawdown_90d":      round(dd90 * 100, 1),
        "max_bear_streak_90d":   max_bear,
        "recovery_from_90d_low": round(rec90, 1),
        "volatility_ann":        round(vol_ann, 1),
    }


def _fetch_coin_detail(coin_id, cg):
    data = cg.get_coin_by_id(
        coin_id,
        localization=False, tickers=False,
        market_data=True, community_data=True,
        developer_data=False, sparkline=False,
    )
    md = data.get("market_data", {})

    def get_aud(field):
        v = md.get(field)
        if isinstance(v, dict):
            return float(v.get("aud") or v.get("usd") or 0)
        return float(v or 0)

    circ         = float(md.get("circulating_supply") or 0)
    total        = float(md.get("total_supply") or circ or 1)
    supply_ratio = circ / total if total > 0 else 1.0

    ath_chg      = get_aud("ath_change_percentage")
    atl_chg      = get_aud("atl_change_percentage")
    rank         = int(data.get("market_cap_rank") or 9999)
    mcap_chg_24h = float(md.get("market_cap_change_percentage_24h") or 0)

    ath_age_days = 0
    ath_date_raw = (md.get("ath_date") or {})
    ath_date_s   = ath_date_raw.get("aud") or ath_date_raw.get("usd") or ""
    if ath_date_s:
        try:
            ath_dt       = datetime.fromisoformat(ath_date_s.replace("Z", "+00:00"))
            ath_age_days = (datetime.now(tz=timezone.utc) - ath_dt).days
        except Exception:
            pass

    genesis_age_days = 0
    genesis = data.get("genesis_date") or ""
    if genesis:
        try:
            gdt = datetime.strptime(genesis, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            genesis_age_days = (datetime.now(tz=timezone.utc) - gdt).days
        except Exception:
            pass

    sentiment_up = float(data.get("sentiment_votes_up_percentage") or 50)
    watchlist    = int(data.get("watchlist_portfolio_users") or 0)

    mom_stack = {
        "1y":   get_aud("price_change_percentage_1y"),
        "200d": get_aud("price_change_percentage_200d"),
        "30d":  get_aud("price_change_percentage_30d"),
        "14d":  get_aud("price_change_percentage_14d"),
        "7d":   get_aud("price_change_percentage_7d_in_currency"),
        "24h":  get_aud("price_change_percentage_24h"),
    }

    return {
        "supply_ratio":     supply_ratio,
        "ath_change_pct":   ath_chg,
        "ath_age_days":     ath_age_days,
        "atl_change_pct":   atl_chg,
        "rank":             rank,
        "mcap_chg_24h":     mcap_chg_24h,
        "genesis_age_days": genesis_age_days,
        "sentiment_up_pct": sentiment_up,
        "watchlist_users":  watchlist,
        "mom_stack":        mom_stack,
    }


def historical_context_vote(row, cg) -> dict:
    try:
        coin_id   = str(row.get("id") or "").lower().strip()
        symbol    = str(row.get("symbol") or "").upper().strip()
        change24  = float(row.get("price_change_percentage_24h") or 0)
        volume    = float(row.get("total_volume") or 0)
        mcap      = float(row.get("market_cap")   or 1)
        vol_ratio = volume / max(mcap, 1e-10)

        if not coin_id:
            return {"action": "hold", "confidence": 0.5, "reason": "HistCtx: no coin ID"}

        fng     = get_fear_greed()
        history = _get_cached(f"hist_{coin_id}",   FULL_HISTORY_TTL,
                              lambda: _fetch_full_history(coin_id, cg))
        detail  = _get_cached(f"detail_{coin_id}", DETAIL_TTL,
                              lambda: _fetch_coin_detail(coin_id, cg))

        if history is None and detail is None:
            return {"action": "hold", "confidence": 0.55,
                    "reason": f"HistCtx: data unavailable for {symbol}"}

        vetoed   = False
        veto_msg = ""

        if detail:
            sr      = detail["supply_ratio"]
            ath_chg = detail["ath_change_pct"]
            ath_age = detail["ath_age_days"]
            atl_chg = detail["atl_change_pct"]
            rank    = detail["rank"]
            gen_age = detail["genesis_age_days"]
            mom     = detail["mom_stack"]

            if gen_age > 0 and gen_age < 180:
                vetoed   = True
                veto_msg = (f"HistCtx VETO: {symbol} only {gen_age}d old — "
                            f"no crash survival history.")
            elif sr < 0.15:
                vetoed   = True
                veto_msg = (f"HistCtx VETO: supply rug risk — only {sr*100:.0f}% "
                            f"of {symbol} circulating.")
            elif ath_chg < -92 and ath_age > 365 and rank > 200:
                vetoed   = True
                veto_msg = (f"HistCtx VETO: dead project — {symbol} is "
                            f"{abs(ath_chg):.0f}% below ATH set {ath_age}d ago.")
            elif 0 < atl_chg < 15:
                vetoed   = True
                veto_msg = (f"HistCtx VETO: near all-time low — {symbol} only "
                            f"{atl_chg:.1f}% above ATL.")
            elif change24 > 40 and vol_ratio > 0.12:
                vetoed   = True
                veto_msg = (f"HistCtx VETO: shill pump — {symbol} +{change24:.0f}% "
                            f"in 24h vol_ratio={vol_ratio:.3f}.")
            elif (mom.get("30d", 0) < -25 and mom.get("14d", 0) < -15
                  and mom.get("7d", 0) < -8 and mom.get("24h", 0) < -3):
                vetoed   = True
                veto_msg = (f"HistCtx VETO: zombie — {symbol} falling all timeframes "
                            f"30d:{mom['30d']:.0f}% 14d:{mom['14d']:.0f}% "
                            f"7d:{mom['7d']:.0f}% 24h:{mom['24h']:.0f}%")

        if history and not vetoed:
            if history["worst_all_time_dd"] > 95:
                vetoed   = True
                veto_msg = (f"HistCtx VETO: {symbol} lost "
                            f"{history['worst_all_time_dd']:.0f}% in single drawdown.")
            elif (history["vol_trend"] == "shrinking"
                  and history["vol_change_6m_pct"] < -60
                  and history["long_trend"] == "down"):
                vetoed   = True
                veto_msg = (f"HistCtx VETO: abandonment — {symbol} volume down "
                            f"{abs(history['vol_change_6m_pct']):.0f}% over 6m.")
            elif history["max_bear_streak_90d"] > 25 and history["recovery_from_90d_low"] < 5:
                vetoed   = True
                veto_msg = (f"HistCtx VETO: free fall — {symbol} fell "
                            f"{history['max_bear_streak_90d']} consecutive days.")

        if vetoed:
            return {"action": "hold", "confidence": 0.0, "reason": veto_msg}

        signals = []
        fng_val = fng["value"]
        fng_lbl = fng["label"]

        if fng_val <= 20:
            signals.append(("buy", 0.85,
                f"Extreme Fear F&G={fng_val} — crowd capitulating"))
        elif fng_val <= 40:
            signals.append(("buy", 0.68, f"Fear F&G={fng_val} — lean buy"))
        elif fng_val >= 85:
            signals.append(("sell", 0.78,
                f"Extreme Greed F&G={fng_val} — every major top occurred here"))
        elif fng_val >= 65:
            signals.append(("hold", 0.55, f"Greed F&G={fng_val} — caution"))
        else:
            signals.append(("hold", 0.55, f"Neutral F&G={fng_val}"))

        if history:
            cyc    = history["cycle_position_pct"]
            lt     = history["long_trend"]
            st     = history["short_trend"]
            crash  = history["major_crashes_count"]
            rec    = history["recoveries_count"]
            avg_r  = history["avg_recovery_days"]
            vol_tr = history["vol_trend"]
            rec90  = history["recovery_from_90d_low"]
            age    = history["coin_age_days"]
            vol_a  = history["volatility_ann"]
            vc6    = history["vol_change_6m_pct"]

            if cyc < 20:
                signals.append(("buy", 0.75,
                    f"Near all-time floor ({cyc:.0f}%) — historically low entry"))
            elif cyc > 80:
                signals.append(("sell", 0.68,
                    f"Near all-time ceiling ({cyc:.0f}%) — limited upside"))
            elif cyc <= 50:
                signals.append(("buy", 0.62,
                    f"Lower half of historical range ({cyc:.0f}%)"))

            if crash >= 2 and rec >= 2:
                signals.append(("buy", 0.72,
                    f"Proven survivor: {crash} crashes, {rec} recoveries, "
                    f"avg {avg_r:.0f}d to recover"))
            elif crash >= 1 and rec == 0:
                signals.append(("sell", 0.60, f"Crashed {crash}x, never fully recovered"))

            if lt == "up":
                signals.append(("buy",  0.75, "50d MA above 200d MA — golden cross"))
            elif lt == "down":
                signals.append(("sell", 0.72, "50d MA below 200d MA — death cross"))

            if st == "up":
                signals.append(("buy",  0.65, "10d MA above 50d MA — short-term up"))
            elif st == "down":
                signals.append(("sell", 0.62, "10d MA below 50d MA — short-term declining"))

            if vol_tr == "growing":
                signals.append(("buy",  0.68, f"Volume +{vc6:.0f}% over 6m"))
            elif vol_tr == "shrinking":
                signals.append(("sell", 0.62, f"Volume {vc6:.0f}% over 6m"))

            if rec90 > 40 and st == "up":
                signals.append(("buy",  0.70, f"{rec90:.0f}% off 90d low and trending up"))
            elif rec90 < 5:
                signals.append(("sell", 0.65, f"Only {rec90:.0f}% off 90d low"))

            if age > 1825:
                signals.append(("buy", 0.60,
                    f"Established: {age // 365}yr old coin, survived multiple cycles"))

            if vol_a > 250:
                signals.append(("hold", 0.52,
                    f"High volatility {vol_a:.0f}% annualised"))

        if detail:
            mom     = detail["mom_stack"]
            ath_chg = detail["ath_change_pct"]
            ath_age = detail["ath_age_days"]
            sent_up = detail["sentiment_up_pct"]

            tf_pos = sum(1 for v in mom.values() if v > 0)
            tf_neg = sum(1 for v in mom.values() if v < 0)

            if tf_pos >= 5:
                signals.append(("buy", 0.75,
                    f"Momentum aligned {tf_pos}/6 timeframes positive: "
                    f"1y:{mom['1y']:.0f}% 200d:{mom['200d']:.0f}% 30d:{mom['30d']:.0f}%"))
            elif tf_neg >= 5:
                signals.append(("sell", 0.70,
                    f"Momentum collapsing: {tf_neg}/6 timeframes negative"))
            else:
                signals.append(("hold", 0.52,
                    f"Mixed momentum {tf_pos} up / {tf_neg} down"))

            if -70 < ath_chg < -15:
                signals.append(("buy", 0.62,
                    f"Healthy discount {ath_chg:.0f}% from ATH ({ath_age}d ago)"))
            elif ath_chg > -10:
                signals.append(("hold", 0.55, f"Near ATH ({ath_chg:.0f}%)"))

            if sent_up > 75:
                signals.append(("buy",  0.58, f"Community: {sent_up:.0f}% bullish"))
            elif sent_up < 35:
                signals.append(("sell", 0.55, f"Community: {sent_up:.0f}% bullish"))

        if not signals:
            return {"action": "hold", "confidence": 0.55,
                    "reason": f"HistCtx HOLD [{fng_lbl}]: no signals"}

        buy_w  = sum(c for a, c, _ in signals if a == "buy")
        sell_w = sum(c for a, c, _ in signals if a == "sell")
        hold_w = sum(c for a, c, _ in signals if a == "hold")
        n      = len(signals)

        if buy_w > sell_w and buy_w > hold_w:
            best = max((s for s in signals if s[0] == "buy"), key=lambda x: x[1])
            return {"action": "buy",
                    "confidence": min(0.92, buy_w / n * 1.5),
                    "reason": f"HistCtx BUY [{fng_lbl}] {symbol}: {best[2][:120]}"}

        if sell_w > buy_w and sell_w > hold_w:
            best = max((s for s in signals if s[0] == "sell"), key=lambda x: x[1])
            return {"action": "sell",
                    "confidence": min(0.92, sell_w / n * 1.5),
                    "reason": f"HistCtx SELL [{fng_lbl}] {symbol}: {best[2][:120]}"}

        return {"action": "hold", "confidence": 0.55,
                "reason": f"HistCtx HOLD [{fng_lbl}] {symbol}: "
                          f"{n} signals balanced (buy:{buy_w:.2f} sell:{sell_w:.2f})"}

    except Exception as e:
        return {"action": "hold", "confidence": 0.5,
                "reason": f"HistCtx error: {str(e)[:100]}"}
