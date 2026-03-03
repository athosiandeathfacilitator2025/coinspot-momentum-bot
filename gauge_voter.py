"""
voters/gauge_voter.py — Tier 7: Gauge Invariance Arbitrage
───────────────────────────────────────────────────────────
WHAT IT DOES:
  Detects price discrepancies between the local Coinspot AUD market
  and the global reference price (via CryptoCompare).

  The math: curvature = log(local_price / global_price_aud)

  Negative curvature → local price is BELOW global → BUY signal (underpriced)
  Positive curvature → local price is ABOVE global → SELL signal (overpriced)
  Near zero          → markets are in sync → HOLD

WHY IT WORKS:
  Coinspot is a smaller AUD exchange. Large coins track global prices tightly,
  but smaller coins (DENT, DFI, etc.) can diverge from global consensus for
  minutes at a time. Buying the local discount before it corrects = pure edge.

INTEGRATION:
  GaugeEngine is instantiated once in bot.py and shared here.
  The engine caches global prices for `cache_ttl` seconds (default 60s)
  so we don't hammer CryptoCompare on every 30s cycle.
"""
import numpy as np
import logging

log = logging.getLogger("GaugeVoter")

# Thresholds — configurable via config.yaml gauge section
_DEFAULT_BUY_THRESHOLD  = -0.003   # log ratio below this = buy
_DEFAULT_SELL_THRESHOLD =  0.003   # log ratio above this = sell


def gauge_vote(row: dict, gauge_engine, config: dict) -> dict:
    """
    row:          single coin dict from market scan
    gauge_engine: engines.gauge_engine.GaugeEngine instance (shared, not recreated)
    config:       full CONFIG dict (reads config['gauge'] thresholds)
    Returns:      {"action": str, "confidence": float, "reason": str}
    """
    try:
        symbol = str(row.get("symbol") or "").upper()
        local_price = float(row.get("current_price") or 0)

        if local_price <= 0:
            return {"action": "hold", "confidence": 0.5, "reason": "Gauge: no local price"}

        # Refresh global prices (engine internally respects cache_ttl)
        gauge_engine.update()
        global_price = gauge_engine.get_global_price(symbol)

        if global_price <= 0:
            return {
                "action":     "hold",
                "confidence": 0.5,
                "reason":     f"Gauge: no global reference price for {symbol}",
            }

        curvature = float(np.log(local_price / global_price))

        gauge_cfg  = config.get("gauge", {})
        buy_thresh  = float(gauge_cfg.get("curvature_buy_threshold",  _DEFAULT_BUY_THRESHOLD))
        sell_thresh = float(gauge_cfg.get("curvature_sell_threshold", _DEFAULT_SELL_THRESHOLD))

        pct_diff = (local_price - global_price) / global_price * 100

        if curvature < buy_thresh:
            # Local is cheaper than global — buy the discount
            confidence = min(0.90, 0.65 + abs(curvature) * 30)
            reason = (
                f"🌀 Gauge BUY: {symbol} local=${local_price:.6f} "
                f"global=${global_price:.6f} ({pct_diff:+.2f}%) "
                f"curvature={curvature:.4f}"
            )
            log.info(reason)
            return {"action": "buy", "confidence": confidence, "reason": reason}

        if curvature > sell_thresh:
            # Local is more expensive than global — avoid / sell
            confidence = min(0.90, 0.65 + abs(curvature) * 30)
            reason = (
                f"🌀 Gauge SELL: {symbol} local=${local_price:.6f} "
                f"global=${global_price:.6f} ({pct_diff:+.2f}%) "
                f"curvature={curvature:.4f}"
            )
            log.info(reason)
            return {"action": "sell", "confidence": confidence, "reason": reason}

        return {
            "action":     "hold",
            "confidence": 0.50,
            "reason":     f"Gauge HOLD: {symbol} in sync ({pct_diff:+.2f}%) curvature={curvature:.4f}",
        }

    except Exception as e:
        return {"action": "hold", "confidence": 0.5, "reason": f"Gauge error: {str(e)[:80]}"}
