"""
voters/geometric_voter.py — Tier 8: Elliptic Curve Snap-Back Detection
────────────────────────────────────────────────────────────────────────
WHAT IT DOES:
  Detects when price and volume have moved so far from their rolling
  statistical norms that the market is "off the curve" and due for a
  violent snap-back reversion.

  The math: normalise price and volume via z-score, then evaluate
  whether the point (norm_price, norm_volume) lies on/near the
  elliptic curve y² = x³ + ax + b.

  High delta (far from the curve) + high snap_prob → mean reversion likely.

  If price has RISEN far above normal + high snap probability → SELL
    (overbought, snap-back downward)
  If price has FALLEN far below normal + high snap probability → BUY
    (oversold, snap-back upward)
  Otherwise → HOLD

WHY IT WORKS:
  Crypto markets are mean-reverting over short windows. When price and
  volume are simultaneously extreme in the same direction, the probability
  of a reversal within the next few candles is elevated. This voter
  captures those moments without needing an external data source.

INTEGRATION:
  GeometricEngine is instantiated once in bot.py and shared here.
  It maintains a rolling price/volume history per symbol internally.
"""
import logging

log = logging.getLogger("GeometricVoter")

_DEFAULT_SNAP_THRESHOLD = 0.65  # configurable via config.yaml geometric.snap_prob_threshold


def geometric_vote(row: dict, geometric_engine, config: dict) -> dict:
    """
    row:             single coin dict from market scan
    geometric_engine: engines.geometric_engine.GeometricEngine instance
    config:          full CONFIG dict
    Returns:         {"action": str, "confidence": float, "reason": str}
    """
    try:
        symbol      = str(row.get("symbol") or "").upper()
        price       = float(row.get("current_price") or 0)
        volume      = float(row.get("total_volume")  or 0)
        change_24h  = float(row.get("price_change_percentage_24h") or 0)

        if price <= 0 or volume <= 0:
            return {"action": "hold", "confidence": 0.5, "reason": "Geometric: no price/volume"}

        result = geometric_engine.check_rationality(symbol, price, volume)

        snap_prob = float(result.get("snap_prob", 0.0))
        delta     = float(result.get("delta",     0.0))
        is_stable = bool(result.get("is_stable",  True))

        geo_cfg        = config.get("geometric", {})
        snap_threshold = float(geo_cfg.get("snap_prob_threshold", _DEFAULT_SNAP_THRESHOLD))

        if is_stable or snap_prob < snap_threshold:
            return {
                "action":     "hold",
                "confidence": 0.50,
                "reason":     (f"Geometric HOLD: {symbol} stable "
                               f"delta={delta:.2f} snap={snap_prob:.2f}"),
            }

        # Market is off-curve. Direction of the snap depends on WHERE price is.
        # If price is above rolling mean (positive z-score → change_24h proxy),
        # the snap-back is downward → sell/avoid.
        # If price is below rolling mean, snap-back is upward → buy.
        if change_24h > 0:
            # Price pumped up → snap-back downward
            confidence = min(0.88, 0.60 + snap_prob * 0.4)
            reason = (
                f"📐 GEOM SELL: {symbol} off-curve (snap={snap_prob:.2f} "
                f"delta={delta:.2f}) price pumped {change_24h:+.1f}% — reversion likely"
            )
            log.info(reason)
            return {"action": "sell", "confidence": confidence, "reason": reason}
        else:
            # Price dumped down → snap-back upward
            confidence = min(0.88, 0.60 + snap_prob * 0.4)
            reason = (
                f"📐 GEOM BUY: {symbol} off-curve (snap={snap_prob:.2f} "
                f"delta={delta:.2f}) price dumped {change_24h:+.1f}% — reversion likely"
            )
            log.info(reason)
            return {"action": "buy", "confidence": confidence, "reason": reason}

    except Exception as e:
        return {"action": "hold", "confidence": 0.5, "reason": f"Geometric error: {str(e)[:80]}"}
