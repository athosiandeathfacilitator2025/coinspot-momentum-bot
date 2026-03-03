# voters/anomaly_guard.py

def detect_anomaly(row, exchange=None, cg=None) -> bool:
    try:
        total_vol  = float(row.get("total_volume")               or 0)
        market_cap = float(row.get("market_cap")                 or 1)
        change_24h = abs(float(row.get("price_change_percentage_24h") or 0))
        price      = float(row.get("current_price")              or 0)
        vol_ratio  = total_vol / market_cap if market_cap > 0 else 0

        if price <= 0:
            return True
        if vol_ratio < 0.001:
            return True
        if change_24h > 50:
            return True
        return False
    except Exception:
        return False
