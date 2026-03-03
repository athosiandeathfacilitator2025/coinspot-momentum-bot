# voters/rule_engine.py

def rule_vote(row, config: dict) -> dict:
    try:
        change_24h = float(row.get("price_change_percentage_24h") or 0)
        change_7d  = float(row.get("price_change_percentage_7d_in_currency") or 0)
        price      = float(row.get("current_price") or 0)

        if price <= 0:
            return {"action": "hold", "confidence": 0.0, "reason": "Rule VETO: zero price"}

        if change_24h < -15 and change_7d < -30:
            return {"action": "hold", "confidence": 0.0,
                    "reason": f"Rule VETO: crash 24h={change_24h:.1f}% 7d={change_7d:.1f}%"}

        if change_24h > 3 and change_7d > 5:
            return {"action": "buy", "confidence": 0.80,
                    "reason": f"Rule BUY: strong momentum 24h={change_24h:.1f}% 7d={change_7d:.1f}%"}

        if change_24h > 0 and change_7d > 0:
            return {"action": "buy", "confidence": 0.60,
                    "reason": f"Rule BUY: positive momentum 24h={change_24h:.1f}% 7d={change_7d:.1f}%"}

        if change_24h < -8 and change_7d < -15:
            return {"action": "sell", "confidence": 0.60,
                    "reason": f"Rule SELL: notable decline 24h={change_24h:.1f}% 7d={change_7d:.1f}%"}

        return {"action": "hold", "confidence": 0.50,
                "reason": f"Rule HOLD: mixed 24h={change_24h:.1f}% 7d={change_7d:.1f}%"}
    except Exception as e:
        return {"action": "hold", "confidence": 0.5, "reason": f"Rule error: {str(e)[:60]}"}
