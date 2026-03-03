# voters/ta_engine.py
import pandas as pd

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


def ta_vote(row, df: pd.DataFrame) -> dict:
    try:
        change_24h = float(row.get("price_change_percentage_24h") or 0)
        total_vol  = float(row.get("total_volume")  or 0)
        market_cap = float(row.get("market_cap")    or 1)
        vol_ratio  = total_vol / market_cap if market_cap > 0 else 0

        rsi = 50.0
        if TA_AVAILABLE and df is not None and not df.empty and len(df) >= 5:
            try:
                changes    = df["price_change_percentage_24h"].fillna(0).values
                close_vals = pd.Series(100.0 * (1 + changes / 100)).cumprod()
                rsi_series = ta.rsi(close_vals, length=min(len(close_vals) - 1, 14))
                if rsi_series is not None and not rsi_series.empty:
                    rsi = float(rsi_series.iloc[-1])
                    if pd.isna(rsi):
                        rsi = 50.0
            except Exception:
                rsi = 50.0

        if rsi < 28:
            return {"action": "buy", "confidence": 0.82,
                    "reason": f"TA BUY: RSI deeply oversold={rsi:.1f}"}
        if rsi < 35:
            return {"action": "buy", "confidence": 0.72,
                    "reason": f"TA BUY: RSI oversold={rsi:.1f}"}
        if rsi > 72 and change_24h < 0:
            return {"action": "sell", "confidence": 0.72,
                    "reason": f"TA SELL: RSI overbought={rsi:.1f} + declining"}
        if vol_ratio > 0.03 and change_24h > 2:
            return {"action": "buy", "confidence": 0.68,
                    "reason": f"TA BUY: volume breakout vol_ratio={vol_ratio:.3f}"}
        if change_24h > 4 and vol_ratio > 0.02:
            return {"action": "buy", "confidence": 0.62,
                    "reason": f"TA BUY: surge change={change_24h:.1f}%"}
        if change_24h < -10:
            return {"action": "sell", "confidence": 0.60,
                    "reason": f"TA SELL: significant decline change={change_24h:.1f}%"}
        return {"action": "hold", "confidence": 0.50,
                "reason": f"TA HOLD: RSI={rsi:.1f} change={change_24h:.1f}%"}
    except Exception as e:
        return {"action": "hold", "confidence": 0.5, "reason": f"TA error: {str(e)[:60]}"}
