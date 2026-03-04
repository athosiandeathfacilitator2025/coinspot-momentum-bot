"""
voters/market_context.py — Tier 4: Market context voter.

bot.py calls:  market_vote(row, regime, market_df)

regime comes from RegimeEngine.detect() which returns "BULL"/"BEAR"/"NEUTRAL".
"""
import pandas as pd
import logging

log = logging.getLogger("MarketVoter")


def market_vote(row, regime: str, df: pd.DataFrame = None) -> dict:
    try:
        change_24h = float(row.get("price_change_percentage_24h") or 0)
        change_7d  = float(row.get("price_change_percentage_7d_in_currency") or 0)
        volume     = float(row.get("total_volume") or 0)
        market_cap = float(row.get("market_cap")   or 1)
        vol_ratio  = volume / max(market_cap, 1)
        signals    = []

        regime_up = regime.upper()   # normalise — RegimeEngine returns uppercase

        if regime_up == "BULL" and change_24h > 1.0:
            signals.append(("buy",  0.70, f"aligned with BULL regime change={change_24h:.1f}%"))
        elif regime_up == "BEAR" and change_24h < -1.0:
            signals.append(("sell", 0.70, f"aligned with BEAR regime change={change_24h:.1f}%"))
        else:
            signals.append(("hold", 0.55, f"regime={regime} neutral"))

        if vol_ratio > 0.08:
            signals.append(("buy" if change_24h > 0 else "sell",
                            0.75 if change_24h > 0 else 0.70,
                            f"volume spike vol_ratio={vol_ratio:.3f}"))
        elif vol_ratio < 0.005:
            signals.append(("hold", 0.52, f"very low volume vol_ratio={vol_ratio:.4f}"))

        if df is not None and not df.empty and len(df) > 1:
            try:
                market_mean = float(df["price_change_percentage_24h"].mean())
                div         = change_24h - market_mean
                if div > 5.0:
                    signals.append(("buy",  0.72, f"outperforming market by {div:.1f}%"))
                elif div < -5.0:
                    signals.append(("sell", 0.68, f"underperforming market by {abs(div):.1f}%"))
                else:
                    signals.append(("hold", 0.54, f"inline with market div={div:.1f}%"))
            except Exception:
                pass

        if change_7d > 5.0 and change_24h > 0:
            signals.append(("buy",  0.65, f"7d trend confirming 24h 7d={change_7d:.1f}%"))
        elif change_7d < -5.0 and change_24h < 0:
            signals.append(("sell", 0.65, f"7d trend confirming decline 7d={change_7d:.1f}%"))

        if not signals:
            return {"action": "hold", "confidence": 0.5, "reason": "MKT: no signal"}

        buy_s  = sum(c for a, c, _ in signals if a == "buy")
        sell_s = sum(c for a, c, _ in signals if a == "sell")
        hold_s = sum(c for a, c, _ in signals if a == "hold")

        if buy_s > sell_s and buy_s > hold_s:
            best = max((s for s in signals if s[0] == "buy"), key=lambda x: x[1])
            return {"action": "buy",  "confidence": min(0.88, buy_s  / len(signals) * 1.3),
                    "reason": f"MKT BUY: {best[2]}"}
        if sell_s > buy_s and sell_s > hold_s:
            best = max((s for s in signals if s[0] == "sell"), key=lambda x: x[1])
            return {"action": "sell", "confidence": min(0.88, sell_s / len(signals) * 1.3),
                    "reason": f"MKT SELL: {best[2]}"}
        return {"action": "hold", "confidence": 0.55,
                "reason": f"MKT HOLD: mixed ({len(signals)} signals)"}

    except Exception as e:
        return {"action": "hold", "confidence": 0.5,
                "reason": f"MKT error: {str(e)[:60]}"}
