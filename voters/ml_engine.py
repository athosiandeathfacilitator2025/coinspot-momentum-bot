# voters/ml_engine.py
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


def _encode_regime(regime: str) -> float:
    return {"bull": 1.0, "bear": -1.0, "range": 0.0}.get(regime, 0.0)


def ml_vote(row, db_session) -> dict:
    try:
        from sqlalchemy import text

        results = db_session.execute(text("""
            SELECT COALESCE(net_pnl, pnl) as pnl, coin, regime,
                   change_24h, change_7d, vol_ratio
            FROM pattern_trades
            ORDER BY timestamp DESC
            LIMIT 300
        """)).fetchall()

        if not results or len(results) < 15:
            if results:
                pnls = [float(r[0]) for r in results if r[0] is not None]
                wr   = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.5
                return {"action": "buy" if wr > 0.55 else "hold", "confidence": 0.55,
                        "reason": f"ML warmup: WR={wr:.2f} ({len(results)}/15 trades needed)"}
            return {"action": "hold", "confidence": 0.5, "reason": "ML: no trade history yet"}

        if not SK_AVAILABLE:
            return {"action": "hold", "confidence": 0.5, "reason": "ML: sklearn not available"}

        rows_data, labels = [], []
        for r in results:
            pnl    = float(r[0]) if r[0] is not None else 0.0
            regime = str(r[2]) if r[2] else "range"
            c24    = float(r[3]) if r[3] is not None else 0.0
            c7d    = float(r[4]) if r[4] is not None else 0.0
            vol_r  = float(r[5]) if r[5] is not None else 0.0
            rows_data.append([c24, c7d, vol_r, _encode_regime(regime)])
            labels.append(1 if pnl > 0 else 0)

        X_train = np.array(rows_data, dtype=float)
        y_train = np.array(labels,    dtype=int)

        if len(np.unique(y_train)) < 2:
            wr = float(y_train.mean())
            return {"action": "buy" if wr > 0.5 else "hold", "confidence": 0.58,
                    "reason": f"ML single-class: WR={wr:.2f} ({len(y_train)} trades)"}

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model    = LogisticRegression(max_iter=500, random_state=42, C=0.8)
        model.fit(X_scaled, y_train)

        current_regime = str(row.get("regime", "range")) if hasattr(row, "get") else "range"
        c24_now  = float(row.get("price_change_percentage_24h") or 0)
        c7d_now  = float(row.get("price_change_percentage_7d_in_currency") or 0)
        vol_now  = float(row.get("total_volume") or 0) / max(float(row.get("market_cap") or 1), 1)

        X_now    = scaler.transform([[c24_now, c7d_now, vol_now, _encode_regime(current_regime)]])
        win_prob = float(model.predict_proba(X_now)[0][1])
        recent_wr = float(y_train[:20].mean()) if len(y_train) >= 20 else 0.5
        overall_wr = float(y_train.mean())

        if win_prob > 0.60 and recent_wr > 0.45:
            return {"action": "buy", "confidence": min(0.88, win_prob),
                    "reason": f"ML BUY: p={win_prob:.2f} recentWR={recent_wr:.2f} n={len(y_train)}"}
        if win_prob < 0.40 and recent_wr < 0.55:
            return {"action": "sell", "confidence": min(0.88, 1.0 - win_prob),
                    "reason": f"ML SELL: p={win_prob:.2f} recentWR={recent_wr:.2f}"}
        return {"action": "hold", "confidence": 0.52,
                "reason": f"ML HOLD: p={win_prob:.2f} WR={overall_wr:.2f} n={len(y_train)}"}
    except Exception as e:
        return {"action": "hold", "confidence": 0.5, "reason": f"ML error: {str(e)[:80]}"}
