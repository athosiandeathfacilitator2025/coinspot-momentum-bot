# feature_aggregator.py

def smooth_momentum(snapshot):
    m24 = snapshot.get("momentum_24h", 0.0)
    m7d = snapshot.get("momentum_7d", 0.0)
    short_term  = m24
    medium_term = (m24 * 0.4) + (m7d * 0.6)
    if short_term > 0 and medium_term > 0:
        regime = "bull"
    elif short_term < 0 and medium_term < 0:
        regime = "bear"
    else:
        regime = "chop"
    return {
        "short_momentum":  short_term,
        "medium_momentum": medium_term,
        "regime":          regime,
    }
