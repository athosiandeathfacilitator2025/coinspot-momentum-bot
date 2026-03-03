# vote_adapter.py

def vote_to_score(vote: dict) -> float:
    if not isinstance(vote, dict):
        return 0.0
    action     = vote.get("action", "hold").lower()
    confidence = float(vote.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    if action == "buy":
        return confidence
    elif action == "sell":
        return -confidence
    return 0.0
