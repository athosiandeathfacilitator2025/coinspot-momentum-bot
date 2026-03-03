# portfolio_state.py

def count_open_positions(trades):
    if not trades:
        return 0
    return len([t for t in trades if t.get("status") == "OPEN"])
