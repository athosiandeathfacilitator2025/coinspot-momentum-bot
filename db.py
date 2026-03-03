"""
lib/db.py — Unified database layer (SQLAlchemy + PostgreSQL/SQLite).
Replaces the old Supabase client. All tables created on first run.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def _make_engine(database_url: str):
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    return create_engine(database_url, pool_pre_ping=True)


class Database:
    """High-level DB helpers used by bot.py."""

    def __init__(self, database_url: str):
        self.engine  = _make_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.init_db()

    def get_engine(self):
        return self.engine

    def get_session(self):
        return self.Session()

    def init_db(self):
        """Create all tables if they don't exist."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    coin        TEXT PRIMARY KEY,
                    entry       REAL NOT NULL,
                    capital     REAL NOT NULL,
                    qty         REAL NOT NULL,
                    opened_at   TEXT NOT NULL,
                    highest_pnl REAL DEFAULT 0,
                    trail_stop  REAL DEFAULT -999
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    time       TEXT NOT NULL,
                    coin       TEXT NOT NULL,
                    action     TEXT NOT NULL,
                    pnl        REAL,
                    reason     TEXT,
                    change_24h REAL,
                    change_7d  REAL,
                    vol_ratio  REAL,
                    regime     TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity    REAL NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS bot_thoughts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category  TEXT NOT NULL,
                    content   TEXT NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pattern_trades (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT NOT NULL,
                    coin       TEXT,
                    regime     TEXT,
                    pnl        REAL,
                    net_pnl    REAL,
                    change_24h REAL,
                    change_7d  REAL,
                    vol_ratio  REAL
                )
            """))
            conn.commit()

    def save_position(self, coin: str, entry: float, capital: float, qty: float):
        from datetime import datetime, timezone
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO open_positions
                    (coin, entry, capital, qty, opened_at, highest_pnl, trail_stop)
                VALUES (:coin, :entry, :capital, :qty, :opened_at, 0, -999)
                ON CONFLICT(coin) DO UPDATE SET
                    entry=excluded.entry, capital=excluded.capital,
                    qty=excluded.qty, opened_at=excluded.opened_at,
                    highest_pnl=0, trail_stop=-999
            """), {"coin": coin, "entry": entry, "capital": capital,
                   "qty": qty, "opened_at": datetime.now(timezone.utc).isoformat()})
            conn.commit()

    def update_position_pnl(self, coin: str, highest_pnl: float, trail_stop: float):
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE open_positions SET highest_pnl=:hp, trail_stop=:ts WHERE coin=:coin
            """), {"coin": coin, "hp": highest_pnl, "ts": trail_stop})
            conn.commit()

    def close_position(self, coin: str):
        with self.engine.connect() as conn:
            conn.execute(text("DELETE FROM open_positions WHERE coin=:coin"), {"coin": coin})
            conn.commit()

    def get_open_positions(self) -> list:
        with self.engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT coin, entry, capital, qty, opened_at, highest_pnl, trail_stop "
                "FROM open_positions"
            )).fetchall()
        return [dict(r._mapping) for r in rows]

    def log_trade(self, coin: str, action: str, pnl: float, reason: str,
                  change_24h=None, change_7d=None, vol_ratio=None, regime=None):
        from datetime import datetime, timezone
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO trade_logs
                    (time, coin, action, pnl, reason, change_24h, change_7d, vol_ratio, regime)
                VALUES (:time, :coin, :action, :pnl, :reason, :c24, :c7d, :vr, :regime)
            """), {"time": datetime.now(timezone.utc).isoformat(),
                   "coin": coin, "action": action, "pnl": pnl, "reason": reason,
                   "c24": change_24h, "c7d": change_7d, "vr": vol_ratio, "regime": regime})
            conn.commit()

    def log_thought(self, category: str, content: str):
        from datetime import datetime, timezone
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO bot_thoughts (timestamp, category, content)
                VALUES (:ts, :cat, :content)
            """), {"ts": datetime.now(timezone.utc).isoformat(),
                   "cat": category, "content": content[:500]})
            conn.commit()

    def record_equity(self, equity: float):
        from datetime import datetime, timezone
        with self.engine.connect() as conn:
            conn.execute(text(
                "INSERT INTO equity_curve (timestamp, equity) VALUES (:ts, :eq)"
            ), {"ts": datetime.now(timezone.utc).isoformat(), "eq": equity})
            conn.commit()

    def log_pattern_trade(self, coin: str, regime: str, pnl: float, net_pnl: float,
                          change_24h=None, change_7d=None, vol_ratio=None):
        from datetime import datetime, timezone
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO pattern_trades
                    (timestamp, coin, regime, pnl, net_pnl, change_24h, change_7d, vol_ratio)
                VALUES (:ts, :coin, :regime, :pnl, :net_pnl, :c24, :c7d, :vr)
            """), {"ts": datetime.now(timezone.utc).isoformat(),
                   "coin": coin, "regime": regime, "pnl": pnl, "net_pnl": net_pnl,
                   "c24": change_24h, "c7d": change_7d, "vr": vol_ratio})
            conn.commit()
