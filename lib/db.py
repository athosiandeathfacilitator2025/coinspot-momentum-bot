"""
lib/db.py — Unified database layer (SQLAlchemy + Supabase PostgreSQL)

IMPORTANT DEPLOYMENT NOTE:
  The DROP TABLE statements in init_db() are intentional for the FIRST
  clean deploy. They wipe the old broken schema (AUTOINCREMENT) and
  recreate with correct PostgreSQL SERIAL syntax.

  After first successful deploy — remove the DROP TABLE lines so that
  trade history is preserved across future restarts.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

log = logging.getLogger("Database")


def _make_engine(database_url: str):
    if not database_url:
        raise ValueError("DATABASE_URL is not set")
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
    )


class Database:

    def __init__(self, database_url: str):
        self.engine  = _make_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self._init_db_safe()

    def get_engine(self):
        return self.engine

    def get_session(self):
        return self.Session()

    def _init_db_safe(self):
        try:
            self.init_db()
        except Exception as e:
            log.error("init_db failed: %s", str(e)[:200])
            raise

    def init_db(self):
        with self.engine.connect() as conn:
            # ── ONE-TIME SCHEMA FIX ──────────────────────────────────────────
            # Drops old tables that were created with SQLite AUTOINCREMENT
            # syntax which PostgreSQL rejects. Remove these DROP lines after
            # first successful deploy to preserve trade history.
            conn.execute(text("DROP TABLE IF EXISTS open_positions CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS trade_logs CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS equity_curve CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS bot_thoughts CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS pattern_trades CASCADE"))
            # ── END ONE-TIME FIX ─────────────────────────────────────────────

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
                    id         SERIAL PRIMARY KEY,
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
                    id        SERIAL PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    equity    REAL NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS bot_thoughts (
                    id        SERIAL PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    category  TEXT NOT NULL,
                    content   TEXT NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pattern_trades (
                    id         SERIAL PRIMARY KEY,
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
            log.info("Database schema ready")

    def save_position(self, coin: str, entry: float, capital: float, qty: float):
        try:
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
                       "qty": qty,
                       "opened_at": datetime.now(timezone.utc).isoformat()})
                conn.commit()
        except Exception as e:
            log.error("save_position failed [%s]: %s", coin, str(e)[:150])

    def update_position_pnl(self, coin: str, highest_pnl: float, trail_stop: float):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE open_positions
                    SET highest_pnl=:hp, trail_stop=:ts
                    WHERE coin=:coin
                """), {"coin": coin, "hp": highest_pnl, "ts": trail_stop})
                conn.commit()
        except Exception as e:
            log.error("update_position_pnl failed [%s]: %s", coin, str(e)[:150])

    def close_position(self, coin: str):
        try:
            with self.engine.connect() as conn:
                conn.execute(text(
                    "DELETE FROM open_positions WHERE coin=:coin"
                ), {"coin": coin})
                conn.commit()
        except Exception as e:
            log.error("close_position failed [%s]: %s", coin, str(e)[:150])

    def get_open_positions(self) -> list:
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT coin, entry, capital, qty, opened_at, "
                    "highest_pnl, trail_stop FROM open_positions"
                )).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as e:
            log.error("get_open_positions failed: %s", str(e)[:150])
            return []

    def log_trade(self, coin: str, action: str, pnl, reason: str,
                  change_24h=None, change_7d=None, vol_ratio=None, regime=None):
        try:
            from datetime import datetime, timezone
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO trade_logs
                        (time, coin, action, pnl, reason,
                         change_24h, change_7d, vol_ratio, regime)
                    VALUES
                        (:time, :coin, :action, :pnl, :reason,
                         :c24, :c7d, :vr, :regime)
                """), {
                    "time":   datetime.now(timezone.utc).isoformat(),
                    "coin":   coin,
                    "action": action,
                    "pnl":    pnl,
                    "reason": reason,
                    "c24":    change_24h,
                    "c7d":    change_7d,
                    "vr":     vol_ratio,
                    "regime": regime,
                })
                conn.commit()
        except Exception as e:
            log.error("log_trade failed [%s %s]: %s", action, coin, str(e)[:150])

    def log_thought(self, category: str, content: str):
        try:
            from datetime import datetime, timezone
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO bot_thoughts (timestamp, category, content)
                    VALUES (:ts, :cat, :content)
                """), {
                    "ts":      datetime.now(timezone.utc).isoformat(),
                    "cat":     category,
                    "content": str(content)[:500],
                })
                conn.commit()
        except Exception as e:
            # Never crash the bot because of a logging failure
            log.warning("log_thought failed [%s]: %s", category, str(e)[:100])

    def record_equity(self, equity: float):
        try:
            from datetime import datetime, timezone
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO equity_curve (timestamp, equity)
                    VALUES (:ts, :eq)
                """), {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "eq": equity,
                })
                conn.commit()
        except Exception as e:
            log.error("record_equity failed: %s", str(e)[:150])

    def log_pattern_trade(self, coin: str, regime: str, pnl: float,
                          net_pnl: float, change_24h=None,
                          change_7d=None, vol_ratio=None):
        try:
            from datetime import datetime, timezone
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO pattern_trades
                        (timestamp, coin, regime, pnl, net_pnl,
                         change_24h, change_7d, vol_ratio)
                    VALUES
                        (:ts, :coin, :regime, :pnl, :net_pnl,
                         :c24, :c7d, :vr)
                """), {
                    "ts":      datetime.now(timezone.utc).isoformat(),
                    "coin":    coin,
                    "regime":  regime,
                    "pnl":     pnl,
                    "net_pnl": net_pnl,
                    "c24":     change_24h,
                    "c7d":     change_7d,
                    "vr":      vol_ratio,
                })
                conn.commit()
        except Exception as e:
            log.error("log_pattern_trade failed [%s]: %s", coin, str(e)[:150])
