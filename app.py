"""
app.py  —  Apex Coinspot AI Dashboard
───────────────────────────────────────
Includes:
  - Live wallet from Coinspot API
  - Fear & Greed Index live display
  - Historical context panel
  - Quant Engine metrics (Gauge & Elliptic Curves)
  - Full error messages so API issues are visible
  - Auto-refresh every 15s
"""

import os
import time
from datetime import datetime, timezone

import ccxt
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Apex Coinspot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Database ──────────────────────────────────────────────────────────────────
_raw_db = os.getenv("DATABASE_URL", "").strip()
if _raw_db.startswith("postgres://"):
    _raw_db = _raw_db.replace("postgres://", "postgresql://", 1)
DB_URL = _raw_db if _raw_db else "sqlite:///apex_learning.db"


@st.cache_resource
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


db_engine = get_engine()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0e1117; color: #e0e0e0; }
@keyframes scroll-left {
    0%   { transform: translateX(100vw); }
    100% { transform: translateX(-100%); }
}
.ticker-wrap {
    width: 100%; overflow: hidden; background: #0e1117;
    border-bottom: 2px solid #00ff41; padding: 7px 0 5px 0; margin-bottom: 14px;
}
.ticker-content {
    display: inline-block; white-space: nowrap;
    animation: scroll-left 45s linear infinite;
    color: #00ff41; font-family: 'Courier New', monospace;
    font-size: 0.9rem; letter-spacing: 0.04em;
}
div[data-testid="metric-container"] {
    background: #1a1d27; border: 1px solid #2e3148;
    border-radius: 6px; padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Live ticker ───────────────────────────────────────────────────────────────
@st.fragment(run_every="5s")
def render_ticker():
    try:
        with db_engine.connect() as conn:
            rows = conn.execute(
                text("SELECT content FROM bot_thoughts ORDER BY timestamp DESC LIMIT 10")
            ).fetchall()
        items = [r[0] for r in rows if r[0]]
        ticker_text = "  ·  ".join(items) if items else "🔍 Apex Engine scanning markets..."
    except Exception as e:
        ticker_text = f"⏳ Connecting... ({str(e)[:50]})"
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker-content">{ticker_text}</div></div>',
        unsafe_allow_html=True,
    )


render_ticker()
st.title("🤖  Apex Coinspot AI Terminal")

# ── Top metrics ───────────────────────────────────────────────────────────────
try:
    # today's date prefix — timestamps stored as ISO text e.g. "2026-03-04T01:47:..."
    today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "%"

    with db_engine.connect() as conn:
        latest   = conn.execute(text("SELECT equity FROM equity_curve ORDER BY timestamp DESC LIMIT 1")).fetchone()
        earliest = conn.execute(text("SELECT equity FROM equity_curve ORDER BY timestamp ASC  LIMIT 1")).fetchone()
        t_total  = conn.execute(text("SELECT COUNT(*) FROM trade_logs")).fetchone()[0]
        t_wins   = conn.execute(text("SELECT COUNT(*) FROM trade_logs WHERE pnl > 0")).fetchone()[0]

        math_res = conn.execute(text("""
            SELECT
                SUM(CASE WHEN content LIKE '%🌀%' THEN 1 ELSE 0 END)                        AS gauge_count,
                SUM(CASE WHEN content LIKE '%📐%' OR content LIKE '%GEOM EXIT%' THEN 1 ELSE 0 END) AS geom_count
            FROM bot_thoughts
            WHERE timestamp LIKE :today
        """), {"today": today_prefix}).fetchone()

    current_eq  = float(latest[0])   if latest   else 0.0
    start_eq    = float(earliest[0]) if earliest else current_eq
    daily_pct   = ((current_eq - start_eq) / start_eq * 100) if start_eq > 0 else 0.0
    win_rate    = (t_wins / t_total * 100) if t_total > 0 else 0.0
    gauge_count = int(math_res[0]) if math_res and math_res[0] else 0
    geom_count  = int(math_res[1]) if math_res and math_res[1] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio AUD",  f"${current_eq:,.2f}")
    c2.metric("Daily Progress", f"{daily_pct:+.2f}%")
    c3.metric("Total Trades",   str(t_total))
    c4.metric("Win Rate",       f"{win_rate:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    mc1.metric("🌀 Daily Gauge Arbitrage Detections", str(gauge_count))
    mc2.metric("📐 Daily Geometric Snap-Backs",       str(geom_count))

except Exception as e:
    st.warning(f"⏳ Waiting for first bot cycle — {str(e)[:80]}")

st.divider()


# ── Fear & Greed Index ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        if r.status_code == 200:
            d = r.json()["data"][0]
            return int(d["value"]), d["value_classification"]
    except Exception:
        pass
    return 50, "Unknown"


# ── Live wallet ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_wallet():
    key    = os.getenv("COINSPOT_API_KEY",    "").strip()
    secret = os.getenv("COINSPOT_API_SECRET", "").strip()

    if not key or not secret:
        return pd.DataFrame(), 0.0, (
            "COINSPOT_API_KEY or COINSPOT_API_SECRET missing from Render environment."
        )

    try:
        ex = ccxt.coinspot({
            "apiKey":          key,
            "secret":          secret,
            "enableRateLimit": True,
            "nonce":           lambda: int(time.time() * 1000),
        })

        bal       = ex.fetch_balance()
        total_aud = float(bal["total"].get("AUD") or 0)
        rows      = []

        for coin, amt in bal["total"].items():
            if coin == "AUD":
                continue
            if not amt or float(amt) <= 0:
                continue
            try:
                ticker = ex.fetch_ticker(f"{coin}/AUD")
                price  = float(ticker["last"] or 0)
                if price <= 0:
                    continue
                value      = float(amt) * price
                total_aud += value
                rows.append({
                    "Coin":        coin,
                    "Balance":     float(amt),
                    "Price (AUD)": price,
                    "Value (AUD)": value,
                })
            except Exception:
                rows.append({
                    "Coin":        coin,
                    "Balance":     float(amt),
                    "Price (AUD)": 0.0,
                    "Value (AUD)": 0.0,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Value (AUD)", ascending=False)
        return df, total_aud, None

    except Exception as e:
        return pd.DataFrame(), 0.0, f"Coinspot API error: {str(e)}"


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰  Wallet & Holdings",
    "📈  Performance",
    "🧠  Engine Intelligence",
    "🧩  Pattern Memory",
    "🧮  Quant Engines",
])

# ── TAB 1 — Live Wallet ───────────────────────────────────────────────────────
with tab1:

    fng_val, fng_lbl = load_fear_greed()
    fng_color = (
        "#00ff41" if fng_val <= 24 else
        "#90ee90" if fng_val <= 45 else
        "#ffa500" if fng_val <= 65 else
        "#ff4444"
    )
    fng_interpretation = (
        "🟢 Historically the BEST time to buy — crowd is panicking"     if fng_val <= 24 else
        "🟡 Below neutral — slight buy lean, caution still warranted"    if fng_val <= 45 else
        "🟠 Neutral territory — no strong crowd signal"                  if fng_val <= 65 else
        "🔴 Crowd is euphoric — historically precedes corrections"
    )
    st.markdown(
        f"<div style='background:#1a1d27;border:1px solid {fng_color};"
        f"border-radius:8px;padding:12px 18px;margin-bottom:16px'>"
        f"<b style='color:{fng_color}'>😨 Fear & Greed Index: "
        f"{fng_val} — {fng_lbl}</b><br>"
        f"<span style='color:#aaa;font-size:0.9em'>{fng_interpretation}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.subheader("💰 Live Coinspot Wallet")
    st.caption("Fetched live from Coinspot every 30 seconds")

    holdings, total_val, wallet_err = load_wallet()

    if wallet_err:
        st.error(f"❌ Wallet fetch failed:\n\n`{wallet_err}`")
        st.info("If you see '401 Unauthorized' — check your API keys in Render environment variables.")
    else:
        if not holdings.empty:
            disp = holdings.copy()
            disp["Balance"]     = disp["Balance"].apply(lambda x: f"{x:,.6f}")
            disp["Price (AUD)"] = disp["Price (AUD)"].apply(
                lambda x: f"${x:,.4f}" if x > 0 else "N/A"
            )
            disp["Value (AUD)"] = disp["Value (AUD)"].apply(
                lambda x: f"${x:,.2f}" if x > 0 else "N/A"
            )
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("💵 No coin holdings — wallet is AUD cash only")
        st.metric("💼 Total Portfolio Value", f"${total_val:,.2f} AUD")

    st.subheader("🎯 Active Position")
    try:
        with db_engine.connect() as conn:
            last_buy = conn.execute(text(
                "SELECT coin, time, reason FROM trade_logs WHERE action='BUY' ORDER BY time DESC LIMIT 1"
            )).fetchone()
            last_sell = conn.execute(text(
                "SELECT time FROM trade_logs WHERE action='SELL' ORDER BY time DESC LIMIT 1"
            )).fetchone()

        if last_buy:
            buy_time  = pd.to_datetime(last_buy[1])
            sell_time = pd.to_datetime(last_sell[0]) if last_sell else None
            if sell_time is None or buy_time > sell_time:
                st.success(f"🟢 HOLDING **{last_buy[0]}** | Entry: {buy_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.caption(str(last_buy[2]))
            else:
                st.info("⚪ No open position — scanning for next entry")
        else:
            st.info("No trades yet — bot warming up")
    except Exception as e:
        st.warning(f"Position check: {str(e)[:100]}")

    st.subheader("📂 Open Positions")
    try:
        open_pos = pd.read_sql(
            "SELECT coin, entry, capital, qty, opened_at, highest_pnl, trail_stop FROM open_positions ORDER BY opened_at DESC",
            db_engine,
        )
        if not open_pos.empty:
            open_pos["entry"]       = open_pos["entry"].apply(lambda x: f"${x:,.4f}")
            open_pos["capital"]     = open_pos["capital"].apply(lambda x: f"${x:,.2f}")
            open_pos["qty"]         = open_pos["qty"].apply(lambda x: f"{x:,.6f}")
            open_pos["highest_pnl"] = open_pos["highest_pnl"].apply(
                lambda x: f"{float(x)*100:+.2f}%" if pd.notna(x) else "—"
            )
            open_pos["trail_stop"]  = open_pos["trail_stop"].apply(
                lambda x: f"{float(x)*100:.2f}% 🟡 ACTIVE"
                if pd.notna(x) and float(x) > -900 else "⏳ waiting for initial TP"
            )
            open_pos = open_pos.rename(columns={
                "highest_pnl": "Peak PnL",
                "trail_stop":  "Trail Stop",
            })
            st.dataframe(open_pos, use_container_width=True, hide_index=True)
            st.caption("Coinspot fee: 0.1% per side (0.2% round trip). All PnL shown is net after fees.")
        else:
            st.info("No open positions")
    except Exception as e:
        st.caption(f"open_positions: {str(e)[:80]}")

    st.subheader("🔭 Historical Intelligence Feed")
    st.caption("What the bot knows about each coin before deciding to trade it")
    try:
        with db_engine.connect() as conn:
            result = conn.execute(text(
                """SELECT timestamp, content
                   FROM bot_thoughts
                   WHERE category = 'VOTE'
                      OR content LIKE '%HistCtx%'
                      OR content LIKE '%VETO%'
                   ORDER BY timestamp DESC LIMIT 40"""
            ))
            hist_thoughts = pd.DataFrame([dict(row) for row in result.mappings()])
        if not hist_thoughts.empty:
            st.dataframe(hist_thoughts, use_container_width=True, hide_index=True)
        else:
            st.info("Historical context populates after first bot cycle (~15s)")
    except Exception as e:
        st.caption(str(e)[:80])


# ── TAB 2 — Performance ───────────────────────────────────────────────────────
with tab2:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("📈 Equity Curve")
        try:
            curve = pd.read_sql(
                "SELECT equity, timestamp FROM equity_curve ORDER BY timestamp ASC",
                db_engine,
            )
            if not curve.empty:
                curve["timestamp"] = pd.to_datetime(curve["timestamp"])
                st.line_chart(curve.set_index("timestamp")["equity"])
            else:
                st.info("Equity curve appears after first bot cycle (~15s)")
        except Exception as e:
            st.warning(str(e)[:100])

    with right:
        st.subheader("📜 Trade Journal")
        try:
            logs = pd.read_sql(
                "SELECT time, coin, action, pnl, reason FROM trade_logs ORDER BY time DESC LIMIT 30",
                db_engine,
            )
            if not logs.empty:
                logs["pnl"] = logs["pnl"].apply(
                    lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "—"
                )
                def highlight_math(row):
                    if pd.notna(row['reason']) and any(
                        kw in str(row['reason']) for kw in ['GEOM', 'Gauge', 'Snap', 'MANDATORY', 'VETO']
                    ):
                        return ['background-color: rgba(0, 255, 65, 0.15)'] * len(row)
                    return [''] * len(row)
                st.dataframe(logs.style.apply(highlight_math, axis=1),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No trades yet")
        except Exception as e:
            st.warning(str(e)[:100])


# ── TAB 3 — Engine Intelligence ───────────────────────────────────────────────
with tab3:
    st.subheader("🧠 Engine Intelligence Feed")
    st.caption("Live voter reasoning, order execution log")

    cat_filter = st.multiselect(
        "Filter by category",
        ["SCAN", "VOTE", "TRADE", "ORDER", "LEARN", "TRAP", "ERROR", "SYSTEM", "SKIP"],
        default=["VOTE", "TRADE", "ORDER", "TRAP", "ERROR"],
    )

    try:
        if cat_filter:
            placeholders = ", ".join(f"'{c}'" for c in cat_filter)
            q = f"SELECT timestamp, category, content FROM bot_thoughts WHERE category IN ({placeholders}) ORDER BY timestamp DESC LIMIT 100"
        else:
            q = "SELECT timestamp, category, content FROM bot_thoughts ORDER BY timestamp DESC LIMIT 100"

        brain = pd.read_sql(q, db_engine)
        if not brain.empty:
            st.dataframe(brain, use_container_width=True, hide_index=True)
        else:
            st.info("Intelligence feed populates within 15s of deployment")
    except Exception as e:
        st.warning(str(e)[:100])


# ── TAB 4 — Pattern Memory ────────────────────────────────────────────────────
with tab4:
    st.subheader("🧩 Learned Pattern Ledger")
    st.caption("Patterns with WR < 45% and Sharpe < 0.2 after 5+ trades are auto-disabled")

    try:
        raw = pd.read_sql("SELECT coin, regime, pnl FROM pattern_trades", db_engine)

        if not raw.empty:
            def summarise(g):
                pnls   = g["pnl"].dropna()
                trades = len(pnls)
                wr     = round(float((pnls > 0).mean() * 100), 1) if trades > 0 else 0.0
                avg    = round(float(pnls.mean() * 100), 3)       if trades > 0 else 0.0
                return pd.Series({"trades": trades, "win_rate_%": wr, "avg_pnl_%": avg})

            patterns = (
                raw.groupby(["coin", "regime"])
                .apply(summarise, include_groups=False)
                .reset_index()
                .sort_values("trades", ascending=False)
                .head(50)
            )
            st.dataframe(patterns, use_container_width=True, hide_index=True)
        else:
            st.info("Pattern ledger fills after first completed trade")

    except Exception as e:
        st.warning(str(e)[:100])


# ── TAB 5 — Quant Engines ─────────────────────────────────────────────────────
with tab5:
    st.subheader("🧮 Math Engine Live Activity")
    st.caption("Monitoring Gauge Invariance (Arbitrage) and Elliptic Snap-Backs — includes TAKEOVER/ELEVATED events")

    try:
        math_query = text("""
            SELECT timestamp, category, content
            FROM bot_thoughts
            WHERE content LIKE '%🌀%'
               OR content LIKE '%📐%'
               OR content LIKE '%GEOM%'
               OR content LIKE '%Gauge%'
               OR content LIKE '%TAKEOVER%'
               OR content LIKE '%ELEVATED%'
               OR content LIKE '%MANDATORY%'
               OR content LIKE '%VETO_SELL%'
            ORDER BY timestamp DESC
            LIMIT 50
        """)
        math_df = pd.read_sql(math_query, db_engine)
        if not math_df.empty:
            math_df['timestamp'] = pd.to_datetime(math_df['timestamp']).dt.strftime('%H:%M:%S')
            st.dataframe(math_df, use_container_width=True, hide_index=True)
        else:
            st.info("Math engines scanning... no anomalies detected yet.")
    except Exception as e:
        st.warning(str(e)[:100])

    st.subheader("🎯 Math-Driven Executions")
    st.caption("Trades entered or exited by MANDATORY_BUY or VETO_SELL signals")
    try:
        math_trades_query = text("""
            SELECT time, coin, action, pnl, reason
            FROM trade_logs
            WHERE reason LIKE '%MANDATORY%'
               OR reason LIKE '%VETO_SELL%'
               OR reason LIKE '%GEOM%'
               OR reason LIKE '%Gauge%'
            ORDER BY time DESC LIMIT 20
        """)
        math_trades = pd.read_sql(math_trades_query, db_engine)
        if not math_trades.empty:
            math_trades['pnl'] = math_trades['pnl'].apply(
                lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "—"
            )
            st.dataframe(math_trades, use_container_width=True, hide_index=True)
        else:
            st.info("No math-driven trades yet.")
    except Exception as e:
        st.warning(str(e)[:100])


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"⏱ Last render: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  "
    "| Dashboard auto-refreshes every 15s via fragment"
)
