"""
bot.py — Apex Coinspot Trading Bot (with Math Authority)
"""

import os
import time
import logging
import traceback
from datetime import datetime, timezone

import ccxt
import pandas as pd
from pycoingecko import CoinGeckoAPI

from config import CONFIG
from lib.db import Database
from lib.risk_engine import calculate_position_size
from lib.regime_engine import RegimeEngine

from engines.gauge_engine     import GaugeEngine
from engines.geometric_engine import GeometricEngine
from engines.math_authority   import MathAuthority

from voters.rule_engine        import rule_vote
from voters.ta_engine          import ta_vote
from voters.ml_engine          import ml_vote
from voters.market_context     import market_vote
from voters.rl_engine          import RLEngine
from voters.historical_context import historical_context_vote
from voters.gauge_voter        import gauge_vote
from voters.geometric_voter    import geometric_vote
from voters.anomaly_guard      import detect_anomaly
from vote_adapter              import vote_to_score

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("ApexBot")

COINSPOT_FEE  = 0.001
FEE_ROUNDTRIP = COINSPOT_FEE * 2


class ApexBot:

    def __init__(self):
        log.info("Apex Bot starting...")
        self.cfg = CONFIG

        self.db       = Database(self.cfg["database_url"])
        self.exchange = ccxt.coinspot({
            "apiKey":          self.cfg["coinspot_api_key"],
            "secret":          self.cfg["coinspot_api_secret"],
            "enableRateLimit": True,
            "nonce":           lambda: int(time.time() * 1000),
        })
        self.cg = CoinGeckoAPI()

        self.gauge_engine     = GaugeEngine(
            cache_ttl=self.cfg.get("gauge", {}).get("cache_ttl", 60)
        )
        self.geometric_engine = GeometricEngine(
            history_len=self.cfg.get("geometric", {}).get("history_len", 30)
        )
        self.regime_engine = RegimeEngine()
        self.rl_engine     = RLEngine()

        self.authority = MathAuthority(
            base_weights       = self.cfg.get("weights", {}),
            takeover_threshold = float(
                self.cfg.get("math_authority", {}).get("takeover_threshold", 0.70)
            ),
        )

        self.positions       = {}
        self.market_cache    = []
        self.market_cache_ts = 0.0
        self.daily_start_eq  = None
        self.circuit_open    = False
        self._dynamic_tp     = {}

        for pos in self.db.get_open_positions():
            self.positions[pos["coin"]] = pos
        log.info("Restored %d positions | RL=%s", len(self.positions),
                 "ACTIVE" if self.rl_engine.ready else "OFF")

    def fetch_market(self) -> list:
        now = time.time()
        ttl = float(self.cfg.get("market_cache_ttl", 90))
        if now - self.market_cache_ts < ttl and self.market_cache:
            return self.market_cache
        pairs = self.cfg.get("trading_pairs", [])
        try:
            data = self.cg.get_coins_markets(
                vs_currency="aud", ids=",".join(pairs),
                order="market_cap_desc", per_page=len(pairs), page=1,
                sparkline=False, price_change_percentage="24h,7d",
            )
            self.market_cache    = data
            self.market_cache_ts = now
        except Exception as e:
            log.warning("CoinGecko fetch failed (using cache): %s", str(e)[:80])
        return self.market_cache

    def detect_regime(self, market: list) -> str:
        prices = [float(c.get("current_price") or 0) for c in market if c.get("current_price")]
        return self.regime_engine.detect(prices) if len(prices) >= 20 else "NEUTRAL"

    def run_voters(self, row: dict, regime: str, market_df,
                   weight_overrides: dict) -> tuple:
        db_session = self.db.get_session()
        coin = str(row.get("symbol") or "").upper()
        pos  = self.positions.get(coin, {})

        votes = {
            "rule":      rule_vote(row, self.cfg),
            "ta":        ta_vote(row, market_df),
            "ml":        ml_vote(row, db_session),
            "market":    market_vote(row, regime, market_df),
            "history":   historical_context_vote(row, self.cg),
            "gauge":     gauge_vote(row, self.gauge_engine, self.cfg),
            "geometric": geometric_vote(row, self.geometric_engine, self.cfg),
        }
        db_session.close()

        rl_pos_state = {
            "in_position":    coin in self.positions,
            "unrealized_pnl": float(pos.get("unrealized_pnl", 0.0)),
            "highest_pnl":    float(pos.get("highest_pnl",    0.0)),
            "trade_duration": float(pos.get("hours_held",      0.0)),
            "daily_pnl":      0.0,
        }
        rl_raw = self.rl_engine.vote(row, rl_pos_state)
        votes["rl"] = {
            "action":     {0: "hold", 1: "buy", 2: "sell"}.get(rl_raw, "hold"),
            "confidence": 0.70,
            "reason":     "RL:" + str(rl_raw),
        }

        for name, vote in votes.items():
            if float(vote.get("confidence", 1.0)) == 0.0:
                reason = vote.get("reason", name + " hard veto")
                self.db.log_thought("TRAP", "VETO by " + name + ": " + reason[:200])
                return 0.0, reason, True

        net_score = 0.0
        parts     = []
        for name, vote in votes.items():
            w     = float(weight_overrides.get(name, 1.0))
            score = vote_to_score(vote) * w
            net_score += score
            if abs(score) >= 0.3:
                parts.append(name + ":" + str(round(score, 2)))

        summary = "score=" + str(round(net_score, 2)) + " [" + ", ".join(parts[:5]) + "]"
        return net_score, summary, False

    def try_enter(self, row: dict, net_score: float, reason: str,
                  forced: bool = False):
        coin   = str(row.get("symbol") or "").upper()
        symbol = coin + "/AUD"

        if coin in self.positions:
            return
        if len(self.positions) >= int(self.cfg.get("max_positions", 5)):
            return

        try:
            balance  = self.exchange.fetch_balance()
            aud_free = float(balance["free"].get("AUD") or 0)
        except Exception as e:
            log.error("Balance fetch failed: %s", e)
            return

        capital = min(
            calculate_position_size(aud_free, row, len(self.positions)),
            float(self.cfg.get("max_trade_aud", 100))
        )
        if capital < 5.0:
            return

        tag = "FORCED" if forced else "VOTED"
        log.info("[%s] %s BUY capital=%.2f score=%+.2f", coin, tag, capital, net_score)

        if self.cfg.get("simulate", False):
            price = float(row.get("current_price") or 1)
            qty   = (capital * (1 - COINSPOT_FEE)) / price
            self._record_entry(coin, price, capital, qty)
            self.db.log_thought("TRADE", "SIM " + tag + " BUY " + coin +
                                " @ $" + str(round(price, 6)) + " [" + reason[:120] + "]")
            return

        try:
            order = self.exchange.create_market_buy_order(symbol, capital)
            time.sleep(2)
            info  = self.exchange.fetch_order(order["id"], symbol)
            qty   = float(info.get("filled")  or 0)
            price = float(info.get("average") or 0)
            if qty <= 0 or price <= 0:
                return
            self._record_entry(coin, price, capital, qty)
            self.db.log_thought("TRADE", tag + " BUY " + coin +
                                " @ $" + str(round(price, 6)) + " [" + reason[:120] + "]")
        except Exception as e:
            log.error("BUY %s failed: %s", coin, e)
            self.db.log_thought("ERROR", "BUY " + coin + " failed: " + str(e)[:150])

    def _record_entry(self, coin, price, capital, qty):
        self.positions[coin] = {
            "coin": coin, "entry": price, "capital": capital, "qty": qty,
            "opened_at":   datetime.now(timezone.utc).isoformat(),
            "highest_pnl": 0.0, "trail_stop": -999.0, "hours_held": 0.0,
        }
        self.db.save_position(coin, price, capital, qty)
        self.db.log_trade(coin, "BUY", None, "entry @ $" + str(round(price, 6)))

    def force_exit(self, coin: str, price: float, reason: str):
        pos = self.positions.get(coin)
        if not pos:
            return
        qty     = float(pos["qty"])
        entry   = float(pos["entry"])
        gross   = (price - entry) / entry
        net_pnl = gross - FEE_ROUNDTRIP
        symbol  = coin + "/AUD"

        log.warning("VETO_SELL %s @ $%.6f net=%.2f%%", coin, price, net_pnl * 100)

        if not self.cfg.get("simulate", False):
            try:
                self.exchange.create_market_sell_order(symbol, qty)
            except Exception as e:
                log.error("VETO_SELL %s failed: %s", coin, e)
                self.db.log_thought("ERROR", "VETO_SELL " + coin + " failed: " + str(e)[:150])
                return

        self.db.log_trade(coin, "VETO_SELL", net_pnl * 100, reason[:200])
        self.db.log_thought("TRADE", "VETO_SELL " + coin +
                            " net=" + str(round(net_pnl * 100, 2)) + "% | " + reason[:120])
        self.db.log_pattern_trade(coin, "TAKEOVER", gross, net_pnl)
        self.db.close_position(coin)
        del self.positions[coin]
        self._dynamic_tp.pop(coin, None)
        self.authority.reset_symbol(coin)

    def check_exits(self, market: list):
        tp_cfg   = float(self.cfg.get("take_profit_pct", 0.8)) / 100
        sl_pct   = float(self.cfg.get("stop_loss_pct",   0.8)) / 100
        trail_b  = float(self.cfg.get("trailing_buffer",  0.005))
        max_min  = float(self.cfg.get("max_hold_minutes", 120))
        trail_on = bool(self.cfg.get("trailing_enabled",  True))

        price_map = {
            str(c.get("symbol") or "").upper(): float(c.get("current_price") or 0)
            for c in market
        }

        for coin, pos in list(self.positions.items()):
            entry = float(pos["entry"])
            qty   = float(pos["qty"])
            price = price_map.get(coin, 0.0)
            if price <= 0:
                try:
                    ticker = self.exchange.fetch_ticker(coin + "/AUD")
                    price  = float(ticker["last"] or 0)
                except Exception:
                    continue
            if price <= 0 or entry <= 0:
                continue

            gross   = (price - entry) / entry
            net_pnl = gross - FEE_ROUNDTRIP

            if gross > pos.get("highest_pnl", 0.0):
                pos["highest_pnl"] = gross
                if trail_on and gross >= tp_cfg * 0.5:
                    pos["trail_stop"] = gross - trail_b
                self.db.update_position_pnl(coin, pos["highest_pnl"],
                                            pos.get("trail_stop", -999))

            opened = datetime.fromisoformat(pos["opened_at"])
            mins_h = (datetime.now(timezone.utc) - opened).total_seconds() / 60
            pos["hours_held"] = mins_h / 60

            effective_tp = self._dynamic_tp.get(coin, tp_cfg)

            exit_reason = None
            if net_pnl >= effective_tp:
                tp_tag = "DynTP" if coin in self._dynamic_tp else "TP"
                exit_reason = tp_tag + " hit: net=" + str(round(net_pnl * 100, 2)) + "%"
            elif gross <= -sl_pct:
                exit_reason = "SL hit: gross=" + str(round(gross * 100, 2)) + "%"
            elif trail_on and pos.get("trail_stop", -999) > -999 and gross <= pos["trail_stop"]:
                exit_reason = "Trail stop: gross=" + str(round(gross * 100, 2)) + "%"
            elif mins_h >= max_min:
                exit_reason = "Timeout " + str(round(mins_h)) + "min net=" + str(round(net_pnl * 100, 2)) + "%"

            if exit_reason:
                self._execute_exit(coin, qty, price, net_pnl, exit_reason, pos)

    def _execute_exit(self, coin, qty, price, net_pnl, reason, pos):
        log.info("EXIT %s: %s", coin, reason)
        if not self.cfg.get("simulate", False):
            try:
                self.exchange.create_market_sell_order(coin + "/AUD", qty)
            except Exception as e:
                log.error("SELL %s failed: %s", coin, e)
                return
        self.db.log_trade(coin, "SELL", net_pnl * 100, reason)
        self.db.log_pattern_trade(coin, "NORMAL", net_pnl, net_pnl)
        self.db.log_thought("TRADE",
            ("WIN" if net_pnl > 0 else "LOSS") + " " + coin +
            " net=" + str(round(net_pnl * 100, 2)) + "% | " + reason)
        self.db.close_position(coin)
        del self.positions[coin]
        self._dynamic_tp.pop(coin, None)
        self.authority.reset_symbol(coin)

    def check_circuit_breaker(self, equity: float) -> bool:
        if self.daily_start_eq is None:
            self.daily_start_eq = equity
            return False
        max_dd = float(self.cfg.get("max_drawdown_pct", 5.0)) / 100
        dd = (self.daily_start_eq - equity) / max(self.daily_start_eq, 1)
        if dd >= max_dd:
            if not self.circuit_open:
                self.circuit_open = True
                self.db.log_thought("SYSTEM", "Circuit breaker: DD=" + str(round(dd * 100, 2)) + "%")
            return True
        self.circuit_open = False
        return False

    def run(self):
        log.info("Main loop starting")
        threshold = float(self.cfg.get("voting", {}).get("net_score_threshold", 1.0))

        while True:
            try:
                t0 = time.time()

                market = self.fetch_market()
                if not market:
                    time.sleep(30)
                    continue

                market_df = pd.DataFrame(market)
                regime    = self.detect_regime(market)
                self.db.log_thought("SCAN",
                    "Regime:" + regime + " coins:" + str(len(market)) +
                    " positions:" + str(len(self.positions)))

                try:
                    bal    = self.exchange.fetch_balance()
                    equity = float(bal["total"].get("AUD") or 0)
                    self.db.record_equity(equity)
                    if self.check_circuit_breaker(equity):
                        self.check_exits(market)
                        time.sleep(30)
                        continue
                except Exception as e:
                    log.warning("Balance fetch failed: %s", str(e)[:60])

                self.check_exits(market)

                for row in market:
                    coin  = str(row.get("symbol") or "").upper()
                    price = float(row.get("current_price") or 0)
                    vol   = float(row.get("total_volume")  or 0)

                    self.authority.push(coin, price, vol)

                    in_pos   = coin in self.positions
                    entry_px = float(self.positions[coin]["entry"]) if in_pos else 0.0

                    decision = self.authority.evaluate(
                        symbol       = coin,
                        row          = row,
                        gauge_engine = self.gauge_engine,
                        entry_price  = entry_px,
                    )

                    if decision.mode != "NORMAL":
                        self.db.log_thought("SYSTEM",
                            "🌀 " + coin + " " + decision.mode +
                            " stress=" + str(round(decision.stress_score, 2)) +
                            " K=" + str(round(decision.K, 2)) +
                            " snap=" + str(round(decision.snap_prob, 2)) +
                            " dev=" + str(round(decision.deviation_z, 2)) +
                            " sbi=" + str(round(decision.sbi, 2)) +
                            " | " + decision.reason[:120])

                    if in_pos and decision.dynamic_tp_pct is not None:
                        old = self._dynamic_tp.get(coin)
                        if old is None or decision.dynamic_tp_pct > old:
                            self._dynamic_tp[coin] = decision.dynamic_tp_pct
                            self.db.log_thought("SYSTEM",
                                "📐 " + coin + " PoI DynTP=" +
                                str(round(decision.dynamic_tp_pct * 100, 2)) +
                                "% target=$" + str(round(decision.dynamic_tp_price, 6)))

                    if decision.signal == "MANDATORY_BUY" and not in_pos:
                        self.db.log_thought("TRADE",
                            "MANDATORY_BUY " + coin + " | " + decision.reason[:180])
                        self.try_enter(row, net_score=9.9, reason=decision.reason, forced=True)
                        continue

                    if decision.signal == "VETO_SELL" and in_pos:
                        self.force_exit(coin, price, decision.reason)
                        continue

                    if in_pos:
                        continue

                    if detect_anomaly(row):
                        self.db.log_thought("SKIP", coin + ": anomaly guard")
                        continue

                    min_pot = float(self.cfg.get("hourly_potential_min", 0.0))
                    if min_pot > 0:
                        h  = float(row.get("high_24h")      or 0)
                        l  = float(row.get("low_24h")       or 0)
                        p  = float(row.get("current_price") or 1)
                        hr = ((h - l) / p * 100 / 24) if p > 0 else 0
                        if hr < min_pot:
                            self.db.log_thought("SKIP",
                                coin + ": hourly_range=" + str(round(hr, 3)) +
                                "% < " + str(min_pot) + "%")
                            continue

                    net_score, summary, hard_vetoed = self.run_voters(
                        row, regime, market_df,
                        weight_overrides=decision.weight_overrides,
                    )

                    mode_tag = " [" + decision.mode + "]" if decision.mode != "NORMAL" else ""
                    self.db.log_thought("VOTE",
                        coin + mode_tag + ": " + summary +
                        " stress=" + str(round(decision.stress_score, 2)))

                    if hard_vetoed:
                        continue

                    if net_score >= threshold:
                        self.try_enter(row, net_score, summary)
                    else:
                        self.db.log_thought("SKIP",
                            coin + ": score=" + str(round(net_score, 2)) +
                            " < " + str(threshold))

                elapsed = time.time() - t0
                time.sleep(max(0, 30 - elapsed))

            except KeyboardInterrupt:
                log.info("Shutdown")
                break
            except Exception as e:
                log.error("Cycle error: %s", traceback.format_exc()[:400])
                self.db.log_thought("ERROR", "Cycle: " + str(e)[:200])
                time.sleep(10)


if __name__ == "__main__":
    bot = ApexBot()
    bot.run()
