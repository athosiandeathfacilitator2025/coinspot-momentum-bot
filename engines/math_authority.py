"""
engines/math_authority.py — Hierarchical Math Engine Authority
═══════════════════════════════════════════════════════════════
This sits ABOVE the voter system. Every cycle it:

  1. Computes a composite stress score (0.0–1.0) from all math signals
  2. Decides the operating mode: NORMAL / ELEVATED / TAKEOVER
  3. Returns adjusted voter weights (suppressing standard voters in takeover)
  4. Issues MANDATORY_BUY or VETO_SELL signals that bot.py acts on directly
  5. Computes dynamic TP via the Point-at-Infinity projector

Operating modes:
  NORMAL    stress < 0.50   Standard weights. Math engines = 2 voters.
  ELEVATED  0.50 - 0.70     Math weights scale up. Others scale down.
  TAKEOVER  stress >= 0.70  Math weights x8. Standard voters x0.1.
                             MANDATORY or VETO signal issued.
"""

import numpy as np
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from lib.math_utils import (
    calculate_gauge_curvature,
    evaluate_elliptic_point,
    fit_elliptic_curve,
    gaussian_curvature_K,
    geodesic_deviation,
    point_at_infinity,
    symmetry_break_index,
    composite_stress_score,
)

log = logging.getLogger("MathAuthority")

STRESS_ELEVATED      = 0.50
STRESS_TAKEOVER      = 0.70
TAKEOVER_MATH_MULT   = 8.0
TAKEOVER_VOTER_SCALE = 0.1

STANDARD_VOTERS = {"rule", "ta", "ml", "market", "rl", "history"}
MATH_VOTERS     = {"gauge", "geometric"}


@dataclass
class AuthorityDecision:
    symbol:           str
    stress_score:     float
    mode:             str
    signal:           Optional[str]
    weight_overrides: dict
    dynamic_tp_pct:   Optional[float]
    dynamic_tp_price: Optional[float]
    K:                float
    snap_prob:        float
    deviation_z:      float
    sbi:              float
    gauge_curv:       float
    is_structured:    bool
    reason:           str


class MathAuthority:

    def __init__(self, base_weights: dict, takeover_threshold: float = STRESS_TAKEOVER):
        self.base_weights       = base_weights.copy()
        self.takeover_threshold = takeover_threshold
        self._price_hist  = defaultdict(lambda: deque(maxlen=30))
        self._volume_hist = defaultdict(lambda: deque(maxlen=30))
        self._last_decision: dict = {}

    def push(self, symbol: str, price: float, volume: float):
        if price > 0 and volume > 0:
            self._price_hist[symbol].append(price)
            self._volume_hist[symbol].append(volume)

    def evaluate(self, symbol, row, gauge_engine, entry_price=0.0) -> AuthorityDecision:
        prices  = np.array(list(self._price_hist[symbol]),  dtype=float)
        volumes = np.array(list(self._volume_hist[symbol]), dtype=float)

        local_price = float(row.get("current_price") or 0)
        volume_now  = float(row.get("total_volume")  or 0)

        gauge_engine.update()
        global_price = gauge_engine.get_global_price(symbol)
        gauge_curv   = calculate_gauge_curvature(local_price, global_price) if global_price > 0 else 0.0

        snap_prob     = 0.0
        is_structured = False

        if len(prices) >= 5:
            pm, ps = float(np.mean(prices)),  max(float(np.std(prices)),  1e-9)
            vm, vs = float(np.mean(volumes)), max(float(np.std(volumes)), 1e-9)
            norm_x = (local_price - pm) / ps
            norm_y = (volume_now  - vm) / vs
            ec            = evaluate_elliptic_point(norm_x, norm_y)
            snap_prob     = ec["snap_prob"]
            ec_fit        = fit_elliptic_curve(prices, volumes)
            is_structured = ec_fit["is_structured"]

        K       = gaussian_curvature_K(prices, volumes) if len(prices) >= 5 else 0.0
        geo     = geodesic_deviation(prices, volumes)   if len(prices) >= 5 else {}
        dev_z   = geo.get("deviation_z", 0.0)
        dev_dir = geo.get("direction",   "inline")
        sym     = symmetry_break_index(prices, volumes) if len(prices) >= 5 else {}
        sbi     = sym.get("sbi", 0.0)

        stress = composite_stress_score(
            K=K, snap_prob=snap_prob, deviation_z=dev_z,
            sbi=sbi, gauge_abs=abs(gauge_curv),
        )

        if stress >= self.takeover_threshold:
            mode = "TAKEOVER"
        elif stress >= STRESS_ELEVATED:
            mode = "ELEVATED"
        else:
            mode = "NORMAL"

        weight_overrides = self._compute_weights(stress, mode)

        signal       = None
        dtp_pct      = None
        dtp_price    = None
        reason_parts = [f"stress={stress:.2f} mode={mode}"]

        if mode == "TAKEOVER":
            signal, reason_parts = self._takeover_signal(
                stress, snap_prob, K, dev_z, dev_dir, sbi,
                gauge_curv, is_structured, prices, entry_price
            )
        elif mode == "ELEVATED":
            reason_parts.append(
                "ELEVATED: snap=" + str(round(snap_prob, 2)) +
                " K=" + str(round(K, 2)) +
                " dev=" + str(round(dev_z, 2)) +
                " sbi=" + str(round(sbi, 2)) +
                " weight shift active"
            )

        if entry_price > 0 and is_structured and len(prices) >= 8:
            pai = point_at_infinity(prices, entry_price)
            if pai["has_valid_signal"]:
                dtp_pct   = pai["dynamic_tp_pct"]
                dtp_price = pai["dynamic_tp_price"]
                reason_parts.append(
                    "PoI TP=" + str(round(dtp_pct * 100, 2)) +
                    "% peak=$" + str(round(pai["projected_peak"], 6)) +
                    " conf=" + str(round(pai["confidence"], 2))
                )

        reason   = " | ".join(reason_parts)
        decision = AuthorityDecision(
            symbol           = symbol,
            stress_score     = stress,
            mode             = mode,
            signal           = signal,
            weight_overrides = weight_overrides,
            dynamic_tp_pct   = dtp_pct,
            dynamic_tp_price = dtp_price,
            K                = K,
            snap_prob        = snap_prob,
            deviation_z      = dev_z,
            sbi              = sbi,
            gauge_curv       = gauge_curv,
            is_structured    = is_structured,
            reason           = reason,
        )
        self._last_decision[symbol] = decision
        if mode != "NORMAL":
            log.info("[%s] %s stress=%.2f signal=%s", symbol, mode, stress, signal)
        return decision

    def _compute_weights(self, stress: float, mode: str) -> dict:
        weights = self.base_weights.copy()
        if mode == "NORMAL":
            return weights
        if mode == "ELEVATED":
            t          = (stress - STRESS_ELEVATED) / (STRESS_TAKEOVER - STRESS_ELEVATED)
            math_mult  = 1.0 + t * (TAKEOVER_MATH_MULT - 1.0)
            voter_scale = 1.0 - t * (1.0 - TAKEOVER_VOTER_SCALE)
            for v in MATH_VOTERS:
                if v in weights:
                    weights[v] = weights[v] * math_mult
            for v in STANDARD_VOTERS:
                if v in weights:
                    weights[v] = weights[v] * voter_scale
            return weights
        for v in MATH_VOTERS:
            if v in weights:
                weights[v] = weights[v] * TAKEOVER_MATH_MULT
        for v in STANDARD_VOTERS:
            if v in weights:
                weights[v] = weights[v] * TAKEOVER_VOTER_SCALE
        return weights

    def _takeover_signal(self, stress, snap_prob, K, dev_z, dev_dir, sbi,
                         gauge_curv, is_structured, prices, entry_price) -> tuple:
        parts       = ["TAKEOVER stress=" + str(round(stress, 2))]
        in_position = entry_price > 0

        if in_position and sbi > 0.6:
            parts.append("VETO_SELL: symmetry break sbi=" + str(round(sbi, 2)) + " fuel running out")
            return "VETO_SELL", parts

        if in_position and dev_dir == "above" and snap_prob > 0.70:
            parts.append(
                "VETO_SELL: geodesic overshoot snap=" + str(round(snap_prob, 2)) +
                " dev=" + str(round(dev_z, 2))
            )
            return "VETO_SELL", parts

        if in_position and K > 2.0 and dev_dir == "above":
            parts.append("VETO_SELL: K spike K=" + str(round(K, 2)) + " at geodesic overshoot")
            return "VETO_SELL", parts

        if snap_prob > 0.75 and gauge_curv < -0.003 and dev_dir == "below" and is_structured:
            parts.append(
                "MANDATORY_BUY: elliptic snap UP + gauge underpriced snap=" +
                str(round(snap_prob, 2)) + " gauge=" + str(round(gauge_curv, 4))
            )
            return "MANDATORY_BUY", parts

        if snap_prob > 0.80 and dev_dir == "below" and not in_position:
            parts.append(
                "MANDATORY_BUY: extreme snap below geodesic snap=" + str(round(snap_prob, 2))
            )
            return "MANDATORY_BUY", parts

        parts.append(
            "WEIGHT SHIFT ONLY: ambiguous snap=" + str(round(snap_prob, 2)) +
            " gauge=" + str(round(gauge_curv, 4)) +
            " dev_dir=" + dev_dir +
            " sbi=" + str(round(sbi, 2))
        )
        return None, parts

    def get_last_decision(self, symbol: str) -> Optional[AuthorityDecision]:
        return self._last_decision.get(symbol)

    def reset_symbol(self, symbol: str):
        self._price_hist[symbol].clear()
        self._volume_hist[symbol].clear()
        if symbol in self._last_decision:
            del self._last_decision[symbol]
        log.info("[%s] Math authority reset", symbol)
