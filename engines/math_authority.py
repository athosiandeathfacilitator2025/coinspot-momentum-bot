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
  ELEVATED  0.50 – 0.70     Math weights scale up. Others scale down.
  TAKEOVER  stress >= 0.70  Math weights × 8. Standard voters × 0.1.
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
            ec         = evaluate_elliptic_point(norm_x, norm_y)
            snap_prob  = ec["snap_prob"]
            ec_fit     = fit_elliptic_curve(prices, volumes)
            is_structured = ec_fit["is_structured"]

        K       = gaussian_curvature_K(prices, volumes) if len(prices) >= 5 else 0.0
        geo     = geodesic_deviation(prices, volumes)   if len(prices) >= 5 else {}
        dev_z   = geo.get("deviation_z", 0.0)
        dev_dir = geo.get("direction",   "inline")
        sym     = symmetry_break_index(prices, volumes) if len(prices) >= 5 else {}
        sbi     = sym.get("sbi", 0.0)

        stress  = composite_stress_score(
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

        signal    = None
        dtp_pct   = None
        dtp_price = None
        reason_parts = [f"stress={stress:.2f} mode={mode}"]

        if mode == "TAKEOVER":
            signal, reason_parts = self._takeover_signal(
                stress, snap_prob, K, dev_z, dev_dir, sbi,
                gauge_curv, is_structured, prices, entry_price
            )
        elif mode == "ELEVATED":
            reason_parts.append(
                f"ELEVATED: snap={snap_prob:.2f} K={K:.2f} "
                f"dev={dev_z:.2f} sbi={sbi:.2f} — weight shift active"
            )
