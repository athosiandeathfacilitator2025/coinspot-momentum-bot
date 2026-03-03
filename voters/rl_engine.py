# voters/rl_engine.py
import numpy as np

MODEL_PATH     = "apex_ppo_btc"
MAX_HOLD_STEPS = 48


def _encode_regime(regime: str) -> float:
    return {"bull": 1.0, "bear": -1.0, "range": 0.0, "chop": 0.0}.get(
        str(regime).lower(), 0.0)


def _simple_rsi(change_24h: float) -> float:
    return float(np.clip(0.5 + (change_24h / 100.0) * 3.0, 0.0, 1.0))


class RLEngine:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.ready = False
        try:
            from stable_baselines3 import PPO
            try:
                self.model = PPO.load(model_path)
                self.ready = True
                print(f"[RL] Model loaded from {model_path}.zip")
            except FileNotFoundError:
                print(f"[RL] Model not found — RL voter disabled")
            except Exception as e:
                print(f"[RL] Load failed: {e} — RL voter disabled")
        except ImportError:
            print("[RL] stable-baselines3 not installed — RL voter disabled")
        except MemoryError:
            print("[RL] OOM — RL voter disabled")
        except Exception as e:
            print(f"[RL] Unexpected: {e} — RL voter disabled")

    def build_observation(self, snapshot: dict, position_state: dict) -> np.ndarray:
        c24       = float(snapshot.get("price_change_percentage_24h") or 0)
        c7d       = float(snapshot.get("price_change_percentage_7d_in_currency") or 0)
        volume    = float(snapshot.get("total_volume")  or 0)
        mcap      = float(snapshot.get("market_cap")    or 1)
        regime    = str(snapshot.get("regime", "range"))
        vol_ratio = volume / max(mcap, 1)
        return np.array([
            np.clip(c24 / 5.0,  -3.0, 3.0),
            np.clip(c7d / 10.0, -3.0, 3.0),
            np.clip(c7d / 20.0, -3.0, 3.0),
            _simple_rsi(c24),
            np.clip(vol_ratio - 0.02, -3.0, 3.0),
            _encode_regime(regime),
            1.0 if position_state.get("in_position") else 0.0,
            np.clip(float(position_state.get("unrealized_pnl",  0.0)) * 100, -3.0, 3.0),
            np.clip(float(position_state.get("highest_pnl",     0.0)) * 100, -3.0, 3.0),
            min(float(position_state.get("trade_duration", 0.0)) / MAX_HOLD_STEPS, 1.0),
            np.clip(float(position_state.get("daily_pnl",   0.0)) / 3.0,  -3.0, 3.0),
        ], dtype=np.float32)

    def vote(self, snapshot: dict, position_state: dict) -> int:
        if not self.ready or self.model is None:
            return 0
        try:
            obs    = self.build_observation(snapshot, position_state)
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action.item() if hasattr(action, "item") else action)
        except Exception as e:
            print(f"[RL] predict failed: {e}")
            return 0
