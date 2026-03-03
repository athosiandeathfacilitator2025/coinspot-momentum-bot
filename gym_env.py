"""
gym_env.py  —  Apex Trading Gymnasium Environment
───────────────────────────────────────────────────
A custom Gymnasium environment that turns Apex's trading logic into an RL
playground. The agent decides each step whether to BUY, HOLD, or SELL
based on what it can observe.

Actions: 0=HOLD  1=BUY  2=SELL
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        class spaces:
            class Discrete:
                def __init__(self, n): self.n = n
                def sample(self): return np.random.randint(self.n)
            class Box:
                def __init__(self, low, high, shape=None, dtype=None):
                    self.shape = shape or (len(low),) if hasattr(low, '__len__') else (1,)
        class gym:
            class Env: pass

# ── Constants ─────────────────────────────────────────────────────────────────
COINSPOT_FEE  = 0.001
FEE_ROUNDTRIP = COINSPOT_FEE * 2

ACTION_HOLD  = 0
ACTION_BUY   = 1
ACTION_SELL  = 2
ACTION_NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}

# ── Reward constants ──────────────────────────────────────────────────────────
REWARD_FEE_COST         = -0.20
REWARD_TRAIL_EXIT       = +0.50
REWARD_HARD_SL          = -0.30
REWARD_VETO_BONUS       = +0.10
REWARD_REGIME_FLIP_EXIT = +0.30
REWARD_HELD_PAST_PEAK   = -0.20
REWARD_HELD_DANGER      = -0.50
REWARD_DAILY_TARGET     = +1.00
REWARD_CIRCUIT_BREAKER  = -2.00
REWARD_TIME_PENALTY     = -0.01


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas   = np.diff(prices)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_regime(prices: np.ndarray, window: int = 20) -> float:
    if len(prices) < window:
        return 0.0
    returns  = np.diff(prices[-window:]) / prices[-window:-1] * 100
    mean_ret = float(np.mean(returns))
    if mean_ret > 1.0:
        return 1.0
    if mean_ret < -1.0:
        return -1.0
    return 0.0


class ApexTradingEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Custom Gymnasium environment for the Apex trading bot.

    Observation space (11 features):
        [0]  price_change_1       — % change from 1 period ago
        [1]  price_change_5       — % change from 5 periods ago
        [2]  price_change_20      — % change from 20 periods ago
        [3]  rsi                  — RSI normalised 0–1
        [4]  vol_ratio            — volume / rolling mean volume
        [5]  regime               — -1 bear / 0 range / +1 bull
        [6]  in_position          — 0 or 1
        [7]  position_pnl         — current unrealised net PnL
        [8]  highest_pnl          — peak PnL seen in this trade
        [9]  days_held            — steps held normalised 0–1
        [10] daily_pnl            — today's running PnL %

    Action space:
        0 = HOLD
        1 = BUY
        2 = SELL
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df:         pd.DataFrame,
        initial_balance:  float = 1000.0,
        trade_fraction:   float = 0.10,
        max_hold_steps:   int   = 48,
        daily_target_pct: float = 3.0,
        max_drawdown_pct: float = 5.0,
        window:           int   = 20,
        render_mode             = None,
    ):
        super().__init__()

        required = {"close", "volume"}
        missing  = required - set(price_df.columns)
        if missing:
            raise ValueError(f"price_df missing columns: {missing}")

        self.df      = price_df.reset_index(drop=True)
        self.closes  = self.df["close"].values.astype(float)
        self.volumes = self.df["volume"].values.astype(float)
        self.n_steps = len(self.df)
        self.window  = window

        self.initial_balance = initial_balance
        self.trade_fraction  = trade_fraction
        self.max_hold_steps  = max_hold_steps
        self.daily_target    = daily_target_pct / 100.0
        self.max_drawdown    = max_drawdown_pct / 100.0
        self.render_mode     = render_mode

        self.observation_space = spaces.Box(
            low   = np.full(11, -10.0, dtype=np.float32),
            high  = np.full(11, +10.0, dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self._reset_state()

    def _reset_state(self):
        self.current_step   = self.window
        self.balance        = self.initial_balance
        self.day_start_bal  = self.initial_balance
        self.in_position    = False
        self.entry_price    = 0.0
        self.position_qty   = 0.0
        self.highest_pnl    = 0.0
        self.days_held      = 0
        self.trade_count    = 0
        self.total_reward   = 0.0
        self.last_day       = 0
        self.daily_realised = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        obs  = self._get_observation()
        info = {"balance": self.balance, "step": self.current_step}
        return obs.astype(np.float32), info

    def step(self, action: int):
        if self.current_step >= self.n_steps - 1:
            obs = self._get_observation()
            return obs.astype(np.float32), 0.0, True, False, self._info()

        current_price = self.closes[self.current_step]
        reward        = 0.0
        terminated    = False
        truncated     = False
        exit_type     = None

        day_num = self.current_step // 24
        if day_num != self.last_day:
            self.day_start_bal  = self.balance
            self.daily_realised = 0.0
            self.last_day       = day_num

        # ── BUY ───────────────────────────────────────────────────────────────
        if action == ACTION_BUY and not self.in_position:
            capital  = self.balance * self.trade_fraction
            buy_cost = capital * (1 + COINSPOT_FEE)
            if buy_cost <= self.balance:
                self.balance      -= buy_cost
                self.position_qty  = capital / current_price
                self.entry_price   = current_price
                self.in_position   = True
                self.highest_pnl   = 0.0
                self.days_held     = 0
                self.trade_count  += 1
                reward            += REWARD_FEE_COST / 2

        # ── SELL ──────────────────────────────────────────────────────────────
        elif action == ACTION_SELL and self.in_position:
            reward, exit_type = self._close_position(current_price, reason="agent_sell")

        # ── HOLD in position ──────────────────────────────────────────────────
        elif action == ACTION_HOLD and self.in_position:
            gross_pnl        = (current_price - self.entry_price) / self.entry_price
            self.highest_pnl = max(self.highest_pnl, gross_pnl)
            self.days_held  += 1
            reward          += REWARD_TIME_PENALTY

            regime  = _compute_regime(self.closes[:self.current_step + 1])
            sl_dist = 0.008 if regime == 0 else (0.012 if regime == -1 else 0.006)
            if gross_pnl < -(sl_dist * 2):
                reward += REWARD_HELD_DANGER

            if self.days_held >= self.max_hold_steps:
                r, exit_type = self._close_position(current_price, reason="timeout")
                reward      += r

        # ── HOLD flat ─────────────────────────────────────────────────────────
        elif action == ACTION_HOLD and not self.in_position:
            reward += REWARD_TIME_PENALTY

        # ── Daily target ──────────────────────────────────────────────────────
        daily_pnl_pct = (self.balance - self.day_start_bal) / self.day_start_bal
        if daily_pnl_pct >= self.daily_target:
            reward += REWARD_DAILY_TARGET * 0.1

        # ── Circuit breaker ───────────────────────────────────────────────────
        total_pnl_pct = (self.balance - self.initial_balance) / self.initial_balance
        if total_pnl_pct <= -self.max_drawdown:
            reward    += REWARD_CIRCUIT_BREAKER
            terminated = True

        self.current_step += 1
        self.total_reward += reward
        obs = self._get_observation()

        if self.current_step >= self.n_steps - 1:
            terminated = True

        info              = self._info()
        info["action"]    = ACTION_NAMES.get(action, str(action))
        info["exit_type"] = exit_type
        info["reward"]    = reward

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _close_position(self, price: float, reason: str = "") -> tuple:
        if not self.in_position:
            return 0.0, None

        gross_pnl = (price - self.entry_price) / self.entry_price
        net_pnl   = gross_pnl - FEE_ROUNDTRIP

        proceeds         = self.position_qty * price * (1 - COINSPOT_FEE)
        self.balance    += proceeds
        self.in_position = False

        reward    = net_pnl * 100
        reward   += REWARD_FEE_COST / 2
        exit_type = "flat"

        if gross_pnl >= 0 and self.highest_pnl > 0 and gross_pnl >= self.highest_pnl * 0.85:
            reward    += REWARD_TRAIL_EXIT
            exit_type  = "trail_peak"
        elif gross_pnl < 0 and abs(gross_pnl) > 0.01:
            reward    += REWARD_HARD_SL
            exit_type  = "stop_loss"
        elif self.highest_pnl > 0 and gross_pnl < self.highest_pnl * 0.7:
            reward    += REWARD_HELD_PAST_PEAK
            exit_type  = "gave_back"

        self.daily_realised += net_pnl
        self.highest_pnl     = 0.0
        self.days_held       = 0
        self.entry_price     = 0.0
        self.position_qty    = 0.0

        return reward, exit_type

    def _get_observation(self) -> np.ndarray:
        idx     = self.current_step
        prices  = self.closes[max(0, idx - self.window):idx + 1]
        vols    = self.volumes[max(0, idx - self.window):idx + 1]
        price   = float(prices[-1]) if len(prices) > 0 else 1.0

        chg_1  = float((prices[-1] / prices[-2] - 1) * 100) if len(prices) >= 2 else 0.0
        chg_5  = float((prices[-1] / prices[-5] - 1) * 100) if len(prices) >= 5 else 0.0
        chg_20 = float((prices[-1] / prices[0]  - 1) * 100) if len(prices) >= 2 else 0.0
        rsi    = _compute_rsi(prices) / 100.0
        vol_mean  = float(np.mean(vols)) if len(vols) > 0 else 1.0
        vol_ratio = float(vols[-1] / max(vol_mean, 1e-8))
        regime    = _compute_regime(prices)
        in_pos    = 1.0 if self.in_position else 0.0

        if self.in_position and self.entry_price > 0:
            cur_pnl = (price - self.entry_price) / self.entry_price - FEE_ROUNDTRIP
        else:
            cur_pnl = 0.0

        peak_pnl  = self.highest_pnl
        days_norm = min(self.days_held / self.max_hold_steps, 1.0)
        daily_pnl = (self.balance - self.day_start_bal) / self.day_start_bal * 100

        obs = np.array([
            np.clip(chg_1  / 5.0,  -3.0, 3.0),
            np.clip(chg_5  / 10.0, -3.0, 3.0),
            np.clip(chg_20 / 20.0, -3.0, 3.0),
            float(rsi),
            np.clip(vol_ratio - 1.0, -3.0, 3.0),
            float(regime),
            float(in_pos),
            np.clip(cur_pnl  * 100, -3.0, 3.0),
            np.clip(peak_pnl * 100, -3.0, 3.0),
            float(days_norm),
            np.clip(daily_pnl / 3.0, -3.0, 3.0),
        ], dtype=np.float32)

        return obs

    def _info(self) -> dict:
        price      = float(self.closes[min(self.current_step, self.n_steps - 1)])
        port_value = self.balance
        if self.in_position and self.entry_price > 0:
            port_value += self.position_qty * price
        return {
            "step":         self.current_step,
            "balance":      self.balance,
            "portfolio":    port_value,
            "total_return": (port_value - self.initial_balance) / self.initial_balance,
            "in_position":  self.in_position,
            "trade_count":  self.trade_count,
            "total_reward": self.total_reward,
        }

    def render(self):
        if self.render_mode == "human":
            info  = self._info()
            price = float(self.closes[min(self.current_step, self.n_steps - 1)])
            print(
                f"Step {self.current_step:5d} | "
                f"Price ${price:,.2f} | "
                f"Balance ${info['balance']:,.2f} | "
                f"Portfolio ${info['portfolio']:,.2f} | "
                f"Return {info['total_return']*100:+.2f}% | "
                f"Trades {info['trade_count']:3d} | "
                f"In pos: {'YES' if info['in_position'] else 'no '}"
            )

    def close(self):
        pass


# ── Fetch historical data from CoinGecko ─────────────────────────────────────
def fetch_coingecko_history(coin_id: str = "bitcoin", days: int = 365) -> pd.DataFrame:
    try:
        from pycoingecko import CoinGeckoAPI
        cg   = CoinGeckoAPI()
        data = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency="aud", days=str(days))
        df   = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        market = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="aud", days=str(days))
        vol_df = pd.DataFrame(market["total_volumes"], columns=["ts", "volume"])
        vol_df["ts"] = pd.to_datetime(vol_df["ts"], unit="ms").dt.floor("D")
        df.index = df.index.floor("D")
        df = df.merge(vol_df.set_index("ts"), left_index=True, right_index=True, how="left")
        df["volume"] = df["volume"].fillna(df["close"] * 1000)
        return df.dropna(subset=["close"])
    except Exception as e:
        raise RuntimeError(f"CoinGecko fetch failed: {e}") from e


# ── Demo run ──────────────────────────────────────────────────────────────────
def demo_random(n_steps: int = 500):
    if not GYM_AVAILABLE:
        print("gymnasium not installed. Run: pip install gymnasium")
        return

    print("=" * 60)
    print("APEX TRADING GYM — DEMO (random agent, synthetic data)")
    print("=" * 60)
    print()

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_steps + 50)
    closes  = 50000.0 * np.cumprod(1 + returns)
    volumes = np.random.uniform(1e8, 5e8, n_steps + 50)
    df      = pd.DataFrame({"close": closes, "volume": volumes})

    env = ApexTradingEnv(df, initial_balance=1000.0, render_mode="human")
    obs, info = env.reset()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}  (0=HOLD, 1=BUY, 2=SELL)")
    print(f"Initial balance:   ${env.initial_balance:,.2f}")
    print()

    total_reward = 0.0
    step_count   = 0
    done         = False

    while not done:
        action                                   = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count   += 1
        done          = terminated or truncated
        if step_count % 50 == 0 or done:
            env.render()

    print()
    print("-" * 60)
    print(f"Episode complete after {step_count} steps")
    print(f"Final portfolio:  ${info['portfolio']:,.2f}")
    print(f"Total return:     {info['total_return']*100:+.2f}%")
    print(f"Total reward:     {total_reward:+.2f}")
    print(f"Trades executed:  {info['trade_count']}")
    env.close()


QUICKSTART = """
╔══════════════════════════════════════════════════════════════════╗
║           APEX GYMNASIUM ENVIRONMENT — QUICK START              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Install:                                                     ║
║     pip install gymnasium stable-baselines3 torch               ║
║                                                                  ║
║  2. Demo (random agent, synthetic data):                         ║
║     python gym_env.py                                            ║
║                                                                  ║
║  3. Train a real agent on historical BTC data:                   ║
║     from gym_env import ApexTradingEnv, fetch_coingecko_history  ║
║     from stable_baselines3 import PPO                            ║
║                                                                  ║
║     df    = fetch_coingecko_history("bitcoin", days=365)         ║
║     env   = ApexTradingEnv(df, initial_balance=1000.0)           ║
║     model = PPO("MlpPolicy", env, verbose=1)                     ║
║     model.learn(total_timesteps=500_000)                         ║
║     model.save("apex_ppo_btc")                                   ║
║                                                                  ║
║  Actions: 0=HOLD  1=BUY  2=SELL                                  ║
║  Reward:  net_pnl×100 + bonuses for good exits                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(QUICKSTART)
    demo_random()
