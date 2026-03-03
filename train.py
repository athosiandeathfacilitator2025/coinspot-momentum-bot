"""
train.py  —  Apex RL Agent Trainer
────────────────────────────────────
Run this in GitHub Codespaces (or locally) to train a PPO reinforcement
learning agent on historical Bitcoin price data.

This file does NOT run on Render. It is a one-time training script.
The output (apex_ppo_btc.zip) gets committed to GitHub and Render
loads it as the 7th voter — no PyTorch needed in production.

Usage:
    python train.py

Output:
    apex_ppo_btc.zip   — trained agent weights (~50-200KB)
    ./logs/            — TensorBoard logs (optional viewing)

After training:
    git add apex_ppo_btc.zip
    git commit -m "add trained RL agent"
    git push
"""

import sys

# ── Dependency check ──────────────────────────────────────────────────────────
missing = []
try:
    import gymnasium
except ImportError:
    missing.append("gymnasium")
try:
    from stable_baselines3 import PPO
except ImportError:
    missing.append("stable-baselines3")
try:
    import torch
except ImportError:
    missing.append("torch")

if missing:
    print("❌  Missing libraries:", ", ".join(missing))
    print()
    print("    Run these two commands first:")
    print("    pip install gymnasium stable-baselines3")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import ApexTradingEnv, fetch_coingecko_history

print("=" * 55)
print("  APEX RL TRAINER")
print("=" * 55)
print()

# ── Step 1: Fetch price data ──────────────────────────────────────────────────
print("Step 1/4  Fetching BTC price history from CoinGecko...")
print("          (fetching 365 days of daily candles)")
print()

try:
    df = fetch_coingecko_history("bitcoin", days=365)
    print(f"          ✅ Got {len(df)} candles")
    print(f"          Date range: {df.index[0].date()} → {df.index[-1].date()}")
except Exception as e:
    print(f"          ❌ CoinGecko fetch failed: {e}")
    print()
    print("          Using synthetic data instead for demo training...")
    np.random.seed(42)
    n       = 400
    closes  = 50000.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    volumes = np.random.uniform(1e8, 5e8, n)
    df      = pd.DataFrame({"close": closes, "volume": volumes})
    print(f"          ✅ Generated {len(df)} synthetic candles")

print()

# ── Step 2: Create environment ────────────────────────────────────────────────
print("Step 2/4  Creating trading environment...")

env = ApexTradingEnv(
    df,
    initial_balance  = 1000.0,
    trade_fraction   = 0.10,
    max_hold_steps   = 48,
    daily_target_pct = 3.0,
    max_drawdown_pct = 5.0,
)

print("          Checking environment validity...")
try:
    check_env(env, warn=True)
    print("          ✅ Environment valid")
except Exception as e:
    print(f"          ⚠️  Warning: {e}")
    print("          Continuing anyway...")

print(f"          Observation space: {env.observation_space.shape}")
print(f"          Action space: {env.action_space.n} actions (0=HOLD, 1=BUY, 2=SELL)")
print(f"          Steps available per episode: {env.n_steps - env.window}")
print()

# ── Step 3: Train ─────────────────────────────────────────────────────────────
TOTAL_STEPS = 200_000

print(f"Step 3/4  Training PPO agent for {TOTAL_STEPS:,} steps...")
print()
print("          What PPO is doing:")
print("          - Running thousands of simulated trading episodes")
print("          - After each episode, adjusting its neural network")
print("          - Learning which actions lead to positive rewards")
print("          - Getting better at BUY/HOLD/SELL decisions over time")
print()
print("          This takes approximately 3-5 minutes in Codespaces.")
print()

model = PPO(
    policy          = "MlpPolicy",
    env             = env,
    verbose         = 1,
    tensorboard_log = "./logs",
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    learning_rate   = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
)

model.learn(total_timesteps=TOTAL_STEPS)
print()

# ── Step 4: Save ──────────────────────────────────────────────────────────────
print("Step 4/4  Saving trained agent...")

SAVE_PATH = "apex_ppo_btc"
model.save(SAVE_PATH)

import os
zip_path = f"{SAVE_PATH}.zip"
size_kb  = os.path.getsize(zip_path) / 1024 if os.path.exists(zip_path) else 0

print(f"          ✅ Saved: {zip_path}  ({size_kb:.0f} KB)")
print()

# ── Quick evaluation ──────────────────────────────────────────────────────────
print("Quick evaluation on training data...")
print()

obs, info = env.reset()
done      = False
steps     = 0

while not done:
    action, _                            = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done   = terminated or truncated
    steps += 1

print(f"  Steps run:       {steps}")
print(f"  Final portfolio: ${info['portfolio']:,.2f}")
print(f"  Total return:    {info['total_return']*100:+.2f}%")
print(f"  Trades made:     {info['trade_count']}")
print(f"  Total reward:    {info['total_reward']:+.2f}")
print()

# ── Instructions ──────────────────────────────────────────────────────────────
print("=" * 55)
print("  TRAINING COMPLETE")
print("=" * 55)
print()
print("  Next steps — run these 3 commands in the terminal:")
print()
print("  git add apex_ppo_btc.zip")
print('  git commit -m "add trained RL agent"')
print("  git push")
print()
print("  Render will pick it up on next deploy.")
print()
