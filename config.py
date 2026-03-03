"""
config.py — Single config loader. Reads config.yaml and merges env vars.
This replaces the old hardcoded CONFIG dict.
"""
import os
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Inject secrets from environment (never stored in yaml)
    cfg["coinspot_api_key"]    = os.getenv("COINSPOT_API_KEY", "").strip()
    cfg["coinspot_api_secret"] = os.getenv("COINSPOT_API_SECRET", "").strip()
    cfg["database_url"]        = os.getenv("DATABASE_URL", "sqlite:///apex_learning.db").strip()

    # Normalise postgres:// → postgresql:// for SQLAlchemy
    if cfg["database_url"].startswith("postgres://"):
        cfg["database_url"] = cfg["database_url"].replace("postgres://", "postgresql://", 1)

    return cfg

CONFIG = load_config()
