import os
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@lru_cache
def load_config() -> dict:
    env = os.getenv("APP_ENV", "dev")
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config.yaml at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        all_configs = yaml.safe_load(f) or {}
    if env not in all_configs:
        raise RuntimeError(f"Environment '{env}' not found in config.yaml")
    cfg = all_configs[env] or {}
    cfg["env"] = env
    return cfg
