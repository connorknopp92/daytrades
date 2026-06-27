"""Load and validate config.yaml into a plain dict with sane defaults."""

from __future__ import annotations

import os
from copy import deepcopy

import yaml

# Defaults mirror config.yaml so the tool works even if the file is missing
# or partially specified.
DEFAULTS = {
    "data": {
        "exchange": "coinbaseexchange",
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD",
                    "ADA/USD", "LTC/USD", "AVAX/USD", "LINK/USD"],
        "stocks": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
                   "SPY", "QQQ"],
        "timeframe": "1d",
        "years": 5,
        "cache_dir": "data/cache",
    },
    "backtest": {
        "initial_capital": 10000.0,
        "leverage": 1.0,
        "fee": 0.001,
        "slippage": 0.0005,
        "funding_rate": 0.0001,
        "maintenance_margin": 0.005,
        "periods_per_year": 365,
    },
    "analysis": {
        "rolling_window": 30,
    },
    "output_dir": "outputs",
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None = None) -> dict:
    """Load configuration, merging the YAML file on top of defaults."""
    path = path or DEFAULT_CONFIG_PATH
    loaded = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    cfg = _deep_merge(DEFAULTS, loaded)
    cfg["_project_root"] = _PROJECT_ROOT
    return cfg


def project_path(cfg: dict, *parts: str) -> str:
    """Resolve a path relative to the project root."""
    return os.path.join(cfg.get("_project_root", _PROJECT_ROOT), *parts)
