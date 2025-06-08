# config.py
"""
Configuration for cursor controller. All parameters are loaded and saved from this file.
"""
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)

# Default values
DEFAULTS = {
    "MIN_DURATION": 0.0001,
    "BASE_DURATION": 0.001,
    "K": 10,
    "PAUSE": 0.001,
    "SMOOTHING_FACTOR": 0.1,
    "MIN_MOVEMENT_THRESHOLD": 5,
    "BASE_RATIO": 1.5,
    "ALPHA": 0.000000000001,
    "BETA": 0.000099
}

# Load config or use defaults
user_cfg = load_config()

class Config:
    """
    Configuration class for adjustable cursor movement parameters.
    """
    MIN_DURATION = user_cfg.get("MIN_DURATION", DEFAULTS["MIN_DURATION"])
    BASE_DURATION = user_cfg.get("BASE_DURATION", DEFAULTS["BASE_DURATION"])
    K = user_cfg.get("K", DEFAULTS["K"])
    PAUSE = user_cfg.get("PAUSE", DEFAULTS["PAUSE"])
    SMOOTHING_FACTOR = user_cfg.get("SMOOTHING_FACTOR", DEFAULTS["SMOOTHING_FACTOR"])
    MIN_MOVEMENT_THRESHOLD = user_cfg.get("MIN_MOVEMENT_THRESHOLD", DEFAULTS["MIN_MOVEMENT_THRESHOLD"])
    BASE_RATIO = user_cfg.get("BASE_RATIO", DEFAULTS["BASE_RATIO"])
    ALPHA = user_cfg.get("ALPHA", DEFAULTS["ALPHA"])
    BETA = user_cfg.get("BETA", DEFAULTS["BETA"])

    @staticmethod
    def update_from_dict(cfg):
        for k, v in cfg.items():
            if hasattr(Config, k):
                setattr(Config, k, v)
        save_config(cfg)
