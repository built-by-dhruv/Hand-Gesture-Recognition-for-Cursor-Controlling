"""
Cursor controller configuration. All parameters are loaded and saved from this file.
"""
import json
import os
import threading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def load_config():
    """Load configuration from config.json if it exists, otherwise return an empty dict."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_config(cfg):
    """Merge and save configuration dictionary to config.json."""
    existing = load_config()
    existing.update(cfg)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(existing, f, indent=2)


# Default values (sensitivity-based model)
DEFAULTS = {
    "PAUSE": 0.001,
    "SMOOTHING_FACTOR": 0.3,
    "MIN_MOVEMENT_THRESHOLD": 10,
    "SENSITIVITY": 0.6,
    "TARGET_FPS": 30,
}


# Load config or use defaults
user_cfg = load_config()


class Config:
    """
    Configuration class for adjustable cursor movement parameters.
    All values are loaded from config.json and can be updated at runtime.
    """
    PAUSE = user_cfg.get("PAUSE", DEFAULTS["PAUSE"])
    SMOOTHING_FACTOR = user_cfg.get("SMOOTHING_FACTOR", DEFAULTS["SMOOTHING_FACTOR"])
    MIN_MOVEMENT_THRESHOLD = user_cfg.get("MIN_MOVEMENT_THRESHOLD", DEFAULTS["MIN_MOVEMENT_THRESHOLD"])
    SENSITIVITY = user_cfg.get("SENSITIVITY", DEFAULTS["SENSITIVITY"])
    TARGET_FPS = user_cfg.get("TARGET_FPS", DEFAULTS["TARGET_FPS"])

    # Shared state for controller
    running = False
    lock = threading.Lock()

    @staticmethod
    def update_from_dict(cfg):
        """Update configuration values from a dictionary and save to config.json."""
        with Config.lock:
            for k, v in cfg.items():
                if hasattr(Config, k):
                    setattr(Config, k, v)
        save_config(cfg)
