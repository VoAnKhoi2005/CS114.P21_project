from dotenv import load_dotenv
import os
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if not _env_path.exists():
    raise FileNotFoundError(f".env file not found at: {_env_path}")
load_dotenv(dotenv_path=_env_path)

def get_env(key: str, default=None):
    return os.getenv(key, default)

def get_env_bool(key: str, default=False):
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")

def get_env_int(key: str, default=None):
    val = os.getenv(key)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default
