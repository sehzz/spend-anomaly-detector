import json
import platform
from enum import Enum
from typing import Optional, Union
from config import SETTINGS_PATH, SECRET_PATH


class Environment(str, Enum):
    """
    Different environment where the code runs.

    Possible values:
    - SERVER: MacMini server.
    - DEVELOPMENT: local machine of developers (laptop).
    """
    DEVELOPMENT = "development"
    SERVER = "server"
    UNKNOWN = "unknown"

def get_conf_for(element: str) -> Optional[Union[dict, list, str]]:
    """Get configuration for a specific element from Settings.json."""

    setting_content = {}
    secret_content = {}

    with open(SETTINGS_PATH, encoding="utf-8") as f:
        setting_content = json.load(f)

    with open(SECRET_PATH, encoding="utf-8") as f:
        secret_content = json.load(f)
    
    setting_val = setting_content.get(element)
    secret_val = secret_content.get(element)

    if isinstance(setting_val, dict) and isinstance(secret_val, dict):
        return {**setting_val, **secret_val}
    
    if setting_val is not None:
        return setting_val
    if secret_val is not None:
        return secret_val
    
    return None

def get_environment() -> Environment:
    """Determine the current environment based on the node name."""

    conf = get_conf_for("nodes")
    node_name = platform.uname().node

    if node_name in conf.get("server_names", []):
        return Environment.SERVER
    
    if node_name in conf.get("dev_names", []):
        return Environment.DEVELOPMENT
    
    return Environment.UNKNOWN

def is_server() -> bool:
    """Return True if we are running on the server environment."""

    return get_environment() == Environment.SERVER

if __name__ == "__main__":
    get_environment()