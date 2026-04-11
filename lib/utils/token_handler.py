import time
from pathlib import Path
from typing import Optional
from config import KEY_DIR

from lib.utils.log import logger

log = logger.get_logger()


def get_token_from_cache(token_name:str, token_dir: Optional[Path] = None) -> Optional[str]:
    """
    Retrieve a cached token for the given application if it exists and is valid.

    Args:
        token_name (str): The name of the application/token.
        token_dir (Optional[Path]): Directory where tokens are stored. Defaults to KEY_DIR.

    Returns:
        Optional[str]: The cached token if valid, otherwise None.
    """
    token_file = get_latest_token_file_for_app(token_name, token_dir)

    if not token_file:
        log.warning(f"No cached token found for {token_name}")
        return None
    
    if not is_token_file_valid(token_file):
        log.info(f"Token for {token_name} has expired. Deleting token file.")
        token_file.unlink(missing_ok=True)
        return None
    
    log.info(f"Using cached token for {token_name} from {token_file}")
    return token_file.read_text(encoding="utf-8")


def get_latest_token_file_for_app(token_name:str, token_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Get the most recently modified token file for the given application.

    Args:
        token_name (str): The name of the application/token.
        token_dir (Optional[Path]): Directory where tokens are stored. Defaults to KEY_DIR.

    Returns:
        Optional[Path]: The path to the latest token file if it exists, otherwise None.
    """
    if token_dir is None:
        token_dir = KEY_DIR

    if not token_dir.is_dir():
        return None
    
    files = list(token_dir.glob(f"{token_name}_expires_at_*"))
    if not files:
        return None
    
    return max(files, key=lambda p: p.stat().st_mtime)


def save_token_to_file(token_name: str, ttl_seconds: int, token: str, token_dir: Optional[Path] = None) -> None:
    """
    Write a new token to a file with an expiration timestamp.

    Args:
        token_name (str): The name of the application/token.
        ttl_seconds (int): Time-to-live in seconds for the token.
        token (str): The token value to be saved.
        token_dir (Optional[Path]): Directory where tokens are stored. Defaults to KEY_DIR.

    Returns:
        None
    """
    if token_dir is None:
        token_dir = KEY_DIR

    token_dir.mkdir(parents=True, exist_ok=True)

    for old_file in token_dir.glob(f"{token_name}_expires_at_*"):
        old_file.unlink()
    
    expire_timestamp = int(time.time() + ttl_seconds)
    new_token_path = token_dir / f"{token_name}_expires_at_{expire_timestamp}"

    temp_path = new_token_path.with_suffix(".tmp")

    try:
        temp_path.write_text(token, encoding="utf-8")
        temp_path.replace(new_token_path)
        log.info(f"Created new token for {token_name} expiring at {expire_timestamp}")
    
    except Exception as e:
        log.error(f"Failed to write token for {token_name}: {e}")
        if temp_path.exists():
            temp_path.unlink()


def is_token_file_valid(token_file: Path) -> bool:
    """
    Check if the token file is still valid based on its expiration timestamp.

    Args:
        token_file (Path): The path to the token file.
    
    Returns:
        bool: True if the token is valid, False if expired or invalid.
    """
    try:
        expire_ts = int(token_file.name.split("expires_at_")[-1])
        return time.time() + 5 < expire_ts
    
    except (ValueError, IndexError):
        return False 
    

if __name__ == "__main__":  #pragma: no cover
    save_token_to_file("example_app", 60, "sample_token_value")
    token = get_token_from_cache("example_app")
    print(f"Retrieved token: {token}")