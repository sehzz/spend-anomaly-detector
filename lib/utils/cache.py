import json
from typing import Optional

from datetime import datetime, timedelta, timezone
from pathlib import Path
from pydantic import BaseModel
from config import RESOURSES_DIR
from lib.utils.log import logger

log = logger.get_logger()
CACHE_TIME_THRESHOLD_HOURS = 24

class JSONFileCache(BaseModel):
    """Class to handle caching of JSON data to files with timestamp validation."""

    name: str
    file_path: Optional[Path] = None

    @property
    def path(self) -> Path:
        """Return the full path to the cache file based on whether it's a key or not."""

        base_dir =  self.file_path or RESOURSES_DIR
        return base_dir.joinpath(self.name)

    @property
    def is_valid(self) -> bool:
        """Check if the cached data is still valid based on timestamp."""
        
        cache_ts = self.get_cache_timestamp()
        
        file_dt_utc = datetime.fromisoformat(cache_ts).replace(tzinfo=timezone.utc)

        threshold = timedelta(hours=CACHE_TIME_THRESHOLD_HOURS)

        if datetime.now(timezone.utc) - file_dt_utc < threshold:
            log.info(f"Cache '{self.name}' is valid. Reading from {self.path}")
            return True

        log.warning(f"Cache '{self.name}' is expired.")
        return False
    
    def save(self, data, save_raw: bool = False) -> None:
        """
        Save data to JSON cache file.

        Args:
            data (dict): Data to be cached.
            save_raw (bool): If True, saves data as is without wrapping in timestamp.
        
        Returns:
            None
        """
        path = self.path
        
        if save_raw:
            cache_data = data

        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4, ensure_ascii=False)
            log.info(f"File saved in {path}")

    def retreive(self):
        """
        Retrieve data from JSON cache file.

        Returns:
            dict: {"timestamp": str, "data": dict} if successfully loaded,
                    else {"timestamp": None, "data": None}.
        """
        content = {}
        path = self.path
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)

        except(OSError, IOError) as e:
            log.error("Failed to read cache file '%s': '%s'", path, e)

        except json.JSONDecodeError:
            log.error("Cache file '%s' is not JSON serialized", self.name)
        
        return {
                "timestamp": content.get("timestamp"),
                "data": content.get("data")
            }

    def get_cache_timestamp(self):
        """Get the timestamp of when the cache was last saved."""

        return self.retreive().get("timestamp")
