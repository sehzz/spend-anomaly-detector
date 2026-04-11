import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from lib.utils.environment import get_environment, Environment
from config import LOG_FILE_PATH


class CustomFormatter(logging.Formatter):
    def format(self, record):
        """Custom log format"""

        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        source = getattr(record, "source", "console").rjust(20)

        log_line = f"   {record.levelname:<7} | {timestamp} | {source} | {record.getMessage()}"
        return log_line


class Logger:
    _logger = None

    @classmethod
    def get_logger(cls, file_path=LOG_FILE_PATH, max_bytes=5_000_000, backup_count=3):
        """Return logger that adapts based on environment"""
        if cls._logger:
            return cls._logger

        logger = logging.getLogger("app_logger")
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = CustomFormatter()
        ch.setFormatter(formatter)
        ch.addFilter(lambda record: setattr(record, "source", "console") or True)
        logger.addHandler(ch)

        env = get_environment()
        if env == Environment.SERVER:
            fh = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            fh.addFilter(lambda record: setattr(record, "source", "file") or True)
            logger.addHandler(fh)

        logger.propagate = False
        cls._logger = logger
        return cls._logger


logger = Logger()
