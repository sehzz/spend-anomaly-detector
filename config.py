from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
LOG_FILE_PATH = BASE_DIR / "app.log"
SETTINGS_PATH = BASE_DIR / "conf" / "Settings.json"
SECRET_PATH = BASE_DIR / "conf" / "Secrets.json"
RESOURSES_DIR = BASE_DIR / "resources"
SERVICE_SPOOL_PATH = Path("/var/spool/monitoring")
KEY_DIR = BASE_DIR / "conf" / "key"