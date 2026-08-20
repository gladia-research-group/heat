import os
from pathlib import Path
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get project root directory from git base path."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")
