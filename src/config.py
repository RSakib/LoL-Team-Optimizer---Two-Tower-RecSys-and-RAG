from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("LOL_DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.getenv("LOL_CHROMA_DIR", PROJECT_ROOT / ".chroma"))
PROCESSED_DIR = Path(os.getenv("LOL_PROCESSED_DIR", DATA_DIR / "processed"))
PROFILE_PATH = Path(os.getenv("LOL_PROFILE_PATH", PROCESSED_DIR / "player_profiles.jsonl"))
API_URL = os.getenv("LOL_API_URL", "http://127.0.0.1:8001")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
