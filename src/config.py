from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = Path(os.getenv("LOL_DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.getenv("LOL_CHROMA_DIR", PROJECT_ROOT / ".chroma"))
PROCESSED_DIR = Path(os.getenv("LOL_PROCESSED_DIR", DATA_DIR / "processed"))
_BUNDLED_PROFILES = PROJECT_ROOT / "deployment" / "player_profiles.jsonl"
PROFILE_PATH = Path(os.getenv(
    "LOL_PROFILE_PATH", _BUNDLED_PROFILES if _BUNDLED_PROFILES.exists() else PROCESSED_DIR / "player_profiles.jsonl"
))
PRECOMPUTED_EMBEDDINGS_PATH = Path(os.getenv("LOL_RAG_EMBEDDINGS_PATH", PROFILE_PATH.parent / "rag_embeddings.npz"))
MODEL_DIR = Path(os.getenv("LOL_MODEL_DIR", PROJECT_ROOT / "artifacts" / "two_tower"))
TWO_TOWER_MODEL_PATH = Path(os.getenv("LOL_TWO_TOWER_MODEL", MODEL_DIR / "model.pt"))
TWO_TOWER_METADATA_PATH = Path(os.getenv("LOL_TWO_TOWER_METADATA", MODEL_DIR / "metadata.json"))
TEAM_MODEL_DIR = Path(os.getenv("LOL_TEAM_MODEL_DIR", PROJECT_ROOT / "artifacts" / "team_model"))
TEAM_MODEL_PATH = Path(os.getenv("LOL_TEAM_MODEL", TEAM_MODEL_DIR / "model.pt"))
TEAM_MODEL_METADATA_PATH = Path(os.getenv("LOL_TEAM_MODEL_METADATA", TEAM_MODEL_DIR / "metadata.json"))
TWO_TOWER_DEVICE = os.getenv("LOL_TWO_TOWER_DEVICE", "auto")
RAG_RRF_WEIGHT = float(os.getenv("LOL_RAG_RRF_WEIGHT", "0.35"))
API_URL = os.getenv("LOL_API_URL", "http://127.0.0.1:8001")
UI_BACKEND = os.getenv("LOL_UI_BACKEND", "local")
SCOUT_MODEL = os.getenv("SCOUT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SCOUT_DEVICE = os.getenv("SCOUT_DEVICE", "zero_gpu" if os.getenv("SPACES_ZERO_GPU") == "1" else "cpu")
SCOUT_MAX_INPUT_TOKENS = int(os.getenv("SCOUT_MAX_INPUT_TOKENS", "8192"))
SCOUT_MAX_NEW_TOKENS = int(os.getenv("SCOUT_MAX_NEW_TOKENS", "192"))
RAG_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RAG_EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "auto")
RAG_EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))
