from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from src.config import CHROMA_DIR, DUO_DB_PATH, PROFILE_PATH, XGB_MODEL_PATH
from src.rag.vector_store import RealPlayerVectorStore
from src.recsys.engine import RecommendationEngine


@dataclass
class Runtime:
    profiles: pd.DataFrame
    vector_store: RealPlayerVectorStore
    recommender: RecommendationEngine


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    if not PROFILE_PATH.exists():
        raise RuntimeError("Preprocessed profiles are missing; run `python -m src.data.preprocess_large` once")
    profiles = pd.read_json(PROFILE_PATH, lines=True)
    if profiles.empty:
        raise RuntimeError("Preprocessed profile file is empty")
    store = RealPlayerVectorStore(CHROMA_DIR)
    if store.collection.count() != len(profiles):
        store.rebuild(profiles)
    return Runtime(profiles, store, RecommendationEngine(None, profiles, DUO_DB_PATH, XGB_MODEL_PATH))


def resolve_player_id(profiles: pd.DataFrame, identity: str | None) -> str | None:
    if not identity:
        return None
    needle = identity.strip().casefold()
    for column in ("summoner_id", "puuid", "player_name"):
        match = profiles[profiles[column].fillna("").astype(str).str.casefold() == needle]
        if not match.empty:
            return str(match.iloc[0]["summoner_id"])
    return identity.strip()


def clear_runtime() -> None:
    get_runtime.cache_clear()
