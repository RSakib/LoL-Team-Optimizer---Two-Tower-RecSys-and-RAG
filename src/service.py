from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from src.config import CHROMA_DIR, PROFILE_PATH
from src.rag.vector_store import RealPlayerVectorStore
from src.recsys.engine import TwoTowerRecommendationEngine


@dataclass
class Runtime:
    profiles: pd.DataFrame
    vector_store: RealPlayerVectorStore
    recommender: TwoTowerRecommendationEngine


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    if not PROFILE_PATH.exists():
        raise RuntimeError("Preprocessed profiles are missing; run `python -m src.data.preprocess_large` once")
    profiles = pd.read_json(PROFILE_PATH, lines=True)
    if profiles.empty:
        raise RuntimeError("Preprocessed profile file is empty")
    store = RealPlayerVectorStore(CHROMA_DIR)
    if store.requires_rebuild(profiles):
        store.rebuild(profiles)
    return Runtime(profiles, store, TwoTowerRecommendationEngine(profiles))


def clear_runtime() -> None:
    get_runtime.cache_clear()
