from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import pandas as pd

from src.config import CHROMA_DIR, PROFILE_PATH, PRECOMPUTED_EMBEDDINGS_PATH
from src.rag.vector_store import RealPlayerVectorStore
from src.recsys.engine import TwoTowerRecommendationEngine
from src.ranks import MATCHMAKING_TIERS


@dataclass
class Runtime:
    profiles: pd.DataFrame
    vector_store: RealPlayerVectorStore
    recommender: TwoTowerRecommendationEngine


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    started = perf_counter()
    print("[startup] Loading compact real-player profiles", flush=True)
    if not PROFILE_PATH.exists():
        raise RuntimeError(
            "Real preprocessed profiles are missing. Upload deployment/player_profiles.jsonl from the Space bundle, "
            "or locally run `python -m src.data.preprocess_large` once. No candidates were generated."
        )
    profiles = pd.read_json(PROFILE_PATH, lines=True)
    if profiles.empty:
        raise RuntimeError("Preprocessed profile file is empty")
    store = RealPlayerVectorStore(CHROMA_DIR)
    if store.requires_rebuild(profiles):
        if PRECOMPUTED_EMBEDDINGS_PATH.exists():
            print("[startup] Importing precomputed vectors into Chroma (no profile encoding)", flush=True)
            store.rebuild(profiles, precomputed_path=PRECOMPUTED_EMBEDDINGS_PATH)
        elif (PROFILE_PATH.parent / "manifest.json").exists():
            raise RuntimeError("Deployment is missing rag_embeddings.npz; upload the complete fast-start bundle")
        else:
            print("[startup] Local index requires profile encoding; export vectors before packaging", flush=True)
            store.rebuild(profiles)
    print(f"[startup] Real-profile index ready in {perf_counter() - started:.2f}s; loading trained rankers", flush=True)
    runtime = Runtime(profiles, store, TwoTowerRecommendationEngine(profiles, allowed_tiers=MATCHMAKING_TIERS))
    print(f"[startup] Recommendation runtime loaded in {perf_counter() - started:.2f}s", flush=True)
    return runtime


def warm_runtime() -> Runtime:
    """Finish model initialization before the UI opens; never generate scout prose."""
    runtime = get_runtime()
    started = perf_counter()
    print("[startup] Warming MiniLM query encoder", flush=True)
    # Operational warmup text is not added to the index or used as player evidence.
    runtime.vector_store.search_teammates("vision control", top_k=1)
    print(f"[startup] Query encoder ready in {perf_counter() - started:.2f}s; warming lineup scorer", flush=True)
    runtime.recommender.recommend_team("FILL", tier="PLATINUM", division="IV", candidates_per_role=1)
    print(f"[startup] Warmup complete in {perf_counter() - started:.2f}s; ready for requests", flush=True)
    return runtime


def clear_runtime() -> None:
    get_runtime.cache_clear()
