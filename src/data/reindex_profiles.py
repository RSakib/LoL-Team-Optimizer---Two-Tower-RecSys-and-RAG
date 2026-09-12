from __future__ import annotations

import json

import pandas as pd

from src.config import CHROMA_DIR, PROFILE_PATH
from src.data.ingestion import write_profiles_jsonl
from src.data.profile_documents import enrich_profiles_for_rag
from src.rag.vector_store import RealPlayerVectorStore


def main() -> None:
    if not PROFILE_PATH.exists():
        raise RuntimeError("Preprocessed profiles are missing; run the large-file preprocessor first")
    profiles = pd.read_json(PROFILE_PATH, lines=True)
    if profiles.empty:
        raise RuntimeError("Preprocessed profile file is empty")
    enriched = enrich_profiles_for_rag(profiles)
    write_profiles_jsonl(enriched, PROFILE_PATH)
    store = RealPlayerVectorStore(CHROMA_DIR)
    indexed = store.rebuild(enriched)
    print(json.dumps({
        "profiles_enriched": len(enriched), "profiles_indexed": indexed,
        "embedding_model": store.embedding_model_name, "device": store.device,
        "collection": store.collection.name,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
