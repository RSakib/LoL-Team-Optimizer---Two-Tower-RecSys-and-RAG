"""Portable real-profile vectors; independent of Chroma's on-disk format."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_embeddings(path: Path, profiles: pd.DataFrame, model: str, pipeline: str, fingerprint: str) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"Precomputed real-profile embeddings are missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if (
            str(archive["embedding_model"].item()) != model
            or str(archive["pipeline_version"].item()) != pipeline
            or str(archive["profile_fingerprint"].item()) != fingerprint
        ):
            raise RuntimeError("Precomputed embeddings do not match the real profiles or embedding configuration; export them again")
        ids = archive["ids"].astype(str).tolist()
        vectors = np.asarray(archive["embeddings"], dtype=np.float32)
    expected_ids = profiles["summoner_id"].astype(str).tolist()
    if not ids or len(ids) != len(set(ids)) or len(expected_ids) != len(set(expected_ids)) or set(ids) != set(expected_ids):
        raise RuntimeError("Precomputed embeddings have missing, duplicate, or unexpected player IDs")
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise RuntimeError("Precomputed embeddings have invalid dimensions or non-finite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise RuntimeError("Precomputed embeddings must be normalized real-profile vectors")
    positions = {player_id: index for index, player_id in enumerate(ids)}
    return vectors[[positions[player_id] for player_id in expected_ids]]


def export_existing_index(store: Any, profiles: pd.DataFrame, path: Path) -> int:
    """Copy only verified existing embeddings; never load a model or encode profiles."""
    if profiles.empty or store.requires_rebuild(profiles):
        raise RuntimeError("The real-profile index is missing or stale; run `python -m src.data.reindex_profiles` first")
    batches = []
    ids = profiles["summoner_id"].astype(str).tolist()
    documents = profiles["rag_document"].astype(str).tolist()
    for start in range(0, len(ids), 128):
        batch_ids = ids[start:start + 128]
        result = store.collection.get(ids=batch_ids, include=["embeddings", "documents"])
        positions = {str(player_id): index for index, player_id in enumerate(result["ids"])}
        if set(positions) != set(batch_ids) or result.get("embeddings") is None:
            raise RuntimeError("Existing Chroma index is missing real-profile vectors")
        for offset, player_id in enumerate(batch_ids):
            if result["documents"][positions[player_id]] != documents[start + offset]:
                raise RuntimeError("Indexed documents differ from the real profiles; rebuild before exporting")
        batches.append(np.asarray([result["embeddings"][positions[player_id]] for player_id in batch_ids], dtype=np.float32))
    vectors = np.concatenate(batches)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, ids=np.asarray(ids), embeddings=vectors,
        embedding_model=store.embedding_model_name, pipeline_version=store.PIPELINE_VERSION,
        profile_fingerprint=store._fingerprint(profiles),
    )
    load_embeddings(path, profiles, store.embedding_model_name, store.PIPELINE_VERSION, store._fingerprint(profiles))
    return len(ids)
