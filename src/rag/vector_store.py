from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CHROMA_DIR,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_EMBEDDING_DEVICE,
    RAG_EMBEDDING_MODEL,
)


class RealPlayerVectorStore:
    """Persistent semantic Chroma index containing only computed real-player profiles."""

    PIPELINE_VERSION = "sentence-transformer-hybrid-role-percentiles-v1"
    STRUCTURED_INTENTS = {
        "role_experience_percentile": ("experience", "veteran", "many matches", "long history"),
        "role_win_rate_percentile": ("win rate", "winning", "wins", "reliable"),
        "role_kda_percentile": ("kda", "survive", "survivability", "consistent"),
        "role_vision_percentile": ("vision", "ward", "warding", "map awareness"),
        "role_damage_percentile": ("damage", "dps", "carry", "high damage"),
        "role_assists_percentile": ("assist", "assists", "team-oriented", "team oriented", "setup"),
    }

    def __init__(self, path: str | Path = CHROMA_DIR, collection_name: str | None = None):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is not installed; run `pip install -r requirements.txt`") from exc
        self.embedding_model_name = RAG_EMBEDDING_MODEL
        self.batch_size = max(1, RAG_EMBEDDING_BATCH_SIZE)
        self.device = self._resolve_device(RAG_EMBEDDING_DEVICE)
        self._embedding_model: Any | None = None
        self._query_embedding_cache: dict[str, np.ndarray] = {}
        model_slug = re.sub(r"[^a-z0-9]+", "_", self.embedding_model_name.lower()).strip("_")[-36:]
        default_collection = f"lol_players_semantic_{model_slug}"
        self.client = chromadb.PersistentClient(path=str(Path(path)))
        self.collection = self.client.get_or_create_collection(
            name=collection_name or default_collection,
            metadata={
                "hnsw:space": "cosine", "source": "local-real-match-data",
                "embedding_model": self.embedding_model_name, "pipeline_version": self.PIPELINE_VERSION,
            },
        )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested.strip().lower() != "auto":
            return requested.strip().lower()
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def embedding_model(self) -> Any:
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is not installed; run `pip install -r requirements.txt`"
                ) from exc
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._embedding_model

    @staticmethod
    def _fingerprint(profiles: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        ordered = profiles[["summoner_id", "rag_document"]].astype(str).sort_values("summoner_id")
        for player_id, document in ordered.itertuples(index=False, name=None):
            digest.update(player_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def requires_rebuild(self, profiles: pd.DataFrame) -> bool:
        metadata = self.collection.metadata or {}
        return (
            self.collection.count() != len(profiles)
            or metadata.get("embedding_model") != self.embedding_model_name
            or metadata.get("pipeline_version") != self.PIPELINE_VERSION
            or metadata.get("profile_fingerprint") != self._fingerprint(profiles)
        )

    def _encode_documents(self, documents: list[str]) -> np.ndarray:
        model = self.embedding_model
        encoder = getattr(model, "encode_document", model.encode)
        return np.asarray(encoder(
            documents, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ), dtype=np.float32)

    def _encode_query(self, query: str) -> np.ndarray:
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached
        model = self.embedding_model
        encoder = getattr(model, "encode_query", model.encode)
        embedding = np.asarray(encoder(
            [query], batch_size=1, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ), dtype=np.float32)
        if len(self._query_embedding_cache) >= 128:
            self._query_embedding_cache.pop(next(iter(self._query_embedding_cache)))
        self._query_embedding_cache[query] = embedding
        return embedding

    def rebuild(self, profiles: pd.DataFrame, precomputed_path: Path | None = None) -> int:
        vectors = None
        if precomputed_path is not None:
            from src.rag.portable import load_embeddings
            # Validate before touching an existing index. Never re-encode on failure.
            vectors = load_embeddings(
                precomputed_path, profiles, self.embedding_model_name,
                self.PIPELINE_VERSION, self._fingerprint(profiles),
            )
        existing = self.collection.get(include=[]).get("ids", [])
        if existing:
            self.collection.delete(ids=existing)
        if profiles.empty:
            return 0
        records = profiles.to_dict("records")
        embedding_dimensions: int | None = None
        # Imported vectors need no transformer activations; use modest larger
        # writes to avoid one database transaction per tiny encoding batch.
        write_batch_size = 256 if vectors is not None else self.batch_size
        for start in range(0, len(records), write_batch_size):
            batch = records[start:start + write_batch_size]
            documents = [str(row["rag_document"]) for row in batch]
            embeddings = vectors[start:start + len(batch)] if vectors is not None else self._encode_documents(documents)
            embedding_dimensions = int(embeddings.shape[1])
            self.collection.add(
                ids=[str(row["summoner_id"]) for row in batch], documents=documents,
                embeddings=embeddings.tolist(), metadatas=[self._metadata(row) for row in batch],
            )
        self.collection.modify(metadata={
            "source": "local-real-match-data", "embedding_model": self.embedding_model_name,
            "embedding_dimensions": embedding_dimensions or 0,
            "pipeline_version": self.PIPELINE_VERSION,
            "profile_fingerprint": self._fingerprint(profiles),
        })
        return len(records)

    @staticmethod
    def _metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
        keep = (
            "player_name", "tier", "rank", "role", "matches", "win_rate", "kda",
            "role_experience_percentile", "role_win_rate_percentile", "role_kda_percentile",
            "role_vision_percentile", "role_damage_percentile", "role_assists_percentile",
        )
        return {key: value for key in keep if (value := row.get(key)) is not None and not pd.isna(value)}

    def _intent_fields(self, query_text: str) -> list[str]:
        query = query_text.casefold()
        return [
            field for field, phrases in self.STRUCTURED_INTENTS.items()
            if any(phrase in query for phrase in phrases)
        ]

    def get_profile_document(self, summoner_id: str) -> dict[str, Any] | None:
        """Retrieve one candidate's exact indexed evidence for grounded generation."""
        result = self.collection.get(
            ids=[str(summoner_id)], include=["documents", "metadatas"]
        )
        ids = result.get("ids") or []
        if not ids:
            return None
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return {
            "summoner_id": str(ids[0]),
            "rag_document": str(documents[0]) if documents else "",
            "metadata": metadatas[0] if metadatas else {},
        }

    def search_teammates(
        self, query_text: str, top_k: int = 10, role: str | None = None,
        use_structured: bool = True,
    ) -> list[dict[str, Any]]:
        if top_k < 1 or not query_text.strip() or self.collection.count() == 0:
            return []
        where = {"role": role.upper()} if role else None
        count = len(self.collection.get(where=where, include=[]).get("ids", []))
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=self._encode_query(query_text).tolist(), n_results=count,
            where=where, include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        intent_fields = self._intent_fields(query_text) if use_structured else []
        items = [
            self._scored_result(player_id, document, metadata or {}, distance, intent_fields)
            for player_id, document, metadata, distance
            in zip(ids, documents, metadatas, distances)
        ]
        items.sort(key=lambda item: item["retrieval_similarity"], reverse=True)
        return items[:top_k]

    @staticmethod
    def _scored_result(
        player_id: str, document: str, metadata: dict[str, Any], distance: float,
        intent_fields: list[str],
    ) -> dict[str, Any]:
        semantic = max(0.0, 1.0 - float(distance))
        structured_values = [float(metadata[field]) for field in intent_fields if field in metadata]
        structured = float(np.mean(structured_values)) if structured_values else None
        hybrid = 0.55 * semantic + 0.45 * structured if structured is not None else semantic
        return {
            "summoner_id": player_id, "rag_document": document, "metadata": metadata,
            "semantic_similarity": semantic, "structured_preference_score": structured,
            "retrieval_similarity": hybrid,
        }
