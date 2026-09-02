from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from src.config import CHROMA_DIR


class RealPlayerVectorStore:
    """Persistent Chroma collection containing only computed real-player profiles."""

    def __init__(self, path: str | Path = CHROMA_DIR, collection_name: str = "real_lol_players"):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is not installed; run `pip install -r requirements.txt`") from exc
        self.client = chromadb.PersistentClient(path=str(Path(path)))
        self.vectorizer = HashingVectorizer(
            n_features=768, alternate_sign=False, norm="l2", ngram_range=(1, 2), lowercase=True
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "source": "local-real-match-data", "embedding": "hashing-768"},
        )

    def rebuild(self, profiles: pd.DataFrame) -> int:
        existing = self.collection.get(include=[]).get("ids", [])
        if existing:
            self.collection.delete(ids=existing)
        if profiles.empty:
            return 0
        records = profiles.to_dict("records")
        for start in range(0, len(records), 128):
            batch = records[start:start + 128]
            documents = [str(row["rag_document"]) for row in batch]
            embeddings = self.vectorizer.transform(documents).toarray().astype("float32").tolist()
            self.collection.add(
                ids=[str(row["summoner_id"]) for row in batch], documents=documents,
                embeddings=embeddings, metadatas=[self._metadata(row) for row in batch],
            )
        return len(records)

    @staticmethod
    def _metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
        keep = ("player_name", "tier", "rank", "role", "matches", "win_rate", "kda")
        return {key: value for key in keep if (value := row.get(key)) is not None and not pd.isna(value)}

    def search_teammates(self, query_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k < 1 or not query_text.strip() or self.collection.count() == 0:
            return []
        count = min(top_k, self.collection.count())
        result = self.collection.query(
            query_embeddings=self.vectorizer.transform([query_text]).toarray().astype("float32").tolist(), n_results=count,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {"summoner_id": pid, "rag_document": document, "metadata": metadata or {},
             "retrieval_similarity": max(0.0, 1.0 - float(distance))}
            for pid, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]
