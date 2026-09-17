"""Use compact real artifacts to test vector portability without model downloads."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.package_space import ROOT
from src.rag.portable import load_embeddings
from src.rag.vector_store import RealPlayerVectorStore

PROFILE_PATH = ROOT / "data/processed/player_profiles.jsonl"
VECTOR_PATH = PROFILE_PATH.parent / "rag_embeddings.npz"


@pytest.fixture
def real_profiles():
    if not PROFILE_PATH.exists() or not VECTOR_PATH.exists():
        pytest.skip("exported real profile vectors required")
    return pd.read_json(PROFILE_PATH, lines=True)


def load(path, profiles):
    from src.config import RAG_EMBEDDING_MODEL
    return load_embeddings(path, profiles, RAG_EMBEDDING_MODEL,
                           RealPlayerVectorStore.PIPELINE_VERSION, RealPlayerVectorStore._fingerprint(profiles))


def test_import_real_vectors_without_running_document_encoder(tmp_path, real_profiles, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("portable import must not load or call a document encoder")

    monkeypatch.setattr(RealPlayerVectorStore, "_encode_documents", forbidden)
    store = RealPlayerVectorStore(tmp_path / "chroma")
    assert store.rebuild(real_profiles, precomputed_path=VECTOR_PATH) == len(real_profiles)
    assert not store.requires_rebuild(real_profiles)
    assert store._embedding_model is None
    ids = real_profiles["summoner_id"].astype(str).tolist()[:8]
    retrieved = store.collection.get(ids=ids, include=["embeddings", "documents"])
    expected = load(VECTOR_PATH, real_profiles)
    locations = dict(zip(real_profiles["summoner_id"].astype(str), range(len(real_profiles))))
    for index, player_id in enumerate(retrieved["ids"]):
        np.testing.assert_allclose(retrieved["embeddings"][index], expected[locations[player_id]], atol=1e-6)
        assert retrieved["documents"][index] == real_profiles.iloc[locations[player_id]]["rag_document"]


def test_portable_vectors_follow_profile_order(real_profiles):
    original = load(VECTOR_PATH, real_profiles)
    reversed_profiles = real_profiles.iloc[::-1].reset_index(drop=True)
    np.testing.assert_array_equal(load(VECTOR_PATH, reversed_profiles), original[::-1])


def test_stale_evidence_is_rejected_before_index_is_touched(tmp_path, real_profiles):
    altered = real_profiles.copy()
    altered.loc[0, "rag_document"] += " Changed evidence."
    store = RealPlayerVectorStore(tmp_path / "chroma")
    with pytest.raises(RuntimeError, match="do not match"):
        store.rebuild(altered, precomputed_path=VECTOR_PATH)
    assert store.collection.count() == 0


@pytest.mark.parametrize("damage", ["duplicate_id", "nan_vector", "wrong_model"])
def test_invalid_portable_vectors_fail_explicitly(tmp_path, real_profiles, damage):
    with np.load(VECTOR_PATH, allow_pickle=False) as archive:
        contents = {name: archive[name] for name in archive.files}
    if damage == "duplicate_id":
        contents["ids"][0] = contents["ids"][1]
    elif damage == "nan_vector":
        contents["embeddings"][0, 0] = np.nan
    else:
        contents["embedding_model"] = np.asarray("different-model")
    damaged = tmp_path / "damaged.npz"
    np.savez_compressed(damaged, **contents)
    with pytest.raises(RuntimeError):
        load(damaged, real_profiles)


def test_warmup_uses_shared_runtime_and_does_not_generate_scout(monkeypatch):
    from src import service
    called = []
    runtime = SimpleNamespace(
        vector_store=SimpleNamespace(search_teammates=lambda *args, **kwargs: called.append("query")),
        recommender=SimpleNamespace(recommend_team=lambda *args, **kwargs: called.append("lineup")),
    )
    monkeypatch.setattr(service, "get_runtime", lambda: runtime)
    assert service.warm_runtime() is runtime
    assert called == ["query", "lineup"]


def test_bundled_missing_vectors_do_not_trigger_document_encoding(tmp_path, real_profiles, monkeypatch):
    from src import service
    profile_path = tmp_path / "player_profiles.jsonl"
    real_profiles.to_json(profile_path, orient="records", lines=True)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(service, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(service, "PRECOMPUTED_EMBEDDINGS_PATH", tmp_path / "missing.npz")
    monkeypatch.setattr(service, "CHROMA_DIR", tmp_path / "chroma")
    service.clear_runtime()
    with pytest.raises(RuntimeError, match="missing rag_embeddings"):
        service.get_runtime()
    service.clear_runtime()
