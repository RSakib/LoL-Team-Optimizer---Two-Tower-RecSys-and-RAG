import pandas as pd
import torch

from src.recsys.engine import TwoTowerRecommendationEngine
from src.recsys.two_tower import ROLES, TwoTowerModel, build_metadata, candidate_tensors, save_artifact


def _profile(player, role, tier="GOLD", champion="Lulu", matches=10):
    return {
        "summoner_id": player,
        "player_name": player,
        "tier": tier,
        "rank": "I",
        "role": role,
        "matches": matches,
        "win_rate": 0.6,
        "kda": 3.0,
        "avg_kills": 5.0,
        "avg_deaths": 4.0,
        "avg_assists": 7.0,
        "avg_vision_score": 20.0,
        "avg_gold_earned": 11000.0,
        "avg_damage_dealt": 18000.0,
        "top_champions": {champion: max(matches // 2, 1)},
        "rag_document": "real profile",
    }


def _engine(tmp_path, profiles, rag_weight=0.35):
    metadata = build_metadata(profiles, embedding_dim=16, hidden_dim=32)
    model = TwoTowerModel(metadata)
    model_path = tmp_path / "model.pt"
    metadata_path = tmp_path / "metadata.json"
    save_artifact(model, metadata, model_path, metadata_path)
    return TwoTowerRecommendationEngine(
        profiles,
        model_path=model_path,
        metadata_path=metadata_path,
        device="cpu",
        rag_rrf_weight=rag_weight,
    )


def test_candidate_tower_embeddings_are_normalized():
    profiles = pd.DataFrame([_profile("top", "TOP"), _profile("support", "SUPPORT")])
    metadata = build_metadata(profiles, embedding_dim=16, hidden_dim=32)
    model = TwoTowerModel(metadata)
    embeddings = model.encode_candidate(**candidate_tensors(profiles, metadata, "cpu"))
    torch.testing.assert_close(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2))


def test_slot_hard_filter_never_falls_back_to_wrong_role(tmp_path):
    profiles = pd.DataFrame([_profile("top", "TOP"), _profile("support", "SUPPORT")])
    engine = _engine(tmp_path, profiles)
    assert engine.recommend_slot("TOP", "JUNGLE") == []


def test_team_builder_reserves_finder_role_and_returns_other_four(tmp_path):
    profiles = pd.DataFrame([_profile(f"player-{role.lower()}", role) for role in ROLES])
    team = _engine(tmp_path, profiles).recommend_team("MID", candidates_per_role=1)
    assert {slot["role"] for slot in team["slots"]} == {"TOP", "JUNGLE", "BOTTOM", "SUPPORT"}
    assert len(team["suggested_lineup"]) == 4
    assert team["complete"] is True


def test_rag_rank_is_fused_only_when_preference_results_exist(tmp_path):
    profiles = pd.DataFrame([
        _profile("support-a", "SUPPORT", matches=20),
        _profile("support-b", "SUPPORT", matches=30),
        _profile("mid", "MID"),
    ])
    engine = _engine(tmp_path, profiles, rag_weight=10.0)
    results = engine.recommend_slot(
        "MID",
        "SUPPORT",
        top_k=2,
        rag_results=[{"summoner_id": "support-b", "retrieval_similarity": 0.9}],
    )
    assert results[0]["summoner_id"] == "support-b"
    assert results[0]["rag_rank"] == 1
