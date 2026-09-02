import pandas as pd

from src.recsys.engine import RecommendationEngine


def _profile(player, role, tier="GOLD"):
    return {
        "summoner_id": player, "player_name": player, "tier": tier, "rank": "I", "role": role,
        "matches": 10, "win_rate": .6, "kda": 3., "avg_kills": 5., "avg_deaths": 4.,
        "avg_assists": 7., "avg_vision_score": 20., "avg_gold_earned": 11000.,
        "avg_damage_dealt": 18000., "top_champions": {"Lulu": 4}, "rag_document": "real profile",
    }


def test_filters_return_empty_instead_of_fallback():
    profiles = pd.DataFrame([_profile("a", "TOP"), _profile("b", "SUPPORT")])
    engine = RecommendationEngine(pd.DataFrame(columns=["summoner_id", "match_id", "team_id", "win"]), profiles)
    assert engine.recommend("a", role="JUNGLE") == []


def test_self_is_excluded_and_real_candidate_is_scored():
    profiles = pd.DataFrame([_profile("a", "SUPPORT"), _profile("b", "SUPPORT")])
    matches = pd.DataFrame([
        {"summoner_id": "a", "match_id": "m1", "team_id": 100, "win": True},
        {"summoner_id": "b", "match_id": "m1", "team_id": 100, "win": True},
    ])
    result = RecommendationEngine(matches, profiles).recommend("a", role="SUPPORT", top_k=5)
    assert [item["summoner_id"] for item in result] == ["b"]
    assert result[0]["duo_games"] == 1
    assert result[0]["duo_win_rate"] == 1.0

