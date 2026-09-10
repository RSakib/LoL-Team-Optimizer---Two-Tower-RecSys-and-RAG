import pandas as pd

from src.recsys.engine import ROLES, TeamBuilderEngine


def _profile(player, role, tier="GOLD", champion="Lulu", matches=10):
    return {
        "summoner_id": player, "player_name": player, "tier": tier, "rank": "I", "role": role,
        "matches": matches, "win_rate": .6, "kda": 3., "avg_kills": 5., "avg_deaths": 4.,
        "avg_assists": 7., "avg_vision_score": 20., "avg_gold_earned": 11000.,
        "avg_damage_dealt": 18000., "top_champions": {champion: max(matches // 2, 1)},
        "rag_document": "real profile",
    }


def _full_profiles():
    return pd.DataFrame([_profile(f"player-{role.lower()}", role) for role in ROLES])


def test_slot_filter_returns_empty_instead_of_cross_role_fallback():
    profiles = pd.DataFrame([_profile("top", "TOP"), _profile("support", "SUPPORT")])
    engine = TeamBuilderEngine(profiles)
    assert engine.recommend_slot("top", role="JUNGLE") == []


def test_team_builder_reserves_finder_role_and_fills_other_four():
    profiles = _full_profiles()
    engine = TeamBuilderEngine(profiles)
    team = engine.recommend_team("player-mid", "MID", candidates_per_role=1)
    assert team["finder_primary_role"] == "MID"
    assert {slot["role"] for slot in team["slots"]} == {"TOP", "JUNGLE", "BOTTOM", "SUPPORT"}
    assert len(team["suggested_lineup"]) == 4
    assert all(candidate["summoner_id"] != "player-mid" for candidate in team["suggested_lineup"])
    assert team["complete"] is True


def test_target_champion_affinity_ranks_recorded_specialist_first():
    profiles = pd.DataFrame([
        _profile("lulu", "SUPPORT", champion="Lulu", matches=20),
        _profile("nautilus", "SUPPORT", champion="Nautilus", matches=30),
        _profile("finder", "MID", champion="Ahri", matches=20),
    ])
    results = TeamBuilderEngine(profiles).recommend_slot(
        "finder", "SUPPORT", target_champion="Lulu", top_k=2
    )
    assert results[0]["summoner_id"] == "lulu"
    assert results[0]["champion_affinity_score"] > results[1]["champion_affinity_score"]


def test_fill_selects_a_valid_assignment_and_returns_scenarios():
    team = TeamBuilderEngine(_full_profiles()).recommend_team("player-mid", "FILL", candidates_per_role=1)
    assert team["requested_primary_role"] == "FILL"
    assert team["fill_assignment"] in ROLES
    assert len(team["fill_scenarios"]) == 5
    assert len(team["slots"]) == 4
