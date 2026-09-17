from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.recsys.two_tower import ROLES


def _profiles() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "summoner_id": f"player-{role.lower()}",
            "player_name": f"Player {role}",
            "tier": "PLATINUM",
            "rank": "I",
            "role": role,
            "rag_document": "real profile",
        }
        for role in ROLES
    ])


class _Store:
    embedding_model_name = "test-local-embedding"

    class collection:
        @staticmethod
        def count():
            return 5

    def search_teammates(self, query, top_k, role=None):
        return []

    def get_profile_document(self, summoner_id):
        return {
            "summoner_id": summoner_id,
            "rag_document": "indexed real profile",
            "metadata": {"role": "MID"},
        }


class _Recommender:
    loaded = SimpleNamespace(device="cpu", metadata={"training": {"development_pairs": 10}})
    team_loaded = SimpleNamespace(metadata={"training": {"development_complete_teams": 10}})

    def recommend_team(self, finder_primary_role, **kwargs):
        assigned = "MID" if finder_primary_role == "FILL" else finder_primary_role
        roles = [role for role in ROLES if role != assigned]
        slots = [
            {
                "role": role,
                "target_champion": None,
                "candidates": [{"summoner_id": f"player-{role.lower()}", "recommendation_score": 80.0}],
            }
            for role in roles
        ]
        return {
            "finder": {"primary_role": assigned, "tier": kwargs.get("tier"), "rank": kwargs.get("division")},
            "finder_primary_role": assigned,
            "requested_primary_role": finder_primary_role,
            "fill_assignment": assigned if finder_primary_role == "FILL" else None,
            "fill_scenarios": [],
            "slots": slots,
            "suggested_lineup": [slot["candidates"][0] for slot in slots],
            "complete": True,
            "missing_roles": [],
            "team_fit_score": 80.0,
            "predicted_performance": 80.0,
            "compatibility_score": 75.0,
            "pair_compatibility": [],
            "lineups_evaluated": 1,
            "champion_pool_evidence": {},
        }

    def score_lineup(self, finder_primary_role, lineup, **kwargs):
        return {
            "predicted_performance": 80.0,
            "compatibility_score": 75.0,
            "pair_compatibility": [],
            "champion_pool_evidence": {},
        }


def _runtime():
    return SimpleNamespace(profiles=_profiles(), vector_store=_Store(), recommender=_Recommender())


def test_health_reports_configured_scout_provider(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    monkeypatch.setattr(
        api_module,
        "scout_generation_status",
        lambda: {"provider": "local_transformers", "model": "Qwen/Qwen2.5-0.5B-Instruct", "configured": True},
    )
    response = TestClient(api_module.app).get("/health")
    assert response.status_code == 200
    assert response.json()["scout_generation_provider"] == "local_transformers"
    assert response.json()["scout_generation_model"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_team_endpoint_returns_four_open_role_slots(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    response = TestClient(api_module.app).post(
        "/team/recommend",
        json={"primary_role": "MID", "tier": "PLATINUM", "candidates_per_role": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert len(payload["team"]["slots"]) == 4
    assert {slot["role"] for slot in payload["team"]["slots"]} == {
        "TOP", "JUNGLE", "BOTTOM", "SUPPORT",
    }


def test_team_endpoint_accepts_fill(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    response = TestClient(api_module.app).post(
        "/team/recommend",
        json={"primary_role": "FILL", "tier": "PLATINUM", "candidates_per_role": 1},
    )
    assert response.status_code == 200
    assert response.json()["team"]["fill_assignment"] in ROLES


def test_team_endpoint_requires_a_rank_tier(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    response = TestClient(api_module.app).post(
        "/team/recommend",
        json={"primary_role": "FILL", "candidates_per_role": 1},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("tier", ["IRON", "BRONZE", "SILVER", "GOLD"])
def test_live_api_rejects_ranks_below_platinum_before_loading_models(monkeypatch, tier):
    def unexpected_runtime():
        raise AssertionError("invalid ranks must be rejected before loading models")

    monkeypatch.setattr(api_module, "get_runtime", unexpected_runtime)
    for route in ("/team/recommend", "/recommend"):
        response = TestClient(api_module.app).post(route, json={"primary_role": "FILL", "tier": tier})
        assert response.status_code == 422


def test_scout_uses_exact_chroma_evidence(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    captured = {}

    def fake_report(candidate, preference):
        captured.update(candidate)
        return "grounded report"

    monkeypatch.setattr(api_module, "generate_scout_report", fake_report)
    response = TestClient(api_module.app).post(
        "/scout",
        json={"summoner_id": "player-mid", "slot_role": "MID", "preference": "vision"},
    )
    assert response.status_code == 200
    assert captured["rag_document"] == "indexed real profile"
    assert response.json()["grounded_profile"] == "indexed real profile"


def test_team_scout_uses_four_exact_chroma_profiles(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", _runtime)
    captured = {}

    def fake_report(facts, preference):
        captured.update(facts)
        return "grounded lineup report"

    monkeypatch.setattr(api_module, "generate_lineup_scout_report", fake_report)
    lineup = {
        role: f"player-{role.lower()}"
        for role in ROLES if role != "MID"
    }
    response = TestClient(api_module.app).post(
        "/team/scout",
        json={"primary_role": "MID", "lineup": lineup, "preference": "balanced team"},
    )
    assert response.status_code == 200
    assert len(captured["retrieved_real_profiles"]) == 4
    assert all(
        item["rag_document"] == "indexed real profile"
        for item in captured["retrieved_real_profiles"]
    )


def test_cached_scout_still_retrieves_and_rejects_missing_current_evidence(monkeypatch):
    from src.llm import scout
    runtime = _runtime()
    reads, generations = [], []
    available = [True]

    def get_document(player_id):
        reads.append(player_id)
        return {"rag_document": "indexed evidence", "metadata": {}} if available[0] else None

    monkeypatch.setattr(runtime.vector_store, "get_profile_document", get_document)
    monkeypatch.setattr(api_module, "get_runtime", lambda: runtime)
    monkeypatch.setattr(scout, "generate_text", lambda messages: generations.append(messages) or "report")
    scout.clear_scout_cache()
    try:
        client = TestClient(api_module.app)
        payload = {"summoner_id": "player-mid", "slot_role": "MID"}
        assert client.post("/scout", json=payload).status_code == 200
        assert client.post("/scout", json=payload).status_code == 200
        assert len(reads) == 2 and len(generations) == 1
        available[0] = False
        assert client.post("/scout", json=payload).status_code == 503
        assert len(reads) == 3 and len(generations) == 1
    finally:
        scout.clear_scout_cache()
