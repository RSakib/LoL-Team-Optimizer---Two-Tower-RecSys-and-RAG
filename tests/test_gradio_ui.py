from __future__ import annotations

from typing import Any

from src.ui import app as ui


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def test_matchmaking_defaults_require_platinum_four_and_fill() -> None:
    assert ui.TIERS[0] == "IRON"
    assert "No rank filter" not in ui.TIERS
    config = ui.demo.get_config_file()
    dropdowns = {
        component["props"].get("label"): component["props"].get("value")
        for component in config["components"]
        if component["type"] == "dropdown"
    }
    assert dropdowns["Finder’s primary role"] == "Fill"
    assert dropdowns["Your rank tier"] == "PLATINUM"
    assert dropdowns["Your division"] == "IV"


def test_recommendation_uses_role_queue_contract_and_real_empty_state(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    api_result = {
        "status": "no_matches",
        "message": "No matching real candidates found for any open role",
        "candidate_count": 0,
        "team": {"slots": []},
    }

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> FakeResponse:
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(api_result)

    monkeypatch.setattr(ui.requests, "post", fake_post)
    rendered, state, selector, lineup_report, candidate_report = ui.recommend_team(
        "Mid", "GOLD", "II", 3, 1, "objective control",
        "Ornn", "Vi", "Ahri", "Jinx", "Nautilus",
    )

    assert captured["url"].endswith("/team/recommend")
    assert captured["json"] == {
        "primary_role": "MID",
        "target_champions": {
            "TOP": "Ornn",
            "JUNGLE": "Vi",
            "BOTTOM": "Jinx",
            "SUPPORT": "Nautilus",
        },
        "tier": "GOLD",
        "rank": "II",
        "preference": "objective control",
        "candidates_per_role": 3,
        "max_tier_gap": 1,
    }
    assert "No matching real candidates found" in rendered
    assert state["response"] == api_result
    assert selector["choices"] == []
    assert lineup_report == candidate_report == ""


def test_candidate_scout_forwards_only_selected_real_candidate(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> FakeResponse:
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"report": "Grounded report"})

    monkeypatch.setattr(ui.requests, "post", fake_post)
    state = {
        "preference": "vision control",
        "candidates": {
            "SUPPORT::real-player-id": {
                "summoner_id": "real-player-id",
                "slot_role": "SUPPORT",
                "target_champion": "Nautilus",
            }
        },
    }

    report = ui.scout_candidate("SUPPORT::real-player-id", state)

    assert captured["url"].endswith("/scout")
    assert captured["json"] == {
        "summoner_id": "real-player-id",
        "slot_role": "SUPPORT",
        "target_champion": "Nautilus",
        "preference": "vision control",
    }
    assert "Grounded report" in report


def test_lineup_scout_refuses_partial_lineup_without_api_call(monkeypatch) -> None:
    def unexpected_post(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("partial lineups must not be sent to the scout endpoint")

    monkeypatch.setattr(ui.requests, "post", unexpected_post)
    state = {"response": {"team": {"complete": False, "suggested_lineup": []}}}

    assert "complete real lineup is required" in ui.scout_lineup(state)


def test_each_recommended_player_renders_as_an_individual_card() -> None:
    candidate = {
        "summoner_id": "real-player-id",
        "player_name": "Indexed Player",
        "tier": "GOLD",
        "rank": "II",
        "recommendation_score": 81.2,
        "kda": 3.4,
        "win_rate": 0.56,
        "matches": 25,
        "top_champions": {"Nautilus": 12},
        "two_tower_rank": 1,
        "two_tower_similarity": 0.73,
        "rag_document": "Profile derived from indexed match statistics.",
        "selected_for_lineup": True,
    }
    rendered = ui._render_team({
        "status": "partial",
        "message": "No matching real candidates found for: TOP, JUNGLE, BOTTOM",
        "team": {
            "finder_primary_role": "MID",
            "slots": [{"role": "SUPPORT", "target_champion": None, "candidates": [candidate]}],
            "suggested_lineup": [],
        },
    })

    assert rendered.count('<article class="player-card selected">') == 1
    assert "Indexed Player" in rendered
    assert "Recorded top champions" in rendered
    assert "Why this candidate" in rendered
