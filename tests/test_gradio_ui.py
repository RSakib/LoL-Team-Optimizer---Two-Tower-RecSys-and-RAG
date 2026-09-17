from __future__ import annotations

from typing import Any

from src.ui import app as ui
from src.ui import theme


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def test_league_title_and_font_are_scoped_to_hero():
    config = ui.demo.get_config_file()
    assert config["title"] == "League of Legends: Team Recommender"
    assert 'id="league-title"' in theme.HERO_HTML
    assert "League of Legends: " in theme.HERO_HTML and "Team Recommender" in theme.HERO_HTML
    assert "YOUR NEXT CHAPTER" not in theme.HERO_HTML
    assert "hero-eyebrow" not in theme.HERO_HTML
    assert "playstyle preferences.<br>Trained off data from LoL patch 25.14</p>" in theme.HERO_HTML
    assert "#league-title { font-family:'Beaufort for LoL'" in theme.HERO_CSS
    assert "Beaufort" not in ui.APP_CSS
    assert "data:font/ttf;base64," in theme.FONT_CSS
    assert theme.FONT_PATH.is_file()
    assert "<article" not in theme.EMPTY_HTML


def test_header_has_writeup_link_without_removed_tagline_or_rank_badge():
    from html.parser import HTMLParser

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.anchors = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.anchors.append(dict(attrs))

    parser = Links()
    parser.feed(theme.HERO_HTML)
    link = next(a for a in parser.anchors if a.get("class") == "hero-writeup")
    assert link["href"] == "https://github.com/RSakib/LoL-Team-Optimizer---Two-Tower-RecSys-and-RAG"
    assert link["target"] == "_blank"
    assert set(link["rel"].split()) >= {"noopener", "noreferrer"}
    assert link["title"] == "Read the Write-up"
    assert 'aria-hidden="true">?</span>' in theme.HERO_HTML
    assert "Find your four" not in theme.HERO_HTML
    assert "PLATINUM" not in theme.HERO_HTML
    assert "REAL MATCH DATA" in theme.HERO_HTML and "RAG SCOUT REPORTS" in theme.HERO_HTML
    assert ".hero-writeup:focus-visible" in theme.HERO_CSS


def test_page_gradient_is_not_clipped_to_the_content_column():
    import re

    page = re.search(r"body\s*\{([^}]+)\}", theme.GLOBAL_CSS).group(1)
    container = re.search(r"\.gradio-container\s*\{([^}]+)\}", theme.GLOBAL_CSS).group(1)
    assert "radial-gradient" in page and "min-height: 100vh" in page
    assert "ellipse 1000px 720px" in page  # Independent of result/page height.
    assert "radial-gradient" not in container
    assert "gradio-app { background: transparent !important; }" in theme.GLOBAL_CSS
    assert "background: transparent !important" in container
    assert "max-width: 1240px" in container  # Preserve readable content width.


def test_matchmaking_defaults_require_platinum_four_and_fill() -> None:
    assert ui.TIERS == ["PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
    assert "No rank filter" not in ui.TIERS
    config = ui.demo.get_config_file()
    dropdowns = {
        component["props"].get("label"): component["props"].get("value")
        for component in config["components"]
        if component["type"] in {"dropdown", "html"}
    }
    assert dropdowns["Your Primary Role"] == "Fill"
    assert dropdowns["Your rank tier"] == "PLATINUM"
    assert dropdowns["Your division"] == "IV"


def test_advanced_search_contains_every_secondary_setting_and_starts_collapsed():
    config = ui.demo.get_config_file()
    components = {component["id"]: component for component in config["components"]}
    advanced = next(c for c in components.values() if c["props"].get("elem_id") == "advanced-search-settings")
    assert advanced["type"] == "accordion"
    assert advanced["props"]["label"] == "Advanced search settings"
    assert advanced["props"]["open"] is False

    def find_node(node, target):
        if node["id"] == target:
            return node
        for child in node.get("children", []):
            match = find_node(child, target)
            if match is not None:
                return match
        return None

    def descendants(node):
        yield node["id"]
        for child in node.get("children", []):
            yield from descendants(child)

    inside = set(descendants(find_node(config["layout"], advanced["id"])))
    labels = {components[key]["props"].get("label") for key in inside}
    assert {"Candidates per open role", "Maximum tier gap", "Team preference", "Top champion",
            "Jungle champion", "Mid champion", "Bottom champion", "Support champion"} <= labels
    assert not {"Your Primary Role", "Your rank tier", "Your division"} & labels
    build_button = next(c for c in components.values() if c["props"].get("elem_id") == "build-team")
    assert build_button["id"] not in inside
    event = next(dep for dep in config["dependencies"] if dep.get("api_name") == "recommend_team")
    assert len(event["inputs"]) == 11  # Hidden settings still submit normally.


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

    monkeypatch.setattr(ui.backend, "post", fake_post)
    rendered, state, lineup_report = ui.recommend_team(
        "Mid", "PLATINUM", "II", 3, 1, "objective control",
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
        "tier": "PLATINUM",
        "rank": "II",
        "preference": "objective control",
        "candidates_per_role": 3,
        "max_tier_gap": 1,
    }
    assert "No matching real candidates found" in rendered
    assert state["response"] == api_result
    assert state["candidate_reports"] == {}
    assert state["revision"]
    assert lineup_report == ""


def test_candidate_scout_forwards_only_selected_real_candidate(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> FakeResponse:
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"report": "Grounded report"})

    monkeypatch.setattr(ui.backend, "post", fake_post)
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

    monkeypatch.setattr(ui.backend, "post", unexpected_post)
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
    assert 'class="candidate-scout-button"' in rendered
    assert 'data-candidate-key="SUPPORT::real-player-id"' in rendered
    assert "Inspect recorded evidence" not in rendered
    assert "Profile derived from indexed match statistics." not in rendered
    assert "Two-tower rank:" not in rendered
    assert 'class="candidate-report"' in rendered


def test_card_click_routes_specific_player_and_rejects_stale_keys(monkeypatch):
    import gradio as gr
    calls = []
    monkeypatch.setattr(ui, "scout_candidate", lambda key, state: calls.append(key) or "report")
    state = _card_state()
    outputs = list(ui.scout_card(state, gr.EventData(None, {"candidate_key": "TOP::id", "revision": "current"})))
    assert len(outputs) == 2 and 'aria-busy="true"' in outputs[0][0]
    assert 'class="candidate-report" role="status" aria-live="polite">report</div>' in outputs[-1][0]
    assert outputs[-1][1]["candidate_reports"] == {"TOP::id": "report"}
    assert state["candidate_reports"] == {}  # Do not mutate the previous state.
    assert calls == ["TOP::id"]
    for event in ({"candidate_key": "TOP::stale", "revision": "current"},
                  {"candidate_key": "TOP::id", "revision": "old"}):
        invalid = list(ui.scout_card(state, gr.EventData(None, event)))
        assert invalid[0][0] == gr.skip() and len(calls) == 1
    config = ui.demo.get_config_file()
    card_event = next(dep for dep in config["dependencies"] if any(event == "scout" for _, event in dep["targets"]))
    lineup_event = next(dep for dep in config["dependencies"] if dep.get("api_name") == "scout_lineup_ui")
    assert set(card_event["outputs"]).isdisjoint(lineup_event["outputs"])
    assert card_event["scroll_to_output"] is False
    assert not any(c["type"] == "tabs" for c in config["components"])


def _card_state():
    roles = ["TOP", "JUNGLE", "BOTTOM", "SUPPORT"]
    candidates = [{"summoner_id": "id" if role == "TOP" else role, "slot_role": role} for role in roles]
    return {
        "revision": "current", "candidate_reports": {}, "preference": "vision",
        "candidates": {f'{c["slot_role"]}::{c["summoner_id"]}': c for c in candidates},
        "response": {"team": {"complete": True, "finder_primary_role": "MID",
                               "suggested_lineup": candidates,
                               "slots": [{"role": c["slot_role"], "candidates": [c]} for c in candidates]}},
    }


def test_player_then_team_then_other_player_never_overwrite_each_other(monkeypatch):
    import gradio as gr
    calls = []

    def post(url, *, json, timeout):
        calls.append((url, json))
        if url.endswith("/team/scout"):
            return FakeResponse({"report": "TEAM-ONLY result"})
        return FakeResponse({"report": "PLAYER-ONLY " + json["summoner_id"]})

    monkeypatch.setattr(ui.backend, "post", post)
    state = _card_state()
    result, state = list(ui.scout_card(state, gr.EventData(None, {"candidate_key": "TOP::id", "revision": "current"})))[-1]
    assert "PLAYER-ONLY id" in result
    team_report = list(ui.scout_lineup_ui(state))[-1]
    assert "TEAM-ONLY result" in team_report and "PLAYER-ONLY" not in team_report
    assert set(calls[-1][1]["lineup"]) == {"TOP", "JUNGLE", "BOTTOM", "SUPPORT"}
    result, state = list(ui.scout_card(state, gr.EventData(None, {"candidate_key": "SUPPORT::SUPPORT", "revision": "current"})))[-1]
    assert "PLAYER-ONLY id" in result and "PLAYER-ONLY SUPPORT" in result
    assert "TEAM-ONLY" not in result
    articles = result.split("<article")[1:]
    assert "PLAYER-ONLY id" in articles[0] and "PLAYER-ONLY SUPPORT" not in articles[0]
    assert "PLAYER-ONLY SUPPORT" in articles[-1] and "PLAYER-ONLY id" not in articles[-1]


def test_inline_reports_escape_model_html_and_preserve_censorship():
    rendered = ui._inline_report_html("## Scout\n**Strength:** &#42;&#42;&#42;&#42;Jungle\n<script>alert(1)</script>")
    assert "<strong>Scout</strong>" in rendered
    assert "<strong>Strength:</strong>" in rendered
    assert "****Jungle" in rendered and "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_names_masked_in_cards_selectors_and_reports_not_identity_keys():
    candidate = {"summoner_id": "actual-id", "player_name": "Sh1tJungle", "rag_document": "Sh1tJungle has 12 matches."}
    data = {"team": {"slots": [{"role": "JUNGLE", "candidates": [candidate]}], "suggested_lineup": [candidate]}}
    choices, candidates = ui._candidate_selector(data)
    assert "Sh1t" not in choices[0][0] and "****Jungle" in choices[0][0]
    assert choices[0][1] == "JUNGLE::actual-id"
    assert candidates["JUNGLE::actual-id"]["summoner_id"] == "actual-id"
    rendered = ui._render_team(data)
    assert "Sh1t" not in rendered and "****Jungle" in rendered
    assert "actual-id" in rendered
    report = ui._display_report("Sh1tJungle: 12 matches", {"response": data})
    assert "Sh1t" not in report and "&#42;&#42;&#42;&#42;Jungle" in report
    assert candidate["player_name"] == "Sh1tJungle"
