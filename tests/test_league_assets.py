"""Small/offline UI tests; no model or raw match data loading."""
import json
from pathlib import Path

from src.ui import app as ui
from src.ui import league_assets as assets


def test_catalog_is_derived_from_profiles_not_asset_roster(tmp_path, monkeypatch):
    path = tmp_path / "profiles.jsonl"
    path.write_text(json.dumps({"top_champions": {"Ahri": 5, "Lux": 0}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(assets, "PROFILE_PATH", path)
    assert assets.recorded_champions() == ("Ahri",)
    path.write_text(json.dumps({"top_champions": {"Ornn": 123}}) + "\n", encoding="utf-8")
    assert assets.recorded_champions() == ("Ornn",)
    monkeypatch.setattr(assets, "PROFILE_PATH", tmp_path / "missing.jsonl")
    assert assets.recorded_champions() == ()


def test_every_bundled_champion_and_supported_tier_has_real_art():
    assert len(assets.artwork()["champions"]) == 171
    for name, entry in assets.artwork()["champions"].items():
        assert assets.champion_url(name)
        assert (assets.ASSET_ROOT / entry["file"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for tier in ui.TIERS:
        assert assets.rank_url(tier)
    for division in ("I", "II", "III", "IV"):
        assert assets.division_url(division)
        assert f"Division {division}" in assets.rank_html("PLATINUM", division)
    assert assets.asset_url("../../.env") == ""
    assert assets.champion_url("<script>") == ""


def test_icon_inputs_only_offer_recorded_champions_and_preserve_ids():
    components = ui.demo.get_config_file()["components"]
    selectors = [c for c in components if (c["props"].get("elem_id") or "").startswith("target-")]
    assert len(selectors) == 5
    for component in selectors:
        assert component["type"] == "html"
        props = component["props"]
        assert props["value"] == ""
        custom = props["props"]
        assert custom["optional"] and custom["searchable"]
        assert {c["value"] for c in custom["choices"]} == set(assets.recorded_champions())
        assert all(c["icon"] for c in custom["choices"])
        wukong = next(c for c in custom["choices"] if c["value"] == "MonkeyKing")
        assert wukong["label"] == "Wukong"


def test_invalid_custom_champion_never_reaches_backend(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Must reject invalid target before the backend call")
    monkeypatch.setattr(ui.backend, "post", unexpected)
    rendered, state, report = ui.recommend_team("Fill", "PLATINUM", "IV", 3, 1, "", "not-a-real-champion", "", "", "", "")
    assert "Invalid champion" in rendered and state == {} and report == ""


def test_champion_mentions_in_cards_evidence_and_both_reports_have_portraits():
    player = {"summoner_id": "test-only", "tier": "PLATINUM", "rank": "IV", "top_champions": {"Ahri": 1}}
    data = {"team": {"suggested_lineup": [player], "champion_pool_evidence": {"distinct_top_champions": ["Ahri"]},
                     "slots": [{"role": "MID", "target_champion": "Ahri", "candidates": [player]}]}}
    output = ui._render_team(data, {"MID::test-only": "Ahri has recorded matches."})
    assert output.count('class="champion-portrait"') == 4
    assert output.count('class="tier-crest"') == 2
    assert output.count('class="division-badge"') == 2
    report = ui._inline_report_html('Vi provides vision. Wukong and Kai\'Sa. <img src=x onerror="alert(1)">')
    assert report.count('class="champion-portrait"') == 3
    assert '<img src=x' not in report and '&lt;img' in report
    assert "vision" in report


def test_lineup_report_decorates_champions_without_changing_scout_evidence(monkeypatch):
    monkeypatch.setattr(ui, "scout_lineup", lambda state: "## Evidence\nAhri and Ornn complement one another.")
    result = list(ui.scout_lineup_ui({}))[-1]
    assert result.count('class="champion-portrait"') == 2
    assert '<strong>Evidence</strong>' in result
