import pytest

import src.llm.scout as scout


@pytest.fixture(autouse=True)
def isolated_report_cache():
    scout.clear_scout_cache()
    yield
    scout.clear_scout_cache()


def test_local_generation_receives_exact_retrieved_evidence(monkeypatch):
    captured = []

    def generate(messages):
        captured.extend(messages)
        return "grounded local report"

    monkeypatch.setattr(scout, "generate_text", generate)

    report = scout.generate_scout_report(
        {"summoner_id": "real-player", "rag_document": "real retrieved evidence"},
        "vision-focused teammate",
    )

    assert report == "grounded local report"
    assert "real retrieved evidence" in captured[1]["content"]
    assert "vision-focused teammate" in captured[1]["content"]
    assert "not instructions" in captured[0]["content"]


def test_generation_failure_never_becomes_a_fallback_report(monkeypatch):
    def offline(*args, **kwargs):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(scout, "generate_text", offline)

    with pytest.raises(scout.ScoutConfigurationError, match="No fallback report"):
        scout.generate_lineup_scout_report({"retrieved_real_profiles": [{"rag_document": "indexed evidence"}]}, "")


def test_local_generator_refuses_empty_output(monkeypatch):
    monkeypatch.setattr(scout, "generate_text", lambda *args: "  ")
    with pytest.raises(scout.ScoutConfigurationError, match="no scout-report text"):
        scout.generate_scout_report({"summoner_id": "real-player", "rag_document": "indexed evidence"}, "")


def test_successful_reports_cached_but_context_and_evidence_changes_invalidate(monkeypatch):
    calls = []
    monkeypatch.setattr(scout, "generate_text", lambda messages: calls.append(messages) or "report")
    candidate = {"summoner_id": "id", "rag_document": "indexed evidence", "slot_role": "MID"}
    assert scout.generate_scout_report(candidate, "vision") == "report"
    assert scout.generate_scout_report(candidate, "vision") == "report"
    assert len(calls) == 1
    scout.generate_scout_report(candidate, "damage")
    scout.generate_scout_report({**candidate, "target_champion": "Ahri"}, "vision")
    scout.generate_scout_report({**candidate, "rag_document": "updated indexed evidence"}, "vision")
    # Even source fields omitted from the compact prompt invalidate cached output.
    scout.generate_scout_report({**candidate, "matches": 20}, "vision")
    monkeypatch.setattr(scout, "SCOUT_MAX_NEW_TOKENS", 100)
    scout.generate_scout_report(candidate, "vision")
    assert len(calls) == 6
    assert scout._cached_report.cache_info().maxsize == 64


def test_failed_or_empty_reports_are_never_cached(monkeypatch):
    calls = []

    def generate(messages):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("offline")
        if len(calls) == 2:
            return " "
        return "report"

    monkeypatch.setattr(scout, "generate_text", generate)
    candidate = {"rag_document": "indexed evidence"}
    for _ in range(2):
        with pytest.raises(scout.ScoutConfigurationError):
            scout.generate_scout_report(candidate, "")
    assert scout.generate_scout_report(candidate, "") == "report"
    assert len(calls) == 3


def test_compact_lineup_keeps_documents_and_removes_duplicate_metadata(monkeypatch):
    import json
    captured = []
    monkeypatch.setattr(scout, "generate_text", lambda messages: captured.extend(messages) or "report")
    facts = {
        "retrieved_real_profiles": [{"role": "TOP", "summoner_id": "opaque-id", "rag_document": "exact indexed narrative",
                                     "retrieval_metadata": {"player_name": "Player", "matches": 23}}],
        "pair_compatibility_model_estimates": [{"roles": ["TOP", "MID"], "players": ["opaque-id", "id2"], "model_compatibility": 60.123456}],
        "champion_pool_and_playstyle_evidence": {"distinct_top_champions": ["Ornn"], "role_relative_playstyle_percentiles": {"TOP": {"vision": 0.6}}},
    }
    before = json.dumps(facts)
    scout.generate_lineup_scout_report(facts, "vision")
    prompt = captured[1]["content"]
    assert "exact indexed narrative" in prompt
    assert "opaque-id" not in prompt and "retrieval_metadata" not in prompt
    assert "60.12" in prompt and "60.123456" not in prompt
    assert "Ornn" in prompt
    assert json.dumps(facts) == before


def test_missing_evidence_does_not_call_generator(monkeypatch):
    monkeypatch.setattr(scout, "generate_text", lambda _: pytest.fail("Must not generate without evidence"))
    with pytest.raises(scout.ScoutConfigurationError, match="missing"):
        scout.generate_scout_report({}, "")
    with pytest.raises(scout.ScoutConfigurationError, match="missing"):
        scout.generate_lineup_scout_report({"retrieved_real_profiles": []}, "")


def test_candidate_prompt_masks_name_without_mutating_raw_profile(monkeypatch):
    captured = []
    monkeypatch.setattr(scout, "generate_text", lambda messages: captured.extend(messages) or "report")
    candidate = {"player_name": "Sh1tJungle", "summoner_id": "keep-this-id", "rag_document": "Sh1tJungle has 12 recorded matches."}
    scout.generate_scout_report(candidate, "")
    assert "Sh1tJungle" not in captured[1]["content"]
    assert "****Jungle has 12 recorded matches" in captured[1]["content"]
    assert candidate["player_name"] == "Sh1tJungle" and candidate["summoner_id"] == "keep-this-id"


def test_runtime_model_override_cannot_download_arbitrary_weights():
    with pytest.raises(scout.ScoutConfigurationError, match="before startup"):
        scout.generate_scout_report({}, "", model="unconfigured/model")


def test_lineup_prompt_labels_scores_as_not_probabilities(monkeypatch):
    captured = []
    monkeypatch.setattr(scout, "generate_text", lambda messages: captured.extend(messages) or "report")
    scout.generate_lineup_scout_report({"retrieved_real_profiles": [{"rag_document": "selected evidence"}]}, "")
    assert "not probabilities" in captured[0]["content"]
    assert "selected evidence" in captured[1]["content"]


def test_candidate_then_lineup_have_distinct_cached_reports(monkeypatch):
    calls = []

    def generate(messages):
        calls.append(messages)
        return "TEAM REPORT" if "Report type: lineup." in messages[0]["content"] else "CANDIDATE REPORT"

    monkeypatch.setattr(scout, "generate_text", generate)
    candidate = {"summoner_id": "id", "role": "TOP", "rag_document": "indexed TOP evidence"}
    lineup = {"retrieved_real_profiles": [
        {"role": role, "rag_document": f"indexed {role} evidence"}
        for role in ("TOP", "JUNGLE", "BOTTOM", "SUPPORT")
    ]}
    for _ in range(2):
        assert scout.generate_scout_report(candidate, "vision") == "CANDIDATE REPORT"
        assert scout.generate_lineup_scout_report(lineup, "vision") == "TEAM REPORT"
    assert len(calls) == 2
    assert "all four roles" in calls[1][0]["content"]
    assert all(f"indexed {role} evidence" in calls[1][1]["content"] for role in ("TOP", "JUNGLE", "BOTTOM", "SUPPORT"))


def test_cache_namespace_separates_even_identical_prompts(monkeypatch):
    calls = []
    monkeypatch.setattr(scout, "generate_text", lambda messages: calls.append(messages) or messages[0]["content"])
    candidate = scout._generate_grounded_text("rules", "evidence", report_kind="candidate")
    lineup = scout._generate_grounded_text("rules", "evidence", report_kind="lineup")
    assert candidate != lineup and len(calls) == 2
