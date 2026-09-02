import json

import pandas as pd
import pytest

from src.data.ingestion import DataIngestionError, build_player_profiles, load_real_match_data


def _match(match_id="NA1_1"):
    return {
        "metadata": {"matchId": match_id},
        "info": {"participants": [{
            "summonerId": "sum-1", "puuid": "p-1", "riotIdGameName": "Real Player",
            "teamPosition": "UTILITY", "championName": "Lulu", "kills": 2, "deaths": 4,
            "assists": 12, "visionScore": 31, "goldEarned": 9000,
            "totalDamageDealtToChampions": 7000, "win": True, "teamId": 100,
        }]},
    }


def test_nested_riot_json_and_rank_join(tmp_path):
    (tmp_path / "matches.jsonl").write_text(json.dumps(_match()) + "\n", encoding="utf-8")
    (tmp_path / "players.csv").write_text("tier,rank,puuid\nPLATINUM,I,p-1\n", encoding="utf-8")
    frame = load_real_match_data(tmp_path)
    assert len(frame) == 1
    assert frame.loc[0, "role"] == "SUPPORT"
    assert frame.loc[0, "tier"] == "PLATINUM"
    assert frame.loc[0, "damage_dealt"] == 7000


def test_overlapping_sources_are_deduplicated(tmp_path):
    line = json.dumps(_match()) + "\n"
    (tmp_path / "a.jsonl").write_text(line, encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(line, encoding="utf-8")
    assert len(load_real_match_data(tmp_path)) == 1


def test_empty_data_never_creates_candidates(tmp_path):
    with pytest.raises(DataIngestionError, match="No usable real participant records"):
        load_real_match_data(tmp_path)


def test_profile_document_is_computed_from_matches(tmp_path):
    (tmp_path / "matches.jsonl").write_text(json.dumps(_match()) + "\n", encoding="utf-8")
    profiles = build_player_profiles(load_real_match_data(tmp_path))
    assert profiles.loc[0, "matches"] == 1
    assert profiles.loc[0, "win_rate"] == 1.0
    assert profiles.loc[0, "top_champions"] == {"Lulu": 1}
    assert "31.0 vision score" in profiles.loc[0, "rag_document"]

