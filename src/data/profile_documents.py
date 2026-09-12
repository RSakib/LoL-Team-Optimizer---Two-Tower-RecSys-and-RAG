from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ROLE_PERCENTILE_FIELDS = {
    "matches": "role_experience_percentile",
    "win_rate": "role_win_rate_percentile",
    "kda": "role_kda_percentile",
    "avg_vision_score": "role_vision_percentile",
    "avg_damage_dealt": "role_damage_percentile",
    "avg_assists": "role_assists_percentile",
}


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    return not bool(missing) if isinstance(missing, (bool, np.bool_)) else True


def _fmt(value: Any, pattern: str, missing: str = "unavailable") -> str:
    return format(value, pattern) if _present(value) else missing


def _percentile_description(value: Any) -> str:
    if not _present(value):
        return "unavailable"
    percentile = int(round(100 * float(value)))
    if percentile >= 90:
        band = "very high"
    elif percentile >= 75:
        band = "high"
    elif percentile >= 50:
        band = "above median"
    elif percentile >= 25:
        band = "below median"
    else:
        band = "low"
    return f"{band} (percentile {percentile} of 100)"


def _sample_reliability(matches: int) -> str:
    if matches >= 50:
        return "high"
    if matches >= 20:
        return "moderate"
    return "limited"


def build_rag_document(row: dict[str, Any]) -> str:
    name = row.get("player_name") or row["summoner_id"]
    tier = " ".join(str(value) for value in (row.get("tier"), row.get("rank")) if _present(value))
    tier = tier or "rank unavailable"
    role = row.get("role") or "role unavailable"
    champions = row.get("top_champions") if isinstance(row.get("top_champions"), dict) else {}
    champion_text = ", ".join(
        f"{champion} ({games} matches)" for champion, games in champions.items()
    ) or "unavailable"
    matches = int(row.get("matches") or 0)
    return (
        f"{name} is a {tier} League of Legends player. Recorded primary role: {role}. "
        f"Evidence quality is {_sample_reliability(matches)} based on {matches} recorded real matches. "
        f"Recorded averages: win rate {_fmt(row.get('win_rate'), '.1%')}, KDA {_fmt(row.get('kda'), '.2f')}, "
        f"kills {_fmt(row.get('avg_kills'), '.2f')}, deaths {_fmt(row.get('avg_deaths'), '.2f')}, "
        f"assists {_fmt(row.get('avg_assists'), '.2f')}, vision score {_fmt(row.get('avg_vision_score'), '.1f')}, "
        f"gold {_fmt(row.get('avg_gold_earned'), '.0f')}, and champion damage "
        f"{_fmt(row.get('avg_damage_dealt'), '.0f')} per match. "
        f"Role-relative profile among recorded {role} players: experience "
        f"{_percentile_description(row.get('role_experience_percentile'))}; win rate "
        f"{_percentile_description(row.get('role_win_rate_percentile'))}; KDA "
        f"{_percentile_description(row.get('role_kda_percentile'))}; vision "
        f"{_percentile_description(row.get('role_vision_percentile'))}; champion damage "
        f"{_percentile_description(row.get('role_damage_percentile'))}; assists "
        f"{_percentile_description(row.get('role_assists_percentile'))}. "
        f"Most played champions: {champion_text}."
    )


def enrich_profiles_for_rag(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add role-relative facts and rebuild narrative documents without inventing data."""
    result = profiles.copy().reset_index(drop=True)
    roles = result["role"].fillna("").astype(str).str.upper()
    for source, destination in ROLE_PERCENTILE_FIELDS.items():
        values = pd.to_numeric(result[source], errors="coerce")
        result[destination] = values.groupby(roles).rank(method="average", pct=True)
    result["rag_document"] = [build_rag_document(row) for row in result.to_dict("records")]
    return result
