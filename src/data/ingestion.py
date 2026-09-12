from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.data.profile_documents import enrich_profiles_for_rag


class DataIngestionError(RuntimeError):
    """Raised when the local data directory contains no usable real match rows."""


MATCH_COLUMNS = [
    "match_id", "summoner_id", "puuid", "player_name", "tier", "rank", "role",
    "champion_name", "kills", "deaths", "assists", "vision_score", "gold_earned",
    "damage_dealt", "win", "team_id",
]

ALIASES = {
    "summoner_id": ("summoner_id", "summonerId", "puuid"),
    "puuid": ("puuid",),
    "player_name": ("player_name", "riotIdGameName", "summonerName"),
    "tier": ("tier",),
    "rank": ("rank",),
    "role": ("role", "teamPosition", "individualPosition", "lane"),
    "champion_name": ("champion_name", "championName"),
    "kills": ("kills",),
    "deaths": ("deaths",),
    "assists": ("assists",),
    "vision_score": ("vision_score", "visionScore"),
    "gold_earned": ("gold_earned", "goldEarned"),
    "damage_dealt": ("damage_dealt", "totalDamageDealtToChampions", "totalDamageDealt"),
    "win": ("win",),
    "team_id": ("team_id", "teamId"),
}


def _first(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return None


def _normalise_role(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    role = str(value).strip().upper()
    mapping = {"MIDDLE": "MID", "UTILITY": "SUPPORT", "CARRY": "BOTTOM", "BOT": "BOTTOM"}
    role = mapping.get(role, role)
    return None if role in {"", "NONE", "UNKNOWN", "INVALID"} else role


def _to_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "win"}:
        return True
    if text in {"false", "0", "no", "fail", "loss"}:
        return False
    return None


def _participant_row(participant: dict[str, Any], match_id: Any, match_tier: Any = None,
                     match_rank: Any = None) -> dict[str, Any] | None:
    row = {key: _first(participant, aliases) for key, aliases in ALIASES.items()}
    row["match_id"] = match_id
    row["tier"] = row["tier"] or match_tier
    row["rank"] = row["rank"] or match_rank
    row["role"] = _normalise_role(row["role"])
    row["win"] = _to_bool(row["win"])
    if not row["summoner_id"] or not row["champion_name"]:
        return None
    return row


def _iter_json(path: Path) -> Iterator[Any]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DataIngestionError(f"Invalid JSON in {path.name} line {number}: {exc}") from exc
        return
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataIngestionError(f"Invalid JSON in {path.name}: {exc}") from exc
    if isinstance(value, list):
        yield from value
    else:
        yield value


def _rows_from_json(path: Path, match_meta: dict[str, tuple[Any, Any]]) -> Iterator[dict[str, Any]]:
    for obj in _iter_json(path):
        if not isinstance(obj, dict):
            continue
        info = obj.get("info") if isinstance(obj.get("info"), dict) else obj
        participants = info.get("participants") if isinstance(info, dict) else None
        if isinstance(participants, list):
            match_id = (obj.get("metadata") or {}).get("matchId") or info.get("matchId") or info.get("gameId")
            tier, rank = match_meta.get(str(match_id), (None, None))
            for participant in participants:
                if isinstance(participant, dict):
                    row = _participant_row(participant, match_id, tier, rank)
                    if row:
                        yield row
        elif any(key in obj for key in ("summoner_id", "summonerId")):
            match_id = obj.get("match_id") or obj.get("matchId") or path.stem
            tier, rank = match_meta.get(str(match_id), (None, None))
            row = _participant_row(obj, match_id, tier, rank)
            if row:
                yield row


def _rank_maps(data_dir: Path) -> tuple[dict[str, tuple[Any, Any]], dict[str, tuple[Any, Any]]]:
    players: dict[str, tuple[Any, Any]] = {}
    matches: dict[str, tuple[Any, Any]] = {}
    for path in sorted(data_dir.glob("*.csv")):
        try:
            header = list(pd.read_csv(path, nrows=0).columns)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        columns = set(header)
        if "puuid" in columns and "tier" in columns and not any(c.startswith("participant0") for c in columns):
            frame = pd.read_csv(path, usecols=[c for c in ("puuid", "tier", "rank") if c in columns], dtype=str)
            for row in frame.to_dict("records"):
                if row.get("puuid"):
                    players[str(row["puuid"])] = (row.get("tier"), row.get("rank"))
        if "matchId" in columns and "tier" in columns and len(columns) <= 10:
            frame = pd.read_csv(path, usecols=[c for c in ("matchId", "tier", "rank") if c in columns], dtype=str)
            for row in frame.to_dict("records"):
                matches[str(row["matchId"])] = (row.get("tier"), row.get("rank"))
    return players, matches


def _rows_from_wide_csv(path: Path, match_meta: dict[str, tuple[Any, Any]]) -> Iterator[dict[str, Any]]:
    header = list(pd.read_csv(path, nrows=0).columns)
    indices = sorted({int(m.group(1)) for c in header if (m := re.match(r"participant(\d+)SummonerId$", c))})
    participant_fields = {name for aliases in ALIASES.values() for name in aliases}
    usecols = [c for c in header if c in {"matchId", "gameId"} or any(
        c == f"participant{index}{field}" for index in indices for field in participant_fields
    )]
    for frame in pd.read_csv(path, usecols=usecols, chunksize=500, low_memory=False):
        for raw in frame.to_dict("records"):
            match_id = raw.get("matchId") or raw.get("gameId")
            tier, rank = match_meta.get(str(match_id), (None, None))
            for index in indices:
                prefix = f"participant{index}"
                participant = {c[len(prefix):]: value for c, value in raw.items() if c.startswith(prefix)}
                row = _participant_row(participant, match_id, tier, rank)
                if row:
                    yield row


def _rows_from_flat_csv(path: Path, match_meta: dict[str, tuple[Any, Any]]) -> Iterator[dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    for raw in frame.to_dict("records"):
        match_id = raw.get("match_id") or raw.get("matchId") or path.stem
        tier, rank = match_meta.get(str(match_id), (None, None))
        row = _participant_row(raw, match_id, tier, rank)
        if row:
            yield row


def load_real_match_data(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Parse local Riot CSV/JSON files and return deduplicated participant-match rows."""
    root = Path(data_dir)
    if not root.is_dir():
        raise DataIngestionError(f"Real data directory does not exist: {root}")
    player_ranks, match_ranks = _rank_maps(root)
    rows: list[dict[str, Any]] = []
    paths = sorted(root.iterdir())
    wide_csvs: list[Path] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            header = pd.read_csv(path, nrows=0).columns
            if any(re.match(r"participant\d+SummonerId$", c) for c in header):
                wide_csvs.append(path)
    # A nested Riot JSONL export and a wide CSV export in the same directory
    # commonly contain the same matches. Prefer the smaller, column-selectable
    # CSV representation; JSON/JSONL remains supported when it is the raw source.
    for path in paths:
        if path.name == "player_profiles.jsonl":
            continue  # derived output, never an ingestion source
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl", ".ndjson"}:
            if not wide_csvs:
                rows.extend(_rows_from_json(path, match_ranks))
        elif suffix == ".csv":
            header = pd.read_csv(path, nrows=0).columns
            if any(re.match(r"participant\d+SummonerId$", c) for c in header):
                rows.extend(_rows_from_wide_csv(path, match_ranks))
            elif any(c in header for c in ("summoner_id", "summonerId")):
                rows.extend(_rows_from_flat_csv(path, match_ranks))
    if not rows:
        raise DataIngestionError(f"No usable real participant records found in {root}")
    frame = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    for puuid, (tier, rank) in player_ranks.items():
        mask = frame["puuid"].astype(str) == puuid
        frame.loc[mask & frame["tier"].isna(), "tier"] = tier
        frame.loc[mask & frame["rank"].isna(), "rank"] = rank
    numeric = ["kills", "deaths", "assists", "vision_score", "gold_earned", "damage_dealt", "team_id"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["tier"] = frame["tier"].astype("string").str.upper()
    frame["rank"] = frame["rank"].astype("string").str.upper()
    frame["champion_name"] = frame["champion_name"].astype("string")
    frame = frame.drop_duplicates(["match_id", "summoner_id"], keep="first").reset_index(drop=True)
    return frame


def _mode(series: pd.Series) -> Any:
    values = series.dropna().astype(str)
    return values.mode().iat[0] if not values.empty else None


def build_player_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        raise DataIngestionError("Cannot build profiles: no real match rows were loaded")
    profiles: list[dict[str, Any]] = []
    for summoner_id, group in matches.groupby("summoner_id", sort=False):
        required = group.dropna(subset=["kills", "deaths", "assists", "win"])
        if required.empty:
            continue
        champions = Counter(group["champion_name"].dropna().astype(str)).most_common(3)
        kills = float(required["kills"].mean())
        deaths = float(required["deaths"].mean())
        assists = float(required["assists"].mean())
        row = {
            "summoner_id": str(summoner_id), "puuid": _mode(group["puuid"]),
            "player_name": _mode(group["player_name"]), "tier": _mode(group["tier"]),
            "rank": _mode(group["rank"]), "role": _mode(group["role"]),
            "matches": int(len(group)), "wins": int(required["win"].astype(bool).sum()),
            "win_rate": float(required["win"].astype(bool).mean()), "avg_kills": kills,
            "avg_deaths": deaths, "avg_assists": assists,
            "kda": float((kills + assists) / max(deaths, 1e-9)),
            "avg_vision_score": float(group["vision_score"].mean()),
            "avg_gold_earned": float(group["gold_earned"].mean()),
            "avg_damage_dealt": float(group["damage_dealt"].mean()),
            "top_champions": dict(champions),
        }
        profiles.append(row)
    if not profiles:
        raise DataIngestionError("No profiles had the real fields needed for aggregation")
    return enrich_profiles_for_rag(pd.DataFrame(profiles))


def write_profiles_jsonl(profiles: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in profiles.to_dict("records"):
            clean = {key: _json_safe(value) for key, value in record.items()}
            handle.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(destination)
    return destination


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
