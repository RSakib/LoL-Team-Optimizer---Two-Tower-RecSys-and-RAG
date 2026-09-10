from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.data.preprocess_large import FIELDS, _bool, _metadata, _present, _role


EVALUATION_DIR = DATA_DIR / "evaluation"
EVENT_PATH = EVALUATION_DIR / "ranked_match_events.parquet"
EVENT_SCHEMA = {
    "match_id": "string", "timestamp": "int64", "summoner_id": "string", "puuid": "string",
    "player_name": "string", "tier": "string", "rank": "string", "role": "string",
    "champion_name": "string", "kills": "float64", "deaths": "float64", "assists": "float64",
    "vision_score": "float64", "gold_earned": "float64", "damage_dealt": "float64",
    "win": "bool", "team_id": "int64",
}


def _first_present(*values: Any) -> Any:
    return next((value for value in values if _present(value)), None)


def _number(value: Any) -> float | None:
    result = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(result) else float(result)


def build_compact_event_table(
    source: Path,
    destination: Path = EVENT_PATH,
    chunk_size: int = 100,
) -> dict[str, Any]:
    """Stream the real wide Riot CSV into a compact ranked-player event table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    header = list(pd.read_csv(source, nrows=0).columns)
    timestamp_column = next(
        (column for column in ("gameCreation", "gameStartTimestamp", "gameEndTimestamp") if column in header),
        None,
    )
    if timestamp_column is None:
        raise RuntimeError("The raw match export has no timestamp column; temporal evaluation is impossible")
    indices = sorted({
        int(column[len("participant"):].split("SummonerId")[0])
        for column in header if column.startswith("participant") and column.endswith("SummonerId")
    })
    usecols = [column for column in header if column in {"matchId", "gameId", timestamp_column} or any(
        column == f"participant{index}{field}" for index in indices for field in FIELDS
    )]
    puuid_rank, match_rank = _metadata(DATA_DIR)
    candidate_puuids = set(puuid_rank)
    if not candidate_puuids:
        raise RuntimeError("No ranked PUUID cohort was found in local player metadata")

    schema = pa.schema([pa.field(name, pa.type_for_alias(kind), nullable=True) for name, kind in EVENT_SCHEMA.items()])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema=schema, compression="zstd")
    matches_seen = 0
    events_written = 0
    try:
        for chunk in pd.read_csv(source, usecols=usecols, chunksize=chunk_size, low_memory=False):
            records: list[dict[str, Any]] = []
            for raw in chunk.to_dict("records"):
                match_id = str(_first_present(raw.get("matchId"), raw.get("gameId")))
                timestamp = pd.to_numeric(raw.get(timestamp_column), errors="coerce")
                if pd.isna(timestamp):
                    continue
                for index in indices:
                    prefix = f"participant{index}"
                    participant = {key[len(prefix):]: value for key, value in raw.items() if key.startswith(prefix)}
                    puuid = str(participant.get("Puuid")) if _present(participant.get("Puuid")) else None
                    if puuid not in candidate_puuids:
                        continue
                    sid = participant.get("SummonerId")
                    champion = participant.get("ChampionName")
                    team_id = pd.to_numeric(participant.get("TeamId"), errors="coerce")
                    win = _bool(participant.get("Win"))
                    if not _present(sid) or not _present(champion) or pd.isna(team_id) or win is None:
                        continue
                    tier, division = puuid_rank.get(puuid, match_rank.get(match_id, (None, None)))
                    records.append({
                        "match_id": match_id, "timestamp": int(timestamp), "summoner_id": str(sid), "puuid": puuid,
                        "player_name": _first_present(participant.get("RiotIdGameName"), participant.get("SummonerName")),
                        "tier": tier, "rank": division,
                        "role": _role(_first_present(participant.get("TeamPosition"), participant.get("IndividualPosition"),
                                                     participant.get("Role"), participant.get("Lane"))),
                        "champion_name": str(champion), "kills": _number(participant.get("Kills")),
                        "deaths": _number(participant.get("Deaths")), "assists": _number(participant.get("Assists")),
                        "vision_score": _number(participant.get("VisionScore")),
                        "gold_earned": _number(participant.get("GoldEarned")),
                        "damage_dealt": _number(participant.get("TotalDamageDealtToChampions")),
                        "win": bool(win), "team_id": int(team_id),
                    })
                matches_seen += 1
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
                events_written += len(records)
            if matches_seen and matches_seen % 5000 == 0:
                print(f"Compacted {matches_seen} matches into {events_written} ranked-player events", flush=True)
    finally:
        writer.close()
    temporary.replace(destination)
    manifest = {
        "source_file": source.name, "matches_scanned": matches_seen, "ranked_player_events": events_written,
        "ranked_puuid_cohort": len(candidate_puuids), "chunk_size": chunk_size,
    }
    destination.with_name("event_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def temporal_three_way_split(
    events: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a non-empty test fraction")
    match_times = events[["match_id", "timestamp"]].drop_duplicates("match_id").sort_values("timestamp")
    if len(match_times) < 3:
        raise RuntimeError("At least three timestamped matches are required")
    train_index = min(max(int(len(match_times) * train_fraction), 1), len(match_times) - 2)
    validation_index = min(
        max(int(len(match_times) * (train_fraction + validation_fraction)), train_index + 1), len(match_times) - 1
    )
    validation_cutoff = int(match_times.iloc[train_index]["timestamp"])
    test_cutoff = int(match_times.iloc[validation_index]["timestamp"])
    train = events[events["timestamp"] < validation_cutoff].copy()
    validation = events[
        (events["timestamp"] >= validation_cutoff) & (events["timestamp"] < test_cutoff)
    ].copy()
    test = events[events["timestamp"] >= test_cutoff].copy()
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("Three-way temporal split produced an empty partition")
    return train, validation, test, validation_cutoff, test_cutoff
