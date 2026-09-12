from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CHROMA_DIR, DATA_DIR, PROFILE_PATH
from src.data.ingestion import write_profiles_jsonl
from src.data.profile_documents import enrich_profiles_for_rag
from src.rag.vector_store import RealPlayerVectorStore


FIELDS = {
    "SummonerId", "Puuid", "RiotIdGameName", "SummonerName", "TeamPosition", "IndividualPosition",
    "Role", "Lane", "ChampionName", "Kills", "Deaths", "Assists", "VisionScore", "GoldEarned",
    "TotalDamageDealtToChampions", "Win", "TeamId",
}


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _role(value: Any) -> str | None:
    if not _present(value):
        return None
    role = str(value).upper()
    role = {"MIDDLE": "MID", "UTILITY": "SUPPORT", "CARRY": "BOTTOM", "BOT": "BOTTOM"}.get(role, role)
    return None if role in {"NONE", "UNKNOWN", "INVALID"} else role


def _bool(value: Any) -> bool | None:
    if not _present(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    return True if text in {"true", "1", "win"} else False if text in {"false", "0", "loss", "fail"} else None


@dataclass
class PlayerTotals:
    summoner_id: str
    puuid: str | None = None
    names: Counter = field(default_factory=Counter)
    tiers: Counter = field(default_factory=Counter)
    ranks: Counter = field(default_factory=Counter)
    roles: Counter = field(default_factory=Counter)
    champions: Counter = field(default_factory=Counter)
    matches: int = 0
    wins: int = 0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, participant: dict[str, Any], tier: str | None, rank: str | None) -> None:
        self.matches += 1
        self.puuid = self.puuid or participant.get("Puuid")
        win = _bool(participant.get("Win"))
        if win is not None:
            self.wins += int(win)
            self.counts["win"] += 1
        for value, counter in ((participant.get("name"), self.names), (tier, self.tiers),
                               (rank, self.ranks), (_role(participant.get("role")), self.roles),
                               (participant.get("ChampionName"), self.champions)):
            if _present(value):
                counter[str(value)] += 1
        for output, source in (("kills", "Kills"), ("deaths", "Deaths"), ("assists", "Assists"),
                               ("vision", "VisionScore"), ("gold", "GoldEarned"),
                               ("damage", "TotalDamageDealtToChampions")):
            value = pd.to_numeric(participant.get(source), errors="coerce")
            if not pd.isna(value):
                self.sums[output] += float(value)
                self.counts[output] += 1

    def average(self, key: str) -> float | None:
        return self.sums[key] / self.counts[key] if self.counts[key] else None

    @staticmethod
    def mode(counter: Counter) -> str | None:
        return counter.most_common(1)[0][0] if counter else None

    def profile(self) -> dict[str, Any]:
        kills, deaths, assists = self.average("kills"), self.average("deaths"), self.average("assists")
        result = {
            "summoner_id": self.summoner_id, "puuid": self.puuid, "player_name": self.mode(self.names),
            "tier": self.mode(self.tiers), "rank": self.mode(self.ranks), "role": self.mode(self.roles),
            "matches": self.matches, "wins": self.wins,
            "win_rate": self.wins / self.counts["win"] if self.counts["win"] else None,
            "avg_kills": kills, "avg_deaths": deaths, "avg_assists": assists,
            "kda": (kills + assists) / max(deaths, 1e-9) if None not in (kills, deaths, assists) else None,
            "avg_vision_score": self.average("vision"), "avg_gold_earned": self.average("gold"),
            "avg_damage_dealt": self.average("damage"), "top_champions": dict(self.champions.most_common(3)),
        }
        return result


def _fmt(value: Any, pattern: str = ".2f", missing: str = "unavailable") -> str:
    return format(value, pattern) if value is not None else missing


def _document(row: dict[str, Any]) -> str:
    name = row["player_name"] or row["summoner_id"]
    rank = " ".join(x for x in (row["tier"], row["rank"]) if x) or "rank unavailable"
    champs = ", ".join(f"{name} ({games} matches)" for name, games in row["top_champions"].items()) or "unavailable"
    return (
        f"{name} is a {rank} League of Legends player. Most recorded role: {row['role'] or 'unavailable'}. "
        f"Across {row['matches']} recorded real matches: win rate {_fmt(row['win_rate'], '.1%')}, KDA {_fmt(row['kda'])}, "
        f"average kills {_fmt(row['avg_kills'])}, deaths {_fmt(row['avg_deaths'])}, assists {_fmt(row['avg_assists'])}, "
        f"vision score {_fmt(row['avg_vision_score'], '.1f')}, gold {_fmt(row['avg_gold_earned'], '.0f')}, "
        f"and champion damage {_fmt(row['avg_damage_dealt'], '.0f')}. Top champions: {champs}."
    )


def _metadata(data_dir: Path) -> tuple[dict[str, tuple[str | None, str | None]], dict[str, tuple[str | None, str | None]]]:
    by_puuid: dict[str, tuple[str | None, str | None]] = {}
    by_match: dict[str, tuple[str | None, str | None]] = {}
    for path in data_dir.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {"puuid", "tier"}.issubset(columns) and len(columns) < 20:
            for row in pd.read_csv(path, usecols=[c for c in ("puuid", "tier", "rank") if c in columns], dtype=str).to_dict("records"):
                by_puuid[row["puuid"]] = (row.get("tier"), row.get("rank"))
        elif {"matchId", "tier"}.issubset(columns) and len(columns) < 20:
            for row in pd.read_csv(path, usecols=[c for c in ("matchId", "tier", "rank") if c in columns], dtype=str).to_dict("records"):
                by_match[row["matchId"]] = (row.get("tier"), row.get("rank"))
    return by_puuid, by_match


def preprocess(source: Path, data_dir: Path, profile_path: Path, chunk_size: int = 250) -> list[dict[str, Any]]:
    header = list(pd.read_csv(source, nrows=0).columns)
    indices = sorted({int(c[len("participant"):].split("SummonerId")[0]) for c in header if c.startswith("participant") and c.endswith("SummonerId")})
    usecols = [c for c in header if c in {"matchId", "gameId"} or any(c == f"participant{i}{field}" for i in indices for field in FIELDS)]
    puuid_rank, match_rank = _metadata(data_dir)
    candidate_puuids = set(puuid_rank)
    if not candidate_puuids:
        raise RuntimeError("No real ranked PUUID cohort was found in the local player metadata")
    players: dict[str, PlayerTotals] = {}
    processed_matches = 0
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=chunk_size, low_memory=False):
        for raw in chunk.to_dict("records"):
            match_id = str(raw.get("matchId") or raw.get("gameId"))
            for index in indices:
                prefix = f"participant{index}"
                participant = {c[len(prefix):]: value for c, value in raw.items() if c.startswith(prefix)}
                sid = participant.get("SummonerId")
                champion = participant.get("ChampionName")
                if not _present(sid) or not _present(champion):
                    continue
                sid = str(sid)
                participant["name"] = participant.get("RiotIdGameName") or participant.get("SummonerName")
                participant["role"] = participant.get("TeamPosition") or participant.get("IndividualPosition") or participant.get("Role") or participant.get("Lane")
                puuid = str(participant.get("Puuid")) if _present(participant.get("Puuid")) else None
                if puuid not in candidate_puuids:
                    continue
                tier, rank = puuid_rank.get(puuid, match_rank.get(match_id, (None, None)))
                players.setdefault(sid, PlayerTotals(sid)).add(participant, tier, rank)
            processed_matches += 1
        if processed_matches % 1000 == 0:
            print(f"Processed {processed_matches} matches; {len(players)} ranked candidate players", flush=True)
    profile_frame = enrich_profiles_for_rag(pd.DataFrame(
        [total.profile() for total in players.values() if total.counts["win"]]
    ))
    write_profiles_jsonl(profile_frame, profile_path)
    profiles = profile_frame.to_dict("records")
    manifest = {"source": str(source), "matches": processed_matches, "players": len(profiles), "chunk_size": chunk_size}
    profile_path.with_name("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory preprocessing for large Riot exports")
    parser.add_argument("--source", type=Path, default=DATA_DIR / "matchData.csv")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()
    profiles = preprocess(args.source, DATA_DIR, PROFILE_PATH, args.chunk_size)
    if not args.skip_index:
        count = RealPlayerVectorStore(CHROMA_DIR).rebuild(pd.DataFrame(profiles))
        print(f"Indexed {count} real player profiles in ChromaDB", flush=True)


if __name__ == "__main__":
    main()
