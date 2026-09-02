from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROLE_ALIASES = {"ADC": "BOTTOM", "BOT": "BOTTOM", "MIDDLE": "MID", "UTILITY": "SUPPORT"}
TIER_ORDER = {name: index for index, name in enumerate(
    ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
)}


class RecommendationEngine:
    def __init__(self, matches: pd.DataFrame | None, profiles: pd.DataFrame, duo_db_path: str | Path | None = None,
                 xgb_model_path: str | Path | None = None):
        self.matches = matches.copy() if matches is not None else pd.DataFrame()
        self.profiles = profiles.copy()
        self.duo_db_path = Path(duo_db_path) if duo_db_path else None
        self.xgb_model = self._load_xgb(xgb_model_path)

    @staticmethod
    def _load_xgb(path: str | Path | None):
        if not path or not Path(path).exists():
            return None
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_jobs=1)
            model.load_model(path)
            return model
        except (ImportError, ValueError):
            return None

    def recommend(
        self,
        summoner_id: str | None,
        candidate_ids: Iterable[str] | None = None,
        role: str | None = None,
        champion: str | None = None,
        tier: str | None = None,
        division: str | None = None,
        max_tier_gap: int = 0,
        top_k: int = 5,
        retrieval_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self.profiles.copy()
        if candidate_ids is not None:
            allowed = {str(value) for value in candidate_ids}
            candidates = candidates[candidates["summoner_id"].astype(str).isin(allowed)]
        if summoner_id:
            candidates = candidates[candidates["summoner_id"].astype(str) != str(summoner_id)]
        if role:
            requested_role = ROLE_ALIASES.get(role.upper(), role.upper())
            candidates = candidates[candidates["role"].fillna("").str.upper() == requested_role]
        if tier:
            requested_tier = tier.upper()
            target = TIER_ORDER.get(requested_tier)
            if target is None:
                candidates = candidates[candidates["tier"].fillna("").str.upper() == requested_tier]
            else:
                gaps = candidates["tier"].map(lambda value: abs(TIER_ORDER.get(str(value).upper(), -99) - target))
                candidates = candidates[gaps <= max_tier_gap]
        if division:
            candidates = candidates[candidates["rank"].fillna("").str.upper() == division.upper()]
        if candidates.empty or top_k < 1:
            return []

        candidates = candidates.copy().reset_index(drop=True)
        candidates["performance_score"] = self._performance(candidates)
        candidates["compatibility_score"] = self._compatibility(summoner_id, candidates)
        duo = candidates["summoner_id"].map(lambda cid: self._duo_stats(summoner_id, str(cid)))
        candidates["duo_games"] = duo.map(lambda value: value[0])
        candidates["duo_win_rate"] = duo.map(lambda value: value[1])
        candidates["duo_score"] = pd.to_numeric(candidates["duo_win_rate"], errors="coerce").fillna(0.0)
        if champion:
            needle = champion.casefold()
            candidates["champion_score"] = candidates["top_champions"].map(
                lambda value: 1.0 if any(str(name).casefold() == needle for name in value) else 0.0
            )
        else:
            candidates["champion_score"] = 0.0
        retrieval_scores = retrieval_scores or {}
        candidates["retrieval_score"] = candidates["summoner_id"].map(retrieval_scores).fillna(0.0)
        candidates["match_score"] = 100 * (
            0.35 * candidates["performance_score"] +
            0.25 * candidates["compatibility_score"] +
            0.20 * candidates["duo_score"] +
            0.15 * candidates["retrieval_score"] +
            0.05 * candidates["champion_score"]
        )
        candidates = candidates.sort_values(["match_score", "matches"], ascending=False).head(top_k)
        return [self._serialise(row) for row in candidates.to_dict("records")]

    @staticmethod
    def _performance(frame: pd.DataFrame) -> np.ndarray:
        features = frame[["win_rate", "kda", "avg_vision_score", "avg_damage_dealt"]].fillna(0.0)
        if len(frame) == 1:
            return np.array([float(features.iloc[0]["win_rate"])])
        scaled = MinMaxScaler().fit_transform(features)
        return scaled @ np.array([0.45, 0.25, 0.15, 0.15])

    def _compatibility(self, summoner_id: str | None, candidates: pd.DataFrame) -> np.ndarray:
        seeker = self.profiles[self.profiles["summoner_id"].astype(str) == str(summoner_id)] if summoner_id else pd.DataFrame()
        if seeker.empty:
            return np.zeros(len(candidates))
        columns = ["win_rate", "kda", "avg_vision_score", "avg_gold_earned", "avg_damage_dealt"]
        matrix = pd.concat([seeker.iloc[[0]][columns], candidates[columns]], ignore_index=True).fillna(0.0)
        scaled = StandardScaler().fit_transform(matrix)
        matrix_score = cosine_similarity(scaled[0:1], scaled[1:])[0].clip(0.0, 1.0)
        if self.xgb_model is None:
            return matrix_score
        from src.recsys.xgb_model import pair_features
        left = pd.concat([seeker.iloc[[0]]] * len(candidates), ignore_index=True)
        learned = np.clip(self.xgb_model.predict(pair_features(left, candidates.reset_index(drop=True))), 0.0, 1.0)
        return 0.5 * matrix_score + 0.5 * learned

    def _duo_stats(self, summoner_id: str | None, candidate_id: str) -> tuple[int, float | None]:
        if not summoner_id:
            return 0, None
        if self.duo_db_path and self.duo_db_path.exists():
            left, right = sorted((str(summoner_id), candidate_id))
            with sqlite3.connect(self.duo_db_path) as connection:
                row = connection.execute(
                    "SELECT games, wins FROM duo_synergy WHERE player_a=? AND player_b=?", (left, right)
                ).fetchone()
            return (int(row[0]), float(row[1] / row[0])) if row and row[0] else (0, None)
        if self.matches.empty:
            return 0, None
        seeker = self.matches[self.matches["summoner_id"].astype(str) == str(summoner_id)]
        candidate = self.matches[self.matches["summoner_id"].astype(str) == candidate_id]
        if seeker.empty or candidate.empty:
            return 0, None
        joined = seeker.merge(candidate, on=["match_id", "team_id"], suffixes=("_seeker", "_candidate"))
        if joined.empty:
            return 0, None
        return len(joined), float(joined["win_candidate"].astype(bool).mean())

    @staticmethod
    def _serialise(row: dict[str, Any]) -> dict[str, Any]:
        wanted = ["summoner_id", "player_name", "tier", "rank", "role", "matches", "win_rate", "kda",
                  "avg_kills", "avg_deaths", "avg_assists", "avg_vision_score", "avg_gold_earned",
                  "avg_damage_dealt", "top_champions", "rag_document", "duo_games", "duo_win_rate",
                  "retrieval_score", "performance_score", "compatibility_score", "match_score"]
        result: dict[str, Any] = {}
        for key in wanted:
            value = row.get(key)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                result[key] = None
            elif isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value
        return result
