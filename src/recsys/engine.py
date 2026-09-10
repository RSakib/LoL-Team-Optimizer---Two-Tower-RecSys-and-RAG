from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


ROLES = ("TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT")
QUEUE_ROLES = (*ROLES, "FILL")
ROLE_ALIASES = {"ADC": "BOTTOM", "BOT": "BOTTOM", "MIDDLE": "MID", "UTILITY": "SUPPORT"}
TIER_ORDER = {name: index for index, name in enumerate(
    ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
)}


def canonical_role(role: str) -> str:
    value = ROLE_ALIASES.get(str(role).strip().upper(), str(role).strip().upper())
    if value not in ROLES:
        raise ValueError(f"Unknown League role: {role}")
    return value


def canonical_queue_role(role: str) -> str:
    value = ROLE_ALIASES.get(str(role).strip().upper(), str(role).strip().upper())
    if value not in QUEUE_ROLES:
        raise ValueError(f"Unknown League queue role: {role}")
    return value


class TeamBuilderEngine:
    """Fill the four missing role-queue slots around one finder using real player profiles."""

    DEFAULT_SCORE_WEIGHTS = {
        "experience_score": 0.12808757860140432,
        "performance_score": 0.004965775629813836,
        "rank_fit_score": 0.590182608420347,
        "champion_affinity_score": 0.2767640373484348,
        "preference_score": 0.05,
    }

    def __init__(self, profiles: pd.DataFrame, score_weights: dict[str, float] | None = None):
        self.profiles = profiles.copy().reset_index(drop=True)
        self.score_weights = dict(self.DEFAULT_SCORE_WEIGHTS)
        if score_weights:
            self.score_weights.update(score_weights)
        self.profiles["role"] = self.profiles["role"].fillna("").map(
            lambda value: ROLE_ALIASES.get(str(value).upper(), str(value).upper())
        )
        self._prepare_role_relative_scores()

    def _prepare_role_relative_scores(self) -> None:
        grouped = self.profiles.groupby("role", dropna=False)
        self.profiles["experience_score"] = grouped["matches"].rank(method="average", pct=True).fillna(0.0)
        metric_weights = {
            "win_rate": 0.35, "kda": 0.25, "avg_vision_score": 0.20, "avg_damage_dealt": 0.20,
        }
        performance = np.zeros(len(self.profiles), dtype=float)
        for column, weight in metric_weights.items():
            values = pd.to_numeric(self.profiles[column], errors="coerce")
            performance += weight * values.groupby(self.profiles["role"]).rank(
                method="average", pct=True
            ).fillna(0.0)
        self.profiles["performance_score"] = performance

    def recommend_slot(
        self,
        finder_id: str | None,
        role: str,
        target_champion: str | None = None,
        tier: str | None = None,
        division: str | None = None,
        max_tier_gap: int = 1,
        top_k: int = 5,
        candidate_ids: Iterable[str] | None = None,
        retrieval_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        requested_role = canonical_role(role)
        candidates = self.profiles[self.profiles["role"] == requested_role].copy()
        if candidate_ids is not None:
            allowed = {str(value) for value in candidate_ids}
            candidates = candidates[candidates["summoner_id"].astype(str).isin(allowed)]
        if finder_id:
            candidates = candidates[candidates["summoner_id"].astype(str) != str(finder_id)]

        requested_tier = str(tier).upper() if tier else self._finder_tier(finder_id)
        if requested_tier:
            target = TIER_ORDER.get(requested_tier)
            if target is None:
                candidates = candidates[candidates["tier"].fillna("").str.upper() == requested_tier]
                candidates["rank_fit_score"] = 1.0
                candidates["tier_gap"] = 0
            else:
                gaps = candidates["tier"].map(
                    lambda value: abs(TIER_ORDER.get(str(value).upper(), -99) - target)
                )
                candidates = candidates[gaps <= max_tier_gap].copy()
                candidates["tier_gap"] = gaps.loc[candidates.index].astype(int)
                candidates["rank_fit_score"] = 1.0 - candidates["tier_gap"] / max(max_tier_gap + 1, 1)
        else:
            candidates["tier_gap"] = None
            candidates["rank_fit_score"] = 0.5
        if division:
            candidates = candidates[candidates["rank"].fillna("").str.upper() == str(division).upper()]
        if candidates.empty or top_k < 1:
            return []

        candidates["champion_affinity_score"] = candidates["top_champions"].map(
            lambda value: self._champion_affinity(value, target_champion)
        )
        retrieval_scores = retrieval_scores or {}
        candidates["preference_score"] = candidates["summoner_id"].astype(str).map(
            retrieval_scores
        ).fillna(0.0)
        weights = dict(self.score_weights)
        if not target_champion:
            weights["champion_affinity_score"] = 0.0
        if not retrieval_scores:
            weights["preference_score"] = 0.0
        active_total = sum(weights.values())
        candidates["team_fit_score"] = 100.0 * sum(
            weight * candidates[column] for column, weight in weights.items()
        ) / active_total
        candidates = candidates.sort_values(
            ["team_fit_score", "experience_score", "matches"], ascending=False
        ).head(top_k)
        candidates["slot_role"] = requested_role
        candidates["target_champion"] = target_champion
        return [self._serialise(row) for row in candidates.to_dict("records")]

    def recommend_team(
        self,
        finder_id: str | None,
        finder_primary_role: str,
        target_champions: dict[str, str] | None = None,
        tier: str | None = None,
        division: str | None = None,
        max_tier_gap: int = 1,
        candidates_per_role: int = 3,
        retrieval_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        queue_role = canonical_queue_role(finder_primary_role)
        target_champions = {
            canonical_role(role): champion.strip()
            for role, champion in (target_champions or {}).items()
            if str(champion).strip()
        }
        if queue_role == "FILL":
            scenarios = [
                self._recommend_for_primary(
                    finder_id, role, target_champions, tier, division, max_tier_gap,
                    candidates_per_role, retrieval_scores,
                )
                for role in ROLES
            ]
            best = max(
                scenarios,
                key=lambda scenario: (
                    scenario["complete"],
                    scenario["team_fit_score"] if scenario["team_fit_score"] is not None else -1.0,
                ),
            )
            best["requested_primary_role"] = "FILL"
            best["fill_assignment"] = best["finder_primary_role"]
            best["fill_scenarios"] = [
                {
                    "assigned_role": scenario["finder_primary_role"],
                    "complete": scenario["complete"],
                    "missing_roles": scenario["missing_roles"],
                    "team_fit_score": scenario["team_fit_score"],
                }
                for scenario in scenarios
            ]
            return best
        result = self._recommend_for_primary(
            finder_id, queue_role, target_champions, tier, division, max_tier_gap,
            candidates_per_role, retrieval_scores,
        )
        result["requested_primary_role"] = queue_role
        result["fill_assignment"] = None
        result["fill_scenarios"] = []
        return result

    def _recommend_for_primary(
        self,
        finder_id: str | None,
        primary_role: str,
        target_champions: dict[str, str],
        tier: str | None,
        division: str | None,
        max_tier_gap: int,
        candidates_per_role: int,
        retrieval_scores: dict[str, float] | None,
    ) -> dict[str, Any]:
        slots: list[dict[str, Any]] = []
        starters: list[dict[str, Any]] = []
        missing_roles: list[str] = []
        for role in ROLES:
            if role == primary_role:
                continue
            candidates = self.recommend_slot(
                finder_id=finder_id,
                role=role,
                target_champion=target_champions.get(role),
                tier=tier,
                division=division,
                max_tier_gap=max_tier_gap,
                top_k=candidates_per_role,
                retrieval_scores=retrieval_scores,
            )
            slots.append({"role": role, "target_champion": target_champions.get(role), "candidates": candidates})
            if candidates:
                starters.append(candidates[0])
            else:
                missing_roles.append(role)
        return {
            "finder": self._finder_summary(finder_id, primary_role),
            "finder_primary_role": primary_role,
            "slots": slots,
            "suggested_lineup": starters,
            "complete": not missing_roles,
            "missing_roles": missing_roles,
            "team_fit_score": float(np.mean([item["team_fit_score"] for item in starters])) if starters else None,
        }

    def _finder_tier(self, finder_id: str | None) -> str | None:
        if not finder_id:
            return None
        match = self.profiles[self.profiles["summoner_id"].astype(str) == str(finder_id)]
        if match.empty or pd.isna(match.iloc[0].get("tier")):
            return None
        return str(match.iloc[0]["tier"]).upper()

    def _finder_summary(self, finder_id: str | None, primary_role: str) -> dict[str, Any]:
        if not finder_id:
            return {"summoner_id": None, "player_name": None, "primary_role": primary_role, "profile_found": False}
        match = self.profiles[self.profiles["summoner_id"].astype(str) == str(finder_id)]
        if match.empty:
            return {"summoner_id": str(finder_id), "player_name": None, "primary_role": primary_role, "profile_found": False}
        row = match.iloc[0]
        return {
            "summoner_id": str(row["summoner_id"]), "player_name": row.get("player_name"),
            "tier": row.get("tier"), "rank": row.get("rank"), "primary_role": primary_role,
            "profile_found": True,
        }

    @staticmethod
    def _champion_affinity(top_champions: Any, champion: str | None) -> float:
        if not champion or not isinstance(top_champions, dict):
            return 0.0
        total = sum(max(float(games), 0.0) for games in top_champions.values())
        if total <= 0:
            return 0.0
        needle = champion.casefold()
        games = next(
            (float(count) for name, count in top_champions.items() if str(name).casefold() == needle), 0.0
        )
        return games / total

    @staticmethod
    def _serialise(row: dict[str, Any]) -> dict[str, Any]:
        wanted = [
            "summoner_id", "player_name", "tier", "rank", "role", "slot_role", "target_champion",
            "matches", "win_rate", "kda", "avg_kills", "avg_deaths", "avg_assists",
            "avg_vision_score", "avg_gold_earned", "avg_damage_dealt", "top_champions", "rag_document",
            "experience_score", "performance_score", "rank_fit_score", "tier_gap",
            "champion_affinity_score", "preference_score", "team_fit_score",
        ]
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


RecommendationEngine = TeamBuilderEngine
