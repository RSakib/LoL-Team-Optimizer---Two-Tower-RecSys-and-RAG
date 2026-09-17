from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import (
    RAG_RRF_WEIGHT,
    TEAM_MODEL_METADATA_PATH,
    TEAM_MODEL_PATH,
    TWO_TOWER_DEVICE,
    TWO_TOWER_METADATA_PATH,
    TWO_TOWER_MODEL_PATH,
)
from src.recsys.team_model import (
    PAIR_INDICES,
    STYLE_COLUMNS,
    LoadedTeamModel,
    build_lineup_feature_bank,
    lineup_tensors,
)
from src.recsys.two_tower import (
    ROLES,
    TIER_ORDER,
    LoadedTwoTower,
    candidate_tensors,
    canonical_queue_role,
    canonical_role,
    query_tensors,
)


class TwoTowerRecommendationEngine:
    """Serve a trained dual encoder and optionally fuse its ranking with RAG."""

    def __init__(
        self,
        profiles: pd.DataFrame,
        model_path=TWO_TOWER_MODEL_PATH,
        metadata_path=TWO_TOWER_METADATA_PATH,
        team_model_path=TEAM_MODEL_PATH,
        team_metadata_path=TEAM_MODEL_METADATA_PATH,
        device: str = TWO_TOWER_DEVICE,
        rag_rrf_weight: float = RAG_RRF_WEIGHT,
        lineup_pool_size: int = 6,
        allowed_tiers: tuple[str, ...] | None = None,
    ):
        # Optional serving policy; offline evaluation keeps historical ranks intact.
        self.allowed_tiers = allowed_tiers
        self.profiles = profiles.copy().reset_index(drop=True)
        self.loaded = LoadedTwoTower.load(model_path, metadata_path, device)
        self.rag_rrf_weight = max(0.0, float(rag_rrf_weight))
        with torch.inference_mode():
            tensors = candidate_tensors(self.profiles, self.loaded.metadata, self.loaded.device)
            self.candidate_embeddings = self.loaded.model.encode_candidate(**tensors).detach().cpu().numpy()
        self.team_loaded = LoadedTeamModel.load(
            team_model_path,
            team_metadata_path,
            model_path,
            requested_device=self.loaded.device,
        )
        self.team_bank = build_lineup_feature_bank(
            self.profiles, self.loaded, self.candidate_embeddings
        )
        self.lineup_pool_size = max(2, int(lineup_pool_size))

    def recommend_slot(
        self,
        finder_primary_role: str,
        role: str,
        target_champion: str | None = None,
        tier: str | None = None,
        division: str | None = None,
        max_tier_gap: int = 1,
        top_k: int = 5,
        rag_results: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        target_role = canonical_role(role)
        finder_role = canonical_role(finder_primary_role)
        eligible = self.profiles.index[
            (self.profiles["role"].fillna("").astype(str).str.upper() == target_role)
            & self.team_bank.valid
        ].to_numpy(dtype=np.int64)
        if self.allowed_tiers is not None:
            candidate_tiers = self.profiles.loc[eligible, "tier"].fillna("").astype(str).str.upper()
            eligible = eligible[candidate_tiers.isin(self.allowed_tiers).to_numpy()]
        requested_tier = str(tier).upper() if tier else None
        if requested_tier and self.allowed_tiers is not None and requested_tier not in self.allowed_tiers:
            raise ValueError("Finder rank must be Platinum or above")
        if requested_tier:
            target_tier = TIER_ORDER.get(requested_tier)
            if target_tier is None:
                eligible = eligible[
                    self.profiles.loc[eligible, "tier"].fillna("").astype(str).str.upper().to_numpy()
                    == requested_tier
                ]
            else:
                candidate_tiers = self.profiles.loc[eligible, "tier"].fillna("").astype(str).str.upper()
                keep = candidate_tiers.map(
                    lambda value: value in TIER_ORDER and abs(TIER_ORDER[value] - target_tier) <= max_tier_gap
                ).to_numpy(dtype=bool)
                eligible = eligible[keep]
        if division:
            candidate_divisions = self.profiles.loc[eligible, "rank"].fillna("").astype(str).str.upper()
            eligible = eligible[candidate_divisions.to_numpy() == str(division).upper()]
        if len(eligible) == 0 or top_k < 1:
            return []

        with torch.inference_mode():
            query = self.loaded.model.encode_query(**query_tensors(
                self.loaded.metadata,
                finder_role=finder_role,
                finder_tier=requested_tier,
                finder_division=division,
                target_role=target_role,
                target_champion=target_champion,
                device=self.loaded.device,
            )).detach().cpu().numpy()[0]
        similarities = self.candidate_embeddings[eligible] @ query
        model_order = np.argsort(-similarities, kind="stable")
        model_rank = {int(eligible[position]): rank for rank, position in enumerate(model_order, 1)}
        rag_results = rag_results or []
        rag_rank_by_id = {
            str(item["summoner_id"]): rank for rank, item in enumerate(rag_results, 1)
        }
        rag_similarity_by_id = {
            str(item["summoner_id"]): float(item["retrieval_similarity"]) for item in rag_results
        }
        use_rag = bool(rag_results)
        scored: list[tuple[float, int]] = []
        rrf_k = 60.0
        maximum = 1.0 / (rrf_k + 1.0)
        if use_rag:
            maximum += self.rag_rrf_weight / (rrf_k + 1.0)
        for index in eligible:
            player_id = str(self.profiles.at[index, "summoner_id"])
            score = 1.0 / (rrf_k + model_rank[int(index)])
            rag_rank = rag_rank_by_id.get(player_id)
            if rag_rank is not None:
                score += self.rag_rrf_weight / (rrf_k + rag_rank)
            scored.append((100.0 * score / maximum, int(index)))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        similarity_by_index = {int(eligible[position]): float(similarities[position]) for position in range(len(eligible))}
        for recommendation_score, index in scored[:top_k]:
            row = self.profiles.loc[index].to_dict()
            player_id = str(row["summoner_id"])
            row.update({
                "slot_role": target_role,
                "target_champion": target_champion,
                "two_tower_similarity": similarity_by_index[index],
                "two_tower_rank": model_rank[index],
                "rag_similarity": rag_similarity_by_id.get(player_id),
                "rag_rank": rag_rank_by_id.get(player_id),
                "recommendation_score": recommendation_score,
            })
            results.append(self._serialise(row))
        return results

    def recommend_team(
        self,
        finder_primary_role: str,
        target_champions: dict[str, str] | None = None,
        tier: str | None = None,
        division: str | None = None,
        max_tier_gap: int = 1,
        candidates_per_role: int = 3,
        rag_results_by_role: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        queue_role = canonical_queue_role(finder_primary_role)
        targets = {
            canonical_role(role): champion.strip()
            for role, champion in (target_champions or {}).items()
            if str(champion).strip()
        }
        rag_results_by_role = rag_results_by_role or {}
        if queue_role == "FILL":
            scenarios = [
                self._recommend_for_primary(
                    role, targets, tier, division, max_tier_gap,
                    candidates_per_role, rag_results_by_role,
                )
                for role in ROLES
            ]
            best = max(
                scenarios,
                key=lambda item: (item["complete"], item["team_fit_score"] or -1.0),
            )
            best["requested_primary_role"] = "FILL"
            best["fill_assignment"] = best["finder_primary_role"]
            best["fill_scenarios"] = [
                {
                    "assigned_role": item["finder_primary_role"],
                    "complete": item["complete"],
                    "missing_roles": item["missing_roles"],
                    "team_fit_score": item["team_fit_score"],
                }
                for item in scenarios
            ]
            return best
        result = self._recommend_for_primary(
            queue_role, targets, tier, division, max_tier_gap,
            candidates_per_role, rag_results_by_role,
        )
        result["requested_primary_role"] = queue_role
        result["fill_assignment"] = None
        result["fill_scenarios"] = []
        return result

    def score_lineup(
        self,
        finder_primary_role: str,
        lineup: dict[str, str],
        tier: str | None = None,
        division: str | None = None,
    ) -> dict[str, Any]:
        """Score one exact four-player lineup; every ID must resolve to real profile data."""
        finder_role = canonical_role(finder_primary_role)
        expected = set(ROLES).difference({finder_role})
        normalized = {canonical_role(role): str(player_id) for role, player_id in lineup.items()}
        if set(normalized) != expected or len(set(normalized.values())) != len(expected):
            raise ValueError("Lineup must contain four distinct real players for exactly the open roles")
        pools: dict[str, list[dict[str, Any]]] = {}
        for role, player_id in normalized.items():
            match = self.profiles[
                (self.profiles["summoner_id"].astype(str) == player_id)
                & (self.profiles["role"].fillna("").astype(str).str.upper() == role)
            ]
            if match.empty:
                raise ValueError(f"No matching real {role} profile found for {player_id}")
            if self.allowed_tiers is not None and str(match.iloc[0].get("tier", "")).upper() not in self.allowed_tiers:
                raise ValueError("Lineup players must be Platinum or above")
            row = self._serialise(match.iloc[0].to_dict())
            row.update({
                "slot_role": role,
                "recommendation_score": 0.0,
                "selected_for_lineup": True,
            })
            pools[role] = [row]
        return self._optimise_lineup(finder_role, tier, division, pools)

    def _recommend_for_primary(
        self,
        primary_role: str,
        targets: dict[str, str],
        tier: str | None,
        division: str | None,
        max_tier_gap: int,
        candidates_per_role: int,
        rag_results_by_role: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        slots: list[dict[str, Any]] = []
        lineup: list[dict[str, Any]] = []
        missing: list[str] = []
        pools_by_role: dict[str, list[dict[str, Any]]] = {}
        for role in ROLES:
            if role == primary_role:
                continue
            candidates = self.recommend_slot(
                finder_primary_role=primary_role,
                role=role,
                target_champion=targets.get(role),
                tier=tier,
                division=division,
                max_tier_gap=max_tier_gap,
                top_k=max(candidates_per_role, self.lineup_pool_size),
                rag_results=rag_results_by_role.get(role),
            )
            pools_by_role[role] = candidates
            if not candidates:
                missing.append(role)
        joint: dict[str, Any] | None = None
        if not missing:
            joint = self._optimise_lineup(primary_role, tier, division, pools_by_role)
            selected_by_role = {
                item["slot_role"]: item for item in joint["suggested_lineup"]
            }
            lineup = joint["suggested_lineup"]
        else:
            selected_by_role = {}
        for role in ROLES:
            if role == primary_role:
                continue
            candidates = pools_by_role[role]
            selected = selected_by_role.get(role)
            if selected is not None:
                candidates = [selected] + [
                    candidate for candidate in candidates
                    if candidate["summoner_id"] != selected["summoner_id"]
                ]
            slots.append({
                "role": role,
                "target_champion": targets.get(role),
                "candidates": candidates[:candidates_per_role],
            })
        return {
            "finder": {"primary_role": primary_role, "tier": tier, "rank": division},
            "finder_primary_role": primary_role,
            "slots": slots,
            "suggested_lineup": lineup,
            "complete": not missing,
            "missing_roles": missing,
            "team_fit_score": joint["joint_lineup_score"] if joint else None,
            "predicted_performance": joint["predicted_performance"] if joint else None,
            "compatibility_score": joint["compatibility_score"] if joint else None,
            "pair_compatibility": joint["pair_compatibility"] if joint else [],
            "lineups_evaluated": joint["lineups_evaluated"] if joint else 0,
            "champion_pool_evidence": joint["champion_pool_evidence"] if joint else None,
        }

    @torch.inference_mode()
    def _optimise_lineup(
        self,
        finder_role: str,
        tier: str | None,
        division: str | None,
        pools_by_role: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        open_roles = [role for role in ROLES if role != finder_role]
        combinations = list(product(*(pools_by_role[role] for role in open_roles)))
        if not combinations:
            raise RuntimeError("No complete real-candidate lineup can be formed")
        lineups = [
            {role: str(candidate["summoner_id"]) for role, candidate in zip(open_roles, combination)}
            for combination in combinations
        ]
        tensors = lineup_tensors(
            lineups,
            [finder_role] * len(lineups),
            [tier] * len(lineups),
            [division] * len(lineups),
            self.team_bank,
            self.loaded.metadata,
            self.team_loaded.device,
        )
        performance_logits, pair_logits, pair_mask = self.team_loaded.model(**tensors)
        performance = torch.sigmoid(performance_logits).detach().cpu().numpy()
        pair_probabilities = torch.sigmoid(pair_logits)
        compatibility = (
            (pair_probabilities * pair_mask.float()).sum(dim=1)
            / pair_mask.float().sum(dim=1).clamp_min(1.0)
        ).detach().cpu().numpy()
        relevance = np.asarray([
            np.mean([float(candidate["recommendation_score"]) / 100.0 for candidate in combination])
            for combination in combinations
        ])
        best = max(
            range(len(combinations)),
            key=lambda index: (float(performance[index]), float(compatibility[index]), float(relevance[index])),
        )
        selected = [dict(candidate) for candidate in combinations[best]]
        for candidate in selected:
            candidate["selected_for_lineup"] = True
        pair_values = pair_probabilities[best].detach().cpu().numpy()
        valid_values = pair_mask[best].detach().cpu().numpy()
        pair_evidence: list[dict[str, Any]] = []
        for pair_index, (left, right) in enumerate(PAIR_INDICES):
            if not valid_values[pair_index]:
                continue
            pair_evidence.append({
                "roles": [ROLES[left], ROLES[right]],
                "players": [lineups[best][ROLES[left]], lineups[best][ROLES[right]]],
                "model_compatibility": float(pair_values[pair_index] * 100.0),
            })
        return {
            "suggested_lineup": selected,
            "joint_lineup_score": float(performance[best] * 100.0),
            "predicted_performance": float(performance[best] * 100.0),
            "compatibility_score": float(compatibility[best] * 100.0),
            "pair_compatibility": pair_evidence,
            "lineups_evaluated": len(combinations),
            "champion_pool_evidence": self._champion_pool_evidence(selected),
        }

    @staticmethod
    def _champion_pool_evidence(lineup: list[dict[str, Any]]) -> dict[str, Any]:
        role_pools: dict[str, list[str]] = {}
        role_styles: dict[str, dict[str, float]] = {}
        for candidate in lineup:
            role = str(candidate["slot_role"])
            champions = candidate.get("top_champions")
            role_pools[role] = list(champions) if isinstance(champions, dict) else []
            role_styles[role] = {
                name: float(candidate[name])
                for name in STYLE_COLUMNS
                if candidate.get(name) is not None and not pd.isna(candidate.get(name))
            }
        shared: list[dict[str, Any]] = []
        for left, right in product(role_pools, role_pools):
            if ROLES.index(left) >= ROLES.index(right):
                continue
            overlap = sorted(set(role_pools[left]).intersection(role_pools[right]))
            if overlap:
                shared.append({"roles": [left, right], "champions": overlap})
        return {
            "top_champion_pools_by_role": role_pools,
            "distinct_top_champions": sorted({champion for pool in role_pools.values() for champion in pool}),
            "shared_top_champions": shared,
            "role_relative_playstyle_percentiles": role_styles,
        }

    @staticmethod
    def _serialise(row: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                result[key] = None
            elif isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value
        return result


RecommendationEngine = TwoTowerRecommendationEngine
