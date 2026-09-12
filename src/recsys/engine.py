from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import RAG_RRF_WEIGHT, TWO_TOWER_DEVICE, TWO_TOWER_METADATA_PATH, TWO_TOWER_MODEL_PATH
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
        device: str = TWO_TOWER_DEVICE,
        rag_rrf_weight: float = RAG_RRF_WEIGHT,
    ):
        self.profiles = profiles.copy().reset_index(drop=True)
        self.loaded = LoadedTwoTower.load(model_path, metadata_path, device)
        self.rag_rrf_weight = max(0.0, float(rag_rrf_weight))
        with torch.inference_mode():
            tensors = candidate_tensors(self.profiles, self.loaded.metadata, self.loaded.device)
            self.candidate_embeddings = self.loaded.model.encode_candidate(**tensors).detach().cpu().numpy()

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
            self.profiles["role"].fillna("").astype(str).str.upper() == target_role
        ].to_numpy(dtype=np.int64)
        requested_tier = str(tier).upper() if tier else None
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
                ).to_numpy()
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
                top_k=candidates_per_role,
                rag_results=rag_results_by_role.get(role),
            )
            slots.append({"role": role, "target_champion": targets.get(role), "candidates": candidates})
            if candidates:
                lineup.append(candidates[0])
            else:
                missing.append(role)
        return {
            "finder": {"primary_role": primary_role, "tier": tier, "rank": division},
            "finder_primary_role": primary_role,
            "slots": slots,
            "suggested_lineup": lineup,
            "complete": not missing,
            "missing_roles": missing,
            "team_fit_score": float(np.mean([item["recommendation_score"] for item in lineup])) if lineup else None,
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
