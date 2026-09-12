from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.ingestion import DataIngestionError
from src.llm.scout import ScoutConfigurationError, generate_scout_report
from src.recsys.engine import ROLES
from src.service import get_runtime


QueueRole = Literal["TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT", "FILL"]

app = FastAPI(title="LoL Two-Tower Team Recommender & RAG Scout", version="3.0.0")


class TeamRecommendRequest(BaseModel):
    primary_role: QueueRole
    target_champions: dict[str, str] = Field(default_factory=dict)
    tier: str | None = None
    rank: str | None = None
    preference: str = ""
    candidates_per_role: int = Field(default=3, ge=1, le=10)
    max_tier_gap: int = Field(default=1, ge=0, le=9)


class ScoutRequest(BaseModel):
    summoner_id: str
    slot_role: str | None = None
    target_champion: str | None = None
    preference: str = ""


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        runtime = get_runtime()
        role_counts = runtime.profiles["role"].fillna("UNKNOWN").value_counts().to_dict()
        return {
            "status": "ok",
            "mode": "trained_two_tower_with_rag",
            "real_player_profiles": len(runtime.profiles),
            "profiles_by_primary_role": role_counts,
            "recommendation_model": "trained_two_tower",
            "model_device": runtime.recommender.loaded.device,
            "model_training": runtime.recommender.loaded.metadata.get("training", {}),
            "rag_embedding_model": runtime.vector_store.embedding_model_name,
            "rag_indexed_profiles": runtime.vector_store.collection.count(),
            "scout_generation_configured": bool(os.getenv("OPENAI_API_KEY")),
            "source": "preprocessed local real data",
        }
    except (DataIngestionError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _build_team(request: TeamRecommendRequest) -> dict[str, Any]:
    try:
        runtime = get_runtime()
    except (DataIngestionError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    retrieved_by_role: dict[str, list[dict[str, Any]]] = {}
    if request.preference.strip():
        open_roles = ROLES if request.primary_role == "FILL" else tuple(
            role for role in ROLES if role != request.primary_role
        )
        per_role = max(request.candidates_per_role * 50, 100)
        for role in open_roles:
            retrieved_by_role[role] = runtime.vector_store.search_teammates(
                request.preference, per_role, role=role
            )
    try:
        team = runtime.recommender.recommend_team(
            finder_primary_role=request.primary_role,
            target_champions=request.target_champions,
            tier=request.tier,
            division=request.rank,
            max_tier_gap=request.max_tier_gap,
            candidates_per_role=request.candidates_per_role,
            rag_results_by_role=retrieved_by_role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    candidate_count = sum(len(slot["candidates"]) for slot in team["slots"])
    if candidate_count == 0:
        status, message = "no_matches", "No matching real candidates found for any open role"
    elif team["complete"]:
        status, message = "complete", None
    else:
        status = "partial"
        message = "No matching real candidates found for: " + ", ".join(team["missing_roles"])
    return {"status": status, "message": message, "candidate_count": candidate_count, "team": team}


@app.post("/team/recommend")
def recommend_team(request: TeamRecommendRequest) -> dict[str, Any]:
    return _build_team(request)


@app.post("/recommend", deprecated=True)
def recommend_team_legacy_route(request: TeamRecommendRequest) -> dict[str, Any]:
    """Compatibility route for the original UI; uses the role-queue team contract."""
    return _build_team(request)


@app.post("/scout")
def scout(request: ScoutRequest) -> dict[str, Any]:
    runtime = get_runtime()
    match = runtime.profiles[runtime.profiles["summoner_id"].astype(str) == request.summoner_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="No matching real candidate found")
    evidence = runtime.vector_store.get_profile_document(request.summoner_id)
    if evidence is None or not evidence["rag_document"]:
        raise HTTPException(
            status_code=503,
            detail="Candidate evidence is missing from the RAG index; rebuild the real-profile index",
        )
    candidate = match.iloc[0].to_dict()
    candidate["rag_document"] = evidence["rag_document"]
    candidate["retrieval_metadata"] = evidence["metadata"]
    candidate["slot_role"] = request.slot_role
    candidate["target_champion"] = request.target_champion
    try:
        report = generate_scout_report(candidate, request.preference)
    except ScoutConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "summoner_id": request.summoner_id,
        "slot_role": request.slot_role,
        "report": report,
        "grounded_profile": evidence["rag_document"],
    }
