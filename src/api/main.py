from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.ingestion import DataIngestionError
from src.llm.scout import ScoutConfigurationError, generate_scout_report
from src.service import get_runtime, resolve_player_id


QueueRole = Literal["TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT", "FILL"]

app = FastAPI(title="LoL Role Queue Team Builder & Tactical Scout", version="2.0.0")


class TeamRecommendRequest(BaseModel):
    finder: str | None = None
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
            "mode": "role_queue_team_builder",
            "real_player_profiles": len(runtime.profiles),
            "profiles_by_primary_role": role_counts,
            "source": "preprocessed local real data",
        }
    except (DataIngestionError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _build_team(request: TeamRecommendRequest) -> dict[str, Any]:
    try:
        runtime = get_runtime()
    except (DataIngestionError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finder_id = resolve_player_id(runtime.profiles, request.finder)
    retrieved = runtime.vector_store.search_teammates(
        request.preference, min(max(request.candidates_per_role * 50, 250), len(runtime.profiles))
    ) if request.preference.strip() else []
    retrieval_scores = {item["summoner_id"]: item["retrieval_similarity"] for item in retrieved}
    try:
        team = runtime.recommender.recommend_team(
            finder_id=finder_id,
            finder_primary_role=request.primary_role,
            target_champions=request.target_champions,
            tier=request.tier,
            division=request.rank,
            max_tier_gap=request.max_tier_gap,
            candidates_per_role=request.candidates_per_role,
            retrieval_scores=retrieval_scores,
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
    candidate = match.iloc[0].to_dict()
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
        "grounded_profile": candidate["rag_document"],
    }
