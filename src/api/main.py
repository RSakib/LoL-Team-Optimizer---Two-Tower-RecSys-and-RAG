from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.ingestion import DataIngestionError
from src.llm.scout import ScoutConfigurationError, generate_scout_report
from src.service import get_runtime, resolve_player_id

app = FastAPI(title="LoL Matchmaker & Tactical Scout", version="1.0.0")


class RecommendRequest(BaseModel):
    summoner: str | None = None
    role: str | None = None
    target_champion: str | None = None
    tier: str | None = None
    rank: str | None = None
    preference: str = ""
    top_k: int = Field(default=5, ge=1, le=50)
    max_tier_gap: int = Field(default=0, ge=0, le=9)


class ScoutRequest(BaseModel):
    summoner_id: str
    preference: str = ""


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        runtime = get_runtime()
        return {"status": "ok", "real_player_profiles": len(runtime.profiles), "source": "preprocessed local real data"}
    except (DataIngestionError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict[str, Any]:
    try:
        runtime = get_runtime()
    except (DataIngestionError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    summoner_id = resolve_player_id(runtime.profiles, request.summoner)
    retrieved = runtime.vector_store.search_teammates(request.preference, max(request.top_k * 10, 50)) if request.preference.strip() else []
    candidate_ids = [item["summoner_id"] for item in retrieved] if request.preference.strip() else None
    retrieval_scores = {item["summoner_id"]: item["retrieval_similarity"] for item in retrieved}
    results = runtime.recommender.recommend(
        summoner_id=summoner_id, candidate_ids=candidate_ids, role=request.role,
        champion=request.target_champion, tier=request.tier, division=request.rank,
        max_tier_gap=request.max_tier_gap, top_k=request.top_k, retrieval_scores=retrieval_scores,
    )
    return {
        "status": "ok" if results else "no_matches",
        "message": None if results else "No matching real candidates found",
        "count": len(results), "candidates": results,
    }


@app.post("/scout")
def scout(request: ScoutRequest) -> dict[str, Any]:
    runtime = get_runtime()
    match = runtime.profiles[runtime.profiles["summoner_id"].astype(str) == request.summoner_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="No matching real candidate found")
    candidate = match.iloc[0].to_dict()
    duo_result = runtime.recommender.recommend(None, [request.summoner_id], top_k=1)
    if duo_result:
        candidate.update(duo_result[0])
    try:
        report = generate_scout_report(candidate, request.preference)
    except ScoutConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"summoner_id": request.summoner_id, "report": report, "grounded_profile": candidate["rag_document"]}
