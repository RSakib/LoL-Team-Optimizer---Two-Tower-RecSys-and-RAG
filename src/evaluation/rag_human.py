from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import PROFILE_PATH, PROJECT_ROOT
from src.evaluation.events import EVENT_PATH
from src.llm.scout import generate_lineup_scout_report
from src.recsys.engine import TwoTowerRecommendationEngine
from src.recsys.team_model import build_observed_lineups
from src.recsys.two_tower import ROLES


REVIEW_DIR = PROJECT_ROOT / "data" / "evaluation" / "rag_human"
CASE_PATH = REVIEW_DIR / "lineup_cases.jsonl"
RATING_PATH = REVIEW_DIR / "ratings.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "rag_generation_evaluation.md"
RATING_COLUMNS = (
    "grounding",
    "tactical_usefulness",
    "preference_alignment",
    "limitation_clarity",
)


def _lineup_from_row(row: Any) -> dict[str, str]:
    return {
        role: str(getattr(row, f"player_{role}"))
        for role in ROLES
        if role != str(row.finder_role)
    }


def _preference(index: int) -> str:
    preferences = (
        "Explain the recorded vision and assist balance across this lineup.",
        "Explain the recorded damage and KDA balance across this lineup.",
        "Explain the breadth and overlap of the recorded champion pools.",
        "Explain the lineup's recorded experience distribution and statistical risks.",
    )
    return preferences[index % len(preferences)]


def prepare_cases(
    event_path: Path = EVENT_PATH,
    profile_path: Path = PROFILE_PATH,
    cases: int = 50,
    reviewers: int = 2,
    seed: int = 42,
    generate: bool = False,
) -> dict[str, Any]:
    """Prepare real-lineup review cases; human labels always remain reviewer supplied."""
    if cases < 1 or reviewers < 1:
        raise ValueError("Case and reviewer counts must be positive")
    events = pd.read_parquet(event_path)
    recent_cutoff = float(events["timestamp"].quantile(0.80))
    events = events[events["timestamp"] >= recent_cutoff].copy()
    profiles = pd.read_json(profile_path, lines=True)
    if profiles.empty:
        raise RuntimeError("Real player profiles are unavailable")
    profiles["summoner_id"] = profiles["summoner_id"].astype(str)
    profile_roles = dict(zip(
        profiles["summoner_id"],
        profiles["role"].fillna("").astype(str).str.upper(),
    ))
    examples = build_observed_lineups(events, profile_roles)
    if examples.empty:
        raise RuntimeError("No fully observed real lineups are available for RAG review")
    examples = examples.sort_values(["timestamp", "match_id", "team_id"], ascending=[False, True, True])
    examples = examples.drop_duplicates(["match_id", "team_id"]).head(cases)
    if len(examples) < cases:
        raise RuntimeError(f"Only {len(examples)} real lineup cases are available; requested {cases}")
    engine = TwoTowerRecommendationEngine(profiles)
    profile_by_id = profiles.set_index("summoner_id").to_dict("index")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    case_records: list[dict[str, Any]] = []
    for case_number, row in enumerate(examples.itertuples(index=False), start=1):
        lineup = _lineup_from_row(row)
        scores = engine.score_lineup(
            finder_primary_role=str(row.finder_role),
            lineup=lineup,
            tier=str(row.finder_tier),
            division=str(row.finder_division),
        )
        retrieved_profiles = [
            {
                "role": role,
                "summoner_id": player_id,
                "rag_document": profile_by_id[player_id]["rag_document"],
            }
            for role, player_id in lineup.items()
        ]
        facts = {
            "finder": {
                "primary_role": str(row.finder_role),
                "tier": str(row.finder_tier),
                "rank": str(row.finder_division),
            },
            "performance_ranking_score_percent": scores["predicted_performance"],
            "compatibility_ranking_score_percent": scores["compatibility_score"],
            "pair_interaction_scores": scores["pair_compatibility"],
            "champion_pool_and_playstyle_evidence": scores["champion_pool_evidence"],
            "retrieved_real_profiles": retrieved_profiles,
        }
        preference = _preference(case_number - 1)
        report = generate_lineup_scout_report(facts, preference) if generate else ""
        case_records.append({
            "case_id": f"lineup-{case_number:03d}",
            "source_match_id": str(row.match_id),
            "source_team_id": int(row.team_id),
            "preference": preference,
            "facts": facts,
            "generated_report": report,
        })
    order = rng.permutation(len(case_records))
    ordered = [case_records[index] for index in order]
    CASE_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )
    fields = [
        "case_id", "reviewer_id", *RATING_COLUMNS,
        "unsupported_claim_count", "notes",
    ]
    with RATING_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ordered:
            for reviewer in range(1, reviewers + 1):
                writer.writerow({
                    "case_id": record["case_id"],
                    "reviewer_id": f"reviewer-{reviewer}",
                    **{column: "" for column in RATING_COLUMNS},
                    "unsupported_claim_count": "",
                    "notes": "",
                })
    write_pending_report(len(ordered), reviewers, generated=generate)
    return {
        "cases": len(ordered),
        "reviewers_per_case": reviewers,
        "reports_generated": generate,
        "case_path": str(CASE_PATH),
        "rating_path": str(RATING_PATH),
    }


def write_pending_report(cases: int, reviewers: int, generated: bool) -> None:
    lines = [
        "# RAG Generation Human Evaluation", "",
        "Status: **awaiting human ratings**.", "",
        f"- Real lineup cases prepared: **{cases}**",
        f"- Reviewers requested per case: **{reviewers}**",
        f"- Scout reports generated: **{'yes' if generated else 'no'}**", "",
        "No grounding, usefulness, preference-alignment, or agreement score is reported until every required",
        "human field is completed. Blank ratings are never replaced with generated or default values.", "",
        "## Rubric", "",
        "Each 1–5 score must be supplied independently by a human reviewer:", "",
        "- **Grounding:** every tactical claim is supported by the retrieved real-player evidence.",
        "- **Tactical usefulness:** the explanation gives actionable, understandable lineup information.",
        "- **Preference alignment:** the report addresses the case's stated preference.",
        "- **Limitation clarity:** model scores and observational limitations are presented honestly.",
        "- **Unsupported claims:** count every factual or causal claim absent from the supplied evidence.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_reviews(
    case_path: Path = CASE_PATH,
    rating_path: Path = RATING_PATH,
) -> dict[str, Any]:
    cases = [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or any(not str(case.get("generated_report", "")).strip() for case in cases):
        raise RuntimeError("Every real case must contain a generated report before human scores can be computed")
    ratings = pd.read_csv(rating_path, dtype={"case_id": str, "reviewer_id": str})
    required = {"case_id", "reviewer_id", *RATING_COLUMNS, "unsupported_claim_count"}
    missing = sorted(required.difference(ratings.columns))
    if missing:
        raise RuntimeError(f"Human rating columns are missing: {missing}")
    for column in (*RATING_COLUMNS, "unsupported_claim_count"):
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    if ratings[[*RATING_COLUMNS, "unsupported_claim_count"]].isna().any().any():
        raise RuntimeError("Human ratings are incomplete; no aggregate score was produced")
    if not all(ratings[column].between(1, 5).all() for column in RATING_COLUMNS):
        raise RuntimeError("Human rubric values must be between 1 and 5")
    expected_cases = {str(case["case_id"]) for case in cases}
    if set(ratings["case_id"]) != expected_cases:
        raise RuntimeError("Human ratings do not cover exactly the prepared real lineup cases")
    reviewer_counts = ratings.groupby("case_id")["reviewer_id"].nunique()
    if reviewer_counts.nunique() != 1 or int(reviewer_counts.iloc[0]) < 2:
        raise RuntimeError("At least two distinct reviewers must score every case")
    means = {column: float(ratings[column].mean()) for column in RATING_COLUMNS}
    agreements: dict[str, float] = {}
    for column in RATING_COLUMNS:
        pivot = ratings.pivot(index="case_id", columns="reviewer_id", values=column)
        first, second = pivot.columns[:2]
        agreements[column] = float(cohen_kappa_score(
            pivot[first].astype(int),
            pivot[second].astype(int),
            weights="quadratic",
        ))
    result = {
        "cases": len(cases),
        "ratings": len(ratings),
        "reviewers_per_case": int(reviewer_counts.iloc[0]),
        "mean_scores": means,
        "quadratic_weighted_kappa_first_two_reviewers": agreements,
        "unsupported_claims_total": int(ratings["unsupported_claim_count"].sum()),
        "reports_with_any_unsupported_claim": float(
            ratings.groupby("case_id")["unsupported_claim_count"].max().gt(0).mean()
        ),
    }
    lines = [
        "# RAG Generation Human Evaluation", "",
        f"- Real lineup cases: **{result['cases']}**",
        f"- Human ratings: **{result['ratings']}**",
        f"- Reviewers per case: **{result['reviewers_per_case']}**", "",
        "| Dimension | Mean (1–5) | Quadratic weighted κ |", "|---|---:|---:|",
    ]
    for column in RATING_COLUMNS:
        lines.append(f"| {column} | {means[column]:.3f} | {agreements[column]:.3f} |")
    lines.extend([
        "",
        f"Unsupported claims counted across ratings: **{result['unsupported_claims_total']}**.",
        f"Reports with at least one unsupported claim: **{result['reports_with_any_unsupported_claim']:.1%}**.",
        "", "These are human judgments over real retrieved evidence; no LLM-generated labels are included.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or score a human RAG-generation evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cases", type=int, default=50)
    prepare.add_argument("--reviewers", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument(
        "--generate",
        action="store_true",
        help="Call the configured local Transformers model for each real case; omitted by default.",
    )
    subparsers.add_parser("score")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_cases(cases=args.cases, reviewers=args.reviewers, seed=args.seed, generate=args.generate)
    else:
        result = score_reviews()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
