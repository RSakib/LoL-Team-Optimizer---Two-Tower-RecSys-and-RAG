from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROJECT_ROOT
from src.data.ingestion import build_player_profiles
from src.evaluation.events import EVENT_PATH, build_compact_event_table, temporal_three_way_split
from src.recsys.engine import TIER_ORDER, TeamBuilderEngine


METRICS_PATH = DATA_DIR / "evaluation" / "role_queue_metrics.csv"
REPORT_PATH = DATA_DIR / "evaluation" / "role_queue_report.json"
PORTFOLIO_PATH = PROJECT_ROOT / "docs" / "role_queue_evaluation.md"
WEIGHT_NAMES = ("experience_score", "performance_score", "rank_fit_score", "champion_affinity_score")


def ranking_metrics(
    scores: np.ndarray, candidate_ids: list[str], positives: set[str], k: int
) -> tuple[float, float]:
    order = np.argsort(-scores, kind="stable")[:k]
    ranked = [candidate_ids[index] for index in order]
    hits = sum(candidate in positives for candidate in ranked)
    recall = hits / len(positives)
    dcg = sum(1.0 / math.log2(rank + 2) for rank, candidate in enumerate(ranked) if candidate in positives)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(positives), k)))
    return recall, dcg / ideal if ideal else 0.0


def _profiles(events: pd.DataFrame, minimum_matches: int) -> pd.DataFrame:
    profiles = build_player_profiles(events)
    return profiles[profiles["matches"] >= minimum_matches].reset_index(drop=True)


def build_role_fill_queries(
    engine: TeamBuilderEngine,
    events: pd.DataFrame,
    max_queries: int,
    seed: int,
    successful_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = events[events["win"].astype(bool)] if successful_only else events
    queries: list[dict[str, Any]] = []
    opportunities = 0
    profile_ids = set(engine.profiles["summoner_id"].astype(str))
    role_ids = {
        role: set(frame["summoner_id"].astype(str))
        for role, frame in engine.profiles.groupby(engine.profiles["role"].fillna("").str.upper())
    }
    player_tier_order = {
        str(row.summoner_id): TIER_ORDER.get(str(row.tier).upper())
        for row in engine.profiles[["summoner_id", "tier"]].itertuples(index=False)
    }
    eligible_cache: dict[tuple[str, int | None], set[str]] = {}
    for (match_id, team_id), group in source.groupby(["match_id", "team_id"], sort=False):
        members = group.drop_duplicates("summoner_id")
        if len(members) < 2:
            continue
        for finder in members.itertuples(index=False):
            finder_id = str(finder.summoner_id)
            if finder_id not in profile_ids:
                continue
            teammates = members[members["summoner_id"].astype(str) != finder_id]
            for (target_role, champion), positive_frame in teammates.groupby(
                ["role", "champion_name"], dropna=True
            ):
                if not target_role or str(target_role).upper() == str(finder.role).upper():
                    continue
                opportunities += 1
                positives = set(positive_frame["summoner_id"].astype(str)) & profile_ids
                if not positives:
                    continue
                target_role_key = str(target_role).upper()
                finder_tier_order = player_tier_order.get(finder_id)
                cache_key = (target_role_key, finder_tier_order)
                if cache_key not in eligible_cache:
                    ids = role_ids.get(target_role_key, set())
                    eligible_cache[cache_key] = {
                        player_id for player_id in ids
                        if finder_tier_order is None
                        or (
                            player_tier_order.get(player_id) is not None
                            and abs(player_tier_order[player_id] - finder_tier_order) <= 1
                        )
                    }
                eligible_ids = eligible_cache[cache_key] - {finder_id}
                positives &= eligible_ids
                if positives and len(eligible_ids) > len(positives):
                    queries.append({
                        "match_id": str(match_id), "finder_id": finder_id,
                        "finder_role": str(finder.role), "target_role": str(target_role),
                        "target_champion": str(champion), "positives": positives,
                    })
    eligible = len(queries)
    rng = np.random.default_rng(seed)
    if len(queries) > max_queries:
        selected = rng.choice(len(queries), size=max_queries, replace=False)
        queries = [queries[index] for index in selected]
    if not queries:
        raise RuntimeError("No eligible future role-fill queries were found")
    return queries, {
        "opportunities": opportunities,
        "eligible_before_sampling": eligible,
        "evaluated_queries": len(queries),
        "coverage_before_sampling": eligible / opportunities if opportunities else 0.0,
    }


def _weight_candidates(trials: int, seed: int) -> np.ndarray:
    fixed = np.asarray([
        [0.40, 0.12, 0.08, 0.35],
        [0.45, 0.15, 0.10, 0.30],
        [0.30, 0.15, 0.10, 0.45],
        [1.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
    ], dtype=float)
    fixed /= fixed.sum(axis=1, keepdims=True)
    if trials <= len(fixed):
        return fixed[:trials]
    sampled = np.random.default_rng(seed).dirichlet(np.full(len(WEIGHT_NAMES), 0.8), trials - len(fixed))
    return np.vstack((fixed, sampled))


def tune_role_queue_weights(
    profiles: pd.DataFrame,
    validation_events: pd.DataFrame,
    trials: int,
    max_queries: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    engine = TeamBuilderEngine(profiles)
    queries, coverage = build_role_fill_queries(engine, validation_events, max_queries, seed, successful_only=True)
    weights = _weight_candidates(trials, seed)
    ndcg_sum = np.zeros(len(weights), dtype=float)
    recall_sum = np.zeros(len(weights), dtype=float)
    discounts = 1.0 / np.log2(np.arange(5) + 2.0)
    for query in queries:
        candidates = engine.recommend_slot(
            finder_id=query["finder_id"], role=query["target_role"],
            target_champion=query["target_champion"], max_tier_gap=1, top_k=len(engine.profiles),
        )
        components = np.asarray([
            [float(candidate[name]) for name in WEIGHT_NAMES] for candidate in candidates
        ])
        combined = components @ weights.T
        top = np.argsort(-combined, axis=0, kind="stable")[:5]
        relevant = np.asarray([
            str(candidate["summoner_id"]) in query["positives"] for candidate in candidates
        ], dtype=float)
        hits = relevant[top]
        ideal_hits = min(len(query["positives"]), 5)
        ndcg_sum += (hits * discounts[:, None]).sum(axis=0) / discounts[:ideal_hits].sum()
        recall_sum += hits.sum(axis=0) / len(query["positives"])
    ndcg = ndcg_sum / len(queries)
    recall = recall_sum / len(queries)
    best = int(np.lexsort((recall, ndcg))[-1])
    selected = {name: float(weights[best, index]) for index, name in enumerate(WEIGHT_NAMES)}
    selected["preference_score"] = 0.05
    return selected, {
        "trials": len(weights), "queries": len(queries), "coverage": coverage,
        "validation_ndcg_at_5": float(ndcg[best]), "validation_recall_at_5": float(recall[best]),
        "selected_weights": selected,
    }


def _cluster_bootstrap_mean(
    values: list[float], clusters: list[str], seed: int, iterations: int = 500
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    cluster_array = np.asarray(clusters)
    unique, inverse = np.unique(cluster_array, return_inverse=True)
    sums = np.bincount(inverse, weights=array)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        means[index] = sums[sampled].sum() / counts[sampled].sum()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_queries(
    engine: TeamBuilderEngine, queries: list[dict[str, Any]], task: str, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    per_query: dict[tuple[str, str], list[float]] = defaultdict(list)
    clusters = [query["match_id"] for query in queries]
    for query in queries:
        candidates = engine.recommend_slot(
            finder_id=query["finder_id"], role=query["target_role"],
            target_champion=query["target_champion"], max_tier_gap=1, top_k=len(engine.profiles),
        )
        candidate_ids = [str(candidate["summoner_id"]) for candidate in candidates]
        scores = {
            "team_builder": np.asarray([candidate["team_fit_score"] for candidate in candidates], dtype=float),
            "experience": np.asarray([candidate["experience_score"] for candidate in candidates], dtype=float),
            "performance": np.asarray([candidate["performance_score"] for candidate in candidates], dtype=float),
            "champion_affinity": np.asarray([candidate["champion_affinity_score"] for candidate in candidates], dtype=float),
            "random": rng.random(len(candidates)),
        }
        for model, model_scores in scores.items():
            for k in (5, 10):
                recall, ndcg = ranking_metrics(model_scores, candidate_ids, query["positives"], k)
                per_query[(model, f"recall@{k}")].append(recall)
                per_query[(model, f"ndcg@{k}")].append(ndcg)
            order = np.argsort(-model_scores, kind="stable")
            first = next(
                (rank + 1 for rank, index in enumerate(order) if candidate_ids[index] in query["positives"]), None
            )
            per_query[(model, "mrr")].append(1.0 / first if first else 0.0)
    rows: list[dict[str, Any]] = []
    for (model, metric), values in sorted(per_query.items()):
        low, high = _cluster_bootstrap_mean(values, clusters, seed)
        rows.append({
            "task": task, "model": model, "metric": metric, "value": float(np.mean(values)),
            "ci_low": low, "ci_high": high, "samples": len(values),
        })
    for baseline in ("champion_affinity", "experience", "random"):
        for metric in ("ndcg@5", "recall@5"):
            differences = (
                np.asarray(per_query[("team_builder", metric)]) - np.asarray(per_query[(baseline, metric)])
            ).tolist()
            low, high = _cluster_bootstrap_mean(differences, clusters, seed)
            rows.append({
                "task": f"{task}_delta", "model": f"team_builder_vs_{baseline}", "metric": metric,
                "value": float(np.mean(differences)), "ci_low": low, "ci_high": high, "samples": len(differences),
            })
    return rows


def _write_report(report: dict[str, Any], metrics: pd.DataFrame) -> None:
    primary = metrics[(metrics["task"] == "successful_role_fill") & metrics["metric"].isin(["ndcg@5", "recall@5"])]
    pivot = primary.pivot(index="model", columns="metric", values="value").sort_values("ndcg@5", ascending=False)
    observed = metrics[(metrics["task"] == "observed_role_fill") & metrics["metric"].isin(["ndcg@5", "recall@5"])]
    observed_pivot = observed.pivot(index="model", columns="metric", values="value").sort_values("ndcg@5", ascending=False)
    deltas = metrics[(metrics["task"] == "successful_role_fill_delta") & (metrics["metric"] == "ndcg@5")]
    lines = [
        "# Role Queue Team-Builder Evaluation", "",
        "The finder occupies one queued role. Each evaluation query asks the system to fill one different role using",
        "profiles computed only from earlier matches. Weights are selected on validation matches; the final test window",
        "is evaluated separately.", "",
        f"- Train matches: **{report['split']['train_matches']:,}**",
        f"- Validation matches: **{report['split']['validation_matches']:,}**",
        f"- Test matches: **{report['split']['test_matches']:,}**",
        f"- Successful role-fill test queries: **{report['coverage']['successful']['evaluated_queries']:,}**",
        f"- All observed role-fill test queries: **{report['coverage']['observed']['evaluated_queries']:,}**", "",
        "## Successful future lineup retrieval", "",
        "| Model | NDCG@5 | Recall@5 |", "|---|---:|---:|",
    ]
    for model, row in pivot.iterrows():
        lines.append(f"| {model} | {row['ndcg@5']:.4f} | {row['recall@5']:.4f} |")
    lines.extend(["", "## All observed future role fills", "", "| Model | NDCG@5 | Recall@5 |", "|---|---:|---:|"])
    for model, row in observed_pivot.iterrows():
        lines.append(f"| {model} | {row['ndcg@5']:.4f} | {row['recall@5']:.4f} |")
    lines.extend(["", "## Paired NDCG@5 comparisons", "", "| Comparison | Delta | 95% CI |", "|---|---:|---:|"])
    for row in deltas.sort_values("model").itertuples(index=False):
        lines.append(f"| {row.model} | {row.value:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] |")
    lines.extend([
        "", "## Selected validation weights", "",
        f"`{json.dumps(report['selection']['selected_weights'], sort_keys=True)}`", "",
        "## Scope and limitations", "",
        "- This evaluates ranking real candidates for open role-queue slots.",
        "- Future co-teammates are exposure-based proxy labels; successful-lineup queries additionally require a win.",
        "- The target champion is known query context, while candidate champion history comes only from earlier matches.",
        "- Natural-language preference retrieval requires a separate human-labeled evaluation set.",
    ])
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    event_path: Path = EVENT_PATH,
    minimum_profile_matches: int = 5,
    weight_trials: int = 256,
    validation_queries: int = 2500,
    test_queries: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    events = pd.read_parquet(event_path)
    train, validation, test, validation_cutoff, test_cutoff = temporal_three_way_split(events)
    train_profiles = _profiles(train, minimum_profile_matches)
    weights, selection = tune_role_queue_weights(
        train_profiles, validation, weight_trials, validation_queries, seed
    )
    development = pd.concat((train, validation), ignore_index=True)
    final_profiles = _profiles(development, minimum_profile_matches)
    final_engine = TeamBuilderEngine(final_profiles, weights)
    successful_queries, successful_coverage = build_role_fill_queries(
        final_engine, test, test_queries, seed, successful_only=True
    )
    observed_queries, observed_coverage = build_role_fill_queries(
        final_engine, test, test_queries, seed, successful_only=False
    )
    metrics = pd.DataFrame(
        evaluate_queries(final_engine, successful_queries, "successful_role_fill", seed)
        + evaluate_queries(final_engine, observed_queries, "observed_role_fill", seed)
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    report = {
        "protocol": "role_queue_chronological_train_validation_test",
        "split": {
            "validation_cutoff_utc": pd.to_datetime(validation_cutoff, unit="ms", utc=True).isoformat(),
            "test_cutoff_utc": pd.to_datetime(test_cutoff, unit="ms", utc=True).isoformat(),
            "train_matches": int(train["match_id"].nunique()),
            "validation_matches": int(validation["match_id"].nunique()),
            "test_matches": int(test["match_id"].nunique()),
        },
        "training": {"train_profiles": len(train_profiles), "final_profiles": len(final_profiles)},
        "selection": selection,
        "coverage": {"successful": successful_coverage, "observed": observed_coverage},
        "artifacts": {"metrics": str(METRICS_PATH), "portfolio_report": str(PORTFOLIO_PATH)},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report(report, metrics)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate role-queue team-slot recommendation on future matches")
    parser.add_argument("--source", type=Path, default=DATA_DIR / "matchData.csv")
    parser.add_argument("--events", type=Path, default=EVENT_PATH)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--rebuild-events", action="store_true")
    parser.add_argument("--minimum-profile-matches", type=int, default=5)
    parser.add_argument("--weight-trials", type=int, default=256)
    parser.add_argument("--validation-queries", type=int, default=2500)
    parser.add_argument("--test-queries", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rebuild_events or not args.events.exists():
        print(build_compact_event_table(args.source, args.events, args.chunk_size), flush=True)
    report = run_evaluation(
        args.events, args.minimum_profile_matches, args.weight_trials,
        args.validation_queries, args.test_queries, args.seed,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
