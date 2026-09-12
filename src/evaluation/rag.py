from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CHROMA_DIR, PROFILE_PATH, PROJECT_ROOT
from src.rag.vector_store import RealPlayerVectorStore
from src.recsys.two_tower import ROLES


REPORT_PATH = PROJECT_ROOT / "docs" / "rag_evaluation.md"
RESULT_PATH = PROJECT_ROOT / "artifacts" / "two_tower" / "rag_evaluation.json"
TOP_K = 10


def ranking_metrics(ranked_ids: list[str], positives: set[str], k: int = TOP_K) -> dict[str, float]:
    ranked = ranked_ids[:k]
    hits = [player_id in positives for player_id in ranked]
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(positives), k)))
    first = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    return {
        "precision": sum(hits) / k,
        "recall": sum(hits) / len(positives),
        "ndcg": dcg / ideal if ideal else 0.0,
        "hit_rate": float(any(hits)),
        "mrr": 1.0 / first if first else 0.0,
    }


def build_retrieval_cases(profiles: pd.DataFrame) -> list[dict[str, Any]]:
    """Create controlled text queries whose labels come only from recorded profile facts."""
    frame = profiles.copy()
    frame["summoner_id"] = frame["summoner_id"].astype(str)
    frame["role"] = frame["role"].fillna("").astype(str).str.upper()
    cases: list[dict[str, Any]] = []
    templates = {
        "experience": ("matches", "experienced {role} teammate with many recorded matches"),
        "vision": ("avg_vision_score", "{role} teammate with excellent vision and map awareness"),
        "damage": ("avg_damage_dealt", "high damage {role} teammate"),
        "kda": ("kda", "{role} teammate with a strong KDA"),
        "win_rate": ("win_rate", "reliable {role} teammate with a high win rate"),
        "assists": ("avg_assists", "team-oriented {role} player with many assists"),
    }
    for role in ROLES:
        role_frame = frame[frame["role"] == role]
        if role_frame.empty:
            continue
        for category, (column, template) in templates.items():
            values = pd.to_numeric(role_frame[column], errors="coerce")
            threshold = float(values.quantile(0.80))
            positives = set(role_frame.loc[values >= threshold, "summoner_id"])
            if positives:
                cases.append({
                    "category": category,
                    "role": role,
                    "query": template.format(role=role.lower()),
                    "positives": positives,
                })
        champion_totals: defaultdict[str, float] = defaultdict(float)
        for champions in role_frame["top_champions"]:
            if isinstance(champions, dict):
                for champion, games in champions.items():
                    champion_totals[str(champion)] += float(games)
        for champion, _ in sorted(champion_totals.items(), key=lambda item: -item[1])[:2]:
            positives = set(role_frame.loc[role_frame["top_champions"].map(
                lambda value: isinstance(value, dict) and champion in value
            ), "summoner_id"])
            cases.append({
                "category": "champion",
                "role": role,
                "query": f"{role.lower()} teammate experienced on {champion}",
                "positives": positives,
            })
    return cases


def bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(1000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run_evaluation(seed: int = 42) -> dict[str, Any]:
    profiles = pd.read_json(PROFILE_PATH, lines=True)
    profiles["summoner_id"] = profiles["summoner_id"].astype(str)
    documents = profiles["rag_document"].fillna("").astype(str).tolist()
    profile_ids = profiles["summoner_id"].tolist()
    store = RealPlayerVectorStore(CHROMA_DIR)
    if store.requires_rebuild(profiles):
        store.rebuild(profiles)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit(documents)
    document_matrix = tfidf.transform(documents)
    cases = build_retrieval_cases(profiles)
    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list)
        for name in ("semantic_chroma", "hybrid_chroma", "tfidf", "random")
    }
    category_ndcg: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        positives = set(case["positives"])
        semantic = store.search_teammates(
            case["query"], min(250, len(profiles)), role=case["role"], use_structured=False
        )
        hybrid = store.search_teammates(
            case["query"], min(250, len(profiles)), role=case["role"]
        )
        query_vector = tfidf.transform([case["query"]])
        tfidf_scores = (document_matrix @ query_vector.T).toarray().ravel()
        role_mask = profiles["role"].fillna("").astype(str).str.upper().to_numpy() == case["role"]
        role_indices = np.flatnonzero(role_mask)
        tfidf_order = role_indices[np.argsort(-tfidf_scores[role_indices], kind="stable")]
        random_order = rng.permutation(role_indices)
        rankings = {
            "semantic_chroma": [str(item["summoner_id"]) for item in semantic],
            "hybrid_chroma": [str(item["summoner_id"]) for item in hybrid],
            "tfidf": [profile_ids[index] for index in tfidf_order],
            "random": [profile_ids[index] for index in random_order],
        }
        for name, ranking in rankings.items():
            metrics = ranking_metrics(ranking, positives)
            for metric, value in metrics.items():
                values[name][metric].append(value)
            if name == "hybrid_chroma":
                category_ndcg[case["category"]].append(metrics["ndcg"])
    models: dict[str, Any] = {}
    for name, metrics in values.items():
        models[name] = {}
        for metric, samples in metrics.items():
            low, high = bootstrap_ci(samples, seed)
            models[name][metric] = {
                "value": float(np.mean(samples)), "ci_low": low, "ci_high": high,
            }
    report = {
        "protocol": "controlled_real_profile_retrieval",
        "profiles": len(profiles),
        "queries": len(cases),
        "label_source": "recorded real profile fields; top-20-percent role thresholds",
        "embedding_model": store.embedding_model_name,
        "models": models,
        "hybrid_ndcg_by_category": {
            name: float(np.mean(samples)) for name, samples in sorted(category_ndcg.items())
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report)
    return report


def _metric(report: dict[str, Any], model: str, name: str) -> str:
    value = report["models"][model][name]
    return f"{value['value']:.4f} [{value['ci_low']:.4f}, {value['ci_high']:.4f}]"


def write_report(report: dict[str, Any]) -> None:
    lines = [
        "# RAG Retrieval Evaluation", "",
        "This benchmark is separate from two-tower teammate prediction. Queries are authored from real profile",
        "attributes, and relevance labels come from recorded role-relative statistics or champion history.", "",
        f"- Real profiles: **{report['profiles']:,}**",
        f"- Controlled queries: **{report['queries']}**",
        f"- Embedding model: **{report['embedding_model']}**", "",
        "| Retriever | NDCG@10 (95% CI) | Precision@10 | Hit rate@10 |",
        "|---|---:|---:|---:|",
    ]
    for model in ("semantic_chroma", "hybrid_chroma", "tfidf", "random"):
        lines.append(
            f"| {model} | {_metric(report, model, 'ndcg')} | "
            f"{report['models'][model]['precision']['value']:.4f} | "
            f"{report['models'][model]['hit_rate']['value']:.4f} |"
        )
    lines.extend(["", "## Grounded hybrid retrieval by query type", "", "| Type | NDCG@10 |", "|---|---:|"])
    for category, value in report["hybrid_ndcg_by_category"].items():
        lines.append(f"| {category} | {value:.4f} |")
    lines.extend([
        "", "## Limitations", "",
        "- Labels are deterministic judgments derived from real profile fields, not human preference ratings.",
        "- This evaluates retrieval, not the OpenAI-generated scout prose.",
        "- Scout quality still requires a human rubric for grounding, usefulness, and unsupported claims.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = run_evaluation()
    print(json.dumps({
        "profiles": report["profiles"],
        "queries": report["queries"],
        "hybrid_ndcg_at_10": report["models"]["hybrid_chroma"]["ndcg"]["value"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
