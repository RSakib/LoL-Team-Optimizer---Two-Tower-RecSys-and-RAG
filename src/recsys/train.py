from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import PROJECT_ROOT, TWO_TOWER_METADATA_PATH, TWO_TOWER_MODEL_PATH
from src.data.ingestion import build_player_profiles
from src.evaluation.events import EVENT_PATH, temporal_three_way_split
from src.recsys.two_tower import (
    TIER_ORDER,
    TwoTowerModel,
    build_metadata,
    candidate_tensors,
    save_artifact,
)


REPORT_PATH = PROJECT_ROOT / "docs" / "two_tower_evaluation.md"
EVALUATION_PATH = TWO_TOWER_MODEL_PATH.parent / "evaluation.json"
HISTORY_PATH = TWO_TOWER_MODEL_PATH.parent / "training_history.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_positive_pairs(events: pd.DataFrame, profile_ids: set[str]) -> pd.DataFrame:
    """Build directed teammate pairs from real players recorded on the same team."""
    columns = [
        "match_id", "team_id", "summoner_id", "role", "tier", "rank",
        "champion_name", "win",
    ]
    members = events[columns].dropna(
        subset=["match_id", "team_id", "summoner_id", "role", "champion_name"]
    ).copy()
    members["summoner_id"] = members["summoner_id"].astype(str)
    members["role"] = members["role"].astype(str).str.upper()
    members = members[members["summoner_id"].isin(profile_ids)]
    members = members.drop_duplicates(["match_id", "team_id", "summoner_id"])
    paired = members.merge(
        members,
        on=["match_id", "team_id"],
        how="inner",
        suffixes=("_finder", "_candidate"),
    )
    paired = paired[
        (paired["summoner_id_finder"] != paired["summoner_id_candidate"])
        & (paired["role_finder"] != paired["role_candidate"])
    ]
    return pd.DataFrame({
        "match_id": paired["match_id"].astype(str),
        "team_id": paired["team_id"].astype(int),
        "finder_id": paired["summoner_id_finder"].astype(str),
        "finder_role": paired["role_finder"],
        "finder_tier": paired["tier_finder"],
        "finder_division": paired["rank_finder"],
        "target_role": paired["role_candidate"],
        "target_champion": paired["champion_name_candidate"].astype(str),
        "candidate_id": paired["summoner_id_candidate"].astype(str),
        "won": paired["win_candidate"].astype(bool),
    }).reset_index(drop=True)


class PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, profiles: pd.DataFrame, metadata: dict[str, Any], seed: int):
        self.pairs = pairs.reset_index(drop=True)
        self.records = self.pairs.to_dict("records")
        self.profiles = profiles.reset_index(drop=True)
        self.metadata = metadata
        self.seed = seed
        self.epoch = 0
        self.player_row = {
            str(player_id): index for index, player_id in enumerate(self.profiles["summoner_id"])
        }
        self.role_rows = {
            str(role): group.index.to_numpy(dtype=np.int64)
            for role, group in self.profiles.groupby(self.profiles["role"].astype(str).str.upper())
        }
        self.positive_rows = np.asarray(
            [self.player_row[str(value)] for value in self.pairs["candidate_id"]], dtype=np.int64
        )
        self.finder_rows = np.asarray(
            [self.player_row.get(str(value), -1) for value in self.pairs["finder_id"]], dtype=np.int64
        )
        self.pair_rows_by_role = {
            str(role): group.index.to_numpy(dtype=np.int64)
            for role, group in self.pairs.groupby("target_role")
        }
        self.negative_rows = np.zeros(len(self.pairs), dtype=np.int64)
        self.resample_negatives()

    def resample_negatives(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        for role, pair_rows in self.pair_rows_by_role.items():
            choices = self.role_rows[role]
            if len(choices) < 2:
                raise RuntimeError(f"No real negative candidate exists for role {role}")
            draws = rng.choice(choices, size=len(pair_rows), replace=True)
            invalid = (
                (draws == self.positive_rows[pair_rows])
                | (draws == self.finder_rows[pair_rows])
            )
            while invalid.any():
                draws[invalid] = rng.choice(choices, size=int(invalid.sum()), replace=True)
                invalid = (
                    (draws == self.positive_rows[pair_rows])
                    | (draws == self.finder_rows[pair_rows])
                )
            self.negative_rows[pair_rows] = draws
        self.epoch += 1

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, int | float]:
        row = self.records[index]
        vocabs = self.metadata["vocabs"]

        def lookup(name: str, value: Any, default: str = "<UNK>") -> int:
            vocab = vocabs[name]
            if value is None or pd.isna(value):
                return vocab[default]
            text = str(value).strip()
            return vocab.get(text.upper(), vocab.get(text, vocab[default]))

        champion = lookup("champion", row["target_champion"])
        # Mask some champion context so the trained query tower supports the UI's optional field.
        if ((index * 2654435761 + self.epoch + self.seed) % 10) < 2:
            champion = vocabs["champion"]["<ANY>"]
        return {
            "finder_role": lookup("role", row["finder_role"]),
            "finder_tier": lookup("tier", row["finder_tier"]),
            "finder_division": lookup("division", row["finder_division"]),
            "target_role": lookup("role", row["target_role"]),
            "target_champion": champion,
            "positive_row": self.player_row[str(row["candidate_id"])],
            "negative_row": int(self.negative_rows[index]),
            "weight": 1.25 if bool(row["won"]) else 1.0,
        }


def _candidate_slice(tensors: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: values[rows] for name, values in tensors.items()}


def train_epochs(
    profiles: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
) -> tuple[TwoTowerModel, list[dict[str, float]]]:
    seed_everything(seed)
    model = TwoTowerModel(metadata).to(device)
    candidate_features = candidate_tensors(profiles, metadata, device)
    dataset = PairDataset(pairs, profiles, metadata, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        dataset.resample_negatives()
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            query = model.encode_query(
                batch["finder_role"], batch["finder_tier"], batch["finder_division"],
                batch["target_role"], batch["target_champion"],
            )
            positive = _candidate_slice(candidate_features, batch["positive_row"])
            negative = _candidate_slice(candidate_features, batch["negative_row"])
            positive_embedding = model.encode_candidate(**positive)
            negative_embedding = model.encode_candidate(**negative)
            positive_score = (query * positive_embedding).sum(dim=-1)
            negative_score = (query * negative_embedding).sum(dim=-1)
            loss = (F.softplus((negative_score - positive_score) / 0.10) * batch["weight"].float()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(positive_score)
            total_examples += len(positive_score)
        epoch_loss = total_loss / max(total_examples, 1)
        history.append({"epoch": float(epoch), "training_loss": epoch_loss})
        print(f"epoch={epoch} pairwise_loss={epoch_loss:.6f}", flush=True)
    return model, history


def _metric_summary(values: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(500, len(array)), replace=True).mean(axis=1)
    return {
        "value": float(array.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


@torch.inference_mode()
def evaluate_model(
    model: TwoTowerModel,
    metadata: dict[str, Any],
    profiles: pd.DataFrame,
    events: pd.DataFrame,
    device: str,
    max_queries: int,
    seed: int,
    successful_only: bool,
    include_target_champion: bool = True,
) -> dict[str, Any]:
    source = events[events["win"].astype(bool)] if successful_only else events
    pairs = build_positive_pairs(source, set(profiles["summoner_id"].astype(str)))
    if pairs.empty:
        raise RuntimeError("No real future teammate pairs were eligible for evaluation")
    if len(pairs) > max_queries:
        pairs = pairs.sample(n=max_queries, random_state=seed).reset_index(drop=True)
    model.eval()
    features = candidate_tensors(profiles, metadata, device)
    embeddings = model.encode_candidate(**features).detach().cpu().numpy()
    profile_ids = profiles["summoner_id"].astype(str).to_numpy()
    profile_roles = profiles["role"].fillna("").astype(str).str.upper().to_numpy()
    profile_tiers = profiles["tier"].fillna("").astype(str).str.upper().to_numpy()
    popularity = np.log1p(pd.to_numeric(profiles["matches"], errors="coerce").fillna(0).to_numpy())
    row_by_id = {player_id: index for index, player_id in enumerate(profile_ids)}
    vocabs = metadata["vocabs"]
    rng = np.random.default_rng(seed)
    collected: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("two_tower", "popularity", "random")
    }

    def lookup(name: str, value: Any, default: str = "<UNK>") -> int:
        vocab = vocabs[name]
        if value is None or pd.isna(value):
            return vocab[default]
        text = str(value).strip()
        return vocab.get(text.upper(), vocab.get(text, vocab[default]))

    evaluated = 0
    for row in pairs.itertuples(index=False):
        positive_row = row_by_id.get(str(row.candidate_id))
        if positive_row is None:
            continue
        eligible = np.flatnonzero(profile_roles == str(row.target_role).upper())
        finder_tier = TIER_ORDER.get(str(row.finder_tier).upper())
        if finder_tier is not None:
            eligible = np.asarray([
                index for index in eligible
                if profile_tiers[index] in TIER_ORDER
                and abs(TIER_ORDER[profile_tiers[index]] - finder_tier) <= 1
            ], dtype=np.int64)
        eligible = eligible[profile_ids[eligible] != str(row.finder_id)]
        if len(eligible) < 2 or not np.any(eligible == positive_row):
            continue
        query = model.encode_query(
            torch.tensor([lookup("role", row.finder_role)], device=device),
            torch.tensor([lookup("tier", row.finder_tier)], device=device),
            torch.tensor([lookup("division", row.finder_division)], device=device),
            torch.tensor([lookup("role", row.target_role)], device=device),
            torch.tensor([
                lookup("champion", row.target_champion)
                if include_target_champion else vocabs["champion"]["<ANY>"]
            ], device=device),
        ).detach().cpu().numpy()[0]
        scores = {
            "two_tower": embeddings[eligible] @ query,
            "popularity": popularity[eligible],
            "random": rng.random(len(eligible)),
        }
        for name, values in scores.items():
            order = eligible[np.argsort(-values, kind="stable")]
            position = next((rank for rank, index in enumerate(order, 1) if index == positive_row), None)
            for k in (5, 10):
                hit = float(position is not None and position <= k)
                collected[name][f"recall@{k}"].append(hit)
                collected[name][f"ndcg@{k}"].append(
                    1.0 / math.log2(position + 1) if hit and position is not None else 0.0
                )
            collected[name]["mrr"].append(1.0 / position if position else 0.0)
        evaluated += 1
    if evaluated == 0:
        raise RuntimeError("No future teammate remained in an eligible real candidate pool")
    return {
        "queries": evaluated,
        "models": {
            name: {
                metric: _metric_summary(values, seed)
                for metric, values in metrics.items()
            }
            for name, metrics in collected.items()
        },
    }


def _format_metric(result: dict[str, Any], model: str, metric: str) -> str:
    value = result["models"][model][metric]
    return f"{value['value']:.4f} [{value['ci_low']:.4f}, {value['ci_high']:.4f}]"


def write_report(report: dict[str, Any]) -> None:
    successful = report["test"]["successful_with_target_champion"]
    successful_anonymous = report["test"]["successful_role_rank_only"]
    observed = report["test"]["all_with_target_champion"]
    observed_anonymous = report["test"]["all_role_rank_only"]
    lines = [
        "# Trained Two-Tower Evaluation", "",
        "The query tower encodes the anonymous finder's role/rank plus the requested teammate role and optional",
        "champion. The candidate tower encodes each real player's identity, role/rank, main champion, and normalized",
        "historical metrics. Training uses real same-team pairs with role-matched sampled negatives.", "",
        "## Leakage controls", "",
        f"- Train matches: **{report['split']['train_matches']:,}**",
        f"- Validation matches: **{report['split']['validation_matches']:,}**",
        f"- Test matches: **{report['split']['test_matches']:,}**",
        "- Split is chronological at match level.",
        "- Early-stopping epoch is selected on validation NDCG@10 only.",
        "- The deployed checkpoint is retrained on train + validation and evaluated once on the held-out test window.", "",
        "## Held-out successful future teammates with target champion context", "",
        f"Queries: **{successful['queries']:,}**", "",
        "| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("two_tower", "popularity", "random"):
        lines.append(
            f"| {name} | {_format_metric(successful, name, 'ndcg@10')} | "
            f"{_format_metric(successful, name, 'recall@10')} | {_format_metric(successful, name, 'mrr')} |"
        )
    lines.extend([
        "", "## Held-out successful future teammates with role/rank context only", "",
        f"Queries: **{successful_anonymous['queries']:,}**", "",
        "| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |",
        "|---|---:|---:|---:|",
    ])
    for name in ("two_tower", "popularity", "random"):
        lines.append(
            f"| {name} | {_format_metric(successful_anonymous, name, 'ndcg@10')} | "
            f"{_format_metric(successful_anonymous, name, 'recall@10')} | "
            f"{_format_metric(successful_anonymous, name, 'mrr')} |"
        )
    lines.extend([
        "", "## Held-out all future teammates with target champion context", "",
        f"Queries: **{observed['queries']:,}**", "",
        "| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |",
        "|---|---:|---:|---:|",
    ])
    for name in ("two_tower", "popularity", "random"):
        lines.append(
            f"| {name} | {_format_metric(observed, name, 'ndcg@10')} | "
            f"{_format_metric(observed, name, 'recall@10')} | {_format_metric(observed, name, 'mrr')} |"
        )
    lines.extend([
        "", "## Held-out all future teammates with role/rank context only", "",
        f"Queries: **{observed_anonymous['queries']:,}**", "",
        "| Model | NDCG@10 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) |",
        "|---|---:|---:|---:|",
    ])
    for name in ("two_tower", "popularity", "random"):
        lines.append(
            f"| {name} | {_format_metric(observed_anonymous, name, 'ndcg@10')} | "
            f"{_format_metric(observed_anonymous, name, 'recall@10')} | "
            f"{_format_metric(observed_anonymous, name, 'mrr')} |"
        )
    lines.extend([
        "", "## Training", "",
        f"- Real directed training pairs: **{report['training']['train_pairs']:,}**",
        f"- Real directed final-training pairs: **{report['training']['development_pairs']:,}**",
        f"- Selected epochs: **{report['training']['selected_epochs']}**",
        f"- Device: **{report['training']['device']}**",
        "- Objective: pairwise logistic ranking loss with same-role real-player negatives.", "",
        "## RAG boundary", "",
        "The two-tower checkpoint is the recommendation model. Chroma/Sentence Transformer retrieval is evaluated",
        "separately and participates only when a user supplies a natural-language preference. Candidate lists are",
        "combined with weighted reciprocal-rank fusion; the LLM scout receives the selected candidate's exact indexed",
        "evidence. The LLM is not the ranker.", "",
        "## Limitations", "",
        "- Co-occurrence is implicit feedback, not proof that two players intentionally chose one another.",
        "- Unobserved pairs are sampled negatives, not verified incompatibilities.",
        "- The anonymous finder context limits personalization to role, rank, champion request, and RAG preference.",
        "- A human-rated benchmark is still required for generated scout-report quality.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    event_path: Path = EVENT_PATH,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    validation_queries: int = 1500,
    test_queries: int = 5000,
    patience: int = 3,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    events = pd.read_parquet(event_path)
    train, validation, test, validation_cutoff, test_cutoff = temporal_three_way_split(events)
    train_profiles = build_player_profiles(train)
    train_ids = set(train_profiles["summoner_id"].astype(str))
    train_pairs = build_positive_pairs(train, train_ids)
    train_metadata = build_metadata(
        train_profiles, embedding_dim, hidden_dim,
        extra_champions=train["champion_name"].dropna().astype(str).tolist(),
    )
    seed_everything(seed)
    model = TwoTowerModel(train_metadata).to(device)
    candidate_features = candidate_tensors(train_profiles, train_metadata, device)
    dataset = PairDataset(train_pairs, train_profiles, train_metadata, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_metric = float("-inf")
    best_epoch = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        dataset.resample_negatives()
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            query = model.encode_query(
                batch["finder_role"], batch["finder_tier"], batch["finder_division"],
                batch["target_role"], batch["target_champion"],
            )
            positive = model.encode_candidate(**_candidate_slice(candidate_features, batch["positive_row"]))
            negative = model.encode_candidate(**_candidate_slice(candidate_features, batch["negative_row"]))
            positive_score = (query * positive).sum(dim=-1)
            negative_score = (query * negative).sum(dim=-1)
            loss = (F.softplus((negative_score - positive_score) / 0.10) * batch["weight"].float()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(positive_score)
            total_examples += len(positive_score)
        loss_value = total_loss / max(total_examples, 1)
        validation_result = evaluate_model(
            model, train_metadata, train_profiles, validation, device,
            validation_queries, seed, successful_only=False,
        )
        validation_ndcg = validation_result["models"]["two_tower"]["ndcg@10"]["value"]
        history.append({
            "epoch": epoch, "training_loss": loss_value, "validation_ndcg_at_10": validation_ndcg,
        })
        print(
            f"epoch={epoch} pairwise_loss={loss_value:.6f} validation_ndcg@10={validation_ndcg:.4f}",
            flush=True,
        )
        if validation_ndcg > best_metric + 1e-4:
            best_metric = validation_ndcg
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    development = pd.concat((train, validation), ignore_index=True)
    development_profiles = build_player_profiles(development)
    development_ids = set(development_profiles["summoner_id"].astype(str))
    development_pairs = build_positive_pairs(development, development_ids)
    deployment_metadata = build_metadata(
        development_profiles, embedding_dim, hidden_dim,
        extra_champions=development["champion_name"].dropna().astype(str).tolist(),
    )
    deployment_model, final_history = train_epochs(
        development_profiles, development_pairs, deployment_metadata, best_epoch,
        batch_size, learning_rate, device, seed,
    )
    successful_champion = evaluate_model(
        deployment_model, deployment_metadata, development_profiles, test,
        device, test_queries, seed, successful_only=True, include_target_champion=True,
    )
    successful_anonymous = evaluate_model(
        deployment_model, deployment_metadata, development_profiles, test,
        device, test_queries, seed, successful_only=True, include_target_champion=False,
    )
    observed_champion = evaluate_model(
        deployment_model, deployment_metadata, development_profiles, test,
        device, test_queries, seed, successful_only=False, include_target_champion=True,
    )
    observed_anonymous = evaluate_model(
        deployment_model, deployment_metadata, development_profiles, test,
        device, test_queries, seed, successful_only=False, include_target_champion=False,
    )
    deployment_metadata["training"] = {
        "seed": seed,
        "selected_epochs": best_epoch,
        "train_pairs": len(train_pairs),
        "development_pairs": len(development_pairs),
        "trained_through_timestamp": int(development["timestamp"].max()),
        "validation_cutoff_utc": pd.to_datetime(validation_cutoff, unit="ms", utc=True).isoformat(),
        "test_cutoff_utc": pd.to_datetime(test_cutoff, unit="ms", utc=True).isoformat(),
    }
    save_artifact(deployment_model, deployment_metadata, TWO_TOWER_MODEL_PATH, TWO_TOWER_METADATA_PATH)
    report = {
        "protocol": "chronological_two_tower_pairwise_ranking",
        "split": {
            "train_matches": int(train["match_id"].nunique()),
            "validation_matches": int(validation["match_id"].nunique()),
            "test_matches": int(test["match_id"].nunique()),
        },
        "training": {
            "device": device,
            "train_profiles": len(train_profiles),
            "development_profiles": len(development_profiles),
            "train_pairs": len(train_pairs),
            "development_pairs": len(development_pairs),
            "selected_epochs": best_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "validation_history": history,
        "test": {
            "successful_with_target_champion": successful_champion,
            "successful_role_rank_only": successful_anonymous,
            "all_with_target_champion": observed_champion,
            "all_role_rank_only": observed_anonymous,
        },
    }
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps({"selection": history, "final": final_history}, indent=2), encoding="utf-8")
    write_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the real-data LoL two-tower recommender")
    parser.add_argument("--events", type=Path, default=EVENT_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--validation-queries", type=int, default=1500)
    parser.add_argument("--test-queries", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_training(
        event_path=args.events,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        validation_queries=args.validation_queries,
        test_queries=args.test_queries,
        patience=args.patience,
        device_name=args.device,
        seed=args.seed,
    )
    summary = {
        "model_path": str(TWO_TOWER_MODEL_PATH),
        "metadata_path": str(TWO_TOWER_METADATA_PATH),
        "device": report["training"]["device"],
        "selected_epochs": report["training"]["selected_epochs"],
        "successful_with_champion_ndcg_at_10": report["test"]["successful_with_target_champion"]["models"]["two_tower"]["ndcg@10"]["value"],
        "successful_role_rank_only_ndcg_at_10": report["test"]["successful_role_rank_only"]["models"]["two_tower"]["ndcg@10"]["value"],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
