from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    PROJECT_ROOT,
    TEAM_MODEL_METADATA_PATH,
    TEAM_MODEL_PATH,
    TWO_TOWER_METADATA_PATH,
    TWO_TOWER_MODEL_PATH,
)
from src.data.ingestion import build_player_profiles
from src.data.profile_documents import enrich_profiles_for_rag
from src.evaluation.events import EVENT_PATH, temporal_three_way_split
from src.recsys.team_model import (
    TeamLineupModel,
    artifact_sha256,
    build_lineup_feature_bank,
    build_observed_lineups,
    build_team_metadata,
    complete_team_counts,
    lineup_tensors,
    save_team_artifact,
)
from src.recsys.train import build_positive_pairs, resolve_device, seed_everything, train_epochs
from src.recsys.two_tower import LoadedTwoTower, ROLES, build_metadata


REPORT_PATH = PROJECT_ROOT / "docs" / "team_model_evaluation.md"
EVALUATION_PATH = TEAM_MODEL_PATH.parent / "evaluation.json"
HISTORY_PATH = TEAM_MODEL_PATH.parent / "training_history.json"


def _lineup_dict(row: Any) -> dict[str, str]:
    return {
        role: str(getattr(row, f"player_{role}"))
        for role in ROLES
        if role != str(row.finder_role)
    }


def _example_tensors(
    examples: pd.DataFrame,
    bank,
    two_tower_metadata: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rows = list(examples.itertuples(index=False))
    tensors = lineup_tensors(
        [_lineup_dict(row) for row in rows],
        [str(row.finder_role) for row in rows],
        [row.finder_tier for row in rows],
        [row.finder_division for row in rows],
        bank,
        two_tower_metadata,
        device,
    )
    labels = torch.tensor(examples["won"].astype(bool).to_numpy(), dtype=torch.float32, device=device)
    return tensors, labels


def _loader(tensors: dict[str, torch.Tensor], labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        tensors["player_features"], tensors["present_mask"], tensors["finder_role"],
        tensors["finder_tier"], tensors["finder_division"], labels,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _batch_loss(
    performance_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    labels: torch.Tensor,
    pair_loss_weight: float,
) -> torch.Tensor:
    performance_loss = F.binary_cross_entropy_with_logits(performance_logits, labels)
    if pair_loss_weight <= 0:
        return performance_loss
    pair_labels = labels.unsqueeze(1).expand_as(pair_logits)
    pair_loss = F.binary_cross_entropy_with_logits(pair_logits[pair_mask], pair_labels[pair_mask])
    return performance_loss + pair_loss_weight * pair_loss


@torch.inference_mode()
def _predict(model: TeamLineupModel, tensors: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    performance, pairs, pair_mask = model(**tensors)
    probabilities = torch.sigmoid(performance).detach().cpu().numpy()
    pair_probabilities = torch.sigmoid(pairs)
    compatibility = (
        (pair_probabilities * pair_mask.float()).sum(dim=1)
        / pair_mask.float().sum(dim=1).clamp_min(1.0)
    ).detach().cpu().numpy()
    return probabilities, compatibility


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise RuntimeError("AUC requires both recorded winning and losing real teams")
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1 - 1e-6)
    target = labels.astype(np.float64)
    return {
        "roc_auc": _auc(labels, scores),
        "log_loss": float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean()),
        "brier_score": float(np.square(clipped - target).mean()),
        "accuracy_at_0_5": float(((clipped >= 0.5) == labels.astype(bool)).mean()),
    }


def _bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    match_ids: np.ndarray,
    seed: int,
) -> dict[str, float]:
    value = _auc(labels, scores)
    rng = np.random.default_rng(seed)
    unique_matches = np.unique(match_ids)
    indices_by_match = {
        match_id: np.flatnonzero(match_ids == match_id)
        for match_id in unique_matches
    }
    estimates: list[float] = []
    for _ in range(500):
        sampled_matches = rng.choice(unique_matches, size=len(unique_matches), replace=True)
        indices = np.concatenate([indices_by_match[match_id] for match_id in sampled_matches])
        sampled_labels = labels[indices]
        if sampled_labels.astype(bool).all() or (~sampled_labels.astype(bool)).all():
            continue
        estimates.append(_auc(sampled_labels, scores[indices]))
    return {
        "value": value,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def _profile_baseline(examples: pd.DataFrame, profiles: pd.DataFrame) -> np.ndarray:
    win_rate = {
        str(row.summoner_id): float(row.win_rate)
        for row in profiles[["summoner_id", "win_rate"]].itertuples(index=False)
        if not pd.isna(row.win_rate)
    }
    values: list[float] = []
    for row in examples.itertuples(index=False):
        rates = [win_rate.get(player_id) for player_id in _lineup_dict(row).values()]
        if any(value is None for value in rates):
            raise RuntimeError("A real lineup member lacks the historical win-rate baseline")
        values.append(float(np.mean(rates)))
    return np.asarray(values, dtype=np.float64)


def _team_level_frame(
    examples: pd.DataFrame,
    performance: np.ndarray,
    compatibility: np.ndarray,
    baseline: np.ndarray,
) -> pd.DataFrame:
    frame = examples[["match_id", "team_id", "won"]].copy()
    frame["performance"] = performance
    frame["compatibility"] = compatibility
    frame["profile_baseline"] = baseline
    return frame.groupby(["match_id", "team_id"], as_index=False).agg(
        won=("won", "first"),
        performance=("performance", "mean"),
        compatibility=("compatibility", "mean"),
        profile_baseline=("profile_baseline", "mean"),
    )


def _winner_ranking(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    outcomes: list[float] = []
    for _, match in frame.groupby("match_id"):
        if len(match) != 2 or match["won"].astype(bool).sum() != 1:
            continue
        winner = float(match.loc[match["won"].astype(bool), score_column].iloc[0])
        loser = float(match.loc[~match["won"].astype(bool), score_column].iloc[0])
        outcomes.append(1.0 if winner > loser else 0.5 if winner == loser else 0.0)
    return {"matches": len(outcomes), "winner_rank_accuracy": float(np.mean(outcomes)) if outcomes else 0.0}


def evaluate_team_model(
    model: TeamLineupModel,
    examples: pd.DataFrame,
    bank,
    two_tower_metadata: dict[str, Any],
    profiles: pd.DataFrame,
    device: str,
    seed: int,
    prior_probability: float,
) -> dict[str, Any]:
    if examples.empty:
        raise RuntimeError("No complete real lineups are available for team-model evaluation")
    tensors, _ = _example_tensors(examples, bank, two_tower_metadata, device)
    performance, compatibility = _predict(model, tensors)
    baseline = _profile_baseline(examples, profiles)
    teams = _team_level_frame(examples, performance, compatibility, baseline)
    labels = teams["won"].astype(bool).to_numpy()
    if not 0.0 < prior_probability < 1.0:
        raise RuntimeError("The development-set outcome prior must be between zero and one")
    prior = np.full(len(teams), float(prior_probability), dtype=np.float64)
    teams["constant_prior"] = prior
    result: dict[str, Any] = {"teams": len(teams), "scenarios": len(examples), "models": {}}
    for name, scores in (
        ("team_model", teams["performance"].to_numpy()),
        ("learned_pair_compatibility", teams["compatibility"].to_numpy()),
        ("mean_historical_win_rate", teams["profile_baseline"].to_numpy()),
        ("constant_prior", prior),
    ):
        metrics = _binary_metrics(labels, scores)
        metrics["roc_auc"] = _bootstrap_auc(
            labels,
            scores,
            teams["match_id"].astype(str).to_numpy(),
            seed,
        )
        metrics["observed_winner_ranking"] = _winner_ranking(teams, {
            "team_model": "performance",
            "learned_pair_compatibility": "compatibility",
            "mean_historical_win_rate": "profile_baseline",
            "constant_prior": "constant_prior",
        }[name])
        result["models"][name] = metrics
    return result


def train_team_epochs(
    examples: pd.DataFrame,
    bank,
    two_tower_metadata: dict[str, Any],
    metadata: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pair_loss_weight: float,
    device: str,
    seed: int,
) -> tuple[TeamLineupModel, list[dict[str, float]]]:
    seed_everything(seed)
    model = TeamLineupModel(metadata).to(device)
    tensors, labels = _example_tensors(examples, bank, two_tower_metadata, device)
    loader = _loader(tensors, labels, batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for player_features, present_mask, finder_role, finder_tier, finder_division, target in loader:
            performance, pairs, pair_mask = model(
                player_features, present_mask, finder_role, finder_tier, finder_division
            )
            loss = _batch_loss(performance, pairs, pair_mask, target, pair_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_examples += len(target)
        value = total_loss / max(total_examples, 1)
        history.append({"epoch": epoch, "training_loss": value})
        print(f"team_epoch={epoch} loss={value:.6f}", flush=True)
    return model, history


def _selection_train(
    train_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
    train_bank,
    two_tower_metadata: dict[str, Any],
    profiles: pd.DataFrame,
    metadata: dict[str, Any],
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    pair_loss_weight: float,
    device: str,
    seed: int,
) -> tuple[int, list[dict[str, float]]]:
    seed_everything(seed)
    model = TeamLineupModel(metadata).to(device)
    train_tensors, train_labels = _example_tensors(train_examples, train_bank, two_tower_metadata, device)
    validation_tensors, _ = _example_tensors(validation_examples, train_bank, two_tower_metadata, device)
    loader = _loader(train_tensors, train_labels, batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_auc = float("-inf")
    best_epoch = 1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for player_features, present_mask, finder_role, finder_tier, finder_division, target in loader:
            performance, pairs, pair_mask = model(
                player_features, present_mask, finder_role, finder_tier, finder_division
            )
            loss = _batch_loss(performance, pairs, pair_mask, target, pair_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_examples += len(target)
        validation_performance, _ = _predict(model, validation_tensors)
        validation_baseline = _profile_baseline(validation_examples, profiles)
        validation_teams = _team_level_frame(
            validation_examples, validation_performance, validation_performance, validation_baseline
        )
        validation_auc = _auc(
            validation_teams["won"].astype(bool).to_numpy(),
            validation_teams["performance"].to_numpy(),
        )
        loss_value = total_loss / max(total_examples, 1)
        history.append({"epoch": epoch, "training_loss": loss_value, "validation_team_auc": validation_auc})
        print(
            f"team_epoch={epoch} loss={loss_value:.6f} validation_team_auc={validation_auc:.4f}",
            flush=True,
        )
        if validation_auc > best_auc + 1e-4:
            best_auc = validation_auc
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch, history


def write_report(report: dict[str, Any]) -> None:
    lines = [
        "# Joint Team Model Evaluation", "",
        "The former single-window evaluation has been removed because it was superseded after that test window was inspected",
        "during development. Keeping its figures beside the current benchmark would present an obsolete generalization estimate.", "",
        "The current and only reported joint-team evaluation is the four-fold rolling temporal benchmark:", "",
        "- [Latest joint-team benchmark](team_benchmark.md)",
        "- [Calibration plot](team_calibration.png)", "",
        "Run `python -m src.evaluation.team_benchmark --device cuda` to regenerate the current report from the compact real-match",
        "history. No synthetic teams, substituted players, imputed members, or fallback scores are used.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    event_path: Path = EVENT_PATH,
    max_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    pair_loss_weight: float = 0.25,
    hidden_dim: int = 96,
    tower_batch_size: int = 1024,
    tower_learning_rate: float = 3e-4,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    events = pd.read_parquet(event_path)
    train, validation, test, validation_cutoff, test_cutoff = temporal_three_way_split(events)
    source_counts = {
        "train": complete_team_counts(train),
        "validation": complete_team_counts(validation),
        "test": complete_team_counts(test),
    }
    for split_name, counts in source_counts.items():
        if counts["wins"] == 0 or counts["losses"] == 0:
            raise RuntimeError(
                f"Cannot train/evaluate a real team outcome model: {split_name} lacks observed wins or losses"
            )

    deployment_two_tower = LoadedTwoTower.load(
        TWO_TOWER_MODEL_PATH, TWO_TOWER_METADATA_PATH, requested_device=device
    )
    tower_epochs = int(deployment_two_tower.metadata.get("training", {}).get("selected_epochs", 0))
    if tower_epochs < 1:
        raise RuntimeError("Two-tower metadata does not contain a valid selected epoch count")

    train_profiles = enrich_profiles_for_rag(build_player_profiles(train))
    train_ids = set(train_profiles["summoner_id"].astype(str))
    train_pairs = build_positive_pairs(train, train_ids)
    provisional_metadata = build_metadata(
        train_profiles,
        int(deployment_two_tower.metadata["embedding_dim"]),
        int(deployment_two_tower.metadata["hidden_dim"]),
        extra_champions=train["champion_name"].dropna().astype(str).tolist(),
    )
    provisional_model, _ = train_epochs(
        train_profiles, train_pairs, provisional_metadata, tower_epochs,
        tower_batch_size, tower_learning_rate, device, seed,
    )
    provisional_two_tower = LoadedTwoTower(provisional_model, provisional_metadata, device)
    train_bank = build_lineup_feature_bank(train_profiles, provisional_two_tower)
    eligible_train_roles = {
        player_id: str(train_bank.roles[row])
        for player_id, row in train_bank.rows.items() if train_bank.valid[row]
    }
    train_examples = build_observed_lineups(train, eligible_train_roles)
    validation_examples = build_observed_lineups(validation, eligible_train_roles)
    if train_examples.empty or validation_examples.empty:
        raise RuntimeError("Complete real teams did not remain after strict historical feature eligibility checks")
    if train_examples["won"].astype(bool).nunique() != 2 or validation_examples["won"].astype(bool).nunique() != 2:
        raise RuntimeError("Eligible real train/validation lineups must contain both wins and losses")
    selection_metadata = build_team_metadata(provisional_two_tower, hidden_dim)
    selected_epochs, selection_history = _selection_train(
        train_examples, validation_examples, train_bank, provisional_metadata, train_profiles,
        selection_metadata, max_epochs, patience, batch_size, learning_rate,
        pair_loss_weight, device, seed,
    )

    development = pd.concat((train, validation), ignore_index=True)
    development_profiles = enrich_profiles_for_rag(build_player_profiles(development))
    development_bank = build_lineup_feature_bank(development_profiles, deployment_two_tower)
    eligible_development_roles = {
        player_id: str(development_bank.roles[row])
        for player_id, row in development_bank.rows.items() if development_bank.valid[row]
    }
    development_examples = build_observed_lineups(development, eligible_development_roles)
    test_examples = build_observed_lineups(test, eligible_development_roles)
    if development_examples.empty or test_examples.empty:
        raise RuntimeError("No complete real development/test lineups survived strict historical eligibility")
    if development_examples["won"].astype(bool).nunique() != 2 or test_examples["won"].astype(bool).nunique() != 2:
        raise RuntimeError("Eligible real development/test lineups must contain both wins and losses")
    final_metadata = build_team_metadata(deployment_two_tower, hidden_dim)
    final_model, final_history = train_team_epochs(
        development_examples, development_bank, deployment_two_tower.metadata, final_metadata,
        selected_epochs, batch_size, learning_rate, pair_loss_weight, device, seed,
    )
    development_team_labels = development_examples.groupby(
        ["match_id", "team_id"], as_index=False
    )["won"].first()
    development_prior = float(development_team_labels["won"].astype(bool).mean())
    test_result = evaluate_team_model(
        final_model, test_examples, development_bank, deployment_two_tower.metadata,
        development_profiles, device, seed, development_prior,
    )
    final_metadata["two_tower_model_sha256"] = artifact_sha256(TWO_TOWER_MODEL_PATH)
    final_metadata["training"] = {
        "seed": seed,
        "selected_epochs": selected_epochs,
        "train_complete_teams": int(train_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
        "validation_complete_teams": int(validation_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
        "development_complete_teams": int(development_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
        "trained_through_timestamp": int(development["timestamp"].max()),
        "validation_cutoff_utc": pd.to_datetime(validation_cutoff, unit="ms", utc=True).isoformat(),
        "test_cutoff_utc": pd.to_datetime(test_cutoff, unit="ms", utc=True).isoformat(),
        "real_lineups_only": True,
        "constructed_negative_lineups": False,
    }
    save_team_artifact(final_model, final_metadata, TEAM_MODEL_PATH, TEAM_MODEL_METADATA_PATH)
    report = {
        "protocol": "chronological_observed_complete_team_outcome_prediction",
        "source_counts": source_counts,
        "training": {
            "device": device,
            "train_teams": int(train_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
            "validation_teams": int(validation_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
            "development_teams": int(development_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
            "selected_epochs": selected_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "pair_loss_weight": pair_loss_weight,
            "two_tower_epochs": tower_epochs,
            "development_win_prior": development_prior,
            "real_lineups_only": True,
            "constructed_negative_lineups": False,
        },
        "validation_history": selection_history,
        "test": test_result,
    }
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    HISTORY_PATH.write_text(
        json.dumps({"selection": selection_history, "final": final_history}, indent=2), encoding="utf-8"
    )
    write_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real observed-lineup team reranker")
    parser.add_argument("--events", type=Path, default=EVENT_PATH)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--pair-loss-weight", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--tower-batch-size", type=int, default=1024)
    parser.add_argument("--tower-learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_training(
        event_path=args.events,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pair_loss_weight=args.pair_loss_weight,
        hidden_dim=args.hidden_dim,
        tower_batch_size=args.tower_batch_size,
        tower_learning_rate=args.tower_learning_rate,
        device_name=args.device,
        seed=args.seed,
    )
    metric = report["test"]["models"]["team_model"]["roc_auc"]
    print(json.dumps({
        "model_path": str(TEAM_MODEL_PATH),
        "device": report["training"]["device"],
        "selected_epochs": report["training"]["selected_epochs"],
        "test_complete_teams": report["test"]["teams"],
        "test_roc_auc": metric["value"],
        "test_roc_auc_ci": [metric["ci_low"], metric["ci_high"]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
