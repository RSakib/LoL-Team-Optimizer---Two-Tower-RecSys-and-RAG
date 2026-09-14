from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PROJECT_ROOT, TWO_TOWER_METADATA_PATH, TWO_TOWER_MODEL_PATH
from src.data.ingestion import build_player_profiles
from src.data.profile_documents import enrich_profiles_for_rag
from src.evaluation.events import EVENT_PATH
from src.recsys.team_model import (
    FEATURE_GROUPS,
    LineupFeatureBank,
    TeamLineupModel,
    build_lineup_feature_bank,
    build_observed_lineups,
    build_team_metadata,
    complete_team_counts,
    select_feature_groups,
)
from src.recsys.train import (
    build_positive_pairs,
    resolve_device,
    train_epochs,
)
from src.recsys.train_team import (
    _example_tensors,
    _predict,
    _selection_train,
    _team_level_frame,
    train_team_epochs,
)
from src.recsys.two_tower import LoadedTwoTower, ROLES, build_metadata, query_tensors


ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
RESULT_PATH = ARTIFACT_DIR / "team_benchmark.json"
PREDICTION_PATH = ARTIFACT_DIR / "team_benchmark_predictions.parquet"
REPORT_PATH = PROJECT_ROOT / "docs" / "team_benchmark.md"
CALIBRATION_PATH = PROJECT_ROOT / "docs" / "team_calibration.png"


@dataclass
class TemporalFold:
    index: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: int
    validation_end: int
    test_end: int


@dataclass(frozen=True)
class Variant:
    groups: tuple[str, ...]
    pair_interactions: bool


VARIANTS = {
    "joint_full": Variant(FEATURE_GROUPS, True),
    "without_explicit_pair_head": Variant(FEATURE_GROUPS, False),
    "without_explicit_champion_pool": Variant(("two_tower", "playstyle"), True),
    "without_explicit_playstyle": Variant(
        ("two_tower", "champion_pool_embedding", "pool_summary"), True
    ),
}


def rolling_temporal_folds(
    events: pd.DataFrame,
    folds: int = 4,
    initial_train_fraction: float = 0.50,
    block_fraction: float = 0.10,
) -> list[TemporalFold]:
    """Create expanding train windows and non-overlapping future test blocks."""
    if folds < 1 or initial_train_fraction <= 0 or block_fraction <= 0:
        raise ValueError("Rolling-fold parameters must be positive")
    if initial_train_fraction + (folds + 1) * block_fraction > 1.000001:
        raise ValueError("Rolling folds exceed the available chronological history")
    match_times = events[["match_id", "timestamp"]].drop_duplicates("match_id").sort_values(
        ["timestamp", "match_id"]
    )
    total = len(match_times)
    result: list[TemporalFold] = []
    for fold_index in range(folds):
        train_stop = int(total * (initial_train_fraction + fold_index * block_fraction))
        validation_stop = int(total * (initial_train_fraction + (fold_index + 1) * block_fraction))
        test_fraction = initial_train_fraction + (fold_index + 2) * block_fraction
        test_stop = total if test_fraction >= 0.999999 else int(total * test_fraction)
        train_ids = set(match_times.iloc[:train_stop]["match_id"].astype(str))
        validation_ids = set(match_times.iloc[train_stop:validation_stop]["match_id"].astype(str))
        test_ids = set(match_times.iloc[validation_stop:test_stop]["match_id"].astype(str))
        event_ids = events["match_id"].astype(str)
        train = events[event_ids.isin(train_ids)].copy()
        validation = events[event_ids.isin(validation_ids)].copy()
        test = events[event_ids.isin(test_ids)].copy()
        if train.empty or validation.empty or test.empty:
            raise RuntimeError(f"Rolling temporal fold {fold_index + 1} contains an empty partition")
        result.append(TemporalFold(
            index=fold_index + 1,
            train=train,
            validation=validation,
            test=test,
            train_end=int(match_times.iloc[train_stop - 1]["timestamp"]),
            validation_end=int(match_times.iloc[validation_stop - 1]["timestamp"]),
            test_end=int(match_times.iloc[test_stop - 1]["timestamp"]),
        ))
    return result


def _eligible_roles(bank: LineupFeatureBank) -> dict[str, str]:
    return {
        player_id: str(bank.roles[row])
        for player_id, row in bank.rows.items()
        if bank.valid[row]
    }


def _train_fold_tower(
    train_events: pd.DataFrame,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
) -> tuple[pd.DataFrame, LoadedTwoTower, LineupFeatureBank]:
    profiles = enrich_profiles_for_rag(build_player_profiles(train_events))
    profile_ids = set(profiles["summoner_id"].astype(str))
    pairs = build_positive_pairs(train_events, profile_ids)
    if pairs.empty:
        raise RuntimeError("A rolling training fold contains no real teammate pairs")
    metadata = build_metadata(
        profiles,
        embedding_dim,
        hidden_dim,
        extra_champions=train_events["champion_name"].dropna().astype(str).tolist(),
    )
    model, _ = train_epochs(
        profiles,
        pairs,
        metadata,
        epochs,
        batch_size,
        learning_rate,
        device,
        seed,
    )
    loaded = LoadedTwoTower(model=model, metadata=metadata, device=device)
    bank = build_lineup_feature_bank(profiles, loaded)
    return profiles, loaded, bank


def _variant_bank(
    full_bank: LineupFeatureBank,
    two_tower: LoadedTwoTower,
    variant: Variant,
) -> LineupFeatureBank:
    return select_feature_groups(full_bank, two_tower, variant.groups)


def _team_predictions(
    model: TeamLineupModel,
    examples: pd.DataFrame,
    bank: LineupFeatureBank,
    two_tower: LoadedTwoTower,
    profiles: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    tensors, _ = _example_tensors(examples, bank, two_tower.metadata, device)
    performance, compatibility = _predict(model, tensors)
    baseline = _profile_stat_score(examples, profiles, "win_rate", transform=float)
    return _team_level_frame(examples, performance, compatibility, baseline)


def _profile_stat_score(
    examples: pd.DataFrame,
    profiles: pd.DataFrame,
    column: str,
    transform: Callable[[float], float],
) -> np.ndarray:
    values = {
        str(row.summoner_id): transform(float(getattr(row, column)))
        for row in profiles[["summoner_id", column]].itertuples(index=False)
        if not pd.isna(getattr(row, column))
    }
    scores: list[float] = []
    for example in examples.itertuples(index=False):
        lineup_values = [
            values.get(str(getattr(example, f"player_{role}")))
            for role in ROLES
            if role != str(example.finder_role)
        ]
        if any(value is None for value in lineup_values):
            raise RuntimeError(f"A real lineup member lacks historical {column}")
        scores.append(float(np.mean(lineup_values)))
    return np.asarray(scores, dtype=np.float64)


@torch.inference_mode()
def _independent_two_tower_scores(
    examples: pd.DataFrame,
    bank: LineupFeatureBank,
    two_tower: LoadedTwoTower,
) -> np.ndarray:
    embedding_dim = int(two_tower.metadata["embedding_dim"])
    candidate_embeddings = bank.features[:, :embedding_dim]
    results: list[float] = []
    for example in examples.itertuples(index=False):
        scores: list[float] = []
        for role in ROLES:
            if role == str(example.finder_role):
                continue
            player_id = str(getattr(example, f"player_{role}"))
            row = bank.rows[player_id]
            query = two_tower.model.encode_query(**query_tensors(
                two_tower.metadata,
                finder_role=str(example.finder_role),
                finder_tier=example.finder_tier,
                finder_division=example.finder_division,
                target_role=role,
                target_champion=None,
                device=two_tower.device,
            )).detach().cpu().numpy()[0]
            scores.append(float(candidate_embeddings[row] @ query))
        results.append(float(np.mean(scores)))
    return np.asarray(results, dtype=np.float64)


def _aggregate_scenario_scores(examples: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    frame = examples[["match_id", "team_id", "won"]].copy()
    frame["score"] = scores
    return frame.groupby(["match_id", "team_id"], as_index=False).agg(
        won=("won", "first"),
        score=("score", "mean"),
    )


def _logistic_design(
    examples: pd.DataFrame,
    bank: LineupFeatureBank,
    two_tower: LoadedTwoTower,
) -> np.ndarray:
    tensors, _ = _example_tensors(examples, bank, two_tower.metadata, "cpu")
    player = tensors["player_features"].numpy().reshape(len(examples), -1)
    mask = tensors["present_mask"].numpy().astype(np.float32)
    role = np.eye(len(two_tower.metadata["vocabs"]["role"]), dtype=np.float32)[
        tensors["finder_role"].numpy()
    ]
    tier = np.eye(len(two_tower.metadata["vocabs"]["tier"]), dtype=np.float32)[
        tensors["finder_tier"].numpy()
    ]
    division = np.eye(len(two_tower.metadata["vocabs"]["division"]), dtype=np.float32)[
        tensors["finder_division"].numpy()
    ]
    return np.concatenate((player, mask, role, tier, division), axis=1)


def _fit_logistic_baseline(
    train_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
    test_examples: pd.DataFrame,
    bank: LineupFeatureBank,
    two_tower: LoadedTwoTower,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    x_train = _logistic_design(train_examples, bank, two_tower)
    x_validation = _logistic_design(validation_examples, bank, two_tower)
    x_test = _logistic_design(test_examples, bank, two_tower)
    y_train = train_examples["won"].astype(int).to_numpy()
    candidates = (0.001, 0.01, 0.1, 1.0)
    best_c = candidates[0]
    best_auc = float("-inf")
    for c_value in candidates:
        model = LogisticRegression(
            C=c_value,
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
            solver="liblinear",
        ).fit(x_train, y_train)
        validation = _aggregate_scenario_scores(
            validation_examples,
            model.predict_proba(x_validation)[:, 1],
        )
        auc = float(roc_auc_score(validation["won"].astype(int), validation["score"]))
        if auc > best_auc:
            best_auc = auc
            best_c = c_value
    selected = LogisticRegression(
        C=best_c,
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    ).fit(x_train, y_train)
    return _aggregate_scenario_scores(
        test_examples,
        selected.predict_proba(x_test)[:, 1],
    ), float(best_c)


def _latency_ms(
    model: TeamLineupModel,
    examples: pd.DataFrame,
    bank: LineupFeatureBank,
    two_tower: LoadedTwoTower,
    device: str,
    repeats: int = 20,
) -> dict[str, float]:
    tensors, _ = _example_tensors(examples, bank, two_tower.metadata, device)
    model.eval()
    with torch.inference_mode():
        model(**tensors)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(**tensors)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
    return {
        "batch_scenarios": int(len(examples)),
        "batch_latency_ms": elapsed_ms,
        "latency_ms_per_scenario": elapsed_ms / max(len(examples), 1),
    }


def _probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1 - 1e-6)
    target = labels.astype(np.float64)
    return {
        "log_loss": float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean()),
        "brier_score": float(np.square(clipped - target).mean()),
        "accuracy_at_0_5": float(((clipped >= 0.5) == labels.astype(bool)).mean()),
        "ece_10_bin": expected_calibration_error(labels, clipped, 10),
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        inclusive = index == bins - 1
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if inclusive else scores < edges[index + 1]
        )
        if selected.any():
            error += selected.mean() * abs(float(scores[selected].mean()) - float(labels[selected].mean()))
    return float(error if total else 0.0)


def calibration_bins(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[dict[str, float | int]] = []
    for index in range(bins):
        inclusive = index == bins - 1
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if inclusive else scores < edges[index + 1]
        )
        if selected.any():
            result.append({
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "teams": int(selected.sum()),
                "mean_score": float(scores[selected].mean()),
                "observed_win_rate": float(labels[selected].mean()),
            })
    return result


def _cluster_bootstrap(
    frame: pd.DataFrame,
    metric: Callable[[np.ndarray, np.ndarray], float],
    seed: int,
    repeats: int = 1000,
) -> tuple[float, float]:
    groups = frame["fold"].astype(str) + ":" + frame["match_id"].astype(str)
    unique_groups = groups.unique()
    rows = {group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups}
    labels = frame["won"].astype(int).to_numpy()
    scores = frame["score"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([rows[group] for group in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        values.append(metric(labels[indices], scores[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _model_metrics(frame: pd.DataFrame, probability_score: bool, seed: int) -> dict[str, Any]:
    labels = frame["won"].astype(int).to_numpy()
    scores = frame["score"].to_numpy(dtype=np.float64)
    auc = float(roc_auc_score(labels, scores))
    average_precision = float(average_precision_score(labels, scores))
    auc_low, auc_high = _cluster_bootstrap(
        frame, lambda y, value: float(roc_auc_score(y, value)), seed
    )
    ap_low, ap_high = _cluster_bootstrap(
        frame, lambda y, value: float(average_precision_score(y, value)), seed + 1
    )
    result: dict[str, Any] = {
        "teams": int(len(frame)),
        "roc_auc": {"value": auc, "ci_low": auc_low, "ci_high": auc_high},
        "pr_auc": {"value": average_precision, "ci_low": ap_low, "ci_high": ap_high},
    }
    if probability_score:
        result.update(_probability_metrics(labels, scores))
    return result


def _paired_auc_delta(
    reference: pd.DataFrame,
    comparator: pd.DataFrame,
    seed: int,
    repeats: int = 1000,
) -> dict[str, float | int]:
    keys = ["fold", "match_id", "team_id", "won"]
    merged = reference[keys + ["score"]].merge(
        comparator[keys + ["score"]], on=keys, suffixes=("_reference", "_comparator")
    )
    labels = merged["won"].astype(int).to_numpy()
    reference_scores = merged["score_reference"].to_numpy()
    comparator_scores = merged["score_comparator"].to_numpy()
    observed = float(
        roc_auc_score(labels, reference_scores) - roc_auc_score(labels, comparator_scores)
    )
    groups = merged["fold"].astype(str) + ":" + merged["match_id"].astype(str)
    unique_groups = groups.unique()
    rows = {group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([rows[group] for group in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        values.append(float(
            roc_auc_score(labels[indices], reference_scores[indices])
            - roc_auc_score(labels[indices], comparator_scores[indices])
        ))
    array = np.asarray(values, dtype=np.float64)
    return {
        "teams": int(len(merged)),
        "delta": observed,
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "bootstrap_two_sided_p": float(2 * min((array <= 0).mean(), (array >= 0).mean())),
    }


def _annotate(frame: pd.DataFrame, fold: int, seed: int, model: str) -> pd.DataFrame:
    result = frame[["match_id", "team_id", "won", "score"]].copy()
    result.insert(0, "model", model)
    result.insert(0, "seed", seed)
    result.insert(0, "fold", fold)
    return result


def _plot_calibration(bins: list[dict[str, float | int]]) -> None:
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Perfect calibration")
    axis.plot(
        [float(item["mean_score"]) for item in bins],
        [float(item["observed_win_rate"]) for item in bins],
        marker="o",
        color="#2563eb",
        linewidth=2,
        label="Joint model",
    )
    for item in bins:
        axis.annotate(
            f"n={item['teams']}",
            (float(item["mean_score"]), float(item["observed_win_rate"])),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=8,
        )
    axis.set(xlabel="Mean model score", ylabel="Observed win rate", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Rolling out-of-time calibration")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(CALIBRATION_PATH, dpi=180)
    plt.close(figure)


def _metric_text(metric: dict[str, float]) -> str:
    return f"{metric['value']:.4f} [{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]"


def write_report(report: dict[str, Any]) -> None:
    lines = [
        "# Rolling Joint-Team Benchmark", "",
        "This benchmark reuses only the existing compact real-match history. It does not construct teams, substitute",
        "players, impute members, or create outcome labels. Expanding chronological folds use non-overlapping",
        f"future test blocks. The complete joint model is repeated across {len(report['protocol']['team_seeds'])} team-model seeds.", "",
        "## Protocol", "",
        f"- Rolling folds: **{report['protocol']['folds']}**",
        f"- Team-model seeds: **{', '.join(str(value) for value in report['protocol']['team_seeds'])}**",
        f"- Fixed fold-specific two-tower seed: **{report['protocol']['two_tower_seed']}**",
        "- Epoch selection uses only the validation block immediately before each fold's test block.",
        "- Player profiles and embeddings are rebuilt from each fold's training history only.",
        "- Test blocks do not overlap across folds.", "",
        "`joint_full` is seed 42, matching the deployment seed. The five-seed ensemble is shown separately; seed",
        "mean and standard deviation quantify training instability.", "",
        "## Out-of-time results", "",
        "| Model | ROC-AUC (match-bootstrap 95% CI) | PR-AUC (95% CI) | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["models"].items():
        brier = f"{result['brier_score']:.4f}" if "brier_score" in result else "—"
        ece = f"{result['ece_10_bin']:.4f}" if "ece_10_bin" in result else "—"
        lines.append(
            f"| {name} | {_metric_text(result['roc_auc'])} | {_metric_text(result['pr_auc'])} | {brier} | {ece} |"
        )
    lines.extend(["", "## Seed stability", "", "| Seed | Pooled ROC-AUC |", "|---:|---:|"])
    for item in report["seed_stability"]:
        lines.append(f"| {item['seed']} | {item['roc_auc']:.4f} |")
    lines.extend([
        "",
        f"Mean ± standard deviation: **{report['seed_mean_auc']:.4f} ± {report['seed_std_auc']:.4f}**.",
        "", "## Rolling-fold coverage", "",
        "| Fold | Train matches | Validation matches | Test matches | Source complete teams | Eligible teams | Coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fold in report["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_matches']:,} | {fold['validation_matches']:,} | "
            f"{fold['test_matches']:,} | {fold['source_complete_test_teams']:,} | "
            f"{fold['eligible_test_teams']:,} | {fold['coverage']:.1%} |"
        )
    lines.extend(["", "## Paired ROC-AUC differences versus the joint model", "",
                  "Positive values favor the complete joint model.", "",
                  "Every comparison uses the deployment-aligned seed-42 joint model as its reference.", "",
                  "| Comparator | Reference | Δ ROC-AUC (95% CI) | Bootstrap p |", "|---|---|---:|---:|"])
    for name, result in report["paired_auc_deltas"].items():
        lines.append(
            f"| {name} | {result['reference']} | {result['delta']:.4f} "
            f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}] | "
            f"{result['bootstrap_two_sided_p']:.4f} |"
        )
    lines.extend(["", "## Calibration", "", f"![Calibration curve](team_calibration.png)", ""])
    for item in report["calibration_bins"]:
        lines.append(
            f"- Score {item['lower']:.1f}–{item['upper']:.1f}: {item['teams']} teams, "
            f"mean score {item['mean_score']:.3f}, observed win rate {item['observed_win_rate']:.3f}."
        )
    lines.extend(["", "## Inference timing", ""])
    for item in report["latency"]:
        lines.append(
            f"- Fold {item['fold']}: {item['batch_scenarios']} real scenarios in "
            f"{item['batch_latency_ms']:.3f} ms ({item['latency_ms_per_scenario']:.4f} ms/scenario)."
        )
    lines.extend([
        "", "## Interpretation constraints", "",
        "- The original single future test window was inspected during development; rolling evaluation improves stability",
        "  evidence but cannot recreate a never-observed final holdout without new matches.",
        "- Logged outcomes evaluate observed teams. They cannot prove that a newly recommended counterfactual lineup would win.",
        "- Explicit champion-pool and playstyle ablations remove the second-stage feature groups, but the frozen two-tower",
        "  embedding still contains entangled main-champion and historical-statistic information.",
        "- Only the team model is repeated across five seeds; the fold-specific two-tower seed is fixed to bound compute.",
        "- Pair outputs inherit team outcome supervision and are not direct measurements of social chemistry.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    event_path: Path = EVENT_PATH,
    fold_count: int = 4,
    team_seeds: Iterable[int] = (42, 43, 44, 45, 46),
    two_tower_seed: int = 42,
    max_team_epochs: int = 20,
    patience: int = 4,
    team_batch_size: int = 128,
    team_learning_rate: float = 5e-4,
    pair_loss_weight: float = 0.25,
    tower_batch_size: int = 1024,
    tower_learning_rate: float = 3e-4,
    device_name: str = "auto",
    tower_epochs_override: int | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    team_seeds = tuple(int(value) for value in team_seeds)
    if len(team_seeds) < 2:
        raise ValueError("At least two team seeds are required for stability reporting")
    deployed_tower = LoadedTwoTower.load(
        TWO_TOWER_MODEL_PATH, TWO_TOWER_METADATA_PATH, requested_device=device
    )
    tower_epochs = int(deployed_tower.metadata.get("training", {}).get("selected_epochs", 0))
    if tower_epochs_override is not None:
        tower_epochs = int(tower_epochs_override)
    if tower_epochs < 1:
        raise RuntimeError("The deployed two-tower metadata lacks a selected epoch count")
    events = pd.read_parquet(event_path)
    folds = rolling_temporal_folds(events, folds=fold_count)
    prediction_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    latency: list[dict[str, Any]] = []
    selected_logistic_c: list[dict[str, float | int]] = []

    for fold in folds:
        print(f"rolling_fold={fold.index} building_train_only_features", flush=True)
        profiles, two_tower, full_bank = _train_fold_tower(
            fold.train,
            int(deployed_tower.metadata["embedding_dim"]),
            int(deployed_tower.metadata["hidden_dim"]),
            tower_epochs,
            tower_batch_size,
            tower_learning_rate,
            device,
            two_tower_seed,
        )
        eligible_roles = _eligible_roles(full_bank)
        train_examples = build_observed_lineups(fold.train, eligible_roles)
        validation_examples = build_observed_lineups(fold.validation, eligible_roles)
        test_examples = build_observed_lineups(fold.test, eligible_roles)
        for name, examples in (
            ("train", train_examples), ("validation", validation_examples), ("test", test_examples)
        ):
            if examples.empty or examples["won"].astype(bool).nunique() != 2:
                raise RuntimeError(
                    f"Rolling fold {fold.index} {name} examples lack eligible real wins and losses"
                )
        source_test_teams = complete_team_counts(fold.test)["teams"]
        eligible_test_teams = int(test_examples[["match_id", "team_id"]].drop_duplicates().shape[0])
        fold_report: dict[str, Any] = {
            "fold": fold.index,
            "train_matches": int(fold.train["match_id"].nunique()),
            "validation_matches": int(fold.validation["match_id"].nunique()),
            "test_matches": int(fold.test["match_id"].nunique()),
            "source_complete_test_teams": int(source_test_teams),
            "eligible_test_teams": eligible_test_teams,
            "coverage": eligible_test_teams / max(source_test_teams, 1),
            "eligible_train_teams": int(train_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
            "eligible_validation_teams": int(validation_examples[["match_id", "team_id"]].drop_duplicates().shape[0]),
            "selected_epochs": {},
        }

        for variant_name, variant in VARIANTS.items():
            bank = _variant_bank(full_bank, two_tower, variant)
            metadata = build_team_metadata(
                two_tower,
                hidden_dim=96,
                player_feature_dim=bank.features.shape[1],
                feature_groups=variant.groups,
                use_pair_interactions=variant.pair_interactions,
            )
            seeds = team_seeds if variant_name == "joint_full" else (team_seeds[0],)
            for seed in seeds:
                selected_epochs, _ = _selection_train(
                    train_examples,
                    validation_examples,
                    bank,
                    two_tower.metadata,
                    profiles,
                    metadata,
                    max_team_epochs,
                    patience,
                    team_batch_size,
                    team_learning_rate,
                    pair_loss_weight if variant.pair_interactions else 0.0,
                    device,
                    seed,
                )
                model, _ = train_team_epochs(
                    train_examples,
                    bank,
                    two_tower.metadata,
                    metadata,
                    selected_epochs,
                    team_batch_size,
                    team_learning_rate,
                    pair_loss_weight if variant.pair_interactions else 0.0,
                    device,
                    seed,
                )
                team_frame = _team_predictions(
                    model, test_examples, bank, two_tower, profiles, device
                ).rename(columns={"performance": "score"})
                prediction_frames.append(_annotate(team_frame, fold.index, seed, variant_name))
                fold_report["selected_epochs"].setdefault(variant_name, {})[str(seed)] = selected_epochs
                if variant_name == "joint_full" and seed == team_seeds[0]:
                    timing = _latency_ms(model, test_examples, bank, two_tower, device)
                    timing["fold"] = fold.index
                    latency.append(timing)

        baseline = _aggregate_scenario_scores(
            test_examples,
            _profile_stat_score(test_examples, profiles, "win_rate", transform=float),
        )
        prediction_frames.append(_annotate(
            baseline, fold.index, team_seeds[0], "mean_historical_win_rate"
        ))
        popularity = _aggregate_scenario_scores(
            test_examples,
            _profile_stat_score(test_examples, profiles, "matches", transform=lambda value: math.log1p(value)),
        )
        prediction_frames.append(_annotate(
            popularity, fold.index, team_seeds[0], "mean_player_popularity"
        ))
        independent = _aggregate_scenario_scores(
            test_examples,
            _independent_two_tower_scores(test_examples, full_bank, two_tower),
        )
        prediction_frames.append(_annotate(
            independent, fold.index, team_seeds[0], "independent_two_tower_score"
        ))
        logistic, selected_c = _fit_logistic_baseline(
            train_examples,
            validation_examples,
            test_examples,
            full_bank,
            two_tower,
            team_seeds[0],
        )
        selected_logistic_c.append({"fold": fold.index, "C": selected_c})
        prediction_frames.append(_annotate(
            logistic, fold.index, team_seeds[0], "logistic_no_learned_interactions"
        ))
        train_prior = float(
            train_examples.groupby(["match_id", "team_id"])["won"].first().astype(bool).mean()
        )
        constant = baseline.copy()
        constant["score"] = train_prior
        prediction_frames.append(_annotate(
            constant, fold.index, team_seeds[0], "historical_constant_prior"
        ))
        random = baseline.copy()
        random["score"] = np.random.default_rng(team_seeds[0] + fold.index).random(len(random))
        prediction_frames.append(_annotate(random, fold.index, team_seeds[0], "random_score"))
        fold_reports.append(fold_report)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTION_PATH, index=False)
    probability_models = {
        "joint_full",
        "joint_full_five_seed_ensemble",
        "without_explicit_pair_head",
        "without_explicit_champion_pool",
        "without_explicit_playstyle",
        "mean_historical_win_rate",
        "logistic_no_learned_interactions",
        "historical_constant_prior",
    }
    ensemble_frames: dict[str, pd.DataFrame] = {}
    for model_name in predictions["model"].unique():
        model_predictions = predictions[predictions["model"] == model_name]
        ensemble_frames[model_name] = model_predictions.groupby(
            ["fold", "match_id", "team_id", "won"], as_index=False
        )["score"].mean()
    seed_stability: list[dict[str, float | int]] = []
    full_predictions = predictions[predictions["model"] == "joint_full"]
    for seed in team_seeds:
        frame = full_predictions[full_predictions["seed"] == seed]
        seed_stability.append({
            "seed": seed,
            "roc_auc": float(roc_auc_score(frame["won"].astype(int), frame["score"])),
        })
    seed_values = np.asarray([item["roc_auc"] for item in seed_stability], dtype=np.float64)
    reference = full_predictions[
        full_predictions["seed"] == team_seeds[0]
    ][["fold", "match_id", "team_id", "won", "score"]]
    reporting_frames = {
        "joint_full": reference,
        "joint_full_five_seed_ensemble": ensemble_frames["joint_full"],
        **{name: frame for name, frame in ensemble_frames.items() if name != "joint_full"},
    }
    models = {
        name: _model_metrics(frame, name in probability_models, team_seeds[0])
        for name, frame in reporting_frames.items()
    }
    paired: dict[str, Any] = {}
    for name, frame in ensemble_frames.items():
        if name == "joint_full":
            continue
        paired[name] = _paired_auc_delta(reference, frame, team_seeds[0])
        paired[name]["reference"] = f"joint_full_seed_{team_seeds[0]}"
    labels = reference["won"].astype(int).to_numpy()
    scores = reference["score"].to_numpy()
    bins = calibration_bins(labels, scores)
    _plot_calibration(bins)
    report = {
        "protocol": {
            "name": "expanding_window_existing_real_data",
            "folds": fold_count,
            "initial_train_fraction": 0.50,
            "validation_block_fraction": 0.10,
            "test_block_fraction": 0.10,
            "team_seeds": list(team_seeds),
            "two_tower_seed": two_tower_seed,
            "two_tower_epochs": tower_epochs,
            "constructed_team_examples": False,
            "test_blocks_overlap": False,
            "original_holdout_previously_inspected": True,
        },
        "folds": fold_reports,
        "models": models,
        "seed_stability": seed_stability,
        "seed_mean_auc": float(seed_values.mean()),
        "seed_std_auc": float(seed_values.std(ddof=1)),
        "paired_auc_deltas": paired,
        "calibration_bins": bins,
        "latency": latency,
        "selected_logistic_c": selected_logistic_c,
        "prediction_artifact": str(PREDICTION_PATH),
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report)
    return report


def summarize_existing() -> dict[str, Any]:
    """Rebuild derived comparisons and prose without retraining any model."""
    if not RESULT_PATH.exists() or not PREDICTION_PATH.exists():
        raise RuntimeError("Existing benchmark artifacts are missing")
    report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(PREDICTION_PATH)
    team_seeds = tuple(int(value) for value in report["protocol"]["team_seeds"])
    ensemble_frames = {
        name: group.groupby(
            ["fold", "match_id", "team_id", "won"], as_index=False
        )["score"].mean()
        for name, group in predictions.groupby("model")
    }
    full_predictions = predictions[predictions["model"] == "joint_full"]
    reference = full_predictions[
        full_predictions["seed"] == team_seeds[0]
    ][["fold", "match_id", "team_id", "won", "score"]]
    paired: dict[str, Any] = {}
    for name, frame in ensemble_frames.items():
        if name == "joint_full":
            continue
        paired[name] = _paired_auc_delta(reference, frame, team_seeds[0])
        paired[name]["reference"] = f"joint_full_seed_{team_seeds[0]}"
    previous_models = report["models"]
    report["models"] = {
        "joint_full": _model_metrics(reference, True, team_seeds[0]),
        "joint_full_five_seed_ensemble": previous_models["joint_full"],
        **{name: value for name, value in previous_models.items() if name != "joint_full"},
    }
    report["paired_auc_deltas"] = paired
    labels = reference["won"].astype(int).to_numpy()
    scores = reference["score"].to_numpy()
    report["calibration_bins"] = calibration_bins(labels, scores)
    _plot_calibration(report["calibration_bins"])
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-data rolling joint-team benchmark")
    parser.add_argument("--events", type=Path, default=EVENT_PATH)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--team-seeds", default="42,43,44,45,46")
    parser.add_argument("--max-team-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--tower-epochs", type=int, default=None)
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.summarize_existing:
        report = summarize_existing()
    else:
        report = run_benchmark(
            event_path=args.events,
            fold_count=args.folds,
            team_seeds=tuple(int(value.strip()) for value in args.team_seeds.split(",") if value.strip()),
            max_team_epochs=args.max_team_epochs,
            patience=args.patience,
            device_name=args.device,
            tower_epochs_override=args.tower_epochs,
        )
    print(json.dumps({
        "report": str(REPORT_PATH),
        "rolling_teams": report["models"]["joint_full"]["teams"],
        "joint_roc_auc": report["models"]["joint_full"]["roc_auc"],
        "seed_mean_auc": report["seed_mean_auc"],
        "seed_std_auc": report["seed_std_auc"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
