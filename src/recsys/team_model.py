from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.recsys.two_tower import LoadedTwoTower, ROLES


STYLE_COLUMNS = (
    "role_experience_percentile",
    "role_win_rate_percentile",
    "role_kda_percentile",
    "role_vision_percentile",
    "role_damage_percentile",
    "role_assists_percentile",
)
POOL_FEATURES = ("champion_pool_depth", "champion_pool_entropy", "main_champion_share")
FEATURE_GROUPS = ("two_tower", "champion_pool_embedding", "playstyle", "pool_summary")
PAIR_INDICES = tuple(combinations(range(len(ROLES)), 2))


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_observed_lineups(events: pd.DataFrame, profile_roles: dict[str, str]) -> pd.DataFrame:
    """Create leave-one-role-out views of complete teams that actually occurred.

    No players are substituted and no lineups are constructed across teams. Each output
    row contains four players from one recorded five-player team; the omitted real player
    supplies only the anonymous finder role/rank context.
    """
    required = {
        "match_id", "timestamp", "team_id", "summoner_id", "role", "tier", "rank",
        "champion_name", "win",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise RuntimeError(f"Lineup training fields are missing from real events: {missing}")
    members = events[list(required)].dropna(
        subset=["match_id", "timestamp", "team_id", "summoner_id", "role", "tier", "rank", "win"]
    ).copy()
    members["summoner_id"] = members["summoner_id"].astype(str)
    members["role"] = members["role"].astype(str).str.upper()
    members = members[members["role"].isin(ROLES)]
    members = members.drop_duplicates(["match_id", "team_id", "summoner_id"])
    records: list[dict[str, Any]] = []
    expected_roles = set(ROLES)
    for (match_id, team_id), group in members.groupby(["match_id", "team_id"], sort=False):
        if len(group) != len(ROLES) or set(group["role"]) != expected_roles:
            continue
        if group["role"].nunique() != len(ROLES) or group["win"].astype(bool).nunique() != 1:
            continue
        by_role = group.set_index("role")
        for finder_role in ROLES:
            candidate_roles = [role for role in ROLES if role != finder_role]
            candidate_ids = {role: str(by_role.at[role, "summoner_id"]) for role in candidate_roles}
            if any(
                player_id not in profile_roles or profile_roles[player_id] != role
                for role, player_id in candidate_ids.items()
            ):
                continue
            finder = by_role.loc[finder_role]
            record: dict[str, Any] = {
                "match_id": str(match_id),
                "team_id": int(team_id),
                "timestamp": int(group["timestamp"].max()),
                "finder_role": finder_role,
                "finder_tier": str(finder["tier"]).upper(),
                "finder_division": str(finder["rank"]).upper(),
                "won": bool(group["win"].astype(bool).iloc[0]),
            }
            for role in ROLES:
                record[f"player_{role}"] = None if role == finder_role else candidate_ids[role]
                champion = by_role.at[role, "champion_name"]
                record[f"champion_{role}"] = None if role == finder_role or pd.isna(champion) else str(champion)
            records.append(record)
    return pd.DataFrame(records)


def complete_team_counts(events: pd.DataFrame) -> dict[str, int]:
    """Count fully observed real teams without constructing lineup examples."""
    members = events.dropna(subset=["match_id", "team_id", "summoner_id", "role", "win"]).copy()
    members["role"] = members["role"].astype(str).str.upper()
    members = members[members["role"].isin(ROLES)].drop_duplicates(
        ["match_id", "team_id", "summoner_id"]
    )
    grouped = members.groupby(["match_id", "team_id"])
    counts = grouped.agg(
        players=("summoner_id", "nunique"),
        roles=("role", "nunique"),
        outcome_values=("win", "nunique"),
        won=("win", "first"),
    )
    complete = counts[
        (counts["players"] == len(ROLES))
        & (counts["roles"] == len(ROLES))
        & (counts["outcome_values"] == 1)
    ]
    return {
        "teams": int(len(complete)),
        "wins": int(complete["won"].astype(bool).sum()),
        "losses": int((~complete["won"].astype(bool)).sum()),
        "matches_with_both_teams": int(complete.reset_index().groupby("match_id").size().eq(2).sum()),
    }


def _pool_features(value: Any) -> np.ndarray | None:
    if not isinstance(value, dict) or not value:
        return None
    counts = np.asarray([float(count) for count in value.values() if float(count) > 0], dtype=np.float32)
    if len(counts) == 0:
        return None
    probabilities = counts / counts.sum()
    entropy = 0.0 if len(probabilities) == 1 else float(
        -(probabilities * np.log(probabilities)).sum() / math.log(len(probabilities))
    )
    return np.asarray((min(len(counts), 3) / 3.0, entropy, float(probabilities.max())), dtype=np.float32)


@dataclass
class LineupFeatureBank:
    player_ids: np.ndarray
    rows: dict[str, int]
    features: np.ndarray
    valid: np.ndarray
    roles: np.ndarray


def feature_group_slices(two_tower: LoadedTwoTower) -> dict[str, slice]:
    embedding_dim = int(two_tower.metadata["embedding_dim"])
    champion_dim = int(two_tower.model.candidate_champion.embedding_dim)
    champion_end = embedding_dim + champion_dim
    style_end = champion_end + len(STYLE_COLUMNS)
    return {
        "two_tower": slice(0, embedding_dim),
        "champion_pool_embedding": slice(embedding_dim, champion_end),
        "playstyle": slice(champion_end, style_end),
        "pool_summary": slice(style_end, style_end + len(POOL_FEATURES)),
    }


def select_feature_groups(
    bank: LineupFeatureBank,
    two_tower: LoadedTwoTower,
    groups: Iterable[str],
) -> LineupFeatureBank:
    """Return a bank containing only explicitly selected real feature groups."""
    selected = tuple(dict.fromkeys(groups))
    unknown = sorted(set(selected).difference(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown lineup feature groups: {unknown}")
    if not selected:
        raise ValueError("At least one real lineup feature group is required")
    slices = feature_group_slices(two_tower)
    columns = np.concatenate([
        np.arange(slices[name].start, slices[name].stop, dtype=np.int64)
        for name in selected
    ])
    return LineupFeatureBank(
        player_ids=bank.player_ids,
        rows=bank.rows,
        features=bank.features[:, columns].copy(),
        valid=bank.valid,
        roles=bank.roles,
    )


@torch.inference_mode()
def build_lineup_feature_bank(
    profiles: pd.DataFrame,
    two_tower: LoadedTwoTower,
    candidate_embeddings: np.ndarray | None = None,
) -> LineupFeatureBank:
    """Build compact real-player inputs; incomplete feature rows remain ineligible."""
    profiles = profiles.reset_index(drop=True)
    if candidate_embeddings is None:
        from src.recsys.two_tower import candidate_tensors

        tensors = candidate_tensors(profiles, two_tower.metadata, two_tower.device)
        candidate_embeddings = two_tower.model.encode_candidate(**tensors).detach().cpu().numpy()
    champion_vocab = two_tower.metadata["vocabs"]["champion"]
    champion_weights = two_tower.model.candidate_champion.weight.detach().cpu().numpy()
    pool_embeddings: list[np.ndarray] = []
    numeric_features: list[np.ndarray] = []
    valid: list[bool] = []
    for row in profiles.to_dict("records"):
        champions = row.get("top_champions")
        pool = _pool_features(champions)
        style = np.asarray([pd.to_numeric(row.get(name), errors="coerce") for name in STYLE_COLUMNS], dtype=np.float32)
        champion_vectors: list[np.ndarray] = []
        champion_counts: list[float] = []
        if isinstance(champions, dict):
            for champion, count in champions.items():
                index = champion_vocab.get(str(champion).strip().upper(), champion_vocab.get(str(champion).strip()))
                numeric_count = pd.to_numeric(count, errors="coerce")
                if index is not None and not pd.isna(numeric_count) and float(numeric_count) > 0:
                    champion_vectors.append(champion_weights[index])
                    champion_counts.append(float(numeric_count))
        row_valid = pool is not None and np.isfinite(style).all() and bool(champion_vectors)
        if row_valid:
            weights = np.asarray(champion_counts, dtype=np.float32)
            weights /= weights.sum()
            pool_embedding = np.average(np.vstack(champion_vectors), axis=0, weights=weights).astype(np.float32)
            numeric = np.concatenate((style, pool)).astype(np.float32)
        else:
            pool_embedding = np.zeros(champion_weights.shape[1], dtype=np.float32)
            numeric = np.zeros(len(STYLE_COLUMNS) + len(POOL_FEATURES), dtype=np.float32)
        pool_embeddings.append(pool_embedding)
        numeric_features.append(numeric)
        valid.append(row_valid)
    features = np.concatenate(
        (np.asarray(candidate_embeddings, dtype=np.float32), np.vstack(pool_embeddings), np.vstack(numeric_features)),
        axis=1,
    )
    player_ids = profiles["summoner_id"].astype(str).to_numpy()
    return LineupFeatureBank(
        player_ids=player_ids,
        rows={player_id: index for index, player_id in enumerate(player_ids)},
        features=features,
        valid=np.asarray(valid, dtype=bool),
        roles=profiles["role"].fillna("").astype(str).str.upper().to_numpy(),
    )


def build_team_metadata(
    two_tower: LoadedTwoTower,
    hidden_dim: int = 96,
    player_feature_dim: int | None = None,
    feature_groups: Iterable[str] = FEATURE_GROUPS,
    use_pair_interactions: bool = True,
) -> dict[str, Any]:
    vocabs = two_tower.metadata["vocabs"]
    champion_dim = int(two_tower.model.candidate_champion.embedding_dim)
    return {
        "format_version": 1,
        "architecture": "observed-complete-team_pair-interaction_reranker",
        "player_feature_dim": int(player_feature_dim) if player_feature_dim is not None else (
            int(two_tower.metadata["embedding_dim"]) + champion_dim + len(STYLE_COLUMNS) + len(POOL_FEATURES)
        ),
        "hidden_dim": int(hidden_dim),
        "feature_groups": list(feature_groups),
        "use_pair_interactions": bool(use_pair_interactions),
        "style_columns": list(STYLE_COLUMNS),
        "pool_features": list(POOL_FEATURES),
        "role_vocab_size": len(vocabs["role"]),
        "tier_vocab_size": len(vocabs["tier"]),
        "division_vocab_size": len(vocabs["division"]),
    }


class TeamLineupModel(nn.Module):
    """Jointly score four teammates using learned player and pair interactions."""

    def __init__(self, metadata: dict[str, Any]):
        super().__init__()
        hidden = int(metadata["hidden_dim"])
        self.use_pair_interactions = bool(metadata.get("use_pair_interactions", True))
        self.finder_role = nn.Embedding(int(metadata["role_vocab_size"]), 12)
        self.finder_tier = nn.Embedding(int(metadata["tier_vocab_size"]), 8)
        self.finder_division = nn.Embedding(int(metadata["division_vocab_size"]), 4)
        self.player_network = nn.Sequential(
            nn.Linear(int(metadata["player_feature_dim"]), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.pair_network = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        ) if self.use_pair_interactions else None
        team_input = hidden * 7 + 24
        if self.use_pair_interactions:
            team_input += len(PAIR_INDICES) * 2
        self.performance_network = nn.Sequential(
            nn.Linear(team_input, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        player_features: Tensor,
        present_mask: Tensor,
        finder_role: Tensor,
        finder_tier: Tensor,
        finder_division: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        players = self.player_network(player_features)
        mask = present_mask.float().unsqueeze(-1)
        masked = players * mask
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = masked.sum(dim=1) / count
        variance = (((players - mean.unsqueeze(1)) * mask) ** 2).sum(dim=1) / count
        std = torch.sqrt(variance + 1e-6)
        pair_masks: list[Tensor] = []
        for left, right in PAIR_INDICES:
            pair_masks.append(present_mask[:, left] & present_mask[:, right])
        valid_pairs = torch.stack(pair_masks, dim=1)
        context = torch.cat((
            self.finder_role(finder_role),
            self.finder_tier(finder_tier),
            self.finder_division(finder_division),
        ), dim=-1)
        team_parts = [masked.flatten(start_dim=1), mean, std]
        if self.use_pair_interactions:
            pair_logits: list[Tensor] = []
            assert self.pair_network is not None
            for left, right in PAIR_INDICES:
                a, b = players[:, left], players[:, right]
                pair_input = torch.cat((a, b, a * b, torch.abs(a - b)), dim=-1)
                pair_logits.append(self.pair_network(pair_input).squeeze(-1))
            pairs = torch.stack(pair_logits, dim=1)
            pair_count = valid_pairs.float().sum(dim=1).clamp_min(1.0)
            pair_mean = (pairs * valid_pairs.float()).sum(dim=1) / pair_count
            team_parts.extend((pairs * valid_pairs.float(), valid_pairs.float()))
        else:
            pairs = torch.zeros(
                (len(player_features), len(PAIR_INDICES)),
                dtype=players.dtype,
                device=players.device,
            )
            pair_mean = torch.zeros(len(player_features), dtype=players.dtype, device=players.device)
        team_parts.append(context)
        performance_logit = self.performance_network(torch.cat(team_parts, dim=-1)).squeeze(-1) + pair_mean
        return performance_logit, pairs, valid_pairs


def _lookup(vocab: dict[str, int], value: Any) -> int:
    if value is None or pd.isna(value):
        return vocab["<UNK>"]
    text = str(value).strip()
    return vocab.get(text.upper(), vocab.get(text, vocab["<UNK>"]))


def lineup_tensors(
    lineups: Iterable[dict[str, str]],
    finder_roles: Iterable[str],
    finder_tiers: Iterable[str | None],
    finder_divisions: Iterable[str | None],
    bank: LineupFeatureBank,
    two_tower_metadata: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Tensor]:
    lineup_list = list(lineups)
    role_list = list(finder_roles)
    tier_list = list(finder_tiers)
    division_list = list(finder_divisions)
    if not (len(lineup_list) == len(role_list) == len(tier_list) == len(division_list)):
        raise ValueError("Lineup and finder context lengths differ")
    feature_rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for lineup, finder_role in zip(lineup_list, role_list):
        if set(lineup) != set(ROLES).difference({finder_role}):
            raise ValueError("A scored lineup must contain exactly the four open roles")
        player_features = np.zeros((len(ROLES), bank.features.shape[1]), dtype=np.float32)
        present = np.zeros(len(ROLES), dtype=bool)
        for position, role in enumerate(ROLES):
            if role == finder_role:
                continue
            player_id = str(lineup[role])
            row = bank.rows.get(player_id)
            if row is None or not bank.valid[row] or bank.roles[row] != role:
                raise ValueError(f"Player {player_id} lacks complete real features for role {role}")
            player_features[position] = bank.features[row]
            present[position] = True
        feature_rows.append(player_features)
        masks.append(present)
    vocabs = two_tower_metadata["vocabs"]
    return {
        "player_features": torch.tensor(np.stack(feature_rows), dtype=torch.float32, device=device),
        "present_mask": torch.tensor(np.stack(masks), dtype=torch.bool, device=device),
        "finder_role": torch.tensor([_lookup(vocabs["role"], value) for value in role_list], dtype=torch.long, device=device),
        "finder_tier": torch.tensor([_lookup(vocabs["tier"], value) for value in tier_list], dtype=torch.long, device=device),
        "finder_division": torch.tensor([_lookup(vocabs["division"], value) for value in division_list], dtype=torch.long, device=device),
    }


@dataclass
class LoadedTeamModel:
    model: TeamLineupModel
    metadata: dict[str, Any]
    device: str

    @classmethod
    def load(
        cls,
        model_path: Path,
        metadata_path: Path,
        two_tower_model_path: Path,
        requested_device: str = "auto",
    ) -> "LoadedTeamModel":
        if not model_path.exists() or not metadata_path.exists():
            raise RuntimeError("Trained team-model artifacts are missing; run `python -m src.recsys.train_team`")
        device = requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("two_tower_model_sha256")
        actual = artifact_sha256(two_tower_model_path)
        if not expected or expected != actual:
            raise RuntimeError("Team model does not match the current two-tower checkpoint; retrain the team model")
        model = TeamLineupModel(metadata).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return cls(model=model, metadata=metadata, device=device)


def save_team_artifact(
    model: TeamLineupModel,
    metadata: dict[str, Any],
    model_path: Path,
    metadata_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary_model)
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_model.replace(model_path)
    temporary_metadata.replace(metadata_path)
