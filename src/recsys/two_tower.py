from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROLES = ("TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT")
QUEUE_ROLES = (*ROLES, "FILL")
ROLE_ALIASES = {"ADC": "BOTTOM", "BOT": "BOTTOM", "MIDDLE": "MID", "UTILITY": "SUPPORT"}
TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER")
DIVISIONS = ("IV", "III", "II", "I")
TIER_ORDER = {tier: index for index, tier in enumerate(TIERS)}
NUMERIC_COLUMNS = (
    "matches", "win_rate", "kda", "avg_kills", "avg_deaths", "avg_assists",
    "avg_vision_score", "avg_gold_earned", "avg_damage_dealt",
)


def canonical_role(role: str) -> str:
    value = ROLE_ALIASES.get(str(role).strip().upper(), str(role).strip().upper())
    if value not in ROLES:
        raise ValueError(f"Unknown League role: {role}")
    return value


def canonical_queue_role(role: str) -> str:
    value = ROLE_ALIASES.get(str(role).strip().upper(), str(role).strip().upper())
    if value not in QUEUE_ROLES:
        raise ValueError(f"Unknown League queue role: {role}")
    return value


def _vocabulary(values: list[str], special: tuple[str, ...] = ("<UNK>",)) -> dict[str, int]:
    ordered = list(dict.fromkeys((*special, *sorted({value for value in values if value}))))
    return {value: index for index, value in enumerate(ordered)}


def _main_champion(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "<UNK>"
    return str(max(value.items(), key=lambda item: float(item[1]))[0])


def numeric_matrix(profiles: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in NUMERIC_COLUMNS:
        values = pd.to_numeric(profiles[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if name in {"matches", "kda", "avg_vision_score", "avg_gold_earned", "avg_damage_dealt"}:
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)
    return np.column_stack(columns).astype(np.float32)


def build_metadata(
    profiles: pd.DataFrame, embedding_dim: int, hidden_dim: int,
    extra_champions: Iterable[str] | None = None,
) -> dict[str, Any]:
    raw_numeric = numeric_matrix(profiles)
    means = raw_numeric.mean(axis=0)
    scales = raw_numeric.std(axis=0)
    scales[scales < 1e-6] = 1.0
    champions = [_main_champion(value) for value in profiles["top_champions"]]
    champions.extend(str(value) for value in (extra_champions or []) if value)
    return {
        "format_version": 1,
        "architecture": "anonymous-context-query-tower_and_real-player-candidate-tower",
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "numeric_columns": list(NUMERIC_COLUMNS),
        "numeric_mean": means.tolist(),
        "numeric_scale": scales.tolist(),
        "vocabs": {
            "player": _vocabulary(profiles["summoner_id"].astype(str).tolist()),
            "role": _vocabulary(list(ROLES)),
            "tier": _vocabulary(list(TIERS)),
            "division": _vocabulary(list(DIVISIONS)),
            "champion": _vocabulary(champions, special=("<UNK>", "<ANY>")),
        },
    }


class TwoTowerModel(nn.Module):
    """Dual encoder trained with positive teammate pairs and role-matched negatives."""

    def __init__(self, metadata: dict[str, Any]):
        super().__init__()
        vocabs = metadata["vocabs"]
        output = int(metadata["embedding_dim"])
        hidden = int(metadata["hidden_dim"])
        self.query_role = nn.Embedding(len(vocabs["role"]), 12)
        self.query_tier = nn.Embedding(len(vocabs["tier"]), 8)
        self.query_division = nn.Embedding(len(vocabs["division"]), 4)
        self.query_target_role = nn.Embedding(len(vocabs["role"]), 12)
        self.query_champion = nn.Embedding(len(vocabs["champion"]), 24)
        self.candidate_player = nn.Embedding(len(vocabs["player"]), 32)
        self.candidate_role = nn.Embedding(len(vocabs["role"]), 12)
        self.candidate_tier = nn.Embedding(len(vocabs["tier"]), 8)
        self.candidate_division = nn.Embedding(len(vocabs["division"]), 4)
        self.candidate_champion = nn.Embedding(len(vocabs["champion"]), 24)
        self.query_network = nn.Sequential(
            nn.Linear(60, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(hidden, output),
        )
        self.candidate_network = nn.Sequential(
            nn.Linear(80 + len(NUMERIC_COLUMNS), hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(hidden, output),
        )

    def encode_query(
        self, finder_role: Tensor, finder_tier: Tensor, finder_division: Tensor,
        target_role: Tensor, target_champion: Tensor,
    ) -> Tensor:
        features = torch.cat((
            self.query_role(finder_role), self.query_tier(finder_tier),
            self.query_division(finder_division), self.query_target_role(target_role),
            self.query_champion(target_champion),
        ), dim=-1)
        return F.normalize(self.query_network(features), dim=-1)

    def encode_candidate(
        self, player: Tensor, role: Tensor, tier: Tensor, division: Tensor,
        champion: Tensor, numeric: Tensor,
    ) -> Tensor:
        features = torch.cat((
            self.candidate_player(player), self.candidate_role(role), self.candidate_tier(tier),
            self.candidate_division(division), self.candidate_champion(champion), numeric,
        ), dim=-1)
        return F.normalize(self.candidate_network(features), dim=-1)


def _index(vocab: dict[str, int], value: Any, unknown: str = "<UNK>") -> int:
    if value is None or pd.isna(value):
        return vocab[unknown]
    return vocab.get(str(value).strip().upper(), vocab.get(str(value).strip(), vocab[unknown]))


def candidate_tensors(
    profiles: pd.DataFrame, metadata: dict[str, Any], device: str | torch.device,
) -> dict[str, Tensor]:
    vocabs = metadata["vocabs"]
    raw = numeric_matrix(profiles)
    numeric = (raw - np.asarray(metadata["numeric_mean"], dtype=np.float32)) / np.asarray(
        metadata["numeric_scale"], dtype=np.float32
    )
    return {
        "player": torch.tensor([_index(vocabs["player"], value) for value in profiles["summoner_id"]], dtype=torch.long, device=device),
        "role": torch.tensor([_index(vocabs["role"], value) for value in profiles["role"]], dtype=torch.long, device=device),
        "tier": torch.tensor([_index(vocabs["tier"], value) for value in profiles["tier"]], dtype=torch.long, device=device),
        "division": torch.tensor([_index(vocabs["division"], value) for value in profiles["rank"]], dtype=torch.long, device=device),
        "champion": torch.tensor([_index(vocabs["champion"], _main_champion(value)) for value in profiles["top_champions"]], dtype=torch.long, device=device),
        "numeric": torch.tensor(numeric, dtype=torch.float32, device=device),
    }


def query_tensors(
    metadata: dict[str, Any], finder_role: str, finder_tier: str | None,
    finder_division: str | None, target_role: str, target_champion: str | None,
    device: str | torch.device,
) -> dict[str, Tensor]:
    vocabs = metadata["vocabs"]
    champion = target_champion if target_champion else "<ANY>"
    return {
        "finder_role": torch.tensor([_index(vocabs["role"], finder_role)], dtype=torch.long, device=device),
        "finder_tier": torch.tensor([_index(vocabs["tier"], finder_tier)], dtype=torch.long, device=device),
        "finder_division": torch.tensor([_index(vocabs["division"], finder_division)], dtype=torch.long, device=device),
        "target_role": torch.tensor([_index(vocabs["role"], target_role)], dtype=torch.long, device=device),
        "target_champion": torch.tensor([_index(vocabs["champion"], champion)], dtype=torch.long, device=device),
    }


@dataclass
class LoadedTwoTower:
    model: TwoTowerModel
    metadata: dict[str, Any]
    device: str

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path, requested_device: str = "auto") -> "LoadedTwoTower":
        if not model_path.exists() or not metadata_path.exists():
            raise RuntimeError("Trained two-tower artifacts are missing; run `python -m src.recsys.train`")
        device = requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = TwoTowerModel(metadata).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return cls(model=model, metadata=metadata, device=device)


def save_artifact(
    model: TwoTowerModel, metadata: dict[str, Any], model_path: Path, metadata_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary_model)
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_model.replace(model_path)
    temporary_metadata.replace(metadata_path)
