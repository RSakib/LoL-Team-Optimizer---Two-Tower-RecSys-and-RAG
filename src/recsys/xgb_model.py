from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DUO_DB_PATH, PROFILE_PATH, XGB_MODEL_PATH

METRICS = ["win_rate", "kda", "avg_vision_score", "avg_gold_earned", "avg_damage_dealt"]


def pair_features(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    a = left[METRICS].fillna(0.0).to_numpy(dtype=float)
    b = right[METRICS].fillna(0.0).to_numpy(dtype=float)
    return np.hstack(((a + b) / 2.0, np.abs(a - b)))


def train_model(profile_path: Path = PROFILE_PATH, duo_path: Path = DUO_DB_PATH,
                model_path: Path = XGB_MODEL_PATH) -> dict[str, int | float]:
    from xgboost import XGBRegressor

    profiles = pd.read_json(profile_path, lines=True).set_index("summoner_id")
    with sqlite3.connect(duo_path) as connection:
        pairs = pd.read_sql_query("SELECT player_a, player_b, games, wins FROM duo_synergy WHERE games >= 2", connection)
    pairs = pairs[pairs["player_a"].isin(profiles.index) & pairs["player_b"].isin(profiles.index)].reset_index(drop=True)
    if len(pairs) < 100:
        raise RuntimeError("Not enough observed real duo pairs to train the compatibility model")
    left = profiles.loc[pairs["player_a"]].reset_index(drop=True)
    right = profiles.loc[pairs["player_b"]].reset_index(drop=True)
    features = pair_features(left, right)
    target = (pairs["wins"] / pairs["games"]).to_numpy(dtype=float)
    weights = pairs["games"].to_numpy(dtype=float)
    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=120, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=1,
    )
    model.fit(features, target, sample_weight=weights)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    prediction = np.clip(model.predict(features), 0.0, 1.0)
    return {"training_pairs": len(pairs), "weighted_games": int(weights.sum()),
            "training_mae": float(np.average(np.abs(prediction - target), weights=weights))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-threaded XGBoost compatibility model from compact real duo data")
    parser.parse_args()
    print(train_model())


if __name__ == "__main__":
    main()
