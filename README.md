# Production LoL Matchmaker & Tactical Scout

A real-data-only teammate recommender built from local Riot Match-V5 records. It aggregates recorded player performance, indexes fact-grounded profile narratives in ChromaDB, scores compatibility and observed duo outcomes, exposes FastAPI endpoints, and renders results in Streamlit.

## Data contract

Put Riot API CSV, JSON, JSONL, or NDJSON files in `data/`. The loader supports:

- Riot Match-V5 objects containing `metadata.matchId` and `info.participants`.
- Wide CSV exports with columns such as `participant0SummonerId`, `participant0ChampionName`, and `participant0Win`.
- Flat participant tables using snake_case or Riot field names.
- Rank tables keyed by `puuid`, and match-rank tables keyed by `matchId`.

The participant facts used are `summoner_id`, `tier`, `rank`, `role`, `champion_name`, `kills`, `deaths`, `assists`, `vision_score`, `gold_earned`, `damage_dealt`, and `win`. The loader joins tier metadata when it is available, preserves unavailable values as missing, and never creates a row or recommendation to fill a gap. Overlapping JSONL and CSV exports are deduplicated by `(match_id, summoner_id)`.

`data/processed/player_profiles.jsonl` is derived output. It is never used as a raw source and is rebuilt from match facts.

For this repository, `match_data.jsonl` and `matchData.csv` are duplicate representations of the same Riot matches. The production preprocessor intentionally reads the smaller wide CSV and skips the 8.2 GB JSONL. It selects only required participant columns, processes 100 matches at a time, and restricts recommendation profiles to the real ranked PUUID cohort in `players_8-14-25.csv`. This avoids indexing thousands of one-match bystanders while preserving authentic candidates.

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

To enable tactical scout reports, set `OPENAI_API_KEY` in your environment. The integration uses the OpenAI Responses API with only the selected candidate's computed facts in its prompt. Without a key, `/scout` returns an explicit configuration error; it does not generate substitute text. See the [official Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create).

## Build the real-data index

```powershell
python -m src.data.preprocess_large --chunk-size 100
```

This bounded-memory command streams `data/matchData.csv`, writes `data/processed/player_profiles.jsonl`, aggregates observed ranked-player duo outcomes in `data/processed/duo_synergy.sqlite`, records counts in `data/processed/manifest.json`, and rebuilds `.chroma/` in batches of 128 profiles. The deterministic 768-feature hashing embedder requires no model download or fitting corpus in memory.

Train the optional requested XGBoost compatibility layer from the compact duo table (single-threaded; no raw-data scan):

```powershell
python -m src.recsys.xgb_model
```

When `data/processed/compatibility_xgb.json` exists, its real historical duo prediction is blended with the matrix similarity score. If it is absent, deterministic scoring continues; candidates are never fabricated.

Normal API and UI startup read only these compact artifacts. They never rescan the raw CSV or JSONL.

## Run

In one terminal:

```powershell
uvicorn src.api.main:app --reload
```

In another:

```powershell
streamlit run src/ui/app.py
```

API documentation is available at `http://127.0.0.1:8000/docs`.

### Endpoints

- `GET /health` reports loaded real match and profile counts.
- `POST /recommend` accepts identity, role, champion, tier/division, preference text, and result count.
- `POST /scout` generates a fact-grounded tactical report for an existing real candidate.

When filtering or retrieval leaves no eligible player, `/recommend` returns:

```json
{
  "status": "no_matches",
  "message": "No matching real candidates found",
  "count": 0,
  "candidates": []
}
```

## Tests

```powershell
pytest -q
```

The tests assert Riot-schema parsing, rank joins, cross-file deduplication, real metric aggregation, self-exclusion, observed duo scoring, and the required empty-result behavior.
