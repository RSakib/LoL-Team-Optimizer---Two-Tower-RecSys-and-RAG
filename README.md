# League of Legends Role Queue Team Builder

A real-data-only system that fills a League player’s open team roles. The finder queues as **Top, Jungle, Mid,
Bottom, Support, or Fill**. The engine reserves the finder’s assigned role and ranks authentic candidates for every
other role using profiles computed from local Riot match records.

## Product contract

The requester supplies:

- Primary queue role: Top, Jungle, Mid, Bottom, Support, or Fill.
- Optional requested champion for each open role.
- Optional rank constraints and natural-language team preference.

The requester is a reserved team slot, not a candidate profile. For a fixed primary role, the UI returns candidates
for the other four slots. For **Fill**, it evaluates all five possible requester assignments and returns the strongest
set of teammate recommendations, along with the other scenario scores. The API retains an optional `finder` identifier
only for clients that want to exclude that player from candidates or infer their tier from an existing profile.

Candidates are hard-filtered by their most recorded primary role and rank eligibility. Candidates are ranked using:

- Role-relative recorded experience.
- Role-relative historical performance.
- Rank proximity.
- Historical affinity for an optionally requested champion.
- Chroma retrieval similarity for an optional natural-language preference.

Missing data never triggers fabricated candidates or fallback recommendations.

## Real-data contract

Place Riot API CSV, JSON, JSONL, or NDJSON exports in `data/`. Supported sources include nested Match-V5 objects,
wide participant CSV exports, flat participant tables, PUUID rank tables, and match-rank tables.

The production preprocessor streams `data/matchData.csv` in bounded chunks, restricts profiles to the ranked PUUID
cohort, writes `data/processed/player_profiles.jsonl`, and indexes the computed player narratives in `.chroma/`.
It does not load the large raw export into memory.

```powershell
python -m src.data.preprocess_large --chunk-size 100
```

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

To enable optional tactical scout reports, set `OPENAI_API_KEY`. The LLM receives only the selected real profile,
requested role, requested champion, and user preference. Without a key, `/scout` returns an explicit configuration
error and never substitutes invented text.

## Run

Start FastAPI on the project’s configured port:

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
```

Start Streamlit in another terminal:

```powershell
.\.venv\Scripts\streamlit.exe run src/ui/app.py
```

API documentation is available at `http://127.0.0.1:8001/docs`.

## API

- `GET /health` reports the loaded real profiles and counts by primary role.
- `POST /team/recommend` fills the finder’s four open role slots.
- `POST /recommend` is a deprecated alias using the same team-building request.
- `POST /scout` generates a fact-grounded report for one candidate in a requested role slot.

Example request:

```json
{
  "finder": "Example Riot Name",
  "primary_role": "MID",
  "target_champions": {
    "JUNGLE": "Vi",
    "SUPPORT": "Nautilus"
  },
  "preference": "Reliable objective control and vision",
  "candidates_per_role": 3,
  "max_tier_gap": 1
}
```

If one role has no eligible real candidate, the response is marked `partial` and identifies the missing role. If no
role has a candidate, the response is `no_matches`.

## Role-queue evaluation

The evaluation matches the product: given a finder occupying one role in a future match, rank candidates for each
other role. It separately measures all observed future teammates and teammates from successful future lineups.

```powershell
python -m src.evaluation.role_queue --weight-trials 256 --validation-queries 2500 --test-queries 5000
```

The protocol uses chronological train/validation/test windows. Candidate profiles come only from earlier matches,
weights are selected on validation NDCG@5, and the final report includes NDCG@5, Recall@5, MRR, hard baselines,
ablations, and match-clustered confidence intervals.

Outputs:

- `docs/role_queue_evaluation.md`
- `data/evaluation/role_queue_metrics.csv`
- `data/evaluation/role_queue_report.json`

## Tests

```powershell
pytest -q
```

Tests cover Riot-schema parsing, real aggregation, role hard-filtering, four-slot team completion, Fill assignment,
champion affinity, temporal separation, and empty-result behavior.
