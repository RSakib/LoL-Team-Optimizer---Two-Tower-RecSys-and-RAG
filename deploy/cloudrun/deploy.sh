#!/usr/bin/env bash
# Run manually in Cloud Shell from an extracted Cloud Run bundle.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

project_id="${1:-}"
region="${REGION:-us-central1}"
service_name="${SERVICE_NAME:-league-team-recommender}"
if [[ -z "$project_id" || "$project_id" == "YOUR_PROJECT_ID" ]]; then
  printf '%s\n' 'Usage: bash deploy-cloud-run.sh YOUR_PROJECT_ID [--dry-run]'
  exit 2
fi
if [[ ! -f Dockerfile || ! -f deployment/manifest.json ]]; then
  printf '%s\n' 'Use the extracted --target cloud-run ZIP, not the raw repository.' >&2
  exit 2
fi
python3 -c "from pathlib import Path; from src.deployment import validate_bundle; validate_bundle(Path('.'))"

# Private by default. Public exposure is a separate, explicit step in CLOUD_RUN.md.
deploy=(gcloud run deploy "$service_name" --project "$project_id" --region "$region"
  --source . --port 8080 --cpu 2 --memory 8Gi
  --min 0 --max 1 --min-instances 0 --max-instances 1
  --concurrency 8 --cpu-throttling --session-affinity --timeout 900
  --startup-probe 'tcpSocket.port=8080,periodSeconds=10,timeoutSeconds=1,failureThreshold=24'
  --set-env-vars 'SCOUT_DEVICE=cpu,SPACES_ZERO_GPU=0,SCOUT_MAX_NEW_TOKENS=192,LOL_TWO_TOWER_DEVICE=cpu,RAG_EMBEDDING_DEVICE=cpu,LOL_UI_BACKEND=local'
  --no-allow-unauthenticated)

if [[ "${2:-}" == "--dry-run" ]]; then
  printf '%s\n' 'Dry run: no upload, API changes, deployment, or charges.'
  printf '%q ' "${deploy[@]}"
  printf '\n'
  exit 0
fi
if [[ $# -gt 1 ]]; then
  printf '%s\n' 'Only --dry-run is accepted as an optional second argument.' >&2
  exit 2
fi
command -v gcloud >/dev/null || { printf '%s\n' 'Run this script in Google Cloud Shell.' >&2; exit 2; }
printf 'Project: %s\nRegion: %s\nService: %s\n' "$project_id" "$region" "$service_name"
printf '%s\n' 'This uploads the clean bundle, enables APIs, and builds/deploys a private service.'
printf '%s\n' 'Build, storage and usage charges may apply. Confirm your credit and budget first.'
read -r -p 'Type DEPLOY to continue: ' confirmation
[[ "$confirmation" == "DEPLOY" ]] || { printf '%s\n' 'Cancelled.'; exit 1; }

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$project_id"
"${deploy[@]}"
printf '%s\n' 'Deployment finished. Follow CLOUD_RUN.md to test privately before sharing.'
