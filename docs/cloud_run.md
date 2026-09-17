# Google Cloud Run: CPU, request-based billing, scale to zero

This deployment keeps the existing Gradio UI, trained rankers, Chroma retrieval,
and local Qwen generation. No API key, GPU, retraining, raw-match upload, or
synthetic recommendations are involved. Hugging Face deployment remains supported.

## What is (and is not) verified

The local tests verify packaging, real artifact checksums, startup port handling,
cache-download contracts and deployment settings. They do not prove a successful
Google Cloud build or Cloud Run inference. Test both recommendations and scout
reports in the deployed service before sharing it. 2 vCPUs and 8 GiB are starting
allocations, not measured minimum requirements. No cloud resources are created by
running the packaging command below.

## 1. Get the clean upload bundle

Use `dist/lol-cloud-run.zip`, or recreate it from the project root:

```powershell
python scripts/package_space.py --target cloud-run
```

Packaging refuses missing or mismatched real artifacts and refuses to overwrite an
existing ZIP. Use `--output dist/lol-cloud-run-v2.zip` for a later version.
Only source, notices, real compact profiles, vectors, trained checkpoints and
deployment files are included. `.env`, raw `data/`, `.venv`, caches and git history
are excluded. Review dataset/font terms and public player identifiers before
publishing; a public demo exposes the displayed player evidence.

The Cloud Run ZIP has a root `Dockerfile`, `.dockerignore`, `.gcloudignore`,
`deploy-cloud-run.sh`, and this guide as `CLOUD_RUN.md`. Do not upload your whole
working directory. These files are intentionally separate from the HF bundle.

## 2. Prepare Google Cloud

1. Create/select a project at https://console.cloud.google.com/.
2. Link the billing account that contains your promotional credit. Check the
   credit's expiration and eligible services.
3. Configure billing alerts before deployment. Alerts are not a hard spending cap.
4. Open Cloud Shell (terminal icon in the console). This is a Linux Bash terminal;
   the following commands are not Windows PowerShell commands.
5. Use Cloud Shell's upload action to upload only `lol-cloud-run.zip`.

Extract into a NEW directory so old files cannot mix into the deployment:

```bash
mkdir league-cloud-run
unzip lol-cloud-run.zip -d league-cloud-run
cd league-cloud-run
bash deploy-cloud-run.sh YOUR_PROJECT_ID --dry-run
```

Replace `YOUR_PROJECT_ID` with the actual project ID, not its display name.
The dry run validates artifacts and prints the command without contacting Google.
Optional overrides: `REGION=us-central1` and `SERVICE_NAME=league-team-recommender`.

## 3. Build and deploy privately

```bash
bash deploy-cloud-run.sh YOUR_PROJECT_ID
```

Review the displayed project/region and type `DEPLOY` only when ready to incur
build/storage/usage charges. The script enables Cloud Run, Cloud Build and Artifact
Registry APIs, then deploys using the Dockerfile. Google performs the heavy build;
Docker does not need to run on your laptop.

The first build downloads CPU-only PyTorch plus MiniLM and Qwen model files. This
can take several minutes. Models are cached in the container image, not loaded
into inference RAM during the download step. Startup uses offline mode: a missing
model file is a real failure, not an invitation to silently download a fallback.
`/opt/hf-cache/build-model-manifest.json` records resolved model snapshot revisions.

### If Google reports an IAM/build permission error

Do not make the project public or grant Owner/Editor to work around it. Source
deployment requires a builder identity with `roles/run.builder`; the deployer
also needs the source-deployment and service-account-use permissions described in
[Google's source deployment documentation](https://docs.cloud.google.com/run/docs/deploying-source-code).
For a new personal project, Google commonly uses the Compute Engine default
service account as the builder. If the error identifies that account, an account
administrator can grant the documented builder role:

```bash
gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)'
# Substitute the returned number below; confirm it matches the error's builder.
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member='serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com' \
  --role='roles/run.builder'
```

Wait for IAM propagation and rerun deployment. Do not grant roles to an identity
you have not identified in your project. No IAM grants are automated by this bundle.

## 4. Test the private service

The deploy script leaves the service private. In Cloud Shell, start an authenticated
proxy (install the offered proxy component if prompted):

```bash
gcloud run services proxy league-team-recommender \
  --project YOUR_PROJECT_ID --region us-central1 --port 8080
```

Open Cloud Shell Web Preview on port 8080. Keep the proxy terminal running while
testing. If you changed REGION/SERVICE_NAME, use your values here and below.

Check all of these:

- The League-themed UI opens with Fill / Platinum IV defaults.
- Build a lineup, including a nonempty natural-language preference to exercise
  semantic retrieval. Displayed candidates must come from the bundled profiles.
- Generate both a whole-lineup and a candidate scout report. Check that each cites
  the supplied player evidence and does not present ranking scores as win odds.
- Observe memory and latency in Cloud Run Metrics. A page loading successfully
  does not prove that report generation fits in memory or finishes in time.
- Leave the tab idle after results finish and check the idle-network procedure
  below. Revisit after scale-down to check cold-start behavior. Stop the proxy
  when you finish private testing.

Read runtime logs if a report fails:

```bash
gcloud run services logs read league-team-recommender \
  --project YOUR_PROJECT_ID --region us-central1 --limit 100
```

Inspect the deployed service configuration:

```bash
gcloud run services describe league-team-recommender \
  --project YOUR_PROJECT_ID --region us-central1 --format=yaml
```

This CPU image must not use `SCOUT_DEVICE=zero_gpu`, a GPU allocation, or user-set
CUDA visibility overrides. Keep the baked model IDs unchanged unless you rebuild
the matching cache. The trained recommendation artifacts are not retrained here.

## 5. Make it public only after testing (optional)

This grants everyone permission to invoke the service, including inference that
can consume credits. Only run it if you want a public portfolio demo:

```bash
gcloud run services add-iam-policy-binding league-team-recommender \
  --project YOUR_PROJECT_ID --region us-central1 \
  --member=allUsers --role=roles/run.invoker
gcloud run services describe league-team-recommender \
  --project YOUR_PROJECT_ID --region us-central1 --format='value(status.url)'
```

If your university/organization policy forbids public access, do not disable the
policy. Ask the administrator or use a project in which public hosting is permitted.

To revoke that public grant later:

```bash
gcloud run services remove-iam-policy-binding league-team-recommender \
  --project YOUR_PROJECT_ID --region us-central1 \
  --member=allUsers --role=roles/run.invoker
```

## Runtime and cost settings

| Setting | Default | Why |
| --- | --- | --- |
| Billing | Request-based (`--cpu-throttling`) | No CPU allocation for arbitrary background work while idle |
| Minimum instances | 0, at service and revision level | Permit idle scale-to-zero |
| Maximum instances | 1, at service and revision level | Limit normal scaling, not a hard financial cap |
| CPU / RAM | 2 / 8 GiB | Initial allowance for CPU rankers, encoder, Qwen and transient tensors |
| HTTP concurrency | 8 | Gradio needs multiple HTTP/SSE connections; do not set this to 1 |
| Gradio inference concurrency | 1 shared queue | Serialize expensive model operations |
| Idle browser connection | None | Tab-local JSON replaces server session state; no session heartbeat or polling timer |
| Session affinity | Enabled | Help route a session to the same in-memory queue |
| Request timeout | 900 seconds | Initial allowance for CPU scout generation |
| Startup probe | TCP, up to 240 seconds | Allow real-index and encoder initialization |
| Scout output budget | 192 tokens | Shorter CPU-generated reports; may end at the token limit |

Cold starts still load models into memory. The recommendations warm before launch;
Qwen loads lazily on the first report. Chroma is rebuilt in `/tmp/lol-chroma` from
the bundled real vectors after an instance is replaced, without re-embedding the
player documents. Temporary files count against container memory and are not durable.
Never put new user data or authoritative training artifacts only in this directory.

Successful candidate and lineup reports are cached in process (up to 64 entries).
Candidate and lineup reports have explicit, separate cache namespaces and UI outputs.
An identical report-type/evidence/context/model request reuses the cached text; errors are never
cached. Source evidence is revalidated before cache lookup. The cache does not survive
scale-to-zero or restarts. For new requests, CPU generation can still take minutes.
Look for `[scout]` log lines: `load` is model initialization time, `input_tokens` is
the prompt length, `generation` is inference time, and `report=cache_hit` means no
language-model inference was repeated. Queue wait time is additional. Prompts avoid
duplicate profile metadata and default to three short bullets (192-token limit).

For local tests without redeploying, run from the repository in PowerShell:

```powershell
$env:LOL_UI_BACKEND="local"
$env:SCOUT_DEVICE="cpu"
$env:SPACES_ZERO_GPU="0"
$env:SCOUT_MAX_NEW_TOKENS="192"
$env:GRADIO_SERVER_NAME="127.0.0.1"
$env:PORT="7860"
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:7860`, build a real lineup and click a card's **Why this
candidate?** button or the whole-lineup scout. The first local report may download
Qwen weights and needs several GB of free RAM; subsequent reports reuse the loaded
model. This is a functionality check, not a Cloud Run performance benchmark.

Keep the browser connected while generating reports. Request-based CPU may pause
work after clients disconnect; this deployment is not a durable background-job
system. Active queued jobs live in memory and can be lost when a container
restarts. Completed lineup display state stays in that browser tab and is sent
with each explicit scout request, so scouting does not require the old server
session. Each scout revalidates player IDs and retrieves server-owned evidence;
browser-supplied statistics are not accepted as evidence. Refreshing the page
resets the live lineup; the app does not use `gr.BrowserState`/local storage to
restore it. Gradio's separate built-in Runs history may save calls in your browser;
it is not needed to keep the server session alive. A changed deployment/dataset
can invalidate an old lineup, in which case rebuild it.

### Idle tabs do not keep a request open

The UI intentionally uses a hidden `gr.JSON` component instead of `gr.State`.
For the pinned Gradio version, this disables the otherwise persistent session
`/gradio_api/heartbeat/...` connection. No polling timers, automatic inference,
or keep-alive pings are registered. Startup fails if a UI change reintroduces a
heartbeat, timer, server state, or automatic backend listener. Queue SSE is still
used while a button request runs, then closes after completion (also on error).
An idle tab may remain open without keeping this application's request active.

After deploying this version, **refresh any tabs opened on an older version**;
their already-loaded frontend can still use the old heartbeat behavior. Deploy
to the same service name, check that the new revision receives all traffic, and
avoid using an old revision's tagged URL. Existing requests on an old revision
may continue until disconnected or timed out.

To verify on the deployed service:

1. Open browser Developer Tools → Network, then refresh the page once.
2. After page assets load, there should be no pending `heartbeat` request and no
   recurring application requests while idle. Do not enable automatic reload.
3. Click Build my team, then test both scout buttons. `/queue/data` is expected
   while queued/running, but must finish when each result or error arrives.
4. Leave the tab open without clicking. Confirm no new application traffic and
   observe Cloud Run request metrics. Idle scale-down is not instantaneous.

Local HTTP tests exercise the actual Gradio queue with isolated inference test
doubles, including a new browser session for each scout, separate report outputs,
and stream EOF after success/error. They do not measure a deployed Google bill.
Do not run an uptime monitor against the service (including `/health`) to test
idleness: those requests themselves can wake it up.

Scale-to-zero does NOT mean the entire project has zero idle cost. Container image
storage, build/source storage, builds, logs and networking may incur separate charges.
Actual page loads, queued/running inference, and public/bot requests are still
billable activity. Closing a tab is not a guarantee of immediate cancellation of
an already-running Python/model operation. Do not add uptime pings, a minimum warm
instance, a load balancer, a VPC connector, or a GPU just to keep the demo awake.
Startup and shutdown time can also be billed. A max-instance setting reduces
exposure but is not a hard cap; public traffic can keep the service active.

Track actual billing after testing; do not assume a guaranteed free monthly bill.
Budget alerts alone do not cap spending. If your billing account offers Google's
[Cloud Run budget spend caps](https://docs.cloud.google.com/run/docs/configuring/billing-settings#budget-spend-caps),
consider enabling one separately; this preview control pauses Cloud Run workloads
at the cap, but is not a cap on all project services such as image storage/builds.
Set storage cleanup policies once you know which images are needed for rollback.
Removing a Cloud Run service alone does not remove its stored build images.

## Build locally only if you want to

The intended path above builds in Google Cloud. An optional local Docker test,
from the extracted bundle, downloads large dependencies/model files and uses RAM:

```bash
docker build -t league-team-recommender .
docker run --rm --cpus=2 --memory=8g -p 127.0.0.1:8080:8080 \
  -e PORT=8080 league-team-recommender
```

No Docker build or cloud deployment is performed by the unit tests.

## References

- [Gradio Docker deployment](https://www.gradio.app/guides/deploying-gradio-with-docker)
- [Cloud Run source deployment and IAM](https://docs.cloud.google.com/run/docs/deploying-source-code)
- [Cloud Run deploy flags](https://docs.cloud.google.com/sdk/gcloud/reference/run/deploy)
- [Container port, startup and filesystem contract](https://docs.cloud.google.com/run/docs/container-contract)
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
- [Billing budgets](https://docs.cloud.google.com/billing/docs/how-to/budgets)
