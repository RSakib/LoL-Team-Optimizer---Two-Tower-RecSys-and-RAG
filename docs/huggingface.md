# Hugging Face deployment

## What runs in the Space

One Gradio process runs the unchanged trained two-tower retriever and joint-lineup scorer on CPU.
MiniLM embeds real profile summaries for Chroma. The scout retrieves the selected players' exact documents,
then **Qwen2.5-0.5B-Instruct** generates the explanation inside the Space using Transformers.
This is retrieval-augmented generation, not a paid inference API or a connection to your laptop.
The player cards, Fill/Platinum IV defaults, and both scout tabs remain available.
The interface is titled **League of Legends: Team Recommender**, with navy, teal, and gold styling.
The bundled Beaufort for LoL bold font is embedded locally in CSS and used only for the main title;
no external font service is needed. Keep `assets/fonts/` in your upload. Font attribution and applicable
terms are documented in `assets/fonts/NOTICE.md`; the font is not relicensed as project code.
Live matchmaking accepts only Platinum and higher; widening the tier gap never admits lower-ranked candidates.

Qwen is the default instead of Gemma to avoid gated-model approval or a download token. No OpenAI key is needed.
It is a small model: expect limited reasoning and possible unsupported prose. Consult the displayed evidence;
the generator cannot add candidates or change rankings. Retrieval metrics in the README do **not** evaluate
this replacement generator. Existing trained checkpoint and retrieval evaluations have not been rerun or altered.

## Free hosting eligibility (checked September 16, 2026)

Hugging Face's [current ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) says personal
accounts in good standing with a verified email and an account older than 30 days can host up to two
ZeroGPU Gradio Spaces free. GPU usage has daily quotas and queues; free hosting is not unlimited inference.
Do not enable paid hardware or purchase credits for this deployment.

The app also supports CPU Basic without changing code (slower reports, no GPU allocation).
The [current Spaces overview](https://huggingface.co/docs/hub/spaces-overview) lists CPU Basic as having no
hourly charge, but creating a new CPU compute Space requires a paid account plan. Check the options shown
for your account. If you are not eligible for either free option, this repository cannot bypass that restriction;
run locally until eligible. Model weights being free does not guarantee that a hosting account is eligible.

## Build the upload bundle

From this repository with the existing real processed data and trained checkpoints:

```powershell
python scripts/export_rag_embeddings.py
python scripts/package_space.py
```

This creates `dist/lol-huggingface-space.zip` with code, dependency list, Space metadata, documentation,
trained checkpoints/metadata, compact real-profile vectors, and `data/processed/player_profiles.jsonl` copied to
`deployment/player_profiles.jsonl`. It includes SHA-256 integrity metadata and validates checkpoint pairing.
Manifest version 3 also requires `deployment/rag_embeddings.npz`, a portable export of existing Chroma vectors.
The export verifies that indexed documents match the real profiles and does not run an embedding model.
It fails if the existing index is stale; explicitly reindex real profiles before exporting in that case.
The manifest tolerates CRLF/LF conversion of JSON/JSONL during Windows-to-Linux uploads, but still
rejects missing files, changed data, and any modified checkpoint bytes. Upload the validator, manifest and
artifacts from the same bundle; version 1 manifests do not include the newline-normalized checksums.
It never includes `.env`, `.venv`, raw matches, `.git`, model-download caches, or the local Chroma database.
No training or raw-data preprocessing is performed. If required real artifacts are missing, packaging fails.
Existing ZIPs are not overwritten; pass `--output dist/another-name.zip` for another build.

Review the dataset license and player-identifying fields before making the Space public. Profiles are derived
from the [credited Kaggle dataset](https://www.kaggle.com/datasets/californianbill/patch-25-14-lol-league-of-legends-ranked-games).
No new permission to redistribute is implied by this packaging script. The archive is for inference, not for
reproducing training: raw matches and temporal training/evaluation tables stay on your machine.

## Upload

1. Create a **Gradio** Space; choose free **ZeroGPU** if your account is eligible. Docker is not used.
2. Extract the ZIP. Upload its **contents** into the Space repository root. `app.py`, `README.md`,
   and `requirements.txt` must be at the root, with `src/`, `artifacts/`, and `deployment/` beside them.
   Do not upload only the ZIP: Spaces will not unpack it. Include hidden `.gitattributes` when uploading with Git.
3. Keep the YAML header in README. It specifies the SDK, Python version and entry point.
4. No secrets are required. In Settings → Variables you may set `SCOUT_MAX_NEW_TOKENS=384`.
   Leave `SCOUT_DEVICE` unset: it selects `zero_gpu` when `SPACES_ZERO_GPU=1`, otherwise `cpu`.
   Leave `LOL_UI_BACKEND=local`. Never copy your local `.env` to the Space.
5. Wait for dependency installation and model downloads. Click **Build my team**, then either scout button.
   Use a rank represented in your data if the default produces no matches. No fake lineup is substituted.

For Git uploads, install Git LFS and upload actual `.pt` and `.npz` files, not unresolved LFS pointers.
The ZIP contains real checkpoint and vector bytes. Check that these files and the profile JSONL arrived intact.

## Resources and failures

- README's `preload_from_hub` downloads the selected MiniLM and Qwen files during the build, not on the first
  request. Keep the default Hub cache location: changing `HF_HOME` can prevent use of the preloaded cache.
  See the [Hugging Face configuration reference](https://huggingface.co/docs/hub/spaces-config-reference).
- Before Gradio starts accepting requests, Chroma imports the bundled vectors for 2,854 real profiles in
  small batches if the index is absent/stale. It does not re-embed those profiles. The rankers load, then a
  query and real candidate lineup warm the inference paths; the warmup never creates fake players or scout text.
  Startup logs label each stage with `[startup]`. Initialization is moved before the UI opens, not eliminated.
- Subsequent requests reuse the cached runtime/index. After an ephemeral-disk reset, vector import may run
  again, but expensive profile embedding does not. Missing/mismatched vectors fail explicitly, not silently rebuild.
- Recommendations and embedding inference stay on CPU even on ZeroGPU; only text generation requests a GPU.
  The generator is prepared at startup on ZeroGPU, before decorated inference, per Hugging Face guidance.
- One shared Gradio queue permits one model task at a time, with eight waiting requests.
  Preference input is limited to 1,000 characters, candidate alternatives to five per role, generation to
  384 tokens, and evidence input to 8,192 tokens. Oversized evidence is rejected, never silently truncated.
- CPU inference can take minutes. For a shorter report set `SCOUT_MAX_NEW_TOKENS=192`; do not expect identical prose.
- Missing profiles/checkpoints, model-download failures, missing evidence, GPU quota exhaustion, or empty generation
  produce explicit errors; they never produce fabricated recommendations or a template disguised as LLM output.
- The standalone FastAPI server is optional for local/external clients. The Space uses shared Python handlers
  directly, so there is no loopback port 8001 to configure and no duplicate model process.

## Local checks

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python app.py
```

Local CPU generation lazily downloads roughly 1 GB of weights, then uses float32 memory for inference.
No large raw file is loaded. Use a Space for the model download if you do not want this on your laptop.

To measure startup and two real lineup requests locally using a fresh diagnostic index:

```powershell
python scripts/benchmark_startup.py --index-dir data/diagnostics/fast-start-check
```

The benchmark refuses to overwrite an existing index directory and forbids document encoding. It reports
startup separately from first/repeat request latency, does not load Qwen, and leaves its diagnostic index
under ignored `data/`. These local timings are not a prediction of Hugging Face performance.

Latest local smoke measurement (Windows, two CPU threads, cached MiniLM files, a fresh Chroma directory):
64.559 seconds for imports/startup/warmup, 1.740 seconds for the first real lineup request, and 1.639 seconds
for the repeat. Both returned a complete four-player lineup; startup made zero document-encoding calls.
Qwen was not loaded or timed. This is one smoke run, not a production latency distribution or a before/after benchmark.
