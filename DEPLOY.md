# Deployment (Google Cloud) & Access Control

This document covers how to deploy the FeedbackAssistantTool container to Google Cloud and how to defend access control.

## 1. Access control (auth)

The app spends LLM tokens per run, so an open URL is a cost/abuse vector. Three options, from most to least defensible:

### Option A — Cloud Run + IAM (recommended, no app code)

Deploy with authentication required so there is **no anonymous surface**:

```
gcloud run deploy feedback-tool --no-allow-unauthenticated ...
```

Grant only the accounts that need it:

```
gcloud run services add-iam-policy-binding feedback-tool \
  --member="user:someone@example.com" --role="roles/run.invoker"
```

Access it via an identity token or `gcloud run services proxy feedback-tool`.

**Defense:** "The service rejects all unauthenticated requests at the platform edge; access is gated by Google Cloud IAM
and granted per-identity. There is no anonymous entry point."

### Option B — Identity-Aware Proxy (IAP)

Put IAP in front of the service for a browser login wall for named users. Use this if reviewers need to open it in a
browser without gcloud.

**Defense:** "All traffic passes through Google IAP; unauthenticated requests are rejected before reaching the
application."

### Option C — App-level shared secret (weakest; demo only)

A password field checked against an env var before rendering the UI. Only acceptable for a throwaway public demo, not
for anything sensitive.

**Recommendation:** Option A for a research tool; add IAP (B) if you want browser logins.

## 2. Cloud Run deployment

The Dockerfile already: uses Python 3.13, installs CPU-only torch, expands `${PORT}` at runtime, and pins dependency
versions. `.dockerignore` keeps `.venv`, `.git`, `.env`,
`*.db`, and `llm_eksport/` out of the image.

### Build & push

```
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/REPO/feedback-tool
```

### Deploy

```
gcloud run deploy feedback-tool \
  --image REGION-docker.pkg.dev/PROJECT/REPO/feedback-tool \
  --memory 4Gi --cpu 2 --timeout 900 --concurrency 1 \
  --no-allow-unauthenticated \
  --set-env-vars LLM_BENCHMARK_DB=/tmp/llm_benchmark.db,UTILITY_MODEL=gpt-4o-mini \
  --set-secrets OPENAI_API_KEY=openai-key:latest,GEMINI_API_KEY=gemini-key:latest,ANTHROPIC_API_KEY=anthropic-key:latest
```

Why each flag:

- `--memory 4Gi`: torch + faiss + sentence-transformers need headroom or they OOM.
- `--timeout 900`: a swarm scenario can take minutes; the default 300s would kill it. For 40+ min talks on slow local
  models, raise this up to Cloud Run's max of `3600`. This is the ceiling on the whole HTTP request to Streamlit and is
  **independent** of the per-LLM-call timeout below.
- `--concurrency 1`: each run is heavy and Streamlit is session-stateful; avoid sharing a container.
- `LLM_BENCHMARK_DB=/tmp/...`: Cloud Run's filesystem is read-only except `/tmp`.
- `--set-secrets`: inject API keys from Secret Manager. **Do not** ship a `.env` file.

### LLM timeouts (long talks)

litellm applies a **per-request** timeout (default 600s) to *each individual* LLM call — every map agent call,
gatekeeper, RAG call, and the single big reduce/monolith call each get their own window. On 40+ minute talks the
reduce/monolith call alone can exceed 600s. These env vars tune it:

- `LLM_REQUEST_TIMEOUT` (default `1800`): per-request timeout for the small map-phase calls.
- `HEGEMON_REQUEST_TIMEOUT` (default `3600`): per-request timeout for the heavy reduce / monolith calls that generate
  the full essay.
- `HEGEMON_MAX_TOKENS` (default `4096`): max **output** tokens for the report. 4096 is the safe floor (Claude 3.5 Sonnet
  caps at 4096 without a beta header); raise to `8192` for gpt-4o / capable local models on long talks.

Two behaviours to know:

- Timeouts on the **map phase** are retried (a blip may clear); timeouts on the **reduce/monolith**
  call are **not** retried (a call that is slow because the work is large won't get faster on retry), so it fails fast
  instead of burning `4x` the wall-clock.
- If the swarm reduce (Hegemon) fails anyway, the run is **not** lost: a degraded report is returned built from the
  deterministic scorecard + the aggregated map findings.

Set `HEGEMON_REQUEST_TIMEOUT` **≤** the Cloud Run `--timeout`, or the platform will kill the request before litellm's
own timeout fires.

### Secrets (one-time)

```
echo -n "sk-..." | gcloud secrets create openai-key --data-file=-
```

## 3. Local (Ollama) models in the cloud

Cloud Run is not a good host for Ollama model serving. To use `ollama/*` models, run an **Ollama server on a GPU host**
(Compute Engine VM with an L4/T4, or a GKE GPU node), then point the app at it — the endpoint is env-driven:

```
--set-env-vars OLLAMA_API_BASE=http://<ollama-host>:11434
```

Single-box alternative (demo): run Ollama + Streamlit together on one GPU VM via `docker-compose`.

### Choosing the Ollama endpoint (`OLLAMA_API_BASE`)

The app reads `OLLAMA_API_BASE` (see `Config.OLLAMA_API_BASE`) and passes it explicitly to litellm for every `ollama/*`
call. `docker-compose.yml` defaults it to the containerized Ollama and lets you override it from `.env`:

```yaml
OLLAMA_API_BASE=${OLLAMA_API_BASE:-http://ollama:11434}
```

- **GCloud / containerized Ollama:** leave it unset — the default `http://ollama:11434` (the compose service name) is
  used. The `ollama` service has **no host port mapping**, so it never collides with anything on the host.
- **Local dev with a host-installed Ollama:** put `OLLAMA_API_BASE=http://host.docker.internal:11434`
  in `.env`. This reuses the Ollama already running on your machine (with your pulled models + GPU)
  instead of the empty CPU-only container. For this to work, Ollama must listen on all interfaces:
  set `OLLAMA_HOST=0.0.0.0:11434` on the host and restart Ollama (by default it binds `127.0.0.1`, which containers
  cannot reach).
- **Running the app outside Docker entirely:** the code default is `http://localhost:11434`.

Note: the GPU only provides hardware — an Ollama server process must be running and have the models pulled.

## 4. Pre-deploy verification

Run the offline smoke test (no external calls) to confirm the pipeline plumbing works on real ZIP data before deploying:

```
.venv\Scripts\python.exe -m smoke_tests.run_smoke
```

It runs all 5 scenarios + one evaluation over `smoke_tests/*.zip` using a mock gateway and asserts non-empty structured
output. Expect `ALL_SMOKE_TESTS_PASSED`.

For a live check, repeat one scenario against a real model (keys/Ollama configured) via the Streamlit UI before opening
access.
