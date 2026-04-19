# MakersGarments — Logo → Design Generator

FastAPI service. Apps Script webhook in, garment mockup image URL out.

## Layout

```
app/
├── main.py              # FastAPI app + endpoints
├── config.py            # .env → typed dataclasses
├── schemas.py           # Pydantic request/response
├── logging_config.py    # emoji stdout + rotating file logs
├── orchestrator.py      # router → branch agent → evaluator
├── router.py            # embeddings (Chroma + mxbai) → branch
├── s3.py                # boto3 manager, presigned URLs
├── agents/
│   ├── base.py          # manual ReAct loop
│   ├── image_gen_agent.py
│   ├── evaluator_agent.py
│   └── question_agent.py
├── providers/
│   ├── llm.py           # ollama | anthropic | openai
│   ├── image_gen.py     # nano_banana | gpt_image_1 | flux_kontext | stub
│   └── embeddings.py    # ollama | openai
└── tools/
    └── garment_tools.py # fetch_reference, gen_image, upload_s3, pillow_composite
scripts/
└── build_router_index.py
db/chroma_db/            # persistent vector store
logs/                    # rotating file logs
```

## Quickstart

```bash
cd ~/projects/makersgarments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start local Ollama (for LLM + embeddings) and pull the models
ollama pull gemma4:e4b
ollama pull mxbai-embed-large

# Seed router index
python -m scripts.build_router_index

# Run
uvicorn app.main:app --reload --port 8000
```

## Try it

```bash
# Healthcheck
curl localhost:8000/healthz | jq

# Generate (stub image provider — draws the prompt on a canvas so nothing breaks without cloud keys)
curl -X POST localhost:8000/generate \
  -H "content-type: application/json" \
  -d '{
    "client_name": "Acme",
    "logo_url": "https://example.com/logo.png",
    "garment_type": "t-shirt",
    "placement": "front center",
    "colors": ["#0a84ff", "#ffffff"],
    "notes": "embroidered feel, athletic fit"
  }' | jq
```

## Switching providers

Edit `.env`:

| Variable | Options |
|---|---|
| `LLM_PROVIDER` | `ollama` (default) / `anthropic` / `openai` |
| `LLM_MODEL` | `gemma4:e4b` / `claude-sonnet-4-6` / `gpt-5` |
| `IMAGE_GEN_PROVIDER` | `stub` (dev) / `nano_banana` / `gpt_image_1` / `flux_kontext` |
| `EMBED_PROVIDER` | `ollama` / `openai` |

Restart and the whole app picks up the change.

## Router branches

Defined as `SEED_CORPUS` in `app/router.py`. Add new branches by adding a new key with example phrasings. Re-run `python -m scripts.build_router_index`.

Current branches:
- `image_gen` — generate a new garment design
- `evaluate` — score an existing rendered image
- `ask_question` — fallback when the router is unsure

Future: `notify_user`, `lookup_template`, `batch_generate`, ...

## Logs

- stdout: emoji-prefixed step-by-step (router decision, agent THINK/ACT/OBSERVE, provider calls, S3 uploads)
- `logs/app.log` — rotating (5MB × 5 files)

Tail live:
```bash
tail -f logs/app.log
```

## Docker

```bash
docker compose up --build
# API on localhost:8012
```

Ollama still runs on the host; the container reaches it via `host.docker.internal:11434`.

## Assets — Executive Crew Socks

Default asset directories (configurable via `.env`):

```
~/Executive Crew Socks/
├── PNG_3D_Mockup_Exec/       → 10 mockup PNGs (LogoHere_Crew_Mock_[1-10].png)
├── PHOTOSHOP_FILES_FOR_DESIGNERS_Exec/  → 5 PSDs
├── ProdFiles_Exec/           → production-ready AI/PDF per sock
└── TRAINING_DATA/            → 16 client logo→output pairs
```

The image-gen agent loads a chosen mockup as a reference image and asks the
image-edit provider to replace the `LogoHere` placeholder with the client's logo.

## Try it

```bash
# Use a local logo + render mockups 1, 2, 3 only
curl -X POST localhost:8000/generate \
  -H "content-type: application/json" \
  -d '{
    "client_name": "680thefan",
    "logo_path": "/home/ab/Executive Crew Socks/TRAINING_DATA/https_--680thefan.com/logos and design assets/680-the-fan-logo-vertical1200x1200-300x300.png",
    "garment_type": "sock",
    "mockups": [1, 2, 3]
  }' | jq
```

Omit `mockups` to render all 10.

## Known gaps (v1 scaffold)

- `evaluate` branch as a standalone entry-point is stubbed — evaluator currently runs inline after `image_gen`.
- `flux_kontext` provider raises `NotImplementedError` — add fal/replicate client when you pick the hosting.
- PSD layer swapping (via `psd-tools`) not wired — currently only PNG-reference image-edit.
- Evaluator uses host LLM (Gemma 4 multimodal) — confirm local Ollama Gemma 4 handles vision reliably, else flip `EVALUATOR_PROVIDER=anthropic`.
