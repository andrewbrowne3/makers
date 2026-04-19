"""FastAPI entrypoint."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import CFG, PROJECT_ROOT
from app.logging_config import get_logger, setup_logging
from app.orchestrator import handle_generate
from app.router import get_router
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse

setup_logging()
log = get_logger("main")

app = FastAPI(title="MakersGarments — Logo-to-Design Generator", version="0.1.0")

STATIC_DIR = PROJECT_ROOT / "app" / "static"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup() -> None:
    log.info("🚀 startup llm=%s image_gen=%s embed=%s bucket=%s", CFG.llm.provider, CFG.image_gen.provider, CFG.embed.provider, CFG.s3.bucket)
    r = get_router()
    if r.collection.count() == 0:
        log.info("🧱 startup: rebuilding router index from seed corpus")
        r.rebuild()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "logo.png").suffix or ".png"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    dest.write_bytes(await file.read())
    log.info("⬆️  /upload saved=%s size=%d", dest, dest.stat().st_size)
    return {"path": str(dest), "filename": file.filename}


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=f"{CFG.llm.provider}:{CFG.llm.model}",
        image_gen_provider=f"{CFG.image_gen.provider}:{CFG.image_gen.model}",
        embed_provider=f"{CFG.embed.provider}:{CFG.embed.model}",
        bucket=CFG.s3.bucket,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    log.info("📨 POST /generate client=%s garment=%s mockups=%s", req.client_name, req.garment_type, req.mockups)
    try:
        return handle_generate(req)
    except Exception as e:  # noqa: BLE001
        log.exception("generate failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/router/rebuild")
def rebuild_router() -> dict:
    n = get_router().rebuild()
    return {"status": "ok", "seeded": n}
