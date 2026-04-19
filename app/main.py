"""FastAPI entrypoint."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import CFG, PROJECT_ROOT
from app.db import (
    connect,
    get_client_by_name,
    list_clients,
    list_designs,
    list_logos_for_client,
)
from app.logging_config import get_logger, setup_logging
from app.orchestrator import handle_generate
from app.router import get_router
from app.s3 import get_s3
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


@app.get("/gallery")
def gallery() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "gallery.html"))


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


# ── History / catalog endpoints ────────────────────────────────────────
@app.get("/clients")
def get_clients() -> dict:
    return {"clients": list_clients()}


@app.get("/clients/{name}")
def get_client(name: str) -> dict:
    c = get_client_by_name(name)
    if not c:
        raise HTTPException(status_code=404, detail="client not found")
    return c


@app.get("/logos/{logo_id}")
def serve_logo(logo_id: int) -> FileResponse:
    with connect() as con:
        row = con.execute("SELECT storage_path, filename FROM logos WHERE id = ?", (logo_id,)).fetchone()
    if not row or not row["storage_path"]:
        raise HTTPException(status_code=404, detail="logo not found")
    p = Path(row["storage_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"file missing on disk: {p}")
    return FileResponse(str(p), media_type="image/png", filename=row["filename"])


@app.get("/baselines/{design_id}")
def serve_baseline(design_id: int) -> FileResponse:
    with connect() as con:
        row = con.execute(
            "SELECT local_path FROM designs WHERE id = ? AND source = 'baseline'",
            (design_id,),
        ).fetchone()
    if not row or not row["local_path"]:
        raise HTTPException(status_code=404, detail="baseline not found")
    p = Path(row["local_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"file missing on disk: {p}")
    return FileResponse(str(p), media_type="image/png", filename=p.name)


@app.get("/clients/{name}/logos")
def get_logos(name: str) -> dict:
    c = get_client_by_name(name)
    if not c:
        raise HTTPException(status_code=404, detail="client not found")
    logos = list_logos_for_client(c["id"])
    for lg in logos:
        lg["url"] = f"/logos/{lg['id']}"
    return {"client": c, "logos": logos}


@app.get("/clients/{name}/baselines")
def get_baselines(name: str) -> dict:
    c = get_client_by_name(name)
    if not c:
        raise HTTPException(status_code=404, detail="client not found")
    rows = list_designs(client_id=c["id"], source="baseline", limit=500)
    for r in rows:
        r["url"] = f"/baselines/{r['id']}"
    return {"client": c, "baselines": rows}


@app.get("/clients/{name}/designs")
def get_designs(
    name: str,
    mockup: int | None = None,
    logo_id: int | None = None,
    source: str | None = None,
    limit: int = 200,
) -> dict:
    c = get_client_by_name(name)
    if not c:
        raise HTTPException(status_code=404, detail="client not found")
    rows = list_designs(
        client_id=c["id"],
        mockup_index=mockup,
        logo_id=logo_id,
        source=source,
        limit=limit,
    )
    s3 = get_s3()
    # re-mint a fresh presigned URL for every AI-generated row that has an s3_key
    for r in rows:
        if r.get("s3_key") and r.get("source") == "ai_generated":
            try:
                r["image_url"] = s3.presigned_download(r["s3_key"])
            except Exception as e:  # noqa: BLE001
                r["image_url"] = None
                r["url_error"] = str(e)
        else:
            r["image_url"] = None
    return {"client": c, "designs": rows}
