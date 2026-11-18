"""Drishti web app: FastAPI backend for the control-room UI.

uvicorn app.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from drishti import __version__
from drishti.config import settings
from drishti.live import LiveMonitor
from drishti.models import new_detector
from drishti.privacy import blur_crop, blur_regions, to_jpeg
from drishti.records import Camera, Crop, SearchResult
from drishti.search import SearchEngine
from drishti.store import Store
from drishti.trail import SAME_PERSON_SIM

from .schemas import CaseContext, SearchRequest, WatchRequest

STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------- singletons
@lru_cache(maxsize=1)
def engine() -> SearchEngine:
    return SearchEngine(Store())


def config_cameras() -> list[Camera]:
    path = settings.cameras_file
    return json.loads(path.read_text()) if path.exists() else []


@lru_cache(maxsize=1)
def monitor() -> LiveMonitor:
    return LiveMonitor(engine(), config_cameras(), settings.cameras_file.parent)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    engine().search("a person", top_k=1)  # load models + index now, so the first demo is instant
    yield
    monitor().stop()


app = FastAPI(
    title="Drishti",
    version=__version__,
    lifespan=lifespan,
    description="Describe them. We'll find them.",
)


# ---------------------------------------------------------------- accountability
def audit(action: str, ctx: CaseContext, **detail: Any) -> None:
    settings.audit_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "action": action,
        "officer": ctx.officer,
        "reason": ctx.reason,
        **detail,
    }
    with settings.audit_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def require_case(ctx: CaseContext) -> None:
    if not ctx.officer.strip() or not ctx.reason.strip():
        raise HTTPException(400, "Officer ID and case reference are required for every action.")


def jpeg_response(data: bytes) -> Response:
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def crop_or_404(name: str) -> Crop:
    c = engine().store.crop(name)
    if c is None:
        raise HTTPException(404, "unknown crop")
    return c


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(404, "image missing")
    return img


# ---------------------------------------------------------------- overview
@app.get("/api/overview")
def overview() -> dict[str, Any]:
    s = engine().store
    site = settings.cameras_file.parent / "site.json"
    return {
        "version": __version__,
        "cameras": config_cameras(),
        "indexed_cameras": len(s.cameras),
        "people_indexed": len(s.tracks),
        "privacy_blur": settings.privacy_blur,
        "links": json.loads(site.read_text())["links"] if site.exists() else [],
    }


# ---------------------------------------------------------------- search
@app.post("/api/search")
def text_search(req: SearchRequest) -> dict[str, Any]:
    require_case(req)
    t = time.perf_counter()
    res: SearchResult = engine().search(req.query, top_k=req.top_k)
    audit(
        "search", req, query=req.query, understood_as=res["understood_as"], hits=len(res["results"])
    )
    return {**res, "latency_ms": round((time.perf_counter() - t) * 1000)}


@app.post("/api/search_image")
async def photo_search(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI's dependency style
    query: str = Form(""),
    officer: str = Form(""),
    reason: str = Form(""),
    top_k: int = Form(12),
) -> dict[str, Any]:
    ctx = CaseContext(officer=officer, reason=reason)
    require_case(ctx)
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "could not read the image") from exc
    # crop the most prominent person in the photo a family member shares
    r = new_detector().predict(np.array(img)[:, :, ::-1], classes=[0], conf=0.3, verbose=False)[0]
    if r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        x1, y1, x2, y2 = xyxy[int(np.argmax(areas))]
        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
    t = time.perf_counter()
    res = engine().search(query, image=img, top_k=top_k)
    audit("photo_search", ctx, query=query, hits=len(res["results"]))
    return {**res, "latency_ms": round((time.perf_counter() - t) * 1000)}


@app.get("/api/similar/{track}")
def similar(track: str) -> dict[str, Any]:
    """Where else was this person seen? Re-identification across the other cameras."""
    eng = engine()
    try:
        matches = eng.similar(track)
    except KeyError as exc:
        raise HTTPException(404, "unknown person") from exc
    out = []
    for key, sim in matches:
        t = eng.store.track(key)
        cam = eng.store.cameras[t["camera_id"]]
        out.append(
            {
                "track": key,
                "camera": cam.get("name", t["camera_id"]),
                "camera_id": t["camera_id"],
                "thumb": t["thumb"],
                "similarity": round(sim, 3),
                "likely_same": sim >= SAME_PERSON_SIM,
            }
        )
    return {"track": track, "matches": out}


# ---------------------------------------------------------------- images (privacy-blurred)
@app.get("/api/thumb/{name}")
def thumb(name: str, reveal: int = 0, officer: str = "", reason: str = "") -> Response:
    c = crop_or_404(name)
    img = read_image(engine().store.root / "thumbs" / f"{name}.jpg")
    if reveal:
        ctx = CaseContext(officer=officer, reason=reason)
        require_case(ctx)
        audit("reveal_face", ctx, crop=name, camera=c["camera_id"])
    elif settings.privacy_blur:
        img = blur_crop(img, c.get("head"))
    return jpeg_response(to_jpeg(img))


@app.get("/api/frame/{name}")
def frame(name: str, reveal: int = 0, officer: str = "", reason: str = "") -> Response:
    c = crop_or_404(name)
    img = read_image(engine().store.root / "frames" / f"{name}.jpg")
    s = img.shape[1] / c["frame_size"][0]
    if reveal:
        require_case(CaseContext(officer=officer, reason=reason))
    elif settings.privacy_blur:
        blur_regions(img, [[v * s for v in b] for b in c.get("heads", [])])
    x1, y1, x2, y2 = (int(v * s) for v in c["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 255), 2)
    return jpeg_response(to_jpeg(img))


@app.get("/api/video/{camera_id}")
def video(camera_id: str, request: Request, officer: str = "", reason: str = "") -> FileResponse:
    """Raw footage is only served against a case reference, and every viewing is logged."""
    cam = engine().store.cameras.get(camera_id)
    if not cam:
        raise HTTPException(404, "unknown camera")
    ctx = CaseContext(officer=officer, reason=reason)
    require_case(ctx)
    if request.headers.get("range", "bytes=0-").startswith("bytes=0-"):
        audit("view_footage", ctx, camera=camera_id)
    return FileResponse(cam["video"], media_type="video/mp4")


# ---------------------------------------------------------------- live monitoring
@app.post("/api/watch")
def add_watch(req: WatchRequest) -> dict[str, Any]:
    require_case(req)
    if not monitor().cameras:
        raise HTTPException(400, "no live cameras configured")
    w = monitor().add_watch(req.query, req.officer, req.reason)
    audit("watch_start", req, query=req.query, watch=w.id)
    return {
        "id": w.id,
        "query": w.query,
        "understood_as": w.parsed.english,
        "threshold": round(w.threshold, 2),
        "cameras": [c["name"] for c in monitor().cameras],
    }


@app.delete("/api/watch/{watch_id}")
def remove_watch(watch_id: int) -> dict[str, bool]:
    monitor().remove_watch(watch_id)
    return {"ok": True}


@app.get("/api/watches")
def watches() -> dict[str, Any]:
    m = monitor()
    return {
        "running": m.running,
        "watches": [
            {"id": w.id, "query": w.query, "created": w.created} for w in m.watches.values()
        ],
    }


@app.get("/api/events")
async def events(request: Request, after: int = 0) -> StreamingResponse:
    """Server-sent events: one message per new alert."""

    async def stream() -> AsyncIterator[str]:
        last = after
        yield "retry: 2000\n\n"
        while not await request.is_disconnected():
            for a in monitor().alerts_after(last):
                last = a.id
                yield f"id: {a.id}\nevent: alert\ndata: {json.dumps(a.public())}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/alerts/{alert_id}/{kind}")
def alert_image(alert_id: int, kind: str) -> Response:
    a = monitor().alert(alert_id)
    if a is None or kind not in ("snapshot", "frame"):
        raise HTTPException(404)
    return jpeg_response(a.snapshot if kind == "snapshot" else a.frame)


@app.get("/api/live/{camera_id}.mjpg")
def live_stream(camera_id: str) -> StreamingResponse:
    """Motion-JPEG of a live camera, faces blurred, matched people boxed in orange."""

    def frames() -> Iterator[bytes]:
        last: bytes | None = None
        for _ in range(60 * 60 * 10):  # ~1 hour at 10 checks/s, then the browser reconnects
            monitor().last_viewed[camera_id] = time.monotonic()
            data = monitor().frames.get(camera_id)
            if data is not None and data is not last:
                last = data
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            time.sleep(0.1)

    monitor().start()
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/audit")
def audit_log(limit: int = 50) -> dict[str, Any]:
    if not settings.audit_log.exists():
        return {"entries": []}
    lines = settings.audit_log.read_text(encoding="utf-8").strip().splitlines()[-limit:]
    return {"entries": [json.loads(line) for line in reversed(lines)]}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
