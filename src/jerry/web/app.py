from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from jerry.store import ClipStore

STATIC = Path(__file__).parent / "static"


def create_app(clips_dir: Path) -> FastAPI:
    store = ClipStore(clips_dir)
    app = FastAPI(title="Jerry Detector")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/clips")
    def clips() -> list[dict]:
        return store.list()

    @app.get("/clips/{clip_id}.gif")
    def clip(clip_id: str) -> FileResponse:
        path = store.gif_path(clip_id)
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/gif")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
