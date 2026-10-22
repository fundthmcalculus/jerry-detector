from datetime import datetime, timezone

import numpy as np
from fastapi.testclient import TestClient

from jerry.store import ClipStore
from jerry.web.app import create_app


def test_lists_and_serves_clips(tmp_path):
    store = ClipStore(tmp_path)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 3
    meta = store.save("north-bowl", datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc), frames, fps=5)

    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/api/clips").json() == [meta]

    gif = client.get(f"/clips/{meta['id']}.gif")
    assert gif.status_code == 200
    assert gif.headers["content-type"] == "image/gif"
    assert gif.content.startswith(b"GIF8")

    assert client.get("/").status_code == 200


def test_rejects_unknown_and_traversal_ids(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/clips/missing.gif").status_code == 404
    assert client.get("/clips/..%2Fsecret.gif").status_code == 404
