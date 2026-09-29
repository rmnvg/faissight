"""Run with an isolated Python containing the built wheel, FAISS and httpx."""

import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
from fastapi.testclient import TestClient

import faissight
from faissight.server.app import STATIC_DIR, create_app
from faissight.session import Session

assert Path(faissight.__file__).is_relative_to(Path(sys.prefix)), faissight.__file__
assert (STATIC_DIR / "index.html").is_file()
x = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
index = faiss.IndexFlatL2(2)
index.add(x)
with TestClient(create_app(Session(index, vectors=x, disk_cache=False))) as client:
    assert client.get("/api/health").json()["status"] == "ok"
    assert "<html" in client.get("/").text.lower()
    response = client.post("/api/search", json={"query": {"vector": [0, 0]}, "k": 2})
    assert response.status_code == 200, response.text
    assert response.json()["recall"] == 1.0
subprocess.run([sys.executable, "-I", "-m", "faissight", "--version"], check=True)
