"""The demo notebook must run top to bottom (plan acceptance for Phase 8)."""

import json
import os
import subprocess
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "examples" / "notebook_demo.ipynb"


def test_notebook_is_clean() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code)


def test_notebook_runs_top_to_bottom(tmp_path) -> None:
    # Execute the code cells in order in a fresh interpreter (no Jupyter needed).
    nb = json.loads(NOTEBOOK.read_text())
    source = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    script = tmp_path / "notebook.py"
    script.write_text(source)
    env = {**os.environ, "FAISSIGHT_NO_BROWSER": "1", "FAISSIGHT_CACHE_DIR": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120
    )
    assert r.returncode == 0, r.stderr[-3000:]
    assert "recall@10 =" in r.stdout
    assert "recommended nprobe:" in r.stdout
