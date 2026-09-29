import subprocess
import sys

import numpy as np

from faissight import core
from tests.conftest import SMALL


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_core_has_no_web_imports() -> None:
    r = _run(
        "import sys, faissight.core; "
        "bad = [m for m in ('fastapi', 'uvicorn', 'starlette') if m in sys.modules]; "
        "assert not bad, bad"
    )
    assert r.returncode == 0, r.stderr


def test_core_imports_without_faiss() -> None:
    r = _run(
        "import sys; sys.modules['faiss'] = None\n"
        "import faissight.core as c\n"
        "try:\n    c.load_index('x.index')\nexcept c.FaissNotInstalledError as e:\n"
        "    print(e.hint)\n"
    )
    assert r.returncode == 0, r.stderr
    assert "faiss-cpu" in r.stdout


def test_documented_flow(synthetic) -> None:
    li = core.load_index(synthetic["ivf_flat"])
    source = core.from_arrays(li, np.load(synthetic["vectors"]))
    gt = core.GroundTruth(source, li.metric)
    q = core.resolve_query(li, id=123, source=source)
    report = core.explain_query(
        li, q, k=10, nprobe=SMALL["nlist"], ground_truth=gt, assignments=core.assignments(li)
    )
    assert report.recall == 1.0
    assert report.ivf_trace is not None
    assert report.ivf_trace.min_nprobe_for_all >= 1


def test_all_exports_exist() -> None:
    for name in core.__all__:
        assert hasattr(core, name), name


def test_search_module_not_shadowed() -> None:
    import types

    from faissight.core import search

    assert isinstance(search, types.ModuleType)
    assert callable(search.search)
