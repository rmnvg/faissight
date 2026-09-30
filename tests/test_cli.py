import errno
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from faissight import __version__
from faissight.cli import app
from tests.conftest import SMALL

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"faissight {__version__}"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_info_ivf_pq(synthetic) -> None:
    result = runner.invoke(app, ["info", str(synthetic["ivf_pq"])])
    assert result.exit_code == 0, result.output
    for text in ("IVF_PQ", "IndexIVFPQ", "nlist", "PQ nbits", "L2", "2,000"):
        assert text in result.output


def test_info_pretransform(synthetic) -> None:
    result = runner.invoke(app, ["info", str(synthetic["pca_ivf_flat"])])
    assert result.exit_code == 0, result.output
    assert "PCAMatrix" in result.output
    assert "core index: 16" in result.output


def test_info_unsupported(binary_index_path) -> None:
    result = runner.invoke(app, ["info", str(binary_index_path)])
    assert result.exit_code == 0, result.output
    assert "UNSUPPORTED" in result.output
    assert "Binary indexes are not supported" in result.output


def test_info_missing_file(tmp_path) -> None:
    result = runner.invoke(app, ["info", str(tmp_path / "nope.index")])
    assert result.exit_code == 1
    assert "not found" in result.output


# --- serve ---------------------------------------------------------------------------------


class FakeServer:
    """Stands in for uvicorn.Server so tests don't bind ports or block."""

    instances: ClassVar[list["FakeServer"]] = []

    def __init__(self, config) -> None:
        self.config = config
        self.started = False
        FakeServer.instances.append(self)

    def run(self) -> None:
        self.started = True


@pytest.fixture
def fake_uvicorn(monkeypatch):
    import faissight.cli as cli_mod

    FakeServer.instances.clear()
    opened = []
    monkeypatch.setattr(cli_mod.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(cli_mod.webbrowser, "open", opened.append)
    return opened


def test_serve_starts_server(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(
        app,
        [
            "serve",
            str(synthetic["ivf_flat"]),
            "--vectors",
            str(synthetic["vectors"]),
            "--meta",
            str(synthetic["chunks"]),
            "--port",
            "8911",
            "--no-browser",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "IVF_FLAT" in result.output
    assert "http://127.0.0.1:8911/" in result.output
    assert "reconstructed" not in result.output
    (server,) = FakeServer.instances
    assert (server.config.host, server.config.port) == ("127.0.0.1", 8911)
    assert fake_uvicorn == []  # --no-browser


def test_serve_with_compare(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(
        app,
        [
            "serve",
            str(synthetic["ivf_flat"]),
            "--vectors",
            str(synthetic["vectors"]),
            "--compare",
            str(synthetic["ivf_pq"]),
            "--compare",
            str(synthetic["hnsw_flat"]),
            "--port",
            "8916",
            "--no-browser",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Comparing with ivf_pq.index (IVF_PQ)" in result.output
    assert "Comparing with hnsw_flat.index (HNSW_FLAT)" in result.output


def test_serve_mmap(synthetic, fake_uvicorn, tmp_path) -> None:
    base = ["serve", str(synthetic["ivf_flat"]), "--no-browser", "--port", "8917", "--mmap"]
    result = runner.invoke(app, [*base, "--vectors", str(synthetic["vectors"])])
    assert result.exit_code == 0, result.output
    assert "memory-mapped" in result.output
    f64 = tmp_path / "f64.npy"
    np.save(f64, np.load(synthetic["vectors"]).astype(np.float64))
    result = runner.invoke(app, [*base, "--vectors", str(f64)])
    assert result.exit_code == 0, result.output
    assert "--mmap not applied" in result.output


def test_sweep_mmap(synthetic) -> None:
    args = ["sweep", str(synthetic["ivf_flat"]), "--vectors", str(synthetic["vectors"])]
    result = runner.invoke(app, [*args, "--mmap", "--n-queries", "10", "--values", "1,16"])
    assert result.exit_code == 0, result.output


def test_serve_opens_browser_when_ready(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(app, ["serve", str(synthetic["flat_l2"]), "--port", "8912"])
    assert result.exit_code == 0, result.output
    for _ in range(100):
        if fake_uvicorn:
            break
        time.sleep(0.02)
    assert fake_uvicorn == ["http://127.0.0.1:8912/"]


def test_serve_warns_without_vectors(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(
        app, ["serve", str(synthetic["ivf_pq"]), "--no-browser", "--port", "8913"]
    )
    assert result.exit_code == 0, result.output
    assert "reconstructed vectors" in result.output


def test_serve_warns_on_partial_metadata(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(
        app,
        [
            "serve",
            str(synthetic["idmap_flat"]),
            "--meta",
            str(synthetic["chunks"]),
            "--no-browser",
            "--port",
            "8914",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Metadata covers" in result.output


def test_serve_display_url_for_all_interfaces(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(
        app,
        ["serve", str(synthetic["flat_l2"]), "--host", "0.0.0.0", "--port", "8915", "--no-browser"],
    )
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8915/" in result.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--vectors", "QUERIES"], "vectors but the index holds"),
        (["--ids", "IDS"], "--ids needs --vectors"),
        (["--queries", "VECTORS_BAD_DIM"], "Queries have shape"),
        (["--meta", "MISSING"], "not found"),
        (["--max-points", "0"], "max_points"),
        (["--compare", "IVF_PQ"], "needs raw vectors"),
        (["--vectors", "VECTORS", "--compare", "MISSING"], "not found"),
    ],
)
def test_serve_input_errors(synthetic, fake_uvicorn, tmp_path, extra, message) -> None:
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((3, 5), dtype=np.float32))
    subs = {
        "QUERIES": str(synthetic["queries"]),
        "IDS": str(synthetic["ids_idmap"]),
        "VECTORS_BAD_DIM": str(bad),
        "MISSING": str(tmp_path / "nope.jsonl"),
        "IVF_PQ": str(synthetic["ivf_pq"]),
        "VECTORS": str(synthetic["vectors"]),
    }
    args = [subs.get(a, a) for a in extra]
    result = runner.invoke(app, ["serve", str(synthetic["ivf_flat"]), "--no-browser", *args])
    assert result.exit_code == 1
    assert message in result.output
    assert not FakeServer.instances


def test_serve_bad_index(tmp_path, fake_uvicorn) -> None:
    result = runner.invoke(app, ["serve", str(tmp_path / "missing.index"), "--no-browser"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_serve_port_in_use(synthetic, fake_uvicorn) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        result = runner.invoke(app, ["serve", str(synthetic["flat_l2"]), "--port", str(port)])
    assert result.exit_code == 1
    assert "already in use" in result.output


@pytest.mark.parametrize(
    ("code", "port", "message", "hint"),
    [
        (errno.EACCES, 80, "Not allowed to listen on 127.0.0.1:80", "administrator rights"),
        (errno.EPERM, 8914, "Not allowed to listen on 127.0.0.1:8914", "firewall or sandbox"),
        (errno.EADDRNOTAVAIL, 8914, "127.0.0.1 is not an address of this machine", "0.0.0.0"),
        (errno.ENOBUFS, 8914, "Cannot listen on 127.0.0.1:8914", "another --port"),
    ],
)
def test_serve_reports_bind_errors_accurately(
    synthetic, fake_uvicorn, monkeypatch, code, port, message, hint
) -> None:
    import faissight.cli as cli_mod

    class RefusingSocket(socket.socket):
        def bind(self, address) -> None:
            raise OSError(code, os.strerror(code))

    monkeypatch.setattr(cli_mod.socket, "socket", RefusingSocket)
    result = runner.invoke(app, ["serve", str(synthetic["flat_l2"]), "--port", str(port)])
    assert result.exit_code == 1
    output = " ".join(result.output.split())  # Rich wraps long lines
    assert message in output
    assert hint in output
    assert "already in use" not in output


def test_serve_unresolvable_host(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(
        app, ["serve", str(synthetic["flat_l2"]), "--host", "no-such-host.invalid"]
    )
    assert result.exit_code == 1
    assert "Cannot resolve host 'no-such-host.invalid'" in result.output


def test_serve_real_subprocess(synthetic, tmp_path) -> None:
    # Acceptance: `faissight serve` starts and answers /api/info and /api/search.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "FAISSIGHT_CACHE_DIR": str(tmp_path)}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "faissight",
            "serve",
            str(synthetic["ivf_flat"]),
            "--vectors",
            str(synthetic["vectors"]),
            "--meta",
            str(synthetic["chunks"]),
            "--port",
            str(port),
            "--no-browser",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        base = f"http://127.0.0.1:{port}/api"
        for _ in range(200):
            try:
                info = json.load(urllib.request.urlopen(f"{base}/info", timeout=1))
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError(proc.stdout.read().decode() if proc.stdout else "no output")
        assert info["kind"] == "IVF_FLAT"
        req = urllib.request.Request(
            f"{base}/search",
            data=json.dumps({"query": {"id": 3}, "nprobe": SMALL["nlist"]}).encode(),
            headers={"content-type": "application/json"},
        )
        body = json.load(urllib.request.urlopen(req, timeout=5))
        assert body["recall"] == 1.0
    finally:
        proc.terminate()
        proc.wait(10)


# --- sweep ---------------------------------------------------------------------------------


def test_sweep_table(synthetic) -> None:
    result = runner.invoke(
        app,
        [
            "sweep",
            str(synthetic["ivf_flat"]),
            "--vectors",
            str(synthetic["vectors"]),
            "--n-queries",
            "30",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "nprobe sweep" in result.output
    assert "Recommended nprobe=" in result.output
    assert "reconstructed" not in result.output
    assert "held-out queries with --queries" in result.output


def test_sweep_json(synthetic) -> None:
    result = runner.invoke(
        app,
        [
            "sweep",
            str(synthetic["ivf_flat"]),
            "--values",
            "1, 16",
            "--n-queries",
            "20",
            "--json",
            "--target",
            "0.99",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert [p["value"] for p in body["points"]] == [1, 16]
    assert body["recommended"] == 16 or body["points"][0]["recall"] >= 0.99
    assert body["truth_source"] == "reconstructed"
    assert set(body["points"][0]) == {
        "value",
        "recall",
        "latency_mean_ms",
        "latency_p95_ms",
        "recall_ci_low",
        "recall_ci_high",
        "recall_distribution",
        "worst_queries",
        "probe_coverage",
    }
    assert set(body["points"][0]["worst_queries"][0]) == {
        "query_no",
        "id",
        "recall",
        "probe_coverage",
    }
    assert body["points"][0]["recall"] <= body["points"][0]["probe_coverage"] + 1e-9
    assert "fastest_meeting_target" in body
    assert isinstance(body["suggestions"], list)


def test_sweep_given_queries(synthetic) -> None:
    result = runner.invoke(
        app, ["sweep", str(synthetic["ivf_flat"]), "--queries", str(synthetic["queries"]), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["query_origin"] == "given"


def test_sweep_unreachable_target_exits_2(synthetic) -> None:
    result = runner.invoke(
        app,
        [
            "sweep",
            str(synthetic["ivf_pq"]),
            "--vectors",
            str(synthetic["vectors"]),
            "--values",
            "1,2",
            "--n-queries",
            "20",
            "--target",
            "0.999",
        ],
    )
    assert result.exit_code == 2
    assert "No value reached recall 0.999" in result.output
    assert "in probed lists" in result.output
    assert "Suggested next steps" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--param", "bogus"], "Unknown --param"),
        (["--param", "efSearch"], "HNSW"),
        (["--values", "1,x"], "comma-separated"),
        (["--values", "999"], "nprobe must be"),
        (["--k", "0"], ">= 1"),
    ],
)
def test_sweep_errors(synthetic, args, message) -> None:
    result = runner.invoke(app, ["sweep", str(synthetic["ivf_flat"]), *args])
    assert result.exit_code != 0
    assert message in result.output


def test_sweep_flat_and_missing(synthetic, tmp_path) -> None:
    flat = runner.invoke(app, ["sweep", str(synthetic["flat_l2"])])
    assert flat.exit_code == 1
    assert "no search parameter" in flat.output
    missing = runner.invoke(app, ["sweep", str(tmp_path / "x.index")])
    assert missing.exit_code == 1
    assert "not found" in missing.output


def test_error_hints_keep_brackets(synthetic, monkeypatch) -> None:
    # Rich markup would swallow "[faiss-cpu]" in the install hint.
    monkeypatch.setitem(sys.modules, "faiss", None)
    result = runner.invoke(app, ["info", str(synthetic["ivf_flat"])])
    assert result.exit_code == 1
    assert "faissight[faiss-cpu]" in result.output


def test_serve_demo_mode_flag(synthetic, fake_uvicorn) -> None:
    result = runner.invoke(
        app, ["serve", str(synthetic["flat_l2"]), "--demo-mode", "--no-browser", "--port", "8916"]
    )
    assert result.exit_code == 0, result.output
    assert "Demo mode" in result.output
    (server,) = FakeServer.instances
    assert server.config.app.state.session.demo_limits is not None
