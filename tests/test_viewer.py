import json
import socket
import sys
import types
import urllib.request

import faiss
import numpy as np
import pytest

import faissight
from faissight import jupyter
from faissight import viewer as viewer_mod

D = 16


@pytest.fixture(scope="module")
def data():
    x = np.random.default_rng(0).standard_normal((1500, D)).astype(np.float32)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, 8)
    index.train(x)
    index.add(x)
    return index, x


@pytest.fixture
def no_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(viewer_mod.webbrowser, "open", opened.append)
    return opened


def _get(url: str):
    return json.load(urllib.request.urlopen(url, timeout=5))


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def test_launch_twice_uses_different_ports_and_stop_frees_them(data, no_browser) -> None:
    # Acceptance: two viewers in one process coexist; stop() frees each port.
    index, x = data
    a = faissight.launch(index, vectors=x, open_browser=False)
    b = faissight.launch(index, open_browser=False)
    try:
        assert a.port != b.port
        assert _get(a.url + "api/info")["inputs"]["raw_vectors"] is True
        assert _get(b.url + "api/info")["inputs"]["raw_vectors"] is False
        assert a.running
        assert b.running
    finally:
        a.stop()
        b.stop()
    assert not a.running
    assert _port_free(a.port)
    assert _port_free(b.port)
    a.stop()  # idempotent


def test_explicit_port_and_port_in_use(data, no_browser) -> None:
    index, _ = data
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    v = faissight.launch(index, port=port, open_browser=False)
    try:
        assert v.port == port
        assert v.url == f"http://127.0.0.1:{port}/"
        with pytest.raises(OSError, match="Try another port"):
            faissight.launch(index, port=port, open_browser=False)
    finally:
        v.stop()


def test_context_manager_and_repr(data, no_browser) -> None:
    index, _ = data
    with faissight.launch(index, open_browser=False) as v:
        assert "(running)" in repr(v)
        html = v._repr_html_()
        assert f'src="{v.url}"' in html
        assert 'height="800"' in html
    assert "(stopped)" in repr(v)
    assert "stopped" in v._repr_html_()


def test_in_memory_inputs_end_to_end(data, no_browser) -> None:
    index, x = data

    class Frame:  # pandas-like: only to_dict("records") is used
        def to_dict(self, orient):
            return [{"id": i, "text": f"chunk {i}"} for i in range(len(x))]

    with faissight.launch(
        index, vectors=x, metadata=Frame(), embedder=lambda t: np.ones(D), open_browser=False
    ) as v:
        info = _get(v.url + "api/info")
        assert info["inputs"]["metadata_rows"] == len(x)
        assert info["inputs"]["embedder"] == "<lambda>"
        req = urllib.request.Request(
            v.url + "api/search",
            data=json.dumps({"query": {"text": "hello"}, "k": 3}).encode(),
            headers={"content-type": "application/json"},
        )
        body = json.load(urllib.request.urlopen(req, timeout=5))
        assert len(body["results"]) == 3
        assert body["results"][0]["snippet"]["text"].startswith("chunk")


def test_browser_opens_outside_notebooks(data, no_browser, monkeypatch) -> None:
    index, _ = data
    monkeypatch.delenv(viewer_mod.NO_BROWSER_ENV, raising=False)
    monkeypatch.setattr(jupyter, "in_notebook", lambda: False)
    with faissight.launch(index) as v:
        assert no_browser == [v.url]


@pytest.mark.parametrize("reason", ["env", "notebook", "explicit"])
def test_browser_suppressed(data, no_browser, monkeypatch, reason) -> None:
    index, _ = data
    monkeypatch.delenv(viewer_mod.NO_BROWSER_ENV, raising=False)
    monkeypatch.setattr(jupyter, "in_notebook", lambda: reason == "notebook")
    if reason == "env":
        monkeypatch.setenv(viewer_mod.NO_BROWSER_ENV, "1")
    kwargs = {"open_browser": False} if reason == "explicit" else {}
    with faissight.launch(index, **kwargs):
        pass
    assert no_browser == []


def test_proxy_url_template(data, no_browser, monkeypatch) -> None:
    index, _ = data
    monkeypatch.setenv(jupyter.PROXY_ENV, "/proxy/{port}/")
    with faissight.launch(index, open_browser=False) as v:
        assert f'src="/proxy/{v.port}/"' in v._repr_html_()


def test_ipython_display_uses_iframe(data, no_browser, monkeypatch) -> None:
    index, _ = data
    shown = []
    display_mod = types.ModuleType("IPython.display")
    display_mod.HTML = lambda s: ("HTML", s)
    display_mod.display = shown.append
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)
    with faissight.launch(index, open_browser=False, height=500) as v:
        v._ipython_display_()
    assert shown[0][0] == "HTML"
    assert 'height="500"' in shown[0][1]


def test_colab_display(data, no_browser, monkeypatch) -> None:
    index, _ = data
    calls = []
    colab = types.ModuleType("google.colab")
    colab.output = types.SimpleNamespace(
        serve_kernel_port_as_iframe=lambda port, height: calls.append((port, height))
    )
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert jupyter.in_colab()
    assert jupyter.in_notebook()
    with faissight.launch(index, height=600) as v:  # in Colab: no local browser
        v._ipython_display_()
    assert calls == [(v.port, "600")]
    assert no_browser == []


def test_notebook_detection(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    assert not jupyter.in_notebook()

    class ZMQInteractiveShell:
        pass

    class TerminalInteractiveShell:
        pass

    for shell, expected in ((ZMQInteractiveShell(), True), (TerminalInteractiveShell(), False)):
        ipython = types.ModuleType("IPython")
        ipython.get_ipython = lambda shell=shell: shell
        monkeypatch.setitem(sys.modules, "IPython", ipython)
        assert jupyter.in_notebook() is expected


def test_iframe_html_escapes() -> None:
    html = jupyter.iframe_html('http://x/"><script>', 300)
    assert "<script>" not in html
    assert 'height="300"' in html


def test_top_level_exports() -> None:
    assert set(faissight.__all__) == {"Session", "Viewer", "__version__", "launch"}
