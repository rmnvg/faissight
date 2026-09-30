"""``faissight.launch``: run the UI for an in-memory index from Python or a notebook."""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
from pathlib import Path
from types import TracebackType
from typing import Any

from faissight import jupyter
from faissight.core.projection import DEFAULT_MAX_POINTS
from faissight.session import Session

NO_BROWSER_ENV = "FAISSIGHT_NO_BROWSER"
"""Set to 1 to never open a browser from ``launch`` (CI, scripts, doc builds)."""


class Viewer:
    """A running faissight server for one :class:`Session`.

    In a notebook, displaying it embeds the UI. Call :meth:`stop` (or use it as a context
    manager) to shut the server down and free its port.
    """

    def __init__(
        self,
        session: Session,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        height: int = jupyter.DEFAULT_HEIGHT,
        start_timeout: float = 15.0,
    ) -> None:
        import uvicorn

        from faissight.server.app import create_app

        self.session = session
        self.height = height
        # Bind the socket ourselves: port=None/0 picks a free port with no check-then-bind race.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind((host, port or 0))
        except OSError as e:
            self._sock.close()
            raise OSError(f"Can't listen on {host}:{port}: {e}. Try another port.") from e
        self._sock.listen(128)
        self.host = host
        self.port: int = self._sock.getsockname()[1]

        config = uvicorn.Config(create_app(session), log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._sock]},
            name=f"faissight-server-{self.port}",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + start_timeout
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"faissight server failed to start on port {self.port}.")
            time.sleep(0.02)

    @property
    def url(self) -> str:
        shown = "127.0.0.1" if self.host in ("0.0.0.0", "", "::") else self.host
        return f"http://{shown}:{self.port}/"

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and not self._server.should_exit

    def stop(self, timeout: float = 10.0) -> None:
        """Shut the server down and free the port. Safe to call twice."""
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout)
        self._sock.close()

    def open(self) -> None:
        """Open the UI in the default web browser."""
        webbrowser.open(self.url)

    def __enter__(self) -> Viewer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<faissight.Viewer {self.url} ({state})>"

    def _repr_html_(self) -> str:
        if not self.running:
            return "<p><code>faissight</code> viewer stopped.</p>"
        return jupyter.iframe_html(jupyter.notebook_url(self.port, self.url), self.height)

    def _ipython_display_(self) -> None:
        if self.running and jupyter.in_colab():
            jupyter.display_in_colab(self.port, self.height)
            return
        from IPython.display import HTML, display

        display(HTML(self._repr_html_()))


def launch(
    index: Any,
    vectors: Any = None,
    ids: Any = None,
    metadata: Any = None,
    embedder: Any = None,
    *,
    queries: Any = None,
    port: int | None = None,
    host: str = "127.0.0.1",
    open_browser: bool | None = None,
    height: int = jupyter.DEFAULT_HEIGHT,
    max_points: int = DEFAULT_MAX_POINTS,
    cache_root: Path | None = None,
    compare: list[Any] | None = None,
    mmap: bool = False,
) -> Viewer:
    """Inspect a FAISS index in the browser.

    ``index`` is a ``faiss.Index`` or a path. ``vectors``/``ids``/``queries`` take numpy
    arrays or ``.npy`` paths; ``metadata`` takes a path, a list of dicts or a DataFrame with
    an ``id`` column; ``embedder`` is a ``str -> vector`` callable or a sentence-transformers
    model name. ``compare`` takes other indexes (``faiss.Index`` or paths) over the same
    vectors to compare in the Compare view; it needs ``vectors``. ``mmap`` memory-maps a
    ``vectors`` .npy path instead of loading it into RAM.

    Returns a :class:`Viewer`. In a notebook, display it (make it the cell's last
    expression) to embed the UI. Elsewhere the browser opens unless ``open_browser=False``
    or ``FAISSIGHT_NO_BROWSER=1``.
    """
    session = Session(
        index,
        vectors=vectors,
        ids=ids,
        metadata=metadata,
        embedder=embedder,
        queries=queries,
        max_points=max_points,
        cache_root=cache_root,
        compare=compare,
        mmap=mmap,
    )
    session.start_background()
    viewer = Viewer(session, host=host, port=port, height=height)
    if open_browser is None:
        open_browser = not jupyter.in_notebook() and os.environ.get(NO_BROWSER_ENV) != "1"
    if open_browser:
        viewer.open()
    return viewer
