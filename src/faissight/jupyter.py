"""Notebook helpers: detect the environment and embed the UI as an iframe."""

from __future__ import annotations

import html
import os
import sys
from typing import Any

DEFAULT_HEIGHT = 800
PROXY_ENV = "FAISSIGHT_PROXY_URL"
"""Optional URL template for notebooks behind a proxy, e.g. ``/proxy/{port}/`` with
jupyter-server-proxy on JupyterHub. ``{port}`` is replaced by the server port."""


def _ipython_shell() -> Any:
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return None
    try:
        return ipython.get_ipython()
    except Exception:  # pragma: no cover - defensive: odd embedded shells
        return None


def in_colab() -> bool:
    """True inside Google Colab."""
    return "google.colab" in sys.modules


def in_notebook() -> bool:
    """True inside a Jupyter-style kernel (JupyterLab, classic, VS Code, Colab)."""
    if in_colab():
        return True
    shell = _ipython_shell()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def notebook_url(port: int, default_url: str) -> str:
    """URL the notebook's browser should load: the proxy template if set, else the server URL."""
    template = os.environ.get(PROXY_ENV)
    return template.format(port=port) if template else default_url


def iframe_html(url: str, height: int = DEFAULT_HEIGHT) -> str:
    """An iframe embedding the UI, with a fallback link."""
    src = html.escape(url, quote=True)
    return (
        f'<iframe src="{src}" width="100%" height="{int(height)}" '
        'style="border:1px solid rgba(0,0,0,0.1);border-radius:8px" '
        'title="faissight"></iframe>'
        f'<div style="font-size:12px;margin-top:4px"><a href="{src}" target="_blank" '
        'rel="noopener">Open faissight in a new tab</a></div>'
    )


def display_in_colab(port: int, height: int = DEFAULT_HEIGHT) -> None:
    """Colab can't reach localhost directly; it proxies the kernel port into an iframe."""
    from google.colab import output

    output.serve_kernel_port_as_iframe(port, height=str(height))
