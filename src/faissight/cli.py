"""Command-line interface: ``faissight serve | info | sweep``."""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from faissight import __version__
from faissight.core._faiss import FaissNotInstalledError
from faissight.core.loader import IndexLoadError, load_index
from faissight.core.projection import DEFAULT_MAX_POINTS
from faissight.core.types import LoadedIndex
from faissight.server.app import create_app
from faissight.session import InputError, Session

app = typer.Typer(
    name="faissight",
    help="See inside your FAISS index: debug retrieval, tune recall.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"faissight {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """See inside your FAISS index: debug retrieval, tune recall."""


_PARAM_LABELS = {
    "nlist": "nlist",
    "nprobe": "nprobe",
    "by_residual": "by residual",
    "pq_m": "PQ m (sub-quantizers)",
    "pq_nbits": "PQ nbits",
    "sq_type": "SQ type",
    "hnsw_m": "HNSW M",
    "ef_search": "efSearch",
    "ef_construction": "efConstruction",
    "max_level": "max level",
    "entry_point": "entry point (internal)",
}

console = Console()
err_console = Console(stderr=True)


def _fail(message: str, hint: str | None = None) -> typer.Exit:
    err_console.print(f"[bold red]Error:[/] {message}")
    if hint:
        err_console.print(f"[dim]Hint: {hint}[/]")
    return typer.Exit(code=1)


def _load_or_exit(index_path: Path) -> LoadedIndex:
    try:
        return load_index(index_path)
    except FaissNotInstalledError as e:
        raise _fail("FAISS is not installed.", e.hint) from e
    except FileNotFoundError as e:
        raise _fail(str(e)) from e
    except IndexLoadError as e:
        raise _fail(str(e), "Is this a file written by faiss.write_index?") from e


def _info_table(li: LoadedIndex) -> Table:
    table = Table(show_header=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Kind", li.kind.value)
    table.add_row("Class chain", " → ".join(li.class_chain))
    if li.transforms:
        steps = ", ".join(f"{t.name} ({t.d_in}→{t.d_out})" for t in li.transforms)
        table.add_row("Transforms", steps)
    dim = str(li.d) if li.d == li.core_d else f"{li.d} (core index: {li.core_d})"
    table.add_row("Dimension", dim)
    table.add_row("Vectors", f"{li.ntotal:,}")
    table.add_row("Metric", li.metric.value)
    table.add_row("Trained", "yes" if li.is_trained else "no")
    table.add_row("ID map", f"yes ({len(li.ids):,} ids)" if li.ids is not None else "no")
    if li.has_refine:
        table.add_row("Refine", "yes (results re-ranked by a refine index)")
    for key, value in li.params.as_dict().items():
        if isinstance(value, bool):
            value = "yes" if value else "no"
        table.add_row(_PARAM_LABELS.get(key, key), str(value))
    return table


@app.command()
def info(
    index_path: Annotated[Path, typer.Argument(help="Path to a FAISS index file.")],
) -> None:
    """Print a summary of a FAISS index: kind, wrappers, dimensions and parameters."""
    li = _load_or_exit(index_path)
    console.print(f"[bold]{index_path}[/]")
    console.print(_info_table(li))
    if li.unsupported_reason:
        console.print(
            f"[yellow]Unsupported:[/] {li.unsupported_reason} "
            "faissight can only show the basic stats above for this index."
        )


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _display_url(host: str, port: int) -> str:
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{shown}:{port}/"


def _open_when_ready(server: uvicorn.Server, url: str) -> None:
    def wait_then_open() -> None:
        for _ in range(600):  # up to 30 s
            if server.started:
                webbrowser.open(url)
                return
            time.sleep(0.05)

    threading.Thread(target=wait_then_open, name="faissight-browser", daemon=True).start()


def _serve_summary(session: Session, url: str) -> None:
    li = session.li
    console.print(
        f"[bold]faissight[/] {__version__}  ·  {li.kind.value}  ·  {li.ntotal:,} vectors  ·  "
        f"d={li.d}  ·  {li.metric.value}"
    )
    if not li.is_supported:
        console.print(f"[yellow]Unsupported:[/] {li.unsupported_reason} Showing basic stats only.")
    elif not session.has_raw_vectors:
        console.print(
            "[yellow]No --vectors given:[/] ground truth is computed on reconstructed vectors; "
            "PQ/SQ error is not measured."
        )
    coverage = session.metadata_coverage()
    if coverage is not None and coverage < 1.0:
        console.print(
            f"[yellow]Metadata covers {coverage:.0%} of index ids[/] "
            "(check that metadata ids match the index ids)."
        )
    if session.embedder_name:
        console.print(f"Embedder [cyan]{session.embedder_name}[/] is loading in the background.")
    console.print(f"Serving on [bold link={url}]{url}[/]  (Ctrl+C to stop)")


@app.command()
def serve(
    index_path: Annotated[Path, typer.Argument(help="Path to a FAISS index file.")],
    vectors: Annotated[
        Path | None,
        typer.Option(help="Raw vectors .npy (float32, n x d). Enables exact ground truth."),
    ] = None,
    ids: Annotated[
        Path | None, typer.Option(help="Ids .npy (int64) for --vectors when row i != id i.")
    ] = None,
    meta: Annotated[
        Path | None, typer.Option(help="Chunk metadata (.jsonl/.csv/.parquet) with an 'id' column.")
    ] = None,
    embedder: Annotated[
        str | None,
        typer.Option(help="sentence-transformers model for text queries, e.g. all-MiniLM-L6-v2."),
    ] = None,
    queries: Annotated[
        Path | None, typer.Option(help="Query vectors .npy for batch evaluation / sweeps.")
    ] = None,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't open a browser window.")
    ] = False,
    max_points: Annotated[
        int, typer.Option(help="Most points sent to the UI (sampled beyond this).")
    ] = DEFAULT_MAX_POINTS,
    normalize_text: Annotated[
        bool | None,
        typer.Option(
            "--normalize-text/--no-normalize-text",
            help="L2-normalise text-query embeddings. Default: match the stored vectors.",
        ),
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Don't read/write the projection disk cache.")
    ] = False,
) -> None:
    """Start the faissight web UI for a FAISS index."""
    if not _port_available(host, port):
        raise _fail(f"Port {port} on {host} is already in use.", "Pick another with --port.")
    try:
        with console.status("Loading index…"):
            session = Session(
                index_path,
                vectors=vectors,
                ids=ids,
                metadata=meta,
                embedder=embedder,
                queries=queries,
                max_points=max_points,
                disk_cache=not no_cache,
                normalize_text=normalize_text,
            )
    except FaissNotInstalledError as e:
        raise _fail("FAISS is not installed.", e.hint) from e
    except FileNotFoundError as e:
        raise _fail(str(e)) from e
    except IndexLoadError as e:
        raise _fail(str(e), "Is this a file written by faiss.write_index?") from e
    except InputError as e:
        raise _fail(str(e), e.hint) from e

    url = _display_url(host, port)
    _serve_summary(session, url)
    session.start_background()
    config = uvicorn.Config(create_app(session), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    if not no_browser:
        _open_when_ready(server, url)
    server.run()
