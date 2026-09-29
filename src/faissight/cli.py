"""Command-line interface: ``faissight serve | info | sweep``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from faissight import __version__
from faissight.core._faiss import FaissNotInstalledError
from faissight.core.loader import IndexLoadError, load_index
from faissight.core.types import LoadedIndex

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
