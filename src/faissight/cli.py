"""Command-line interface: ``faissight serve | info | sweep``."""

from __future__ import annotations

import typer

from faissight import __version__

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
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """See inside your FAISS index: debug retrieval, tune recall."""
