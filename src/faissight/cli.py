"""Command-line interface: ``faissight serve | info | sweep``."""

from __future__ import annotations

import errno
import json
import socket
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from faissight import __version__
from faissight.core._faiss import FaissNotInstalledError
from faissight.core.advice import Suggestion
from faissight.core.loader import IndexLoadError, load_index
from faissight.core.projection import DEFAULT_MAX_POINTS
from faissight.core.sweep import Choice, SweepResult
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
    # Escape: hints contain things like faissight[all] that Rich would read as markup.
    err_console.print(f"[bold red]Error:[/] {escape(message)}")
    if hint:
        err_console.print(f"[dim]Hint: {escape(hint)}[/]")
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


def _bind_problem(host: str, port: int) -> tuple[str, str] | None:
    """Why ``serve`` couldn't listen on ``host:port`` as ``(message, hint)``, or None if it can."""
    where = f"{host}:{port}"
    try:
        # Resolve first so hostnames and IPv6 addresses (like "::") get the right family.
        family, kind, proto, _, addr = socket.getaddrinfo(
            host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )[0]
    except socket.gaierror:
        return (
            f"Cannot resolve host {host!r}.",
            "Use 127.0.0.1 (this machine only) or 0.0.0.0 (all interfaces).",
        )
    with socket.socket(family, kind, proto) as sock:
        # Like uvicorn: a recently closed port in TIME_WAIT is free, a live listener is not.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(addr)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return f"Port {port} on {host} is already in use.", "Pick another with --port."
            if e.errno in (errno.EACCES, errno.EPERM):
                hint = (
                    "Ports below 1024 need administrator rights; use --port 8765 or higher."
                    if port < 1024
                    else "A firewall or sandbox may block it; try another --port or --host."
                )
                return f"Not allowed to listen on {where}.", hint
            if e.errno == errno.EADDRNOTAVAIL:
                return (
                    f"{host} is not an address of this machine.",
                    "Use 127.0.0.1 (this machine only) or 0.0.0.0 (all interfaces).",
                )
            return f"Cannot listen on {where}: {e.strerror or e}.", "Try another --port or --host."
    return None


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
    if session.demo_limits is not None:
        console.print("[yellow]Demo mode:[/] read-only limits are on (k, sweeps, efSearch, UMAP).")
    if session.vectors_mapped:
        console.print("Raw vectors are memory-mapped from disk (--mmap).")
    elif session.mmap_note:
        console.print(f"[yellow]--mmap not applied:[/] {escape(session.mmap_note)}")
    for c in session.candidates:
        console.print(
            f"Comparing with [cyan]{escape(c.name)}[/] ({c.li.kind.value}) in the Compare view."
        )
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
    demo_mode: Annotated[
        bool,
        typer.Option(
            "--demo-mode",
            help="Read-only public demo: cap k, sweeps and efSearch; UMAP only if precomputed.",
        ),
    ] = False,
    mmap: Annotated[
        bool,
        typer.Option(
            "--mmap",
            help="Memory-map --vectors instead of loading them into RAM (needs a float32 .npy "
            "with rows sorted by id).",
        ),
    ] = False,
    compare: Annotated[
        list[Path] | None,
        typer.Option(
            "--compare",
            help="Another index over the same vectors to compare in the UI (repeatable; "
            "needs --vectors).",
        ),
    ] = None,
) -> None:
    """Start the faissight web UI for a FAISS index."""
    problem = _bind_problem(host, port)
    if problem is not None:
        raise _fail(*problem)
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
                demo_mode=demo_mode,
                compare=list(compare or []),
                mmap=mmap,
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


def _values_option(text: str | None) -> list[int] | None:
    if text is None:
        return None
    try:
        values = [int(v) for v in text.replace(" ", "").split(",") if v]
    except ValueError:
        raise typer.BadParameter("Use comma-separated integers, e.g. 1,2,4,8.") from None
    if not values:
        raise typer.BadParameter("Give at least one value.")
    return values


@app.command()
def sweep(
    index_path: Annotated[Path, typer.Argument(help="Path to a FAISS index file.")],
    vectors: Annotated[
        Path | None, typer.Option(help="Raw vectors .npy for exact ground truth.")
    ] = None,
    ids: Annotated[Path | None, typer.Option(help="Ids .npy for --vectors.")] = None,
    queries: Annotated[
        Path | None, typer.Option(help="Query vectors .npy (default: sample stored vectors).")
    ] = None,
    param: Annotated[
        str | None, typer.Option(help="nprobe (IVF) or efSearch (HNSW). Default: by index kind.")
    ] = None,
    values: Annotated[
        str | None, typer.Option(help="Comma-separated values, e.g. 1,2,4,8. Default: a ladder.")
    ] = None,
    k: Annotated[int, typer.Option(help="Recall@k.")] = 10,
    n_queries: Annotated[int, typer.Option(help="Queries to use.")] = 200,
    repeats: Annotated[int, typer.Option(min=1, max=20, help="Timing repetitions per query.")] = 3,
    seed: Annotated[
        int, typer.Option(min=0, max=2**32 - 1, help="Query sampling and timing-order seed.")
    ] = 0,
    target: Annotated[float, typer.Option(help="Target recall for the recommendation.")] = 0.95,
    max_p95_ms: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Also require p95 latency (single-threaded, per query) within this budget.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
    mmap: Annotated[
        bool,
        typer.Option(
            "--mmap",
            help="Memory-map --vectors instead of loading them into RAM (needs a float32 .npy "
            "with rows sorted by id).",
        ),
    ] = False,
) -> None:
    """Measure recall@k and latency across nprobe/efSearch values.

    Exits with code 2 if no value reaches --target (within --max-p95-ms, if given), so it
    can gate CI.
    """
    parsed_values = _values_option(values)
    if param is not None and param not in ("nprobe", "efSearch"):
        raise _fail(f"Unknown --param {param!r}.", "Use nprobe (IVF) or efSearch (HNSW).")
    try:
        session = Session(
            index_path, vectors=vectors, ids=ids, queries=queries, disk_cache=False, mmap=mmap
        )
        _, job = session.sweep_job(param, parsed_values, k, n_queries, repeats, seed)
    except FaissNotInstalledError as e:
        raise _fail("FAISS is not installed.", e.hint) from e
    except (FileNotFoundError, IndexLoadError) as e:
        raise _fail(str(e)) from e
    except InputError as e:
        raise _fail(str(e), e.hint) from e
    except ValueError as e:
        raise _fail(str(e)) from e

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        console=err_console,
        transient=True,
        disable=as_json,
    ) as bar:
        task = bar.add_task("Sweeping", total=1.0)
        while not job.wait(0.1):
            bar.update(task, completed=job.progress, description=job.message or "Sweeping")
    if job.error is not None:
        raise _fail(f"Sweep failed: {job.error}")
    result = job.result
    assert result is not None
    choice = result.choose(target, max_p95_ms=max_p95_ms)
    rec = choice.point
    fastest = result.fastest(target, max_p95_ms=max_p95_ms)
    suggestions = session.sweep_advice(result, target, max_p95_ms=max_p95_ms)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "param": result.param.value,
                    "k": result.k,
                    "n_queries": result.n_queries,
                    "query_origin": result.query_origin,
                    "truth_source": "reconstructed" if result.truth_reconstructed else "raw",
                    "target": target,
                    "max_p95_ms": max_p95_ms,
                    "status": choice.status,
                    "repeats": result.repeats,
                    "seed": result.seed,
                    "query_sha256": result.query_sha256,
                    "query_seed": result.query_seed,
                    "environment": result.environment,
                    "recommended": rec.value if rec else None,
                    "fastest_meeting_target": fastest.value if fastest else None,
                    "points": [asdict(p) for p in result.points],
                    "suggestions": [{**asdict(s), "kind": s.kind.value} for s in suggestions],
                },
                indent=2,
            )
        )
    else:
        _print_sweep(result, choice, target)
        _print_suggestions(suggestions)
    if rec is None:
        raise typer.Exit(code=2)


def _print_sweep(result: SweepResult, choice: Choice, target: float) -> None:
    rec, budget = choice.point, choice.max_p95_ms
    origin = "given" if result.query_origin == "given" else "sampled stored-vector"
    console.print(
        f"[bold]{result.param.value} sweep[/] · recall@{result.k} over {result.n_queries} "
        f"{origin} queries · single-threaded latency"
    )
    if result.truth_reconstructed:
        console.print(
            "[yellow]Ground truth computed on reconstructed vectors; PQ/SQ error is not "
            "measured.[/] Pass --vectors for exact ground truth."
        )
    if result.query_origin == "sampled":
        console.print(
            "Queries are stored vectors, which can behave differently from real queries. "
            "Pass held-out queries with --queries for a production-representative result."
        )
    table = Table()
    table.add_column(result.param.value, justify="right")
    table.add_column(f"recall@{result.k}", justify="right")
    table.add_column("95% interval", justify="right")
    table.add_column(f"queries < {target:g}", justify="right")
    has_coverage = result.coverage_curve is not None
    if has_coverage:
        table.add_column("in probed lists", justify="right")
    table.add_column("mean ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("")
    for p in result.points:
        mark = (
            "[green]✓ recommended[/]"
            if rec is not None and p.value == rec.value
            else ("meets target" if p.recall >= target else "")
        )
        if budget is not None and p.latency_p95_ms > budget:
            mark = f"{mark}, over budget" if mark else "over budget"
        below = p.fraction_below(target)
        coverage = [f"{p.probe_coverage:.3f}" if p.probe_coverage is not None else ""]
        table.add_row(
            str(p.value),
            f"{p.recall:.3f}",
            (
                f"{p.recall_ci_low:.3f}-{p.recall_ci_high:.3f}"
                if p.recall_ci_low is not None and p.recall_ci_high is not None
                else ""
            ),
            f"{below:.0%}" if below is not None else "",
            *(coverage if has_coverage else []),
            f"{p.latency_mean_ms:.3f}",
            f"{p.latency_p95_ms:.3f}",
            mark,
        )
    console.print(table)
    name = result.param.value
    if choice.status == "latency":
        over = choice.by_recall
        assert over is not None
        console.print(
            f"[red]No value reached recall {target:g} within p95 {budget:g} ms[/]: it first "
            f"comes at {name}={over.value} with p95 {over.latency_p95_ms:.3f} ms."
        )
        return
    if rec is None:
        best = max(result.points, key=lambda p: p.recall)
        console.print(
            f"[red]No value reached recall {target:g}[/] (best {best.recall:.3f} at "
            f"{name}={best.value})."
        )
        return
    slowest = max(result.points, key=lambda p: p.value)
    faster = (
        f", {slowest.latency_mean_ms / rec.latency_mean_ms:.1f}x faster than "
        f"{result.param.value}={slowest.value}"
        if slowest.value != rec.value and rec.latency_mean_ms > 0
        else ""
    )
    console.print(
        f"Recommended [bold]{result.param.value}={rec.value}[/] (smallest value meeting the "
        f"target): recall {rec.recall:.3f} at {rec.latency_mean_ms:.3f} ms/query{faster}."
    )
    fastest = result.fastest(target, max_p95_ms=budget)
    if fastest is not None and fastest.value != rec.value:
        console.print(
            f"Fastest measured: {result.param.value}={fastest.value} at "
            f"{fastest.latency_mean_ms:.3f} ms/query. A larger value measuring faster is "
            "usually timing noise; rerun with more --repeats before relying on it."
        )
    worst = rec.worst_queries[0] if rec.worst_queries else None
    if worst is not None and worst.recall < target:
        who = f"id {worst.id}" if worst.id is not None else f"query #{worst.query_no}"
        console.print(
            f"Worst query there: {who} with recall {worst.recall:.2f}; "
            f"{rec.fraction_below(target) or 0:.0%} of queries are below the target."
        )


def _print_suggestions(suggestions: list[Suggestion]) -> None:
    if not suggestions:
        return
    console.print("\n[bold]Suggested next steps[/]")
    for i, s in enumerate(suggestions, 1):
        console.print(f"{i}. [bold]{escape(s.title)}[/]")
        console.print(f"   {escape(s.detail)}")
        for e in s.evidence:
            console.print(f"   [dim]{escape(e.label)}:[/] {escape(e.value)}")
        if s.sweep_values:
            console.print(f"   [dim]Next sweep:[/] --values {','.join(map(str, s.sweep_values))}")


@app.command()
def compare(
    left: Annotated[Path, typer.Argument(help="First FAISS index.")],
    right: Annotated[Path, typer.Argument(help="Second FAISS index.")],
    vectors: Annotated[Path, typer.Option(help="Shared raw vectors .npy (required).")],
    ids: Annotated[Path | None, typer.Option(help="Ids for rows in --vectors.")] = None,
    queries: Annotated[Path | None, typer.Option(help="Shared query vectors .npy.")] = None,
    k: Annotated[int, typer.Option(min=1, help="Recall@k.")] = 10,
    n_queries: Annotated[int, typer.Option(min=1, help="Queries to evaluate.")] = 200,
    repeats: Annotated[int, typer.Option(min=1, max=20, help="Timing repeats.")] = 3,
    seed: Annotated[int, typer.Option(min=0, max=2**32 - 1, help="Sampling/timing seed.")] = 0,
    left_nprobe: Annotated[int | None, typer.Option(min=1)] = None,
    right_nprobe: Annotated[int | None, typer.Option(min=1)] = None,
    left_ef_search: Annotated[int | None, typer.Option(min=1)] = None,
    right_ef_search: Annotated[int | None, typer.Option(min=1)] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Include per-query neighbour changes.")
    ] = False,
    mmap: Annotated[
        bool,
        typer.Option(
            "--mmap",
            help="Memory-map --vectors instead of loading them into RAM (needs a float32 .npy "
            "with rows sorted by id).",
        ),
    ] = False,
) -> None:
    """Compare recall, latency and serialized size on exactly the same queries and ground truth."""
    from faissight.core.comparison import compare_indexes

    try:
        session = Session(
            left, vectors=vectors, ids=ids, queries=queries, disk_cache=False, mmap=mmap
        )
        result = compare_indexes(
            session.li,
            load_index(right),
            session.source,
            session.sweep_queries(n_queries, seed),
            k=k,
            repeats=repeats,
            seed=seed,
            left_nprobe=left_nprobe,
            right_nprobe=right_nprobe,
            left_ef_search=left_ef_search,
            right_ef_search=right_ef_search,
        )
    except FaissNotInstalledError as e:
        raise _fail("FAISS is not installed.", e.hint) from e
    except (OSError, ValueError) as e:
        raise _fail(str(e), getattr(e, "hint", None)) from e
    if as_json:
        payload = asdict(result)
        payload["left_path"], payload["right_path"] = str(left), str(right)
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(
        f"[bold]Index comparison[/] · recall@{k} · {result.n_queries} identical queries · "
        f"{repeats} timing repeats · seed {seed}"
    )
    table = Table("Metric", left.name, right.name)
    table.add_row("Kind", result.left.kind, result.right.kind)
    table.add_row("Search parameters", str(result.left.params), str(result.right.params))
    table.add_row("Recall", f"{result.left.recall:.4f}", f"{result.right.recall:.4f}")
    table.add_row(
        "Mean ms", f"{result.left.latency_mean_ms:.3f}", f"{result.right.latency_mean_ms:.3f}"
    )
    table.add_row(
        "p95 ms", f"{result.left.latency_p95_ms:.3f}", f"{result.right.latency_p95_ms:.3f}"
    )
    table.add_row(
        "Serialized bytes",
        f"{result.left.serialized_bytes:,}",
        f"{result.right.serialized_bytes:,}",
    )
    console.print(table)
    changed = sum(bool(d.left_only or d.right_only) for d in result.differences)
    console.print(
        f"Neighbour membership changed for {changed}/{result.n_queries} queries. "
        "Use --json for per-query changes and measurement provenance."
    )


DEMO_INDEXES = {"ivf_pq": "ivf_pq.index", "ivf_flat": "ivf_flat.index", "hnsw": "hnsw_flat.index"}
DEMO_HINT = 'The demo needs the text and parquet extras: pip install "faissight[all]".'


@app.command()
def demo(
    index: Annotated[
        str, typer.Option(help="Which demo index to open: ivf_pq, ivf_flat or hnsw.")
    ] = "ivf_pq",
    data_dir: Annotated[
        Path | None, typer.Option(help="Where to keep the demo data (default: the cache).")
    ] = None,
    max_chunks: Annotated[
        int, typer.Option(help="Passage chunks to embed on first build.")
    ] = 10_000,
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Rebuild even if built.")] = False,
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8765,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Don't open a browser.")] = False,
) -> None:
    """Build (first run only) and open a RAG demo: real passages, MiniLM embeddings, 3 indexes."""
    import importlib.util

    from faissight import demo as demo_mod

    if index not in DEMO_INDEXES:
        raise _fail(f"Unknown --index {index!r}.", f"Use one of: {', '.join(DEMO_INDEXES)}.")
    out = data_dir or demo_mod.default_dir()
    if rebuild or not demo_mod.is_built(out):
        missing = [
            m for m in ("sentence_transformers", "pyarrow") if importlib.util.find_spec(m) is None
        ]
        if missing:
            raise _fail(f"Missing {', '.join(missing)}.", DEMO_HINT)
        console.print(f"Building the RAG demo in [bold]{out}[/] (first run only, ~1 minute)…")
        try:
            demo_mod.build_demo(out, max_chunks=max_chunks)
        except OSError as e:
            raise _fail(f"Could not build the demo: {e}", "Check your internet connection.") from e
    serve(
        index_path=out / DEMO_INDEXES[index],
        vectors=out / "vectors.npy",
        ids=None,
        meta=out / "chunks.jsonl",
        embedder="all-MiniLM-L6-v2",
        queries=out / "queries.npy",
        host=host,
        port=port,
        no_browser=no_browser,
        max_points=DEFAULT_MAX_POINTS,
        normalize_text=True,
        no_cache=False,
        demo_mode=False,
        compare=None,
        mmap=False,
    )
