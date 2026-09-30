import faiss
import numpy as np
import pytest
from typer.testing import CliRunner

from faissight.cli import app
from faissight.core.comparison import changed_queries, compare_indexes
from faissight.core.loader import load_index
from faissight.core.search import GroundTruth
from faissight.core.sweep import given_queries, ground_truth_ids, sample_queries
from faissight.core.vectors import from_arrays, reconstruct_all


def test_compare_exact_and_approximate(synthetic) -> None:
    left, right = load_index(synthetic["flat_l2"]), load_index(synthetic["ivf_pq"])
    source = from_arrays(left, np.load(synthetic["vectors"]))
    queries = sample_queries(source, 20, seed=42)
    previous = faiss.omp_get_max_threads()
    report = compare_indexes(left, right, source, queries, repeats=2, seed=42, right_nprobe=1)
    assert report.left.recall == 1.0
    assert report.right.recall < 1.0
    assert report.left.serialized_bytes > report.right.serialized_bytes
    assert report.repeats == 2
    assert report.seed == 42
    assert report.left.latency_mean_ms > 0
    assert report.right.latency_p95_ms > 0
    assert len(report.differences) == 20
    assert any(d.left_only for d in report.differences)
    assert all(queries.exclude_ids[d.query_no] not in d.left_only for d in report.differences)
    assert faiss.omp_get_max_threads() == previous


def test_compare_identical_indexes_and_given_queries(synthetic) -> None:
    li = load_index(synthetic["flat_l2"])
    source = from_arrays(li, np.load(synthetic["vectors"]))
    qs = given_queries(li, np.load(synthetic["queries"])[:5])
    result = compare_indexes(li, li, source, qs)
    assert result.query_origin == "given"
    assert result.left.recall == result.right.recall == 1.0
    assert all(not d.left_only and not d.right_only and d.overlap == 10 for d in result.differences)


def test_compare_rejects_incompatible_inputs(synthetic) -> None:
    li = load_index(synthetic["flat_l2"])
    source = from_arrays(li, np.load(synthetic["vectors"]))
    qs = sample_queries(source, 3)
    with pytest.raises(ValueError, match="dimension and distance metric"):
        compare_indexes(li, load_index(faiss.IndexFlatIP(li.d)), source, qs)
    with pytest.raises(ValueError, match="raw vectors"):
        compare_indexes(li, li, reconstruct_all(li), qs)
    with pytest.raises(ValueError, match="do not match"):
        compare_indexes(li, load_index(synthetic["idmap_flat"]), source, qs)


def test_compare_cli(synthetic) -> None:
    import json

    args = [
        "compare",
        str(synthetic["flat_l2"]),
        str(synthetic["ivf_flat"]),
        "--vectors",
        str(synthetic["vectors"]),
        "--n-queries",
        "5",
        "--right-nprobe",
        "16",
        "--seed",
        "9",
        "--repeats",
        "2",
    ]
    runner = CliRunner()
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["left"]["recall"] == body["right"]["recall"] == 1.0
    assert body["seed"] == 9
    assert body["repeats"] == 2
    assert len(body["differences"]) == 5
    table = runner.invoke(app, args)
    assert table.exit_code == 0, table.output
    assert "Serialized bytes" in table.output


def test_compare_reuses_truth_reports_progress_and_orders_changes(synthetic) -> None:
    left, right = load_index(synthetic["flat_l2"]), load_index(synthetic["ivf_pq"])
    source = from_arrays(left, np.load(synthetic["vectors"]))
    queries = sample_queries(source, 40, seed=3)
    truth = ground_truth_ids(GroundTruth(source, left.metric), queries, 10)
    messages = []
    result = compare_indexes(
        left,
        right,
        source,
        queries,
        right_nprobe=1,
        truth=truth,
        progress=lambda frac, msg: messages.append((frac, msg)),
    )
    assert messages
    assert all(0.0 <= f <= 1.0 for f, _ in messages)
    assert not any("ground truth" in m for _, m in messages)  # truth was given
    assert result.right.recall_ci_low <= result.right.recall <= result.right.recall_ci_high
    assert [d.query_id for d in result.differences] == [int(i) for i in queries.exclude_ids]
    changed = changed_queries(result)
    assert changed
    assert all(d.left_only or d.right_only for d in changed)
    deltas = [abs(d.right_recall - d.left_recall) for d in changed]
    assert deltas == sorted(deltas, reverse=True)
    with pytest.raises(ValueError, match="truth must have shape"):
        compare_indexes(left, right, source, queries, truth=truth[:, :5])
