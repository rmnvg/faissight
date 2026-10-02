import json
import math

import faiss
import numpy as np
import pytest
from typer.testing import CliRunner

from faissight.cli import app
from faissight.core.loader import load_index
from faissight.core.relevance import evaluate_relevance, load_judgements, relevance_metrics
from faissight.core.search import SearchResult, rerank
from faissight.core.types import Metric
from faissight.core.vectors import VectorSource, from_arrays


def test_graded_metrics_and_duplicates():
    labels = {10: 3.0, 20: 1.0, 30: 0.0}
    result = relevance_metrics([30, 20, 20, 10], labels, 3)
    assert result.recall == 0.5
    assert result.mrr == 0.5
    assert result.ndcg == pytest.approx((1 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert relevance_metrics([10, 20], labels, 10).ndcg == pytest.approx(1)
    assert relevance_metrics([], labels, 10).recall == 0
    assert relevance_metrics([999], labels, 1).mrr == 0


@pytest.mark.parametrize("labels", [{}, {1: 0}, {1: -1}, {1: float("nan")}, {1: float("inf")}])
def test_invalid_grades(labels):
    with pytest.raises(ValueError, match=r"grades|positive"):
        relevance_metrics([1], labels, 10)


@pytest.mark.parametrize(
    "line",
    [
        "{}",
        "null",
        "[]",
        '{"row":0,"relevant":{}}',
        '{"row":0,"relevant":{"1":true}}',
        '{"row":0,"relevant":{"-1":1}}',
        '{"row":0,"relevant":{"9223372036854775808":1}}',
        '{"row":true,"relevant":{"1":1}}',
        '{"row":0,"relevant":{"1":NaN}}',
    ],
)
def test_invalid_judgement_files(tmp_path, line):
    path = tmp_path / "labels.jsonl"
    path.write_text(line)
    with pytest.raises(ValueError, match="line 1"):
        load_judgements(path, 1)


def test_complete_unique_rows_and_large_ids(tmp_path):
    path = tmp_path / "labels.jsonl"
    line = json.dumps({"row": 0, "relevant": {str(2**63 - 1): 2}})
    path.write_text(line)
    assert load_judgements(path, 1) == [{2**63 - 1: 2}]
    with pytest.raises(ValueError, match="every"):
        load_judgements(path, 2)
    path.write_text(line + "\n" + line)
    with pytest.raises(ValueError, match="unique"):
        load_judgements(path, 1)


@pytest.mark.parametrize("metric", [Metric.L2, Metric.IP])
def test_rerank_uses_raw_distances_and_custom_ids(metric):
    ids = np.array([2**53 + 1, 2**63 - 1], dtype=np.int64)
    source = VectorSource(np.array([[1, 0], [2, 0]], dtype=np.float32), ids, False)
    pool = SearchResult(ids[::-1], np.array([0, 1], dtype=np.float32), metric, 2)
    ranked = rerank(source, [1, 0], pool, 1)
    assert ranked.ids.tolist() == [int(ids[0] if metric is Metric.L2 else ids[1])]
    assert ranked.distances.tolist() == [0 if metric is Metric.L2 else 2]
    assert ranked.latency_ms >= 2
    with pytest.raises(ValueError, match="raw"):
        rerank(VectorSource(source.vectors, ids, True), [1, 0], pool, 1)


def test_evaluate_and_cli(tmp_path):
    x = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    index = faiss.IndexFlatL2(2)
    index.add(x)
    li = load_index(index)
    labels = [{0: 2, 2: 1}, {2: 1}]
    queries = np.array([[0.1, 0], [1.9, 0]], dtype=np.float32)
    result = evaluate_relevance(li, queries, labels, from_arrays(li, x), k=1, candidates=3)
    assert result["metrics"] == {"recall": 0.75, "mrr": 1, "ndcg": 1}
    assert result["reranked_metrics"] == result["metrics"]
    with pytest.raises(ValueError, match="exist"):
        evaluate_relevance(li, queries, [{999: 1}, {2: 1}], from_arrays(li, x))
    with pytest.raises(ValueError, match="candidates"):
        evaluate_relevance(li, queries, labels, from_arrays(li, x), k=10, candidates=3)
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    np.save(tmp_path / "vectors.npy", x)
    np.save(tmp_path / "queries.npy", queries)
    (tmp_path / "labels.jsonl").write_text(
        "\n".join(json.dumps({"row": row, "relevant": rel}) for row, rel in enumerate(labels))
    )
    args = [
        "evaluate",
        str(tmp_path / "index.faiss"),
        "--vectors",
        str(tmp_path / "vectors.npy"),
        "--queries",
        str(tmp_path / "queries.npy"),
        "--labels",
        str(tmp_path / "labels.jsonl"),
        "--k",
        "1",
        "--candidates",
        "3",
        "--json",
    ]
    run = CliRunner().invoke(app, args)
    assert run.exit_code == 0, run.output
    assert json.loads(run.output)["metrics"]["recall"] == 0.75
    (tmp_path / "labels.jsonl").write_text("{}")
    run = CliRunner().invoke(app, args)
    assert run.exit_code == 1
    assert "line 1" in run.output


def test_evaluation_rejects_mismatched_raw_corpus():
    x = np.array([[0, 0], [1, 0]], dtype=np.float32)
    index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
    index.add_with_ids(x, np.array([10, 20], dtype=np.int64))
    li = load_index(index)
    with pytest.raises(ValueError, match="count and dimension"):
        evaluate_relevance(li, [[0, 0]], [{10: 1}], VectorSource(x[:1], np.array([10]), False))
    with pytest.raises(ValueError, match=r"ids|Ids|IDs"):
        evaluate_relevance(li, [[0, 0]], [{10: 1}], VectorSource(x, np.array([10, 30]), False))
