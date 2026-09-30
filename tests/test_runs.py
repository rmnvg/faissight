import copy
import json

import numpy as np
import pytest

from faissight.core import runs as R
from faissight.core import search as S
from faissight.core import sweep as W
from faissight.core import vectors as V
from faissight.core.loader import load_index
from faissight.core.projection import index_fingerprint


@pytest.fixture(scope="module")
def record(synthetic):
    li = load_index(synthetic["ivf_pq"])
    src = V.from_arrays(li, np.load(synthetic["vectors"]))
    qs = W.sample_queries(src, 30)
    truth = W.ground_truth_ids(S.GroundTruth(src, li.metric), qs, 10)
    result = W.sweep(li, qs, truth, values=[1, 4], k=10, repeats=1)
    return R.run_record(
        result, li, index_fingerprint(li), target_recall=0.5, max_p95_ms=100.0, label="pq"
    )


def test_run_record_round_trips(record, tmp_path) -> None:
    assert record["format"] == R.RUN_FORMAT
    assert record["index"]["kind"] == "IVF_PQ"
    assert record["index"]["name"] == "ivf_pq.index"
    assert len(record["index"]["sha1"]) == 40
    assert record["queries"]["origin"] == "sampled"
    assert record["settings"]["values"] == [1, 4]
    assert record["decision"]["max_p95_ms"] == 100.0
    assert record["decision"]["status"] in ("ok", "recall")
    assert record["points"][0]["worst_queries"]  # example failures travel with the run
    path = R.save_run(record, tmp_path / "run.json")
    assert R.load_run(path) == json.loads(json.dumps(record))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.update(format="other"), "Not a faissight sweep run"),
        (lambda r: r.update(version=99), "version 99"),
        (lambda r: r.pop("index"), "missing or invalid"),
        (lambda r: r["points"][0].pop("recall"), "missing or invalid"),
        (lambda r: r["settings"].pop("k"), "settings.k"),
    ],
)
def test_bad_runs_are_rejected(record, mutate, match) -> None:
    bad = copy.deepcopy(record)
    mutate(bad)
    with pytest.raises(R.RunFormatError, match=match):
        R.check_run(bad)


def test_load_run_rejects_non_json(tmp_path) -> None:
    (tmp_path / "x.json").write_text("{nope")
    with pytest.raises(R.RunFormatError, match="not JSON"):
        R.load_run(tmp_path / "x.json")


def _changed(record, recall=0.0, p95=1.0, **sections):
    run = copy.deepcopy(record)
    for p in run["points"]:
        p["recall"] = max(0.0, p["recall"] + recall)
        p["latency_p95_ms"] *= p95
    for section, values in sections.items():
        run[section].update(values)
    return run


def test_identical_runs_have_no_regressions(record) -> None:
    cmp = R.compare_runs(record, record)
    assert not cmp.regressed
    assert [d.value for d in cmp.points] == [1, 4]
    assert cmp.recall_comparable
    assert cmp.latency_comparable
    assert cmp.notes == []


def test_recall_drop_and_latency_growth_regress(record) -> None:
    worse = R.compare_runs(record, _changed(record, recall=-0.05))
    assert worse.regressed
    assert all(d.recall_regressed and not d.latency_regressed for d in worse.points)
    assert all(p["recall"] > 0.05 for p in record["points"])
    assert worse.points[0].recall_change == pytest.approx(-0.05)
    # Within the threshold: not a regression.
    assert not R.compare_runs(record, _changed(record, recall=-0.005)).regressed
    # These latencies are microseconds: under the default 0.05 ms floor, 50% is noise.
    assert not R.compare_runs(record, _changed(record, p95=1.5)).regressed
    slower = R.compare_runs(
        record, _changed(record, p95=1.5), max_p95_increase=0.2, min_p95_increase_ms=0
    )
    assert [d.latency_regressed for d in slower.points] == [True, True]
    assert slower.points[0].p95_change == pytest.approx(0.5)
    assert not R.compare_runs(
        record, _changed(record, p95=1.5), max_p95_increase=0.6, min_p95_increase_ms=0
    ).regressed
    # Both thresholds apply: +50% of 10 ms is 5 ms, over a 1 ms floor but not a 6 ms one.
    ten_ms = copy.deepcopy(record)
    for p in ten_ms["points"]:
        p["latency_p95_ms"] = 10.0
    assert R.compare_runs(ten_ms, _changed(ten_ms, p95=1.5), min_p95_increase_ms=1).regressed
    assert not R.compare_runs(ten_ms, _changed(ten_ms, p95=1.5), min_p95_increase_ms=6).regressed


def test_incomparable_runs_say_why(record) -> None:
    other_machine = _changed(record, p95=3.0, environment={"machine": "x86_64"})
    cmp = R.compare_runs(record, other_machine)
    assert not cmp.latency_comparable
    assert not cmp.regressed  # latency from another machine isn't judged
    assert any("different environments (machine)" in n for n in cmp.notes)

    other_queries = _changed(record, recall=-0.5, queries={"sha256": "f" * 64})
    cmp = R.compare_runs(record, other_queries)
    assert not cmp.recall_comparable
    assert cmp.points == []
    assert any("Different query sets" in n for n in cmp.notes)

    other_index = _changed(record, index={"sha1": "0" * 40}, decision={"target_recall": 0.9})
    notes = R.compare_runs(record, other_index).notes
    assert any("Different index files" in n for n in notes)
    assert any("different targets" in n for n in notes)

    disjoint = copy.deepcopy(record)
    for p in disjoint["points"]:
        p["value"] += 100
    assert "No setting was measured in both runs." in R.compare_runs(record, disjoint).notes
    with pytest.raises(ValueError, match="non-negative"):
        R.compare_runs(record, record, max_recall_drop=-1)
