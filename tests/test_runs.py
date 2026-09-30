import copy
import json
import math

import numpy as np
import pytest

from faissight.core import runs as R
from faissight.core import search as S
from faissight.core import sweep as W
from faissight.core import vectors as V
from faissight.core.loader import load_index
from faissight.core.projection import index_fingerprint


@pytest.fixture(scope="module")
def src(synthetic):
    li = load_index(synthetic["ivf_pq"])
    return li, V.from_arrays(li, np.load(synthetic["vectors"]))


@pytest.fixture(scope="module")
def record(src):
    li, source = src
    qs = W.sample_queries(source, 30)
    truth = W.ground_truth_ids(S.GroundTruth(source, li.metric), qs, 10)
    result = W.sweep(li, qs, truth, values=[1, 4], k=10, repeats=1)
    return R.run_record(
        result,
        li,
        index_fingerprint(li),
        target_recall=0.5,
        max_p95_ms=100.0,
        label="pq",
        source=source,
    )


def test_run_record_round_trips(record, tmp_path) -> None:
    assert record["format"] == R.RUN_FORMAT
    assert record["index"]["kind"] == "IVF_PQ"
    assert record["index"]["name"] == "ivf_pq.index"
    assert len(record["index"]["sha1"]) == 40
    assert record["queries"]["origin"] == "sampled"
    assert record["settings"]["values"] == [1, 4]
    assert record["settings"]["truth_source"] == "raw"
    assert len(record["settings"]["ground_truth_fingerprint"]) == 40
    assert record["decision"]["max_p95_ms"] == 100.0
    assert record["decision"]["status"] in ("ok", "recall")
    assert record["points"][0]["worst_queries"]  # example failures travel with the run
    path = R.save_run(record, tmp_path / "run.json")
    assert R.load_run(path) == json.loads(json.dumps(record))


def test_run_record_without_source_has_no_ground_truth_fingerprint(src) -> None:
    li, source = src
    qs = W.sample_queries(source, 10)
    truth = W.ground_truth_ids(S.GroundTruth(source, li.metric), qs, 10)
    result = W.sweep(li, qs, truth, values=[1], k=10, repeats=1)
    record = R.run_record(result, li, "x" * 40, target_recall=0.5)
    assert record["settings"]["ground_truth_fingerprint"] is None


# --- corpus fingerprint -----------------------------------------------------------------


def test_corpus_fingerprint_is_stable_and_sensitive(src) -> None:
    _, source = src
    assert R.corpus_fingerprint(source) == R.corpus_fingerprint(source)
    perturbed = V.VectorSource(source.vectors + np.float32(0.01), source.ids, source.reconstructed)
    assert R.corpus_fingerprint(source) != R.corpus_fingerprint(perturbed)
    empty = V.VectorSource(
        np.empty((0, source.vectors.shape[1]), np.float32), np.empty(0, np.int64), False
    )
    assert R.corpus_fingerprint(empty)  # doesn't crash on an empty corpus


# --- validation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.update(format="other"), "Not a faissight sweep run"),
        (lambda r: r.update(version=99), "version 99"),
        (lambda r: r.pop("index"), "'index'"),
        (lambda r: r["points"][0].pop("recall"), "'recall'"),
        (lambda r: r["settings"].pop("k"), "'k'"),
        (lambda r: r["index"].update(sha1=""), "non-empty string"),
        (lambda r: r["queries"].update(sha256=None), "non-empty string"),
        (lambda r: r.update(points=[]), "non-empty list"),
        (lambda r: r["points"][0].update(recall=math.nan), "finite"),
        (lambda r: r["points"][0].update(recall=1.5), "outside \\[0, 1\\]"),
        (lambda r: r["points"][0].update(recall=-0.5), "outside \\[0, 1\\]"),
        (lambda r: r["points"][0].update(latency_p95_ms=-1.0), "negative"),
        (lambda r: r["points"][0].update(latency_p95_ms=math.inf), "finite"),
        (lambda r: r["points"][0].update(latency_mean_ms=-1.0), "negative"),
        (lambda r: r["points"][0].update(probe_coverage=2.0), "outside \\[0, 1\\]"),
        (
            lambda r: r["points"][0].update(recall_ci_low=0.9, recall_ci_high=0.1),
            "recall_ci_low > recall_ci_high",
        ),
        (lambda r: r["points"].append(dict(r["points"][0])), "duplicate point value"),
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


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_run_rejects_nonstandard_json_constants(record, tmp_path, token) -> None:
    bad = copy.deepcopy(record)
    bad["points"][0]["recall"] = 0.5
    text = json.dumps(bad).replace("0.5", token, 1)
    (tmp_path / "bad.json").write_text(text)
    with pytest.raises(R.RunFormatError, match="not allowed"):
        R.load_run(tmp_path / "bad.json")


def test_save_run_rejects_an_invalid_record(record, tmp_path) -> None:
    bad = copy.deepcopy(record)
    bad["points"][0]["recall"] = math.nan
    with pytest.raises(R.RunFormatError):
        R.save_run(bad, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def _changed(record, recall=0.0, p95=1.0, **sections):
    run = copy.deepcopy(record)
    for p in run["points"]:
        p["recall"] = max(0.0, min(1.0, p["recall"] + recall))
        p["latency_p95_ms"] *= p95
    for section, values in sections.items():
        run[section].update(values)
    return run


# --- comparison ----------------------------------------------------------------------------


def test_identical_runs_have_no_regressions(record) -> None:
    cmp = R.compare_runs(record, record)
    assert not cmp.regressed
    assert cmp.comparable
    assert [d.value for d in cmp.points] == [1, 4]
    assert cmp.recall_comparable
    assert cmp.latency_comparable
    assert cmp.notes == []


def test_recall_drop_regresses(record) -> None:
    worse = R.compare_runs(record, _changed(record, recall=-0.05))
    assert worse.regressed
    assert worse.comparable
    assert all(d.recall_regressed and not d.latency_regressed for d in worse.points)
    assert all(p["recall"] > 0.05 for p in record["points"])
    assert worse.points[0].recall_change == pytest.approx(-0.05)
    # Within the threshold: not a regression.
    assert not R.compare_runs(record, _changed(record, recall=-0.005)).regressed


def test_latency_growth_regresses_with_a_noise_floor(record) -> None:
    # Fixed synthetic timings, not the fixture's real measured ones: real wall-clock p95
    # over 30 queries and one repeat is itself noisy, which would make this test flaky.
    microsecond_scale = copy.deepcopy(record)
    for p in microsecond_scale["points"]:
        p["latency_p95_ms"] = 0.02
    # +50% of 0.02 ms is 0.01 ms: under the default 0.05 ms floor, so not a regression.
    assert not R.compare_runs(microsecond_scale, _changed(microsecond_scale, p95=1.5)).regressed
    slower = R.compare_runs(
        microsecond_scale,
        _changed(microsecond_scale, p95=1.5),
        max_p95_increase=0.2,
        min_p95_increase_ms=0,
    )
    assert [d.latency_regressed for d in slower.points] == [True, True]
    assert slower.points[0].p95_change == pytest.approx(0.5)
    assert not R.compare_runs(
        microsecond_scale,
        _changed(microsecond_scale, p95=1.5),
        max_p95_increase=0.6,
        min_p95_increase_ms=0,
    ).regressed
    # Both thresholds apply: +50% of 10 ms is 5 ms, over a 1 ms floor but not a 6 ms one.
    ten_ms = copy.deepcopy(record)
    for p in ten_ms["points"]:
        p["latency_p95_ms"] = 10.0
    assert R.compare_runs(ten_ms, _changed(ten_ms, p95=1.5), min_p95_increase_ms=1).regressed
    assert not R.compare_runs(ten_ms, _changed(ten_ms, p95=1.5), min_p95_increase_ms=6).regressed


def test_incomparable_runs_say_why(record) -> None:
    # Derived from the recorded value, so it differs on any CI runner or laptop.
    machine = f"not-{record['environment']['machine']}"
    other_machine = _changed(record, p95=3.0, environment={"machine": machine})
    cmp = R.compare_runs(record, other_machine)
    assert not cmp.latency_comparable
    assert not cmp.regressed  # latency from another machine isn't judged
    assert cmp.comparable  # recall was still checked
    assert any("different environments (machine)" in n for n in cmp.notes)

    other_queries = _changed(record, recall=-0.5, queries={"sha256": "f" * 64})
    cmp = R.compare_runs(record, other_queries)
    assert not cmp.recall_comparable
    assert not cmp.comparable
    assert cmp.points == []
    assert any("Different query sets" in n for n in cmp.notes)

    other_index = _changed(record, index={"sha1": "0" * 40}, decision={"target_recall": 0.9})
    notes = R.compare_runs(record, other_index).notes
    assert any("Different index files" in n for n in notes)
    assert any("different targets" in n for n in notes)

    disjoint = copy.deepcopy(record)
    for p in disjoint["points"]:
        p["value"] += 100
    disjoint_cmp = R.compare_runs(record, disjoint)
    assert not disjoint_cmp.comparable
    assert "No setting was measured in both runs." in disjoint_cmp.notes
    with pytest.raises(ValueError, match="non-negative"):
        R.compare_runs(record, record, max_recall_drop=-1)


def test_different_metrics_are_not_comparable(record) -> None:
    other_metric = _changed(record, index={**record["index"], "metric": "IP"})
    cmp = R.compare_runs(record, other_metric)
    assert not cmp.recall_comparable
    assert not cmp.comparable
    assert any("Different metrics" in n for n in cmp.notes)
    assert any("L2" in n and "IP" in n for n in cmp.notes)


def test_different_ground_truth_corpus_is_not_comparable(record) -> None:
    assert record["settings"]["ground_truth_fingerprint"] is not None
    other_corpus = _changed(
        record,
        settings={
            **record["settings"],
            "ground_truth_fingerprint": "f" * 40,
        },
    )
    cmp = R.compare_runs(record, other_corpus)
    assert not cmp.recall_comparable
    assert not cmp.comparable
    assert any("--vectors file differs" in n for n in cmp.notes)


def test_missing_ground_truth_fingerprint_is_not_flagged(record) -> None:
    # Older runs (or reconstructed ground truth) have no fingerprint; absence on either
    # side shouldn't itself block a comparison.
    no_fp = _changed(record, settings={**record["settings"], "ground_truth_fingerprint": None})
    cmp = R.compare_runs(record, no_fp)
    assert cmp.recall_comparable
    assert not any("--vectors file differs" in n for n in cmp.notes)
