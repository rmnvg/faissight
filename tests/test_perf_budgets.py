"""The scheduled performance check's pass/fail rules (benchmarks/regression.py)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "benchmarks" / "regression.py"


@pytest.fixture(scope="module")
def perf():
    spec = importlib.util.spec_from_file_location("perf_regression", PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks its module up while defining classes
    spec.loader.exec_module(module)
    return module


def test_absolute_budgets(perf) -> None:
    assert perf.check({"query_p95_ms": 1.0, "sweep_s": 0.5}) == []
    (failure,) = perf.check({"query_p95_ms": 151.0})
    assert "query_p95_ms" in failure
    assert "budget of 150" in failure


def test_slowdown_against_baseline_needs_ratio_and_delta(perf) -> None:
    base = {"sweep_s": 1.0, "query_p95_ms": 1.0, "peak_rss_mib": 300.0}
    # 2.5x but only 1.5 ms slower: noise on a tiny value, not a regression.
    assert perf.check({"query_p95_ms": 2.5}, base) == []
    (failure,) = perf.check({"sweep_s": 2.6}, base)
    assert "2.6x the baseline" in failure
    assert perf.check({"sweep_s": 1.9}, base) == []
    assert perf.check({"peak_rss_mib": 360.0}, base) == []
    assert len(perf.check({"peak_rss_mib": 420.0}, base)) == 1


def test_check_command_exit_code(perf, tmp_path, capsys) -> None:
    def write(name: str, metrics: dict[str, float]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({"metrics": metrics}))
        return path

    base = write("base.json", {"sweep_s": 1.0})
    assert (
        perf.main(["check", str(write("ok.json", {"sweep_s": 1.1})), "--baseline", str(base)]) == 0
    )
    assert (
        perf.main(["check", str(write("slow.json", {"sweep_s": 3.0})), "--baseline", str(base)])
        == 1
    )
    assert "FAIL sweep_s" in capsys.readouterr().out
