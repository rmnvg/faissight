import json

import pytest

from faissight.core.history import RunHistory


def test_archive_restart_rename_delete_and_lossless_ids(tmp_path):
    history = RunHistory(tmp_path)
    run_id = history.save("evaluation", "First", "index", {"ids": [2**63 - 1], "recall": 0.5})
    restarted = RunHistory(tmp_path)
    assert restarted.list()[0]["label"] == "First"
    assert restarted.get(run_id)["data"]["ids"] == [str(2**63 - 1)]
    restarted.rename(run_id, "Better name")
    assert history.get(run_id)["label"] == "Better name"
    assert history.get(run_id)["data"]["recall"] == 0.5
    restarted.delete(run_id)
    assert history.list() == []
    with pytest.raises(FileNotFoundError):
        history.get(run_id)


def test_history_disabled_invalid_and_corrupt(tmp_path):
    history = RunHistory(tmp_path / "disabled", enabled=False)
    assert history.save("sweep", "name", "index", {}) is None
    assert history.list() == []
    assert not history.root.exists()
    with pytest.raises(ValueError, match="id"):
        history.get("../secret")
    with pytest.raises(FileNotFoundError):
        history.get("a" * 32)
    history = RunHistory(tmp_path)
    (tmp_path / ("a" * 32 + ".json")).write_text("{")
    (tmp_path / ("b" * 32 + ".json")).write_text(json.dumps({"data": {}}))
    assert history.list() == []
    run_id = history.save("sweep", "name", "index", {})
    with pytest.raises(ValueError, match="Name"):
        history.rename(run_id, " ")
    assert not list(tmp_path.glob("*.tmp"))


def test_retention_and_failed_atomic_write(tmp_path):
    history = RunHistory(tmp_path)
    first = history.save("sweep", "first", "index", {})
    for i in range(101):
        history.save("sweep", str(i), "index", {})
    assert len(history.list()) == 100
    with pytest.raises(FileNotFoundError):
        history.get(first)
    with pytest.raises(ValueError, match="Out of range float"):
        history.save("sweep", "bad", "index", {"nan": float("nan")})
    assert len(history.list()) == 100
    assert not list(tmp_path.glob("*.tmp"))
