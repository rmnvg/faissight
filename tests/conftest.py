"""Shared fixtures: small synthetic indexes of every type, built once per test session."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

faiss = pytest.importorskip("faiss")

SMALL = {"n": 2_000, "d": 32, "n_centers": 20, "n_queries": 50, "nlist": 16, "hnsw_m": 16}


def _load_make_synthetic() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "make_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_synthetic", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_synthetic"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def make_synthetic() -> ModuleType:
    """The ``examples/make_synthetic.py`` module."""
    return _load_make_synthetic()


@pytest.fixture(scope="session")
def synthetic(
    tmp_path_factory: pytest.TempPathFactory, make_synthetic: ModuleType
) -> dict[str, Path]:
    """Paths to small synthetic indexes and side files (see ``make_synthetic.build_dataset``)."""
    out = tmp_path_factory.mktemp("synthetic")
    return make_synthetic.build_dataset(out, **SMALL)


@pytest.fixture(scope="session")
def binary_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A binary index file (unsupported by faissight)."""
    path = tmp_path_factory.mktemp("binary") / "binary_flat.index"
    faiss.write_index_binary(faiss.IndexBinaryFlat(64), str(path))
    return path
