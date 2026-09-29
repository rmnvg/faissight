"""Lazy FAISS import with an actionable error when it is missing."""

from __future__ import annotations

from types import ModuleType
from typing import Any

INSTALL_HINT = (
    'Install FAISS with `pip install "faissight[faiss-cpu]"`, '
    "or `conda install -c pytorch faiss-cpu` if you use conda."
)


class FaissNotInstalledError(ImportError):
    """Raised when FAISS is required but not importable."""

    def __init__(self) -> None:
        super().__init__(f"FAISS is not installed. {INSTALL_HINT}")
        self.hint = INSTALL_HINT


def import_faiss() -> ModuleType:
    """Import and return the ``faiss`` module, raising :class:`FaissNotInstalledError`."""
    try:
        import faiss
    except ImportError as e:
        raise FaissNotInstalledError() from e
    return faiss


def class_name(obj: Any) -> str:
    """Concrete class name of a (downcast) FAISS object."""
    return type(obj).__name__
