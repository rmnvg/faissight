"""faissight: see inside your FAISS index, debug retrieval, tune recall."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("faissight")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
