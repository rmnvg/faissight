"""faissight: see inside your FAISS index, debug retrieval, tune recall.

import faissight
viewer = faissight.launch(index, vectors=xb, metadata=chunks)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("faissight")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

# Imported after __version__: the server and CLI modules read it from this package.
from faissight.session import Session
from faissight.viewer import Viewer, launch

__all__ = ["Session", "Viewer", "__version__", "launch"]
