"""Test bootstrap: make the ``q1_alignment`` package importable from ``src``.

Running ``python -m unittest discover -s src/q1_alignment/tests`` starts with
``sys.path`` pointing at the discovery directory, not at ``src``. Insert the
``src`` directory here so the tests can import ``q1_alignment`` regardless of
the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
