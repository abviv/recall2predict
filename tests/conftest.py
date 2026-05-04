"""Pytest configuration. Ensures project root is on sys.path so `src` is importable."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_src_root = _root / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))
