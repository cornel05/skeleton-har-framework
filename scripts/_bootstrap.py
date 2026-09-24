"""Make `src/` importable when a script is run as `uv run scripts/<name>.py`."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
