"""Entry point — launch the FastAPI chat server."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from src.ui.server import launch

if __name__ == "__main__":
    launch()
