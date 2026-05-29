"""
Logging setup for LTRaaS.

Call setup_logging() once at startup (main.py does this). After that,
every module just does logging.getLogger(__name__) — nothing else needed.

Console shows INFO and above. The log file captures DEBUG too, so you can
see per-probe detail without cluttering the terminal.
"""

import logging
import logging.handlers
from pathlib import Path


_FMT = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logging(log_dir: str = "logs") -> None:
    """Configure the root logger. Safe to call multiple times (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)

    # ── Console: INFO and above ──────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(_FMT)
    root.addHandler(console)

    # ── File: DEBUG and above, rotated at 5 MB (3 backups kept) ────────────
    Path(log_dir).mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "ltraas.log",
        mode="a",
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FMT)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging initialised — file: %s/ltraas.log", log_dir
    )
