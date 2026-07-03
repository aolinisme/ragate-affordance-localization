#!/usr/bin/env python
"""Entry-point for running the linear probing experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pba.geometry.linear_probe import parse_train_args, run_train  # noqa: E402,F401


def main() -> None:
    run_train()


if __name__ == "__main__":  # pragma: no cover
    main()
