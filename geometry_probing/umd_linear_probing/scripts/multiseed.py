#!/usr/bin/env python
"""Entry-point for sequential geometry multi-seed runs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pba.geometry.multiseed import parse_multiseed_args, run_multiseed  # noqa: E402,F401


def main() -> None:
    run_multiseed()


if __name__ == "__main__":  # pragma: no cover
    main()
