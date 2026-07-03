#!/usr/bin/env python
"""Evaluate a saved linear probe checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pba.geometry.linear_probe import parse_eval_args, run_eval  # noqa: E402,F401


def main() -> None:
    run_eval()


if __name__ == "__main__":  # pragma: no cover
    main()
