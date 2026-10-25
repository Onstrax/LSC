"""Entrada ejecutable: ``python -m lsc_data <subcomando>`` (design §19)."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
