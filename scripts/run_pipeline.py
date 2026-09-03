#!/usr/bin/env python3
"""Run the full factor research pipeline end-to-end.

Usage:
    python scripts/run_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_factor_lab.pipeline import run  # noqa: E402

if __name__ == "__main__":
    run()
