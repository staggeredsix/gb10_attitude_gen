#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from ltx2_backend import validate_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for LTX-2 snapshot validation")
    parser.add_argument("--path", default=os.getenv("LTX2_MODEL_ID", "/models/LTX-2"))
    parser.add_argument("--variant", default=os.getenv("LTX2_VARIANT", "fp4"))
    args = parser.parse_args()

    result = validate_snapshot(args.path, args.variant)
    print("Snapshot validation OK")
    print("Components:", result["components"])
    print("FP4 files:", result["fp4_files"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
