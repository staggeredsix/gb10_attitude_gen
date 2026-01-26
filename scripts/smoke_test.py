#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from ltx2_backend import validate_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for LTX-2 snapshot validation")
    parser.add_argument("--path", default=os.getenv("LTX2_MODEL_ID", "/models/LTX-2"))
    args = parser.parse_args()

    result = validate_snapshot(args.path)
    print("Snapshot validation OK")
    print("Components:", result["components"])
    print("Safetensors:", result["safetensors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
