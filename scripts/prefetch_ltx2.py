#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

from huggingface_hub import snapshot_download


def _allow_patterns(fp4_only: bool) -> list[str]:
    patterns = [
        "model_index.json",
        "*.json",
        "tokenizer/**",
        "scheduler/**",
        "transformer/**",
        "text_encoder/**",
        "vae/**",
        "*fp4*.safetensors",
        "*.fp4.safetensors",
    ]
    if not fp4_only:
        patterns.append("*.safetensors")
    return patterns


def _tree_summary(root: pathlib.Path, max_entries: int = 200) -> list[str]:
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        entries.append(str(rel) + ("/" if path.is_dir() else ""))
        if len(entries) >= max_entries:
            entries.append("... (truncated)")
            break
    return entries


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch LTX-2 snapshot for offline use")
    parser.add_argument("--repo-id", default="Lightricks/LTX-2")
    parser.add_argument("--local-dir", default="./models/LTX-2")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fp4-only", action="store_true", help="Download only fp4 weights (default)")
    group.add_argument("--all-weights", action="store_true", help="Download all safetensors weights")
    args = parser.parse_args(list(argv) if argv is not None else None)

    fp4_only = True if not args.all_weights else False

    local_dir = pathlib.Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {local_dir} (fp4_only={fp4_only})")
    snapshot_dir = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=_allow_patterns(fp4_only=fp4_only),
    )

    model_index = pathlib.Path(snapshot_dir) / "model_index.json"
    if not model_index.is_file():
        print("ERROR: model_index.json not found in snapshot.")
        return 1

    print("Snapshot ready at:", snapshot_dir)
    print("Directory summary:")
    for line in _tree_summary(pathlib.Path(snapshot_dir)):
        print("  ", line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
