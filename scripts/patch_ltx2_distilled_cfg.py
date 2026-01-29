#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MARKER = "# LTX2_DISTILLED_CFG_PATCH"

CALL_SIG_OLD = """    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
"""

CALL_SIG_NEW = """    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        negative_prompt: str = "",
        cfg_guidance_scale: float = 1.0,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
"""

ENCODE_OLD = """        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p
"""

ENCODE_NEW = """        # LTX2_DISTILLED_CFG_PATCH
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
"""

def _replace_block(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"[patch] expected block not found: {label}")
    return text.replace(old, new, 1)


def _ensure_import(text: str, import_line: str) -> str:
    if import_line in text:
        return text
    lines = text.splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.startswith("from __future__"):
            insert_at = idx + 1
            continue
        if line.startswith("import ") or line.startswith("from "):
            insert_at = idx + 1
            continue
        break
    lines.insert(insert_at, import_line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _replace_denoise_fn(text: str) -> str:
    old = """        denoise_fn=simple_denoising_func(
            video_context=video_context,
            audio_context=audio_context,
            transformer=transformer,  # noqa: F821
        ),
"""
    if old in text:
        new = """        denoise_fn=(
            guider_denoising_func(
                CFGGuider(cfg_guidance_scale),
                v_context_p,
                v_context_n,
                a_context_p,
                a_context_n,
                transformer=transformer,  # noqa: F821
            )
            if cfg_guidance_scale and cfg_guidance_scale > 1.0
            else simple_denoising_func(
                video_context=v_context_p,
                audio_context=a_context_p,
                transformer=transformer,  # noqa: F821
            )
        ),
"""
        return text.replace(old, new, 1)

    pattern = re.compile(
        r"denoise_fn\s*=\s*simple_denoising_func\([\s\S]*?\),",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        lines = text.splitlines()
        preview = "\n".join(lines[:200])
        print(preview, file=sys.stderr)
        raise RuntimeError("[patch] expected denoise_fn block not found")

    replacement = """denoise_fn=(
            guider_denoising_func(
                CFGGuider(cfg_guidance_scale),
                v_context_p,
                v_context_n,
                a_context_p,
                a_context_n,
                transformer=transformer,  # noqa: F821
            )
            if cfg_guidance_scale and cfg_guidance_scale > 1.0
            else simple_denoising_func(
                video_context=v_context_p,
                audio_context=a_context_p,
                transformer=transformer,  # noqa: F821
            )
        ),"""
    return text[: match.start()] + replacement + text[match.end() :]


def _compile_or_raise(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "py_compile failed")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: patch_ltx2_distilled_cfg.py /path/to/distilled.py", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    original_text = path.read_text(encoding="utf-8")
    if MARKER in original_text:
        print("already patched")
        return 0

    text = original_text
    text = _ensure_import(text, "from ltx_core.components.guiders import CFGGuider")
    text = _ensure_import(text, "from ltx_pipelines.utils.helpers import guider_denoising_func")
    text = _replace_block(text, CALL_SIG_OLD, CALL_SIG_NEW, label="__call__ signature")
    text = _replace_block(text, ENCODE_OLD, ENCODE_NEW, label="encode_text block")
    text = _replace_denoise_fn(text)

    path.write_text(text, encoding="utf-8")
    _compile_or_raise(path)
    print(f"[patch] patched ok: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
