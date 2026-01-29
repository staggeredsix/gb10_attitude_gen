#!/usr/bin/env python3
"""
Patch LTX-2 ti2vid_one_stage.py to cache models between calls.

Applied during Docker build after the LTX-2 repo is cloned.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MARKER = "# --- LTX2_CACHE_MODELS_PATCH ---"

MODEL_ATTRS = (
    "text_encoder",
    "video_encoder",
    "transformer",
    "video_decoder",
    "audio_decoder",
    "vocoder",
)


def _bool_expr(name: str) -> str:
    # Use cached module when enabled, otherwise call builder each time.
    return f"(self._{name} if self._cache_models else self.model_ledger.{name}())"


def _ensure_import_os(text: str) -> str:
    if re.search(r"^import\s+os\b", text, flags=re.MULTILINE):
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
    lines.insert(insert_at, "import os")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _replace_model_ledger_calls(text: str) -> str:
    """
    Replace only real code tokens `self.model_ledger.NAME()` with cached expression.
    Avoid changing comments and triple-quoted strings.
    """
    lines = text.splitlines()
    out: list[str] = []

    in_triple: str | None = None
    patterns = {
        name: re.compile(rf"\bself\.model_ledger\.{re.escape(name)}\(\)")
        for name in MODEL_ATTRS
    }

    for line in lines:
        stripped = line.lstrip()

        # Skip pure comments.
        if stripped.startswith("#"):
            out.append(line)
            continue

        # Track triple-quoted strings and do not replace within them.
        if in_triple:
            out.append(line)
            if in_triple in line:
                in_triple = None
            continue

        if '"""' in line or "'''" in line:
            # Enter triple-string mode if it's an odd-count open.
            if line.count('"""') == 1 and line.count("'''") == 0:
                in_triple = '"""'
            elif line.count("'''") == 1 and line.count('"""') == 0:
                in_triple = "'''"
            out.append(line)
            continue

        replaced = line
        for name, pattern in patterns.items():
            replaced = pattern.sub(_bool_expr(name), replaced)
        out.append(replaced)

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _find_class_block(lines: list[str], class_name: str) -> tuple[int, int]:
    """
    Return (start_idx, end_idx) line indices for a top-level class block by indentation.
    """
    class_re = re.compile(rf"^class\s+{re.escape(class_name)}\b")
    start = None
    for i, line in enumerate(lines):
        if class_re.search(line):
            start = i
            break
    if start is None:
        raise RuntimeError(f"Could not find class {class_name}")

    # Determine class indent (likely 0) and find where it ends.
    class_indent = re.match(r"^(\s*)", lines[start]).group(1)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        indent = re.match(r"^(\s*)", lines[j]).group(1)
        if lines[j].strip() and len(indent) <= len(class_indent) and not lines[j].lstrip().startswith(("#", "@")):
            # A new top-level block begins.
            end = j
            break
    return start, end


def _insert_cache_init(text: str) -> str:
    if MARKER in text:
        return text  # already patched

    lines = text.splitlines()
    class_start, class_end = _find_class_block(lines, "TI2VidOneStagePipeline")

    # Find def __init__ inside class block.
    def_line_idx = None
    for idx in range(class_start, class_end):
        if lines[idx].lstrip().startswith("def __init__"):
            def_line_idx = idx
            break
    if def_line_idx is None:
        raise RuntimeError("Could not find TI2VidOneStagePipeline.__init__")

    def_indent = re.match(r"^(\s+)", lines[def_line_idx]).group(1)
    body_indent = def_indent + "    "

    # Find end of signature: MUST start with same indent as def_indent then ') ... :'
    end_re = re.compile(rf"^{re.escape(def_indent)}\)\s*(->\s*.+)?\:\s*$")
    sig_end_idx = None
    for idx in range(def_line_idx, class_end):
        line = lines[idx].rstrip()
        if end_re.match(line) or (line.startswith(def_indent) and line.endswith("):")):
            sig_end_idx = idx
            break
    if sig_end_idx is None:
        raise RuntimeError("Could not locate end of __init__ signature")

    body_start_idx = sig_end_idx + 1

    cache_lines = [
        f"{body_indent}{MARKER}",
        f'{body_indent}cache_env = os.getenv("LTX2_PIPELINES_CACHE_MODELS", "1")',
        f'{body_indent}self._cache_models = cache_env.strip().lower() not in {{"0", "false", "no", "off"}}',
    ]
    for name in MODEL_ATTRS:
        cache_lines.append(f"{body_indent}self._{name} = None")
    cache_lines.append(f"{body_indent}if self._cache_models:")
    for name in MODEL_ATTRS:
        cache_lines.append(f"{body_indent}    self._{name} = self.model_ledger.{name}()")

    # Insert after model_ledger assignment, within __init__ body only.
    insert_at = body_start_idx
    assign_idx = None
    for idx in range(body_start_idx, class_end):
        line = lines[idx]
        if not line.strip():
            continue
        indent = re.match(r"^(\s*)", line).group(1)
        # leave __init__ body
        if len(indent) < len(body_indent):
            break
        if "self.model_ledger" in line and "=" in line:
            assign_idx = idx
            break

    if assign_idx is not None:
        balance = lines[assign_idx].count("(") - lines[assign_idx].count(")")
        insert_at = assign_idx + 1
        for idx in range(assign_idx + 1, class_end):
            line = lines[idx]
            indent = re.match(r"^(\s*)", line).group(1)
            if line.strip() and len(indent) < len(body_indent):
                break
            balance += line.count("(") - line.count(")")
            insert_at = idx + 1
            if balance <= 0:
                break

    lines[insert_at:insert_at] = cache_lines + [""]

    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _guard_cleanup_and_del(text: str) -> str:
    """
    Avoid freeing cached models. Only cleanup/del when cache is off.
    """
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        indent = re.match(r"^(\s*)", line).group(1)

        if stripped == "cleanup_memory()":
            out.append(f"{indent}if not self._cache_models:")
            out.append(f"{indent}    cleanup_memory()")
            continue

        if stripped.startswith("del "):
            target = stripped.split(" ", 1)[1].strip()
            if target in MODEL_ATTRS:
                out.append(f"{indent}if not self._cache_models:")
                out.append(f"{indent}    del {target}")
                continue

        out.append(line)

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _compile_or_raise(path: Path, original_text: str) -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Restore original file
        path.write_text(original_text, encoding="utf-8")

        msg = exc.stderr or exc.stdout or "py_compile failed"

        # Attempt to extract line number and print context from the BROKEN (pre-restore) content
        m = re.search(r"line\s+(\d+)", msg)
        if m:
            line_no = int(m.group(1))
            broken = original_text.splitlines()  # original may not match, but gives context in logs
            # We'll also include the compiler message as-is.
            context = []
            lo = max(1, line_no - 10)
            hi = line_no + 10
            context.append(f"\n--- compile error context (approx) lines {lo}-{hi} ---")
            for i in range(lo, min(hi, len(broken)) + 1):
                prefix = ">>" if i == line_no else "  "
                context.append(f"{prefix} {i:4d}: {broken[i-1]}")
            msg = msg + "\n" + "\n".join(context)

        raise RuntimeError(msg) from exc


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: patch_ltx2_cache_models.py /path/to/ti2vid_one_stage.py", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    original_text = path.read_text(encoding="utf-8")
    text = original_text

    # Idempotent skip if already patched
    if MARKER in text:
        return 0

    text = _ensure_import_os(text)
    text = _replace_model_ledger_calls(text)
    text = _insert_cache_init(text)
    text = _guard_cleanup_and_del(text)

    path.write_text(text, encoding="utf-8")
    _compile_or_raise(path, original_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
