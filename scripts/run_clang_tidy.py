#!/usr/bin/env python3
"""Run the supported clang-tidy version on first-party host C++ translation units."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

HOST_SOURCE_SUFFIXES = frozenset({".cc", ".cpp", ".cxx"})
EXCLUDED_PARTS = frozenset({"build", "thirdparty"})
VERSION_RE = re.compile(r"(?:LLVM|clang) version\s+(\d+)", re.IGNORECASE)
CLANG_TIDY_MAJOR_VERSION = 20


def repository_root() -> Path:
    """Return the root of the current Git worktree."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def is_first_party(path: Path, repo_root: Path) -> bool:
    """Return whether path is inside the repository and outside excluded trees."""
    try:
        relative = path.resolve().relative_to(repo_root)
    except ValueError:
        return False
    return not any(part in EXCLUDED_PARTS or part.startswith("build-") for part in relative.parts)


def host_translation_units(compilation_database: Path, repo_root: Path) -> list[Path]:
    """Read unique first-party host C++ files from compile_commands.json."""
    try:
        entries = json.loads(compilation_database.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"compilation database not found: {compilation_database}; configure CMake with "
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid compilation database: {compilation_database}: {error}") from error

    sources: set[Path] = set()
    for entry in entries:
        filename = Path(entry["file"])
        if not filename.is_absolute():
            filename = Path(entry["directory"]) / filename
        filename = filename.resolve()
        if filename.suffix.lower() in HOST_SOURCE_SUFFIXES and is_first_party(filename, repo_root):
            sources.add(filename)
    return sorted(sources)


def find_clang_tidy(explicit: str | None) -> str:
    """Find the supported clang-tidy executable."""
    versioned_executable = f"clang-tidy-{CLANG_TIDY_MAJOR_VERSION}"
    candidates = [explicit] if explicit else [versioned_executable, "clang-tidy"]
    resolved_candidates = (shutil.which(candidate) for candidate in candidates if candidate)
    executable = next((path for path in resolved_candidates if path is not None), None)
    if executable is None:
        raise RuntimeError(
            f"clang-tidy {CLANG_TIDY_MAJOR_VERSION} not found (looked for {versioned_executable} and clang-tidy)"
        )
    version = subprocess.run(
        [executable, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ).stdout
    match = VERSION_RE.search(version)
    if match is None or int(match.group(1)) != CLANG_TIDY_MAJOR_VERSION:
        raise RuntimeError(f"expected clang-tidy {CLANG_TIDY_MAJOR_VERSION}, got: {version.strip()}")
    return executable


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build", type=Path, help="CMake build directory (default: build)")
    parser.add_argument(
        "--clang-tidy",
        help=f"clang-tidy {CLANG_TIDY_MAJOR_VERSION} executable (auto-detected by default)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = repository_root()
        build_dir = args.build_dir if args.build_dir.is_absolute() else repo_root / args.build_dir
        translation_units = host_translation_units(build_dir / "compile_commands.json", repo_root)
        if not translation_units:
            print("No first-party host C++ translation units found in the compilation database.")
            return 0

        clang_tidy = find_clang_tidy(args.clang_tidy)
        command = [
            clang_tidy,
            *map(str, translation_units),
            "-p",
            str(build_dir),
        ]
        return subprocess.run(command, cwd=repo_root, check=False).returncode
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
