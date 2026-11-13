#!/usr/bin/env python3
"""Require a valid DCO Signed-off-by trailer in a Git commit message."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SIGN_OFF_RE = re.compile(
    r"^Signed-off-by:\s+(?P<name>(?=[^<>\r\n]*[^\s<>\r\n])[^<>\r\n]+?)\s+<(?P<email>[^<>\s@]+@[^<>\s@]+)>\s*$",
    re.IGNORECASE,
)


def has_valid_sign_off(message: str) -> bool:
    """Return whether the message contains a well-formed DCO sign-off."""
    return any(SIGN_OFF_RE.fullmatch(line) for line in message.splitlines() if not line.lstrip().startswith("#"))


def main(argv: list[str]) -> int:
    """Validate the commit-message file supplied by Git's commit-msg hook."""
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <commit-message-file>", file=sys.stderr)
        return 2

    message_path = Path(argv[1])
    try:
        message = message_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        print(f"error: cannot read commit message {message_path}: {error}", file=sys.stderr)
        return 2

    if has_valid_sign_off(message):
        return 0

    print(
        "error: commit message is missing a valid 'Signed-off-by: Name <email>' trailer.\n"
        "Create signed-off commits with: git commit -s",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
