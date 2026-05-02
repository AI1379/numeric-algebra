#!/usr/bin/env python3
"""List Exercise section numbers from report.tex files.

Useful for detecting already-covered exercise numbers in previous homework.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import re

SECTION_RE = re.compile(r"\\section\{Exercise\s+([^}]+)\}")


def extract_sections(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return SECTION_RE.findall(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Exercise section labels from report files")
    parser.add_argument("--glob", required=True, help="Glob for report files, e.g. ./complex_analysis/hw*/report.tex")
    args = parser.parse_args()

    paths = sorted(pathlib.Path(p) for p in glob.glob(args.glob))
    if not paths:
        print("No files matched")
        return

    seen: set[str] = set()
    for path in paths:
        labels = extract_sections(path)
        print(f"=== {path.as_posix()} ===")
        if not labels:
            print("(no exercise sections found)")
            continue
        for lb in labels:
            print(f"- Exercise {lb}")
            seen.add(lb)

    print("\nAll unique labels:")
    for lb in sorted(seen):
        print(f"- Exercise {lb}")


if __name__ == "__main__":
    main()
