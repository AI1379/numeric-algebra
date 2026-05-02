#!/usr/bin/env python3
"""Extract exercise markers and nearby context from a PDF handout.

This script uses pdftotext (external binary) and then scans text lines with
regex patterns such as: Exercise 3.[0-9]+.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Iterable


def run_pdftotext(pdf: pathlib.Path, txt: pathlib.Path, layout: bool) -> None:
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd.extend([str(pdf), str(txt)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "pdftotext failed\n")
        raise SystemExit(proc.returncode)


def read_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def find_matches(lines: Iterable[str], regex: re.Pattern[str]) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines, start=1):
        for m in regex.finditer(line):
            rows.append((i, m.group(0), line.rstrip()))
    return rows


def print_context(lines: list[str], line_no: int, context: int) -> None:
    start = max(1, line_no - context)
    end = min(len(lines), line_no + context)
    for i in range(start, end + 1):
        prefix = ">" if i == line_no else " "
        print(f"{prefix}{i:5d}: {lines[i - 1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract exercise markers from PDF handout text")
    parser.add_argument("--pdf", required=True, help="Path to source PDF")
    parser.add_argument(
        "--pattern",
        default=r"Exercise\s+[0-9]+\.[0-9]+",
        help="Regex for exercise markers",
    )
    parser.add_argument("--context", type=int, default=6, help="Context lines around each match")
    parser.add_argument(
        "--layout",
        action="store_true",
        help="Use pdftotext -layout for two-column PDFs",
    )
    parser.add_argument(
        "--keep-txt",
        action="store_true",
        help="Keep generated txt file next to PDF",
    )
    args = parser.parse_args()

    pdf = pathlib.Path(args.pdf).resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    txt = pdf.with_suffix(".layout.txt" if args.layout else ".raw.txt")
    run_pdftotext(pdf, txt, layout=args.layout)

    lines = read_lines(txt)
    regex = re.compile(args.pattern)
    matches = find_matches(lines, regex)

    if not matches:
        print("No exercise markers found. Try --layout or a different --pattern.")
        raise SystemExit(1)

    unique = sorted({m[1] for m in matches})
    print("Found markers:")
    for marker in unique:
        print(f"- {marker}")

    print("\nDetailed contexts:\n")
    for line_no, marker, raw_line in matches:
        print("=" * 72)
        print(f"Marker: {marker} | line {line_no}")
        print_context(lines, line_no, args.context)

    if not args.keep_txt:
        try:
            txt.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
