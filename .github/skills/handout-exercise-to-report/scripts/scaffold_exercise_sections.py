#!/usr/bin/env python3
"""Generate LaTeX exercise sections, optionally with empty solution blocks."""

from __future__ import annotations

import argparse


def parse_labels(raw: str) -> list[str]:
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    if not labels:
        raise SystemExit("No labels provided")
    return labels


def render(labels: list[str], with_solution: bool) -> str:
    parts: list[str] = []
    for label in labels:
        parts.append(f"\\section{{Exercise {label}}}")
        parts.append("")
        parts.append("% TODO: copy exercise statement here")
        parts.append("")
        if with_solution:
            parts.append("\\begin{solution}")
            parts.append("  % TODO: write proof/solution here")
            parts.append("\\end{solution}")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX exercise section scaffolds")
    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated labels, e.g. 3.39,3.41,3.42",
    )
    parser.add_argument(
        "--with-solution",
        action="store_true",
        help="Include empty solution block under each section",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output file path; prints to stdout if omitted",
    )
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    text = render(labels, with_solution=args.with_solution)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
