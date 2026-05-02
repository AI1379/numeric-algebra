---
name: handout-exercise-to-report
description: 'Extract exercise statements from course handout PDFs and synchronize them into LaTeX report.tex with the existing template. Use for tasks like detect correct exercise set, handle duplicate exercise numbers with user confirmation, generate empty solution blocks by default, and verify build with LuaLaTeX.'
argument-hint: 'Course folder, handout PDF path, target report.tex, expected exercise count'
user-invocable: true
disable-model-invocation: false
---

# Handout Exercise To Report

## What This Skill Produces
This skill updates a target `report.tex` by:
- Extracting exercise numbers/statements from a handout PDF
- Selecting the correct assignment set (for example, 8 exercises)
- Detecting duplicate exercise numbers against earlier reports and asking user confirmation
- Writing sections in the existing report style
- Generating empty `solution` template blocks by default
- Verifying the result with LaTeX compilation

## When To Use
Use this skill when the user asks to:
- "copy this homework statement into report"
- "extract exercises from handout PDF"
- "sync handout problems into LaTeX"
- "I copied the wrong exercise numbers"

## Inputs To Collect
- Handout PDF path, for example `complex_analysis/handouts/CA-chap3-2026-04-14.pdf`
- Target report path, for example `complex_analysis/hw5/report.tex`
- Expected problem count, for example 8
- Optional: chapter-specific regex pattern, for example `Exercise\s+3\.[0-9]+`

## Procedure
1. Inspect existing reports and current target report.
2. Extract exercise candidates from the handout PDF.
3. If extraction is noisy (two-column mixing), switch extractor mode and re-check contexts.
4. Determine the final exercise set:
   - Prefer explicit user-provided count/range.
  - Detect exercise-number overlaps with previous reports.
  - If overlap exists, pause and ask user confirmation before continuing (do not auto-drop duplicates).
  - If multiple candidates remain, present the candidate list and ask one targeted confirmation.
5. Update `report.tex` preserving the existing preamble/header/template style.
6. By default, generate one empty template under each exercise:
  - `\begin{solution}`
  - placeholder sentence or blank line
  - `\end{solution}`
7. Run compile check with LuaLaTeX if macros require it.
8. Validate completion criteria.

## Decision Logic
- If file name is uncertain (for example `chap3` vs `chpt3`), check directory and pick existing file.
- If plain `pdftotext` output is garbled, retry with `-layout` and line-context extraction.
- If theorem references are ambiguous, locate theorem statement from matching chapter handout before writing proofs.
- If user says "this homework has N problems", enforce exact section count N as a hard check.
- If duplicate exercise numbers are found in prior reports, always ask user whether this is intentional.

## Completion Checks
- `report.tex` contains exactly the expected number of `\section{Exercise ...}` entries.
- Exercise numbers match the selected set.
- Every exercise section contains an empty `solution` block unless user explicitly requests otherwise.
- Statements are present and readable (no broken mixed-column fragments).
- Compilation succeeds (`latexmk -lualatex -interaction=nonstopmode report.tex`).

## Scripts
- Extract exercise markers and context:
  - [extract_exercises.py](./scripts/extract_exercises.py)
- Parse section exercise numbers from an existing report:
  - [report_exercises.py](./scripts/report_exercises.py)
- Generate LaTeX exercise sections with empty solution blocks:
  - [scaffold_exercise_sections.py](./scripts/scaffold_exercise_sections.py)

## Example Commands
```powershell
# 1) Extract exercise candidates from PDF
python ./.github/skills/handout-exercise-to-report/scripts/extract_exercises.py \
  --pdf ./complex_analysis/handouts/CA-chap3-2026-04-14.pdf \
  --pattern "Exercise 3\\.[0-9]+" --context 8

# 2) List existing exercise sections in previous reports
python ./.github/skills/handout-exercise-to-report/scripts/report_exercises.py \
  --glob "./complex_analysis/hw*/report.tex"

# 3) Generate LaTeX section blocks with empty solution templates
python ./.github/skills/handout-exercise-to-report/scripts/scaffold_exercise_sections.py \
  --labels "3.39,3.41,3.42" --with-solution
```

## Output Style Requirements
- Preserve existing LaTeX template and section naming style.
- Do not rewrite unrelated content.
- Keep math notation consistent with `common/macros.tex`.
- For proof-writing tasks, cite used theorem names/numbers explicitly.
