---
name: literature-reading-workflow
description: Read academic PDF papers and produce Chinese reading notes using the local D:\TREE paper-reading workflow. Use when the user asks to read, summarize,整理,精读, or make notes for PDF literature, especially when outputs should go under D:\TREE\paper reading\output.
---

# Literature Reading Workflow

Use this skill to process PDF papers into reusable Chinese reading notes with explicit parser provenance and quality warnings.

## Workflow

1. Read the active PDF rules before processing.
   - Check `C:\Users\57680\.codex\AGENTS.md` when available.
   - Follow the project rule: confirm the `pdf-reading` conda environment, parse text before summarizing, and report parser/intermediate files.

2. Prepare outputs under `D:\TREE\paper reading\output`.
   - Use stable filenames derived from the PDF stem.
   - Keep extracted text as `<stem>_extracted.txt`.
   - Keep rendered page samples under `rendered_pages/` or `<stem>_rendered_pages/`.

3. Extract text with the agreed local parser.
   - Prefer:
     `C:\Users\57680\.conda\envs\pdf-reading\python.exe "D:\TREE\paper reading\pdf_to_text.py" <input.pdf> <output.txt>`
   - If the script is missing, use PyMuPDF in the same `pdf-reading` environment.
   - Do not summarize directly from memory or filename alone.

4. Render and inspect representative pages.
   - Prefer `pdftoppm` if it works.
   - If `pdftoppm` fails, render with PyMuPDF.
   - Inspect at least: page 1, a table-heavy page, a figure-heavy page, an experiment/results page, and the conclusion/reference tail.

5. Assess extraction quality.
   - Mark quality as good when text order is readable and section structure is preserved.
   - Explicitly warn when tables lose column alignment, double-column order is mixed, formulas are garbled, figures are essential, or scanned/OCR text is weak.
   - If ordinary PyMuPDF extraction is poor, recommend MinerU instead of guessing.

6. Write the Chinese note.
   - Include: paper metadata, one-paragraph summary, research problem, method/dataset, experiments, key numbers, findings, limitations, relevance to the user's project, and follow-up ideas.
   - Include a "解析与核对说明" section naming the parser, intermediate text file, rendered pages, and any content that should be checked against the original PDF.
   - Use concise Markdown headings and tables. Avoid long verbatim quotes.

7. Validate artifacts.
   - Check output files exist and are readable.
   - If a workflow skill was created or edited, validate it with `quick_validate.py`.

## Optional Helper

Use `scripts/prepare_pdf_reading.py` from this skill to extract text and render sample pages in one step:

```powershell
C:\Users\57680\.conda\envs\pdf-reading\python.exe "D:\TREE\paper reading\output\literature-reading-workflow\scripts\prepare_pdf_reading.py" <input.pdf> --output-dir "D:\TREE\paper reading\output"
```

## Note Template

Read `references/chinese_note_template.md` when you need a consistent note structure.
