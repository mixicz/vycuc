# Extraction PoC — design

Date: 2026-10-28 · Status: approved

## Goal

Decide which extraction pipeline Výcuc uses in production by measuring candidates on real documents. The question to answer:

> Which combination of OCR engine and text→fields extractor reads Czech invoices and receipts
> accurately enough, and what CPU time and RAM does it actually need?

The answer feeds the hosting choice: the budget is well under 4 000 Kč/year (see `docs/python-hosting-cz.md`), i.e. roughly 1 vCPU and 1–2 GB RAM. RAM is **measured, not capped** — an engine that needs more is still benchmarked, so we know what bigger hosting would buy.

The harness code is kept: it becomes the app's extraction module and its regression test.

## Context and constraints

- Stack: Python (approach A — Django monolith later). Every candidate below is Python or has a Python binding.
- Production must be CPU-only and Czech-billed; a GPU model is only a quality reference.
- Test data: 67 PDFs in `testdata/zari/` (gitignored, client documents), each with a hand-checked `<file>.expected.json` label (schema: `docs/label-schema.md`).
  - 40 PDFs have a text layer → no OCR needed.
  - 27 are scans/photos (receipts, a few invoices), including hard cases: crumpled receipt (ZA044), printer with broken diacritics (one shop's receipts), 1:5 phone screenshot (ZA003), rotated stamp (ZA017).
- Open accounting questions are tracked in `docs/otazky-pro-hanu.md`; they do not block the PoC.

## Pipeline under test

```
PDF ──► pages ──┬─ text layer present ──────────────► text ─┐
                └─ scan ─► render 300 DPI ─► OCR ───► text ─┼─► extractor ─► fields ─► checks ─► result
                                                            ┘
```

Three stages, measured separately so that a weak OCR cannot hide a good extractor and vice versa.

### Stage 1 — text

- **Text layer:** pypdfium2. Applied to the 40 native PDFs; also run OCR on them as a control.
- **OCR candidates** (all CPU, default settings, Czech models where the engine has them):
  1. Tesseract 5 (`ces` traineddata)
  2. PaddleOCR (current PP-OCR generation with Czech/Latin recognition)
  3. RapidOCR (ONNX Runtime port of PaddleOCR models — lighter runtime)
  4. docTR
  5. EasyOCR

  Exact package versions are looked up and pinned in the implementation plan. No custom preprocessing in round one (engines get the same 300 DPI render); deskew/crop is a round-two option only if the hard cases fail across all engines.

### Stage 2 — text → fields

- **(a) Rules:** regexes + layout heuristics for Czech labels (DUZP, Datum vystavení, Variabilní symbol, IČ/IČO, DIČ, VAT summary table, Celkem / K úhradě) and type/payment keywords.
- **(b) Small LLM on CPU:** llama.cpp with a quantised instruct model (e.g. Qwen3 1.7B and 4B, Q4), prompted with the page text and a JSON schema. RAM measured, expected to exceed 2 GB.
- **(c) Reference, not a production candidate:** a larger vision-language model on the user's GPU reading page images directly — shows the achievable ceiling. Must fit in **≤ 21 GB VRAM total** (weights + KV cache + context + runtime overhead), measured as peak VRAM during the run.

### Stage 3 — checks (no AI, applied to every variant)

- IČO mod-11 checksum; DIČ format.
- VAT summary lines sum to the total (±0.01 per line, ±1 Kč for cash rounding).
- **ARES lookup** by IČO (free state registry): fills/validates supplier name, address, DIČ, so these need not be read reliably from scans. Responses cached on disk for repeatable runs.
- A document failing any check is marked `needs_review`.

## Metrics

Per document and aggregated per engine/extractor combination:

| Metric | How |
|---|---|
| OCR value recall | Share of labelled key values (IČO, number, dates, VAT bases/amounts, total) found in the raw OCR text after normalisation (`1 105,00` → `1105.00`, dates in any common Czech format). Measures OCR independently of the extractor. |
| Field accuracy | Extracted vs labelled, per field: amounts ±0.01, dates/IČO/DIČ/number exact, text fields fuzzy (normalised similarity ≥ 0.9). Items scored separately (count + per-line amounts). |
| Type / payment accuracy | `type` and `payment` vs label. |
| Auto-pass rate | Share of documents with all key fields correct **and** no check failure. |
| Silent errors | Documents with a wrong key field that still passed all checks — the number that must be ~0. |
| Time | Wall-clock seconds per page, single core (`taskset -c 0`, `OMP_NUM_THREADS=1`), plus cold-start (model load) time. |
| Memory | Peak RSS per engine process (`os.wait4` rusage of that child), model size on disk; for the GPU reference also peak VRAM (amdgpu sysfs `mem_info_vram_used`, RX 7900 XTX). |

Key fields: `type`, `number`, `variabilni_symbol`, `issue_date`, `duzp`, `due_date`, `supplier.ico`, `vat[]`, `total`.

## Harness layout

```
poc/
  ocr.py        # engine name → fn(image) -> text; one engine per subprocess
  extract.py    # rules extractor (a); llm extractor (b) behind the same fn(text) -> dict
  checks.py     # IČO checksum, VAT sums, ARES (cached)
  bench.py      # runs combinations over testdata/, writes poc/results/<run>.csv
  score.py      # compares results with *.expected.json, prints the metric tables
```

Each engine runs in its own subprocess so its RSS and load time are isolated. Results are CSV plus a generated markdown summary committed to `docs/poc-results.md` (no client data beyond field names and file IDs).

## Out of scope for this round

- Splitting multi-document pages / ignoring card-terminal slips (harness only records detected document count per file for later).
- Pohoda XML generation, web UI, dedup logic.
- Docker; production packaging.

## Decision rule

Results are always reported separately for native PDFs and scans. Pick the cheapest pipeline (RAM first, then time) with zero silent errors on the test set and an auto-pass rate of at least 90 % on native PDFs and 70 % on scans (starting targets — adjust with the user once real numbers exist). If no CPU pipeline is acceptable, the report states what the best one needs (RAM/time) and how far it is from the GPU reference, so hosting can be re-evaluated against the 4 000 Kč/year ceiling.
