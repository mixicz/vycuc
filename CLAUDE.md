# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Extraction PoC (merged to `main`; development happens on `main` until the app exists): a benchmark harness in `poc/`, no app yet. Results and findings are in `docs/poc-results.md`. Spec: `docs/superpowers/specs/2026-10-28-extraction-poc-design.md`.

## Commands

Three venvs keep conflicting stacks apart: `.venv` (harness, Tesseract, RapidOCR, llama.cpp), `.venv-paddle` (PaddleOCR), `.venv-torch` (docTR, EasyOCR). Requirements are in `poc/requirements*.txt`.

```
.venv/bin/python -m pytest -m 'not slow'          # fast tests
.venv/bin/python -m pytest tests/test_score.py::test_name   # one test
.venv/bin/python -m pytest -m slow                # needs models in poc/models, ollama, testdata
.venv/bin/python -m poc.bench RUN ocr|extract|vlm [--cpus 0]   # results in poc/results/RUN
.venv/bin/python -m poc.score RUN [--report docs/poc-results.md] [--errors poc/results/RUN-errors.md]
.venv/bin/python -m poc.pohoda --fetch-schemas            # once: Stormware XSDs for the schema test
.venv/bin/python -m poc.pohoda --ico ICO -o poc/results/pohoda/X.xml testdata/zari/*.expected.json   # Pohoda XML from labels
```

- Pipeline: `poc/ocr.py` (text per page; the text layer and RapidOCR also store `words`: `[x0, y0, x1, y1, text, confidence, line]` with coordinates as page fractions) → `poc/extract.py` (rules / llama.cpp / ollama VLM) → `poc/checks.py` (IČO, DIČ, VAT sums, ARES) → `poc/score.py` (against `testdata/zari/*.expected.json`). `poc/pohoda.py` writes Pohoda XML (dataPack v2, totals per VAT rate, no line items); format notes in `docs/pohoda-xml.md`.
- A label document may list `not_printed` paths (`vat.0.base`, `issue_date.year`): values the label computed or inferred. OCR recall skips them; extraction is still scored on them.
- Extraction results are cached by input hash in `poc/cache/extract/`; the key ignores prompt and code, so delete the cache after changing either.

## Client data (public repo)

The GitHub repo is public. Nothing from the client documents may enter git: no supplier or client names, IČO/DIČ, addresses, document numbers, VS, or amounts copied from `testdata/`.

- `testdata/`, `poc/cache/`, `poc/results/`, `poc/models/` and `docs/otazky-pro-hanu.md` are gitignored. Committed reports (`docs/poc-results.md`) contain only field names, file IDs and aggregate numbers.
- Test fixtures and prompt examples use invented data (checksum-valid fake IČO such as `12345679`). Tests that need a real document read the expected values from its `.expected.json` label.
- The test client's IČO lives in `testdata/client_ico`; extra forbidden terms go in `testdata/forbidden.txt`.
- `.githooks/` blocks commits and pushes containing any of these values; enable it once per clone with `git config core.hooksPath .githooks`. Never bypass it with `--no-verify`.

## Language conventions

- Source code, identifiers, comments, and commits are in English. Keep Czech accounting terms as they are (e.g. `duzp`, `ico`, `dic`, `variabilniSymbol`, `pokladna`). Don't translate them.
- The UI is Czech only. There is no i18n/l10n layer, so hardcode Czech strings.
- Conversation with the user mixes English and Czech.

## What this is

**Výcuc** (vyčítání účtenek) is a tool for accountants who handle several client companies. It extracts data from received (and sometimes issued) invoices and receipts, then exports it as XML for import into **Pohoda** accounting software (Stormware, run in IPodnik cloud). Similar product: https://vytezovna.cz/. Source spec (Czech): kept in the user's private notes, outside the repo.

## Requirements

- **Users:** 2 accountants, both see all client companies. Data is separated per company. Uploads are batches, including a whole folder at once.
- **Inputs:** scanned PDF/JPEG, phone photos, and native (generated) PDFs. Word is optional. Email ingestion is out of scope for now.
- **Document classification → Pohoda agenda:**
  - Invoice received → faktury přijaté; invoice issued → faktury vydané
  - Advance invoices (zálohové faktury) get their own section
  - Receipt paid in cash → pokladna
  - Receipt paid by card, or payment method unknown → ostatní závazky. Fallback if detection is unreliable: send all receipts to ostatní závazky.
  - Confirmed by the accountant (2026-11): invoices paid in cash or prepaid by card still go to faktury přijaté, and receipts made out to the client company (with its IČO) go to ostatní závazky; she records payments by hand. Proforma invoices → přijaté zálohové faktury. Payment reminders (e.g. insurance) → ostatní závazky. Foreign-currency invoices → faktury přijaté in that currency (reverse charge is booked by hand).
  - Handled by hand, not exported: tax documents for received advances (ZDD, they must be linked to the advance in Pohoda), returns paid with vouchers, private expenses. The review UI marks such a document "not exported" (it stays in the list; whether to remove it entirely is left for UX polish).
  - Card terminal slips attached to a receipt are ignored, never a separate document.
- **Dates:** DUZP missing on the document → use the issue date.
- **Receipt number:** when the receipt has no explicit document number, use the number printed under its barcode (the accountant types it from that text to look documents up). Barcode decoding is deferred until OCR of the printed text proves insufficient.
- **Volume:** about 300 documents a month per accountant, roughly 600 in total. Expected to grow gradually (that is why the app is being built), so size with reserve; even several times this volume should be fine on cheap hardware.
- **Fields to extract:** datum vystavení, DUZP (datum zdanitelného plnění), splatnost; supplier name, IČO, DIČ, address; invoice number, variabilní symbol; line items (split when there are several); base without VAT, VAT rate, VAT amount, total.
- **Multi-currency**, CZK primary.
- **Dedup:** detect an invoice already imported, by invoice number (per company). This is the main reason to persist anything.
- **Split/merge pages:** one scan containing several receipts → separate documents. A multi-page invoice → one document.
- **Export:** Pohoda XML (dataPack format, Stormware XSD schemas).

## Hard constraints

- Every paid service (hosting, cloud, AI/LLM/OCR APIs) must be billed by a **Czech entity**. Foreign billing would force the client to register with the tax office (FÚ) as a non-VAT payer. Check this before picking any external service.
- Still open: where OCR and extraction run (small ARM box on-prem, in-browser, or rented GPU cloud). Don't assume an answer. Ask.
