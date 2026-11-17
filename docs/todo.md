# TODO

Open work and ideas, kept here so they survive between sessions. Public repo: no client data (names, IČO, amounts) in this file.

## Now

- **Review UI mock with Hana.** The mock (`poc/results/mocks/`, gitignored, built by `build.py` from real documents) has two views: "Detail dokladu" (form beside the document; click takes the whole value, drag a part, right-click opens a menu to join a value to a field, take one word, or pick any field) and "Dávka v tabulce" (batch table; findings for fields outside the table link to the detail). Collect her feedback on the UI philosophy, including whether the table should be self-sufficient (expandable rows with all fields) or stay a triage view next to the detail.
- **Design the linking data model** after Hana's feedback. For each confirmed field, store the source tokens (word boxes), the final value, whether the user corrected it, and the supplier IČO. This is what supplier templates (below) learn from, so it must be in place from the start.
- Act on Hana's answers (questions 1–13, recorded in CLAUDE.md Requirements):
  - Review UI: add a "not exported" mark (just a mark for now; removal is a UX-polish question) for documents she books by hand (advance tax documents, voucher returns, private expenses).
  - Receipts: when there is no explicit document number, use the number printed under the barcode (OCR text). Check our receipts to confirm the barcode number is always printed; barcode decoding is deferred until it proves needed. Re-check the receipt `number` labels against this rule and rescore.
  - Agenda rules in the extractor/mock: receipts with the client's IČO → ostatní závazky; invoices stay in faktury přijaté whatever the payment method.
  - Supplier templates are worth it: she estimates about 80 % of documents, mostly invoices, come from repeating suppliers.
- Questions for Hana (local questions file): 14 (do cash receipts without the client's IČO go to pokladna or ostatní závazky?) and 18 (what she fills in Pohoda beyond the document: předkontace, členění DPH, středisko, text, číselná řada; blocks the export design).
- Hana has a test company in IPodnik for trying XML imports, so export tests never touch real books.
- Pohoda export: first version in `poc/pohoda.py`, the test month exports and validates. Next: Hana test-imports `poc/results/pohoda/zari.xml` (generated with her test company's IČO via `--ico`) and reports what lands wrong; then question 18 (předkontace etc.) and 14 (pokladna).

## Ideas

### Supplier templates learned from reviews

Goal: converge to a high success rate with cheap, deterministic parsing, because most suppliers repeat over long periods (the accountant estimates about 80 % of documents).

- Keep the generic rule parser as the baseline and keep tuning it.
- Learn a template per supplier from confirmed documents instead of hand-writing specialized parsers: for each field, where the value sits relative to a nearby label ("Celkem" → number to its right). Relative anchors survive receipts of varying length; absolute positions don't. One or two confirmed documents should be enough; no code per supplier.
- Pipeline: identify the supplier (IČO, checksum + ARES; fallback name or layout similarity) → apply its template if one exists → accept the result only if it passes the checks (VAT sums, dates, IČO), otherwise fall back to the generic parser → review → each confirmation or correction updates the template.
- Templates belong to the supplier, not the client company, so they are shared across all clients.
- Mark a template stale when checks fail or users keep correcting it (the supplier changed its layout), then re-learn.
- Metric: share of fields the user changed, per supplier. It shows where templates are missing or broken.
- Test data (one month, 68 documents): 48 distinct suppliers, 13 of them appear more than once. 33 documents come from repeating suppliers, so a template learned from each supplier's first document would cover 20 documents (29 %) within the month. Invoices repeat more than receipts (21 of 37 invoices vs 10 of 25 receipts come from repeating suppliers). Across months the share should be much higher; measure it once there is a second month of data.
- Limit: OCR quality. A template finds the right place but cannot fix a misread digit.

## Housekeeping

- Push `main` to `origin` (ask first; the leakcheck hooks run on push).
- Delete the pre-rewrite backup bundle outside the repo (`../vycuc-pre-rewrite.bundle`); it contains client data.
- Maybe add the latest analysis summary to the Findings in `docs/poc-results.md`.
