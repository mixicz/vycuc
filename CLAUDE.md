# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Greenfield: no code, stack, or build/test commands yet. Update this file with commands and architecture once they exist.

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
- **Fields to extract:** datum vystavení, DUZP (datum zdanitelného plnění), splatnost; supplier name, IČO, DIČ, address; invoice number, variabilní symbol; line items (split when there are several); base without VAT, VAT rate, VAT amount, total.
- **Multi-currency**, CZK primary.
- **Dedup:** detect an invoice already imported, by invoice number (per company). This is the main reason to persist anything.
- **Split/merge pages:** one scan containing several receipts → separate documents. A multi-page invoice → one document.
- **Export:** Pohoda XML (dataPack format, Stormware XSD schemas).

## Hard constraints

- Every paid service (hosting, cloud, AI/LLM/OCR APIs) must be billed by a **Czech entity**. Foreign billing would force the client to register with the tax office (FÚ) as a non-VAT payer. Check this before picking any external service.
- Still open: where OCR and extraction run (small ARM box on-prem, in-browser, or rented GPU cloud). Don't assume an answer. Ask.
