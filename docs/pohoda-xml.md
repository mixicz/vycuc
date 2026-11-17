# Pohoda XML import: notes

Findings from Stormware's XML schemas (version 2, `https://www.stormware.cz/xml/schema/version_2/*.xsd`) and sample files (`https://www.stormware.cz/xml/samples/version_2/import/`), read on 2026-11-17. Input for the export design; nothing is built yet. Not yet verified by an import (Hana has a test company in IPodnik for that).

## File

- One file is a `dat:dataPack` with `dat:dataPackItem` children, one document each.
- `dataPack@ico` is the IČO of the accounting company the file is for; Pohoda rejects it if it doesn't match the open company. So one file per client company.
- `dataPack@id` together with `dataPackItem@id` is used by Pohoda for duplicate checks on import. Stable IDs give a second dedup safety net on top of ours.
- Stormware's samples are Windows-1250. Whether UTF-8 is accepted: verify with a test import.
- Validation: `xmllint --schema data.xsd` (system `xmllint` is available; `lxml` is not installed in `.venv`).

## Agendas

| Výcuc | Schema | Type value |
|---|---|---|
| faktury přijaté | `invoice.xsd` | `invoiceType=receivedInvoice` (credit note: `receivedCreditNotice`) |
| přijaté zálohové faktury | `invoice.xsd` | `receivedAdvanceInvoice` (there is also `receivedProformaInvoice`; which one Hana means is open) |
| ostatní závazky | `invoice.xsd` | `commitment` |
| pokladna | `voucher.xsd` | `voucherType=expense`; `cashAccount` (which cash register) is required |

## Header fields (invoice)

| Our field | Element | Note |
|---|---|---|
| document number | `originalDocument` | The supplier's number. `number` is Pohoda's own sequence (číselná řada) and is filled from the default series when omitted. |
| variabilní symbol | `symVar` | Derived from `number` when omitted, so always send it. |
| issue date | `date` | |
| DUZP | `dateTax` | Defaults to `date`, which matches Hana's rule. |
| due date | `dateDue` | Defaults from settings when omitted. |
| — | `dateAccounting`, `dateKHDPH` | Accounting date and the date for the VAT control statement; defaults to `date` / unset. |
| — | `accounting` | Předkontace. Optional; defaults from user settings. |
| — | `classificationVAT` | Členění DPH. Optional; defaults to `inland`. Not used for proforma/advance invoices. |
| — | `text` | Document text; required when creating a document. |
| supplier | `partnerIdentity/address` | company, name, street, city, zip, ico, dic. |
| payment | `paymentType` | Defaults to bank order. Hana records payments by hand. |

Voucher (pokladna) header: `voucherType`, `cashAccount`, `date` and `text` are required; `originalDocument` is the receipt number ("číslo paragonu"); `dateTax`, `accounting`, `classificationVAT`, `partnerIdentity` as for invoices.

## Amounts

Without line items, the totals go into `invoiceSummary`:

- `homeCurrency`: `priceNone` (0 %), `priceLow` + `priceLowVAT` (reduced rate), `priceHigh` + `priceHighVAT` (standard rate), `price3` + `price3VAT` (third rate). Our VAT lines map onto these by rate.
- `foreignCurrency`: `currency` (code), `rate`, `amount`, `priceSum`. When a foreign currency is set, the home-currency amounts are ignored on import.
- `roundingDocument` (e.g. `math2one` for receipts rounded to whole Kč) and `roundingVAT`; default from global settings.

## Open

- What Hana fills beyond the document: předkontace, členění DPH, středisko, text, number series (question 18).
- Pokladna: which `cashAccount`, and whether cash receipts go there at all (question 14).
- Line items: not in the first version (decided 2026-11-17); totals per VAT rate only.
- Encoding (UTF-8 vs Windows-1250), and whether the supplier is matched to the address book.
