# Ground-truth label schema

One `<file>.expected.json` next to each test document in `testdata/` (gitignored). A file holds one or more documents (several receipts on one scan → several entries). Values are what is **printed** on the document; `null` = not present/not legible.

```jsonc
{
  "file": "ZA001.pdf",
  "documents": [
    {
      "pages": [1],                 // 1-based pages of this document within the file
      "region": null,               // several docs on one page: "left", "top-right", ...
      "type": "receivedInvoice",    // receivedInvoice | issuedInvoice | receivedCreditNotice |
                                    // receivedAdvanceInvoice | issuedAdvanceInvoice |
                                    // receivedProformaInvoice | receivedAdvanceTaxDocument (daňový doklad k přijaté platbě) |
                                    // commitment (non-invoice payable, e.g. insurance) | receipt
      "payment": "transfer",        // transfer | cash | card | cod | null (not stated)
      "number": "20269001",
      "variabilni_symbol": "20269001",
      "corrects": null,             // credit notes: number of the corrected document
      "issue_date": "2026-07-31",
      "duzp": "2026-07-31",
      "due_date": "2026-08-14",
      "currency": "CZK",
      "supplier": {"name": "", "ico": "", "dic": "", "street": "", "city": "", "zip": "", "country": "CZ"},
      "customer": {"name": "", "ico": "", "dic": ""},
      "vat": [{"rate": 12, "base": 4120.75, "vat": 494.49}],   // [] for non-VAT-payer supplier
      "total": 4615.24,           // amount incl. VAT (before rounding/advance deduction)
      "to_pay": 4615.24,          // "k úhradě" — differs from total after rounding or paid advances
      "items": [
        {"text": "ledové koule", "quantity": 525, "unit": "ks", "unit_price": 10.10,
         "vat_rate": 12, "base": 2000.00, "total": null}
      ],
      "notes": null                 // labeller remarks: uncertainty, duplicates, oddities
    }
  ]
}
```

Pohoda agenda is derived, not labelled; the mapping is `TYPES` in `poc/pohoda.py`. For now every `receipt` → `commitment` (ostatní závazky); whether cash receipts go to pokladna (`voucher`) waits on question 14 for the accountant.
