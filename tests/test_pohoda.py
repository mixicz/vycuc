import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from poc import pohoda

NS = {"dat": "http://www.stormware.cz/schema/version_2/data.xsd",
      "inv": "http://www.stormware.cz/schema/version_2/invoice.xsd",
      "typ": "http://www.stormware.cz/schema/version_2/type.xsd"}


def doc(**kw):
    d = {"type": "receivedInvoice", "payment": "transfer", "number": "FV-2026/0042",
         "variabilni_symbol": "20260042", "issue_date": "2026-09-03", "duzp": None,
         "due_date": "2026-09-17", "currency": "CZK",
         "supplier": {"name": "Pekárna Příklad s.r.o.", "ico": "12345679", "dic": "CZ12345679",
                      "street": "Dlouhá 1", "city": "Brno", "zip": "602 00", "country": "CZ"},
         "vat": [{"rate": 21, "base": 1000.0, "vat": 210.0}, {"rate": 12, "base": 100.0, "vat": 12.0}],
         "total": 1322.0}
    d.update(kw)
    return d


def parse(docs, ico="87654321"):
    return ET.fromstring(pohoda.build([(f"T{i}", d) for i, d in enumerate(docs)], ico, "test"))


def one(root, path):
    return root.find(path, NS).text


def test_agenda_types():
    assert pohoda.invoice_type(doc()) == "receivedInvoice"
    assert pohoda.invoice_type(doc(type="receivedProformaInvoice")) == "receivedAdvanceInvoice"
    assert pohoda.invoice_type(doc(type="receipt", payment="cash")) == "commitment"
    assert pohoda.invoice_type(doc(type="receipt", payment="card")) == "commitment"
    assert pohoda.invoice_type(doc(type="receivedAdvanceTaxDocument")) is None


def test_header_fields():
    root = parse([doc()])
    assert root.get("ico") == "87654321"
    h = "dat:dataPackItem/inv:invoice/inv:invoiceHeader/"
    assert one(root, h + "inv:invoiceType") == "receivedInvoice"
    assert one(root, h + "inv:originalDocument") == "FV-2026/0042"
    assert one(root, h + "inv:symVar") == "20260042"
    assert one(root, h + "inv:dateTax") == "2026-09-03"  # missing DUZP -> issue date
    assert one(root, h + "inv:dateDue") == "2026-09-17"
    assert one(root, h + "inv:text") == "Pekárna Příklad s.r.o."
    assert one(root, h + "inv:partnerIdentity/typ:address/typ:ico") == "12345679"


def test_vat_lines_go_to_rate_slots():
    s = "dat:dataPackItem/inv:invoice/inv:invoiceSummary/inv:homeCurrency/typ:"
    root = parse([doc()])
    assert (one(root, s + "priceHigh"), one(root, s + "priceHighVAT")) == ("1000.00", "210.00")
    assert (one(root, s + "priceLow"), one(root, s + "priceLowVAT")) == ("100.00", "12.00")


def test_non_vat_payer_total_is_zero_rate():
    root = parse([doc(vat=[], total=500.0)])
    assert one(root, "dat:dataPackItem/inv:invoice/inv:invoiceSummary/inv:homeCurrency/typ:priceNone") == "500.00"


def test_foreign_currency():
    root = parse([doc(currency="EUR", vat=[], total=12.5)])
    s = root.find("dat:dataPackItem/inv:invoice/inv:invoiceSummary", NS)
    assert s.find("inv:homeCurrency", NS) is None
    assert one(s, "inv:foreignCurrency/typ:currency/typ:ids") == "EUR"
    assert one(s, "inv:foreignCurrency/typ:priceSum") == "12.50"


def test_skips_manual_documents():
    root = parse([doc(), doc(type="receivedAdvanceTaxDocument")])
    assert len(root.findall("dat:dataPackItem", NS)) == 1


def test_unknown_rate_fails():
    with pytest.raises(ValueError):
        parse([doc(vat=[{"rate": 7, "base": 1.0, "vat": 0.07}])])


def test_encoding_is_windows_1250():
    out = pohoda.build([("T0", doc())], "87654321", "test")
    assert out.startswith(b"<?xml version='1.0' encoding='windows-1250'?>")
    assert "Pekárna".encode("cp1250") in out


@pytest.mark.skipif(not (pohoda.SCHEMAS / "data.xsd").exists() or not shutil.which("xmllint"),
                    reason="run `python -m poc.pohoda --fetch-schemas` first")
def test_valid_against_stormware_schema(tmp_path):
    f = tmp_path / "pack.xml"
    f.write_bytes(pohoda.build([(f"T{i}", d) for i, d in enumerate(
        [doc(), doc(type="receipt", vat=[], number=None, variabilni_symbol=None, due_date=None),
         doc(currency="EUR", type="receivedProformaInvoice")])], "87654321", "test"))
    r = subprocess.run(["xmllint", "--noout", "--schema", str(pohoda.SCHEMAS / "data.xsd"), str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
