import urllib.error

import pytest

from poc import checks
from poc.checks import check, dic_valid, ico_valid

ARES_METRO = {"obchodniJmeno": "Kavárna Vzor s.r.o.", "dic": "CZ12345679",
              "sidlo": {"kodStatu": "CZ", "nazevObce": "Praha", "psc": 10100,
                        "nazevUlice": "Vzorová", "cisloDomovni": 1515,
                        "cisloOrientacni": 1, "cisloOrientacniPismeno": "b"}}


def lookup(ico):
    return ARES_METRO if ico == "12345679" else None


def invoice(**kw):
    doc = {"type": "receivedInvoice", "number": "20269001", "issue_date": "2026-07-31",
           "duzp": "2026-07-31", "supplier": {"ico": "12345679"},
           "vat": [{"rate": 12, "base": 4120.75, "vat": 494.49}],
           "total": 4615.24, "to_pay": 4615.24}
    doc.update(kw)
    return doc


def test_ico_checksum():
    assert ico_valid("76543218") and ico_valid("12345679") and ico_valid("24680249")
    assert not ico_valid("21809789") and not ico_valid("1234567") and not ico_valid("abcdefgh")


def test_dic_format():
    assert dic_valid("CZ76543218") and dic_valid("CZ7501011234") and dic_valid("IE6388047V")
    assert not dic_valid("76543218") and not dic_valid("CZ123")


def test_valid_invoice_passes_and_is_enriched_from_ares():
    doc, fails = check(invoice(), lookup)
    assert fails == []
    assert doc["supplier"]["name"] == "Kavárna Vzor s.r.o."
    assert doc["supplier"]["street"] == "Vzorová 1515/1b"
    assert doc["supplier"]["zip"] == "10100"


def test_failures():
    _, fails = check(invoice(total=11000.0, duzp=None,
                             supplier={"ico": "12345679", "dic": "CZ99999999"}), lookup)
    assert "missing duzp" in fails and "vat sum != total" in fails and "dic != ARES" in fails
    _, fails = check(invoice(supplier={"ico": "21809789"}), lookup)
    assert fails == ["ico checksum"]
    _, fails = check(invoice(supplier={"ico": "24680249"}), lookup)
    assert fails == ["ico not in ARES"]
    _, fails = check(invoice(vat=[{"rate": 12, "base": None, "vat": 1.0}]), lookup)
    assert "vat line incomplete" in fails


def test_receipt_needs_no_number_but_needs_ico():
    receipt = {"type": "receipt", "issue_date": "2026-09-15", "total": 7.8,
               "supplier": {}, "vat": []}
    assert check(receipt, lookup)[1] == ["missing ico"]


def test_foreign_supplier_without_ico_skips_ares():
    def boom(ico):
        raise AssertionError("ARES must not be called")
    doc = {"type": "receivedInvoice", "number": "5", "issue_date": "2026-09-01",
           "duzp": "2026-09-01", "total": 10.0, "vat": [],
           "supplier": {"dic": "IE6388047V", "country": "IE"}}
    assert check(doc, boom)[1] == []


def test_ares_network_error_raises_and_is_not_cached(tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(checks.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        checks.ares("12345679", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_ares_live(tmp_path):
    assert checks.ares("76543218", tmp_path)["dic"] == "CZ76543218"
    assert checks.ares("21809789", tmp_path) is None
    assert (tmp_path / "21809789.json").exists()
