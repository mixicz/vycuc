"""Pohoda XML (dataPack v2) export of received documents: header and totals per VAT rate, no line items.

Format notes: docs/pohoda-xml.md.
"""
import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA_URL = "https://www.stormware.cz/xml/schema/version_2/"
SCHEMAS = Path(__file__).parent / "cache" / "pohoda-xsd"
NS = {p: f"http://www.stormware.cz/schema/version_2/{n}.xsd"
      for p, n in [("dat", "data"), ("inv", "invoice"), ("typ", "type")]}
for _p, _u in NS.items():
    ET.register_namespace(_p, _u)

# label type -> Pohoda invoiceType; None = booked by hand, not exported.
# ponytail: receipts all go to ostatní závazky; pokladna (voucher.xsd) waits on question 14.
TYPES = {"receivedInvoice": "receivedInvoice", "receivedCreditNotice": "receivedCreditNotice",
         "receivedProformaInvoice": "receivedAdvanceInvoice", "receivedAdvanceInvoice": "receivedAdvanceInvoice",
         "commitment": "commitment", "receipt": "commitment", "receivedAdvanceTaxDocument": None}
# VAT rate -> summary slot; 15/10 are the pre-2024 reduced rates
RATES = {21: "High", 15: "Low", 12: "Low", 10: "3", 0: "None"}


def invoice_type(doc: dict) -> str | None:
    return TYPES[doc["type"]]


def _el(parent, tag, text=None):
    p, t = tag.split(":")
    e = ET.SubElement(parent, f"{{{NS[p]}}}{t}")
    if text is not None:
        e.text = str(text)
    return e


def _amt(v: float) -> str:
    return f"{v:.2f}"


def _invoice(item, doc):
    inv = _el(item, "inv:invoice")
    inv.set("version", "2.0")
    h = _el(inv, "inv:invoiceHeader")
    _el(h, "inv:invoiceType", invoice_type(doc))
    vs = re.sub(r"\D", "", doc.get("variabilni_symbol") or "")[:20]
    if vs:
        _el(h, "inv:symVar", vs)
    if doc.get("number"):
        _el(h, "inv:originalDocument", doc["number"][:32])
    _el(h, "inv:date", doc["issue_date"])
    _el(h, "inv:dateTax", doc.get("duzp") or doc["issue_date"])
    if doc.get("due_date"):
        _el(h, "inv:dateDue", doc["due_date"])
    s = doc.get("supplier") or {}
    _el(h, "inv:text", (s.get("name") or f"Doklad {doc.get('number') or ''}").strip()[:240])
    a = _el(_el(h, "inv:partnerIdentity"), "typ:address")
    for k, tag in [("name", "company"), ("city", "city"), ("street", "street"), ("zip", "zip"), ("ico", "ico"), ("dic", "dic")]:
        if s.get(k):
            _el(a, f"typ:{tag}", s[k])
    if s.get("country"):
        _el(_el(a, "typ:country"), "typ:ids", s["country"])

    summ = _el(inv, "inv:invoiceSummary")
    if (doc.get("currency") or "CZK") != "CZK":
        f = _el(summ, "inv:foreignCurrency")
        _el(_el(f, "typ:currency"), "typ:ids", doc["currency"])
        _el(f, "typ:priceSum", _amt(doc["total"]))
        return
    slots = {}
    for line in doc.get("vat") or []:
        if line["rate"] not in RATES:
            raise ValueError(f"unknown VAT rate {line['rate']}")
        b, v = slots.get(RATES[line["rate"]], (0.0, 0.0))
        slots[RATES[line["rate"]]] = (b + (line["base"] or 0), v + (line["vat"] or 0))
    if not slots:  # non-VAT payer: the whole amount is outside VAT
        slots["None"] = (doc["total"], 0.0)
    home = _el(summ, "inv:homeCurrency")
    for slot in ["None", "Low", "High", "3"]:  # schema order
        if slot in slots:
            b, v = slots[slot]
            _el(home, f"typ:price{slot}", _amt(b))
            if slot != "None":
                _el(home, f"typ:price{slot}VAT", _amt(v))


def build(items: list[tuple[str, dict]], ico: str, pack_id: str) -> bytes:
    """items: (stable id, label-format document). ico: the accounting company the file is for."""
    pack = ET.Element(f"{{{NS['dat']}}}dataPack",
                      {"id": pack_id, "ico": ico, "application": "Vycuc", "version": "2.0", "note": "Import z Výcucu"})
    for item_id, doc in items:
        if invoice_type(doc) is None:
            continue
        item = _el(pack, "dat:dataPackItem")
        item.set("id", item_id[:64])
        item.set("version", "2.0")
        _invoice(item, doc)
    ET.indent(pack)
    return ET.tostring(pack, encoding="windows-1250", xml_declaration=True)


def fetch_schemas(dest: Path = SCHEMAS) -> None:
    """Download data.xsd and everything it imports, for validation with xmllint."""
    dest.mkdir(parents=True, exist_ok=True)
    todo, seen = ["data.xsd"], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        with urllib.request.urlopen(SCHEMA_URL + name, timeout=30) as r:
            body = r.read()
        (dest / name).write_bytes(body)
        todo += [n.decode() for n in re.findall(rb'schemaLocation="([\w.]+\.xsd)"', body)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export labelled documents as Pohoda XML")
    ap.add_argument("labels", nargs="*", help="*.expected.json files")
    ap.add_argument("--ico", help="IČO of the accounting company in Pohoda (e.g. Hana's test company)")
    ap.add_argument("-o", "--out", help="output file; must be a git-ignored path (it holds client data)")
    ap.add_argument("--fetch-schemas", action="store_true", help=f"download the XSDs to {SCHEMAS}")
    a = ap.parse_args()
    if a.fetch_schemas:
        fetch_schemas()
    if a.labels:
        items = [(f"{Path(p).name.split('.')[0]}-{i + 1}", d)
                 for p in a.labels for i, d in enumerate(json.loads(Path(p).read_text())["documents"])]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes(build(items, a.ico, Path(a.out).stem))
