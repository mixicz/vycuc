"""Non-AI checks applied to every extracted document; failures => needs_review."""
import copy
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ARES_URL = "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/{}"
ARES_CACHE = Path(__file__).parent / "cache" / "ares"
INVOICE_TYPES = {"receivedInvoice", "issuedInvoice", "receivedCreditNotice",
                 "receivedAdvanceInvoice", "issuedAdvanceInvoice",
                 "receivedProformaInvoice", "receivedAdvanceTaxDocument"}


def ico_valid(ico: str) -> bool:
    if not re.fullmatch(r"\d{8}", ico or ""):
        return False
    s = sum(int(ico[i]) * (8 - i) for i in range(7))
    return (11 - s % 11) % 10 == int(ico[7])


def dic_valid(dic: str) -> bool:
    if (dic or "").startswith("CZ"):
        return re.fullmatch(r"CZ\d{8,10}", dic) is not None
    return re.fullmatch(r"[A-Z]{2}[0-9A-Z]{2,13}", dic or "") is not None


def ares(ico: str, cache_dir: Path = ARES_CACHE) -> dict | None:
    f = cache_dir / f"{ico}.json"
    if f.exists():
        return json.loads(f.read_text()) or None
    try:
        with urllib.request.urlopen(ARES_URL.format(ico), timeout=10) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        data = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, ensure_ascii=False))
    return data or None


def _street(s: dict) -> str | None:
    name = s.get("nazevUlice") or s.get("nazevCastiObce")
    if not name or not s.get("cisloDomovni"):
        return None
    num = str(s["cisloDomovni"])
    if s.get("cisloOrientacni"):
        num += f"/{s['cisloOrientacni']}{s.get('cisloOrientacniPismeno', '')}"
    return f"{name} {num}"


def check(doc: dict, lookup=ares) -> tuple[dict, list[str]]:
    doc = copy.deepcopy(doc)
    sup = doc["supplier"] = doc.get("supplier") or {}
    fails = []

    need = ["issue_date", "total"]
    if doc.get("type") in INVOICE_TYPES:
        need += ["number", "duzp"]
    fails += [f"missing {k}" for k in need if doc.get(k) in (None, "")]

    ico, dic = sup.get("ico"), sup.get("dic")
    if dic and not dic_valid(dic):
        fails.append("dic format")
    if ico:
        if not ico_valid(ico):
            fails.append("ico checksum")
        elif (a := lookup(ico)) is None:
            fails.append("ico not in ARES")
        else:
            if dic and a.get("dic") and dic != a["dic"]:
                fails.append("dic != ARES")
            s = a.get("sidlo") or {}
            found = {"name": a.get("obchodniJmeno"), "dic": a.get("dic"),
                     "street": _street(s), "city": s.get("nazevObce"),
                     "zip": str(s["psc"]) if s.get("psc") else None,
                     "country": s.get("kodStatu")}
            for k, v in found.items():
                if v and not sup.get(k):
                    sup[k] = v
    elif (sup.get("country") or "CZ") == "CZ" and not (dic and not dic.startswith("CZ")):
        fails.append("missing ico")

    vat = doc.get("vat") or []
    if any(None in (v.get("rate"), v.get("base"), v.get("vat")) for v in vat):
        fails.append("vat line incomplete")
    else:
        for v in vat:
            if abs(v["base"] * v["rate"] / 100 - v["vat"]) > 0.01 + 1e-9:
                fails.append(f"vat {v['rate']}% != base*rate")
        if vat and doc.get("total") is not None:
            if abs(sum(v["base"] + v["vat"] for v in vat) - doc["total"]) > 0.01 * len(vat) + 1e-9:
                fails.append("vat sum != total")
    if doc.get("to_pay") is not None and doc.get("total") is not None:
        if abs(doc["to_pay"] - doc["total"]) > 1:  # cash rounding up to 1 Kč
            fails.append("to_pay != total")
    return doc, fails
