"""Text (or page images) -> documents in the label schema.

    python -m poc.extract EXTRACTOR OUT_DIR SOURCE...
SOURCE: poc.ocr output dirs (per page, the first one with text wins), or PDFs for vlm:*.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from poc.checks import ico_valid
from poc.norm import amounts_in, dates_in, fold
from poc.ocr import MODELS, has_text

CLIENT_ICO = "76543218"  # Klient Vzor s.r.o.; production gets this per company
CACHE = Path(__file__).parent / "cache" / "extract"

# --- (a) rules -----------------------------------------------------------------

_ICO = lambda s: [x for x in re.findall(r"(?<!\d)\d{8}(?!\d)", s) if ico_valid(x)]
_DIC = lambda s: re.findall(r"\b[A-Z]{2}\d{8,12}\b", s.upper())
_NUMBER = lambda s: re.findall(r"(?=[A-Za-z0-9/\-]*\d)[A-Za-z0-9][A-Za-z0-9/\-]{3,}", s)
_DIGITS = lambda s: re.findall(r"(?<!\d)\d{1,10}(?!\d)", s)


def _near(text: str, f: str, label: str, finder, before: int = 0) -> list:
    """First value per label hit: rest of the line, then `before` chars before it, then next line."""
    out = []
    for m in re.finditer(label, f):
        rest = text[m.end():].split("\n", 2)
        line_start = text.rfind("\n", 0, m.start()) + 1
        segs = [rest[0], text[max(line_start, m.start() - before):m.start()] if before else "",
                rest[1] if len(rest) > 1 else ""]
        for seg in segs:
            if vals := finder(seg):
                out.append(vals[0])
                break
    return out


def _first(xs):
    return xs[0] if xs else None


def rules(text: str, client_ico: str | None = None) -> dict:
    f = fold(text)
    client_dic = f"CZ{client_ico}" if client_ico else None

    dics = [d for d in _near(text, f, r"\bdic\b|\bic dph\b|\bvat\b", _DIC, before=20) if d != client_dic]
    icos = _near(text, f, r"\bico?\b", _ICO, before=15) + [d[2:] for d in dics if d.startswith("CZ")]
    ico = _first([i for i in icos if ico_valid(i) and i != client_ico])
    dic = _first(dics)

    issue = _first(_near(text, f, r"datum vystaveni|vystaveno|datum vyst", dates_in)) \
        or _first(dates_in(text))
    duzp = _first(_near(text, f, r"duzp|zdanitelneho plneni|datum plneni", dates_in)) or issue
    due = _first(_near(text, f, r"splatnost|datum splatnosti|splatno", dates_in))

    totals = _near(text, f, r"celkem|k uhrade|k zaplaceni|celkova castka|total|suma", amounts_in)
    to_pay = _first(_near(text, f, r"k uhrade|k zaplaceni", amounts_in))
    total = max(totals) if totals else None

    vat, seen = [], set()
    for line in text.splitlines():
        amounts = amounts_in(line)
        for rate in map(int, re.findall(r"(?<![\d,.])(10|12|15|21)\s?%", fold(line))):
            pair = next(((b, v) for i, b in enumerate(amounts) for v in amounts[i + 1:]
                         if b > 0 and abs(b * rate / 100 - v) <= 0.02), None)
            if pair and rate not in seen:  # ponytail: rate in header row is missed; LLM covers it
                seen.add(rate)
                vat.append({"rate": rate, "base": pair[0], "vat": pair[1]})

    has = lambda p: re.search(p, f) is not None
    if has(r"dobropis|opravny danovy doklad"):
        kind = "receivedCreditNotice"
    elif has(r"danovy doklad k prijate platbe"):
        kind = "receivedAdvanceTaxDocument"
    elif has(r"proforma|neni danovy doklad"):
        kind = "receivedProformaInvoice"
    elif has(r"zalohov"):
        kind = "receivedAdvanceInvoice"
    elif has(r"faktur") or (client_ico and client_ico in text):
        kind = "receivedInvoice"
    else:
        kind = "receipt"
    payment = next((p for p, pat in [("cod", r"dobirk"), ("cash", r"hotov"),
                                     ("card", r"kartou|platebni kart|\bvisa\b|mastercard"),
                                     ("transfer", r"prevodem|bankovni prevod")] if has(pat)), None)

    return {
        "type": kind, "payment": payment, "number": _first(_near(
            text, f, r"\bcislo\b|\bc\. ?(?:dokladu|faktury)|(?:faktura|doklad|uctenka) c\.|danovy doklad\b",
            _NUMBER)),
        "variabilni_symbol": _first(_near(text, f, r"variabilni symbol|var\. ?symbol|\bv\.? ?s\.?:", _DIGITS)),
        "issue_date": issue, "duzp": duzp, "due_date": due,
        "currency": "EUR" if has(r"\beur\b|€") else "CZK",
        "supplier": {"name": None, "ico": ico, "dic": dic, "street": None, "city": None, "zip": None,
                     "country": dic[:2] if dic else ("CZ" if ico else None)},
        "vat": vat, "total": total, "to_pay": to_pay if to_pay is not None else total, "items": [],
    }


LOADERS = {"rules": lambda: lambda text: [rules(text, CLIENT_ICO)]}

# --- CLI -----------------------------------------------------------------------


def page_texts(name: str, sources: list[Path]) -> list[str]:
    recs = [json.loads(f.read_text()) for s in sources if (f := s / f"{name}.json").exists()]
    pages = [r["pages"] for r in recs if "pages" in r]
    if not pages:
        raise ValueError("; ".join(r.get("error", "no source") for r in recs) or "no source")
    return [next((p[i] for p in pages if i < len(p) and has_text(p[i])), "")
            for i in range(max(map(len, pages)))]


def main(extractor: str, out_dir: str, sources: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = CACHE / extractor.replace(":", "_")
    cache.mkdir(parents=True, exist_ok=True)
    vlm = extractor.startswith("vlm:")
    if vlm:
        inputs = [(Path(p).name, Path(p)) for p in sources]
    else:
        src = [Path(s) for s in sources]
        inputs = [(p.name[:-5], None) for p in sorted(src[0].glob("*.json")) if p.name != "_meta.json"]
    run, load, meta = None, 0.0, {}
    for name, pdf in inputs:
        try:
            data = pdf.read_bytes() if vlm else "\n\n".join(page_texts(name, src)).encode()
            hit = cache / f"{hashlib.sha256(data).hexdigest()[:16]}.json"
            # ponytail: key ignores prompt/schema/code; rm -r poc/cache/extract after changing them
            if extractor != "rules" and hit.exists():
                rec = json.loads(hit.read_text())
            else:
                if run is None:
                    t = time.perf_counter()
                    run, meta = load_extractor(extractor)
                    load = time.perf_counter() - t
                t = time.perf_counter()
                docs = run(pdf if vlm else data.decode())
                rec = {"documents": docs, "seconds": time.perf_counter() - t}
                hit.write_text(json.dumps(rec, ensure_ascii=False))
        except Exception as e:  # one bad file must not stop the benchmark
            rec = {"error": repr(e)}
        (out / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(extractor, name, "error" if "error" in rec else f"{rec['seconds']:.1f}s", file=sys.stderr)
    if callable(meta.get("finish")):
        meta = {**{k: v for k, v in meta.items() if k != "finish"}, **meta["finish"]()}
    (out / "_meta.json").write_text(json.dumps({"extractor": extractor, "load_seconds": load,
                                                "model_bytes": 0, "peak_vram_bytes": None, **meta}))


def load_extractor(name: str):
    """-> (fn(input) -> list[dict], meta dict). meta may hold 'finish': () -> dict."""
    return LOADERS[name](), {}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
