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

# --- (b) small LLM on CPU, (c) VLM: shared schema and prompt -------------------

TYPES = ["receivedInvoice", "issuedInvoice", "receivedCreditNotice", "receivedAdvanceInvoice",
         "issuedAdvanceInvoice", "receivedProformaInvoice", "receivedAdvanceTaxDocument",
         "commitment", "receipt"]
_S, _N = {"type": ["string", "null"]}, {"type": ["number", "null"]}
_D = {"type": ["string", "null"], "description": "YYYY-MM-DD"}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


DOC_SCHEMA = _obj({
    "type": {"enum": TYPES}, "payment": {"enum": ["transfer", "cash", "card", "cod", None]},
    "number": _S, "variabilni_symbol": _S, "issue_date": _D, "duzp": _D, "due_date": _D,
    "currency": {"type": "string"},
    "supplier": _obj({k: _S for k in ("name", "ico", "dic", "street", "city", "zip", "country")}),
    "vat": {"type": "array", "items": _obj({k: {"type": "number"} for k in ("rate", "base", "vat")})},
    "total": _N, "to_pay": _N,
    "items": {"type": "array", "items": _obj({"text": {"type": "string"}, "quantity": _N, "unit": _S,
                                              "unit_price": _N, "vat_rate": _N, "base": _N, "total": _N})},
})
SCHEMA = _obj({"documents": {"type": "array", "items": DOC_SCHEMA, "minItems": 1}})

PROMPT = """You extract accounting data from Czech invoices and receipts. The input may be OCR text with errors.
Return JSON matching the schema, with one entry in "documents" per distinct document. Ignore card-terminal slips.
- The customer is our client. "supplier" is the seller/issuer (dodavatel, prodávající).
- type: receivedInvoice = faktura; receivedCreditNotice = dobropis / opravný daňový doklad;
  receivedAdvanceInvoice = zálohová faktura; receivedProformaInvoice = proforma / "není daňový doklad";
  receivedAdvanceTaxDocument = daňový doklad k přijaté platbě; commitment = other payable (e.g. insurance
  premium notice); receipt = účtenka / pokladní doklad from a till.
- payment: transfer = převodem, cash = hotově, card = kartou, cod = dobírka, null if not stated.
- Dates as YYYY-MM-DD. Amounts as numbers with a dot (1 105,00 -> 1105.00).
- duzp = DUZP / datum zdanitelného plnění (date of taxable supply), never splatnost (due_date).
  If no DUZP is printed (e.g. receipts), duzp = issue_date.
- ico = IČ/IČO (8 digits). dic = DIČ (e.g. CZ12345678).
- vat: one line per VAT rate from the VAT summary (rekapitulace DPH): rate in %, base without VAT, VAT amount.
  Empty if the supplier is not a VAT payer.
- total = exact amount including VAT before rounding (base + VAT, e.g. 1234.56).
  to_pay = amount to pay (k úhradě) after rounding or deducted advances (e.g. 1235.00).
- Use null for anything not on the document. Never invent values."""

LLMS = {"qwen3.5-2b": MODELS / "Qwen3.5-2B-Q4_K_M.gguf",
        "qwen3.5-4b": MODELS / "Qwen3.5-4B-Q4_K_M.gguf"}
MAX_CHARS = 12000  # ponytail: long multi-page bills are truncated; chunk if it costs accuracy


def parse_documents(s: str) -> list[dict]:
    try:
        docs = json.loads(s)["documents"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"bad model output: {s[:200]!r}") from e
    if not docs:
        raise ValueError("model returned no documents")
    return docs


def _load_llm(path: Path):
    import os

    from llama_cpp import Llama
    n = len(os.sched_getaffinity(0))  # n_threads_batch defaults to cpu_count(), not the pinned cores
    llm = Llama(model_path=str(path), n_ctx=8192, n_threads=n, n_threads_batch=n, verbose=False)

    def run(text: str) -> list[dict]:
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text[:MAX_CHARS]}],
            response_format={"type": "json_object", "schema": SCHEMA},
            temperature=0, max_tokens=3000)
        return parse_documents(r["choices"][0]["message"]["content"])
    run.llm = llm
    return run


OLLAMA = "http://localhost:11434"
VLM_DPI = 150  # ponytail: 300 DPI pages blow the image-token budget; raise if small print is misread
VLM_CTX = 16384
VRAM_LIMIT = 21 * 2**30  # user's cap for the GPU reference, all overhead included


def _ollama(path: str, body: dict, timeout: float = 900) -> dict:
    import urllib.request
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def vram_sampler(model: str):
    """Sample ollama's VRAM allocation for `model` (weights + KV cache + context) every 0.2 s.

    Card-level sysfs counters are useless here: the desktop shares the GPU and ollama's
    runner runs as another user, so its per-process fdinfo is unreadable.
    """
    import threading
    import urllib.request

    def read() -> int:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as r:
            return next((m["size_vram"] for m in json.load(r)["models"] if m["name"] == model), 0)
    peak, done = [read()], threading.Event()

    def loop():
        while not done.wait(0.2):
            peak[0] = max(peak[0], read())
    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def stop():
        done.set()
        th.join()
        return {"peak_vram_bytes": max(peak[0], read())}
    return stop


def _load_vlm(model: str):
    import base64
    import io

    from poc.ocr import render
    import urllib.request
    _ollama("/api/generate", {"model": model, "keep_alive": 0})  # unload -> cold start
    stop = vram_sampler(model)

    def png(img) -> str:
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def run(pdf: Path) -> list[dict]:
        r = _ollama("/api/chat", {
            "model": model, "stream": False, "think": False, "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": VLM_CTX},
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": "Extract the documents from these pages.",
                          "images": [png(i) for i in render(pdf, VLM_DPI)]}]})
        return parse_documents(r["message"]["content"])

    import urllib.request
    with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10) as t:  # GET
        size = next((m["size"] for m in json.load(t)["models"] if m["name"] == model), 0)
    return run, {"finish": stop, "model_bytes": size}


# --- CLI -----------------------------------------------------------------------


def page_texts(name: str, sources: list[Path]) -> list[str]:
    recs = [json.loads(f.read_text()) if (f := s / f"{name}.json").exists() else {"error": f"no {s.name}"}
            for s in sources]
    pages = [r["pages"] for r in recs if "pages" in r]
    errors = "; ".join(r["error"] for r in recs if "error" in r)
    if not pages:
        raise ValueError(errors or "no source")
    out = [next((p[i] for p in pages if i < len(p) and has_text(p[i])), "")
           for i in range(max(map(len, pages)))]
    if errors and not all(map(has_text, out)):  # a scan page whose OCR failed or never ran
        raise ValueError(errors)
    return out


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
    if name.startswith("vlm:"):
        return _load_vlm(name[4:])
    if name in LLMS:
        return _load_llm(LLMS[name]), {"model_bytes": LLMS[name].stat().st_size}
    return LOADERS[name](), {}

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
