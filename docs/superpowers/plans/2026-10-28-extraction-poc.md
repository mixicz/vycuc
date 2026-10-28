# Extraction PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark harness that runs 5 OCR engines and 3 kinds of extractors (rules, a small CPU LLM, a GPU vision-model reference) over `testdata/zari/`, then scores them against the hand labels and writes `docs/poc-results.md`. The point is to choose the production pipeline.

**Architecture:** There are five modules in `poc/`, and every stage writes JSON files to disk:
- `ocr.py` turns each PDF into page texts. Each engine runs in its own process, and in its own venv when its dependencies are heavy.
- `extract.py` turns page texts, or the PDF itself for the vision model, into document dicts in the label schema.
- `checks.py` validates and enriches those dicts without AI, including an ARES lookup.
- `bench.py` launches every stage as a child process pinned to the chosen CPUs and records wall time and peak RSS via `os.wait4`.
- `score.py` compares the results with `*.expected.json` and renders the markdown report.

**Tech Stack:** Python 3.13, pypdfium2, tesserocr, RapidOCR + ONNX Runtime, PaddleOCR + PaddlePaddle, docTR + EasyOCR on PyTorch CPU, llama-cpp-python with Qwen3.5 GGUF, ollama (already on the user's machine, AMD RX 7900 XTX), pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-extraction-poc-design.md`

## Global Constraints

- Python: always call `.venv/bin/python`, `.venv/bin/pip`, `.venv-paddle/bin/...` and `.venv-torch/bin/...` directly. Never `source activate`, and never the system Python for project work.
- Before installing anything new, verify the package with `pip index versions <pkg>` and check that its description matches. All pins below were verified on 2026-10-28.
- Code, identifiers and comments are in English. Czech accounting terms stay as they are (`duzp`, `ico`, `dic`, `variabilni_symbol`).
- Test data is client data. `testdata/`, `poc/cache/`, `poc/results/` and `poc/models/` are never committed. `docs/poc-results.md` may contain only field names, file IDs and numbers.
- Production must be CPU-only. The GPU model is a quality reference only, and it must fit in **≤ 21 GB VRAM in total** (weights + KV cache + context + overhead), measured as peak VRAM during the run.
- Timing is measured on one core: CPU affinity `{0}`, with `OMP_NUM_THREADS=1`.
- RAM is measured, not capped.
- Every result is reported separately for native PDFs and scans.
- Decision rule (starting targets): no silent errors; auto-pass at least 90 % on native PDFs and at least 70 % on scans; among the pipelines that pass, choose the cheapest, comparing RAM first and then time.

### Pinned versions (latest stable on 2026-10-28, from PyPI / Hugging Face / GitHub)

| Component | Version | Note |
|---|---|---|
| pypdfium2 | 5.14.0 | |
| pillow | 13.0.0 | |
| tesserocr | 2.12.0 | the manylinux wheel bundles libtesseract, so no apt or sudo is needed |
| tessdata `ces.traineddata` | 4.1.0 (tesseract-ocr/tessdata) | the standard model, the same one Debian ships |
| rapidocr | 3.10.0 | default PP-OCRv6 "multi" models, which list `cs` |
| onnxruntime | 1.31.1 | |
| paddlepaddle | 3.3.1 | CPU build, has a cp313 wheel |
| paddleocr | 3.7.0 | |
| torch / torchvision | 2.14.1+cpu / 0.29.1+cpu | from the PyTorch CPU index, so no CUDA download |
| python-doctr | 1.1.0 | |
| easyocr | 1.7.2 | latest, even though it is old (2024) |
| llama-cpp-python | 0.3.36 | sdist only, built locally (gcc and cmake are present); vendors llama.cpp `0c1e570` |
| Qwen3.5-2B / 4B Q4_K_M GGUF | `unsloth/Qwen3.5-{2B,4B}-GGUF` @ `f6d5376b…` / `e87f1764…` | the newest small Qwen models (Feb 2026); Qwen 3.6 and 3.8 come only in sizes of 27B and up |
| GPU reference | ollama 0.32.14, `qwen3.8:27b-q4_K_M` (ID `25b843619e94`, 17 GB) | already pulled; the newest vision-capable Qwen that can fit in 21 GB |
| pytest | 9.2.0 | |

Separate venvs (`.venv`, `.venv-paddle`, `.venv-torch`) keep the dependency stacks of PaddlePaddle and PyTorch from conflicting. They also make the on-disk size of each stack visible.

## Review Focus

1. **A file that crashes an engine or an extractor**, or an LLM that returns invalid JSON. The run should continue, and the file should be recorded as an error: `needs_review`, never a pass and never a silent error. Tests are in Task 3 (OCR CLI), Task 7 (extract CLI) and Task 11 (`evaluate`).
2. **A file with mixed pages** (ZA017: one page has a text layer, one is a scan). Text is chosen per page: the text layer where it exists, OCR otherwise. Test: Task 7, `test_page_texts_prefers_first_nonempty_source`.
3. **A labelled `null`.** An extracted `null` counts as correct; an extracted value where the label is `null` counts as wrong. Test: Task 11, `test_same_null_semantics`.
4. **A foreign supplier without an IČO** (Google, EUR). Checks must not query ARES or fail on a missing IČO. **ARES unreachable:** this must raise an error and must not be cached as "not found". Tests: Task 2.
5. **The upper-case `.PDF` extension** (ZA006.PDF, ZA009.PDF) must be picked up. Test: Task 10, `test_pdfs_includes_uppercase_extension`.

---

### Task 1: Scaffold, venv and normalisation helpers

**Files:**
- Create: `pyproject.toml`, `poc/__init__.py` (empty), `poc/requirements.txt`, `poc/norm.py`, `tests/test_norm.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `poc.norm.fold(s: str) -> str`: lower-case, diacritics removed, **same length as `s`**, so spans map back onto the original text
  - `poc.norm.amounts_in(s: str) -> list[float]`
  - `poc.norm.dates_in(s: str) -> list[str]`, returning ISO `YYYY-MM-DD`

- [ ] **Step 1: Write the scaffold files**

`.gitignore` (append):
```
.venv*/
__pycache__/
poc/cache/
poc/results/
poc/models/
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["slow: needs downloaded models or a running ollama"]
```

`poc/requirements.txt`:
```
pypdfium2==5.14.0
pillow==13.0.0
tesserocr==2.12.0
rapidocr==3.10.0
onnxruntime==1.31.1
llama-cpp-python==0.3.36
pytest==9.2.0
```

- [ ] **Step 2: Create the venv and install**

```bash
python3 -m venv .venv
.venv/bin/pip install -r poc/requirements.txt
```
Expected: the install succeeds. llama-cpp-python compiles for a few minutes. Then `.venv/bin/python -c "import pypdfium2, tesserocr, rapidocr, llama_cpp"` prints nothing.

- [ ] **Step 3: Write the failing test**

`tests/test_norm.py`:
```python
from poc.norm import amounts_in, dates_in, fold


def test_fold_keeps_length_and_strips_diacritics():
    s = "Datum zdanitelného plnění: Čtvrtek, IČO"
    assert fold(s) == "datum zdanitelneho plneni: ctvrtek, ico"
    assert len(fold(s)) == len(s)


def test_amounts_czech_and_english_formats():
    text = "12 %  4 120,75  494,49  4 615,24 Kč; 1105.00; 1.105,00; -50,00"
    assert amounts_in(text) == [4120.75, 494.49, 4615.24, 1105.00, 1105.00, -50.00]


def test_amounts_ignore_dates_times_and_integers():
    assert amounts_in("15.09.2026 14:22 qty 3 CELKEM 7,80") == [7.80]


def test_dates_common_formats():
    text = "31.07.2026, 1. 8. 2026, 14/08/26, 2026-09-15, 31.02.2026"
    assert dates_in(text) == ["2026-07-31", "2026-08-01", "2026-08-14", "2026-09-15"]
```

- [ ] **Step 4: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_norm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.norm'`.

- [ ] **Step 5: Implement**

`poc/norm.py`:
```python
"""Czech amount/date normalisation, shared by extractors and scoring."""
import re
import unicodedata
from datetime import date

# Two decimals required: integers are too ambiguous (quantities, years, codes).
_AMOUNT = re.compile(
    r"(?<![\d.,])-?(?:\d{1,3}(?:[  .,']\d{3})+|\d+)[.,]\d{2}(?!\d|[.,]\d)"
)
_DATE = re.compile(
    r"(?<![\d.])(\d{1,2})\s?[./-]\s?(\d{1,2})\s?[./-]\s?(\d{4}|\d{2})(?![\d.])"
    r"|(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"
)


def fold(s: str) -> str:
    """Lower-case without diacritics, one char per char (spans stay valid)."""
    return "".join(unicodedata.normalize("NFKD", c)[0] for c in s).lower()


def amounts_in(s: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(s):
        t = m.group()
        v = int(re.sub(r"\D", "", t)) / 100
        out.append(-v if t.startswith("-") else v)
    return out


def dates_in(s: str) -> list[str]:
    out = []
    for m in _DATE.finditer(s):
        if m.group(4):
            y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
        else:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y += 2000 if y < 100 else 0
        try:
            out.append(date(y, mo, d).isoformat())
        except ValueError:
            pass
    return out
```

- [ ] **Step 6: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_norm.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml poc/__init__.py poc/requirements.txt poc/norm.py tests/test_norm.py
git commit -m "PoC: scaffold and Czech amount/date normalisation"
```

---

### Task 2: Checks (IČO, DIČ, VAT sums, required fields, ARES)

**Files:**
- Create: `poc/checks.py`, `tests/test_checks.py`

**Interfaces:**
- Produces:
  - `poc.checks.ico_valid(ico: str) -> bool`
  - `poc.checks.dic_valid(dic: str) -> bool`
  - `poc.checks.ares(ico: str, cache_dir: Path = ARES_CACHE) -> dict | None`: cached on disk; `None` means 404; a network error raises
  - `poc.checks.check(doc: dict, lookup=ares) -> tuple[dict, list[str]]`: returns `(enriched copy, failure messages)`, where an empty list means the document passes
  - `poc.checks.INVOICE_TYPES: set[str]`

- [ ] **Step 1: Write the failing test**

`tests/test_checks.py`:
```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_checks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.checks'`.

- [ ] **Step 3: Implement**

`poc/checks.py`:
```python
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
```

- [ ] **Step 4: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_checks.py -v`
Expected: 8 passed (`test_ares_live` included; it needs internet).

- [ ] **Step 5: Commit**

```bash
git add poc/checks.py tests/test_checks.py
git commit -m "PoC: IČO/DIČ/VAT checks and cached ARES lookup"
```

---

### Task 3: Page text (text layer, rendering, OCR CLI) with Tesseract

**Files:**
- Create: `poc/ocr.py`, `tests/test_ocr.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `poc.ocr.text_layer(pdf: Path) -> list[str]`
  - `poc.ocr.render(pdf: Path, dpi: int = 300) -> list[PIL.Image.Image]`
  - `poc.ocr.join_boxes(items: list[tuple[float, float, float, float, str]]) -> str`
  - `poc.ocr.has_text(s: str) -> bool`: at least 30 non-whitespace characters
  - `poc.ocr.ENGINES: dict[str, Callable[[], Callable[[Image], str]]]`
  - `poc.ocr.MODELS: Path`, which is `poc/models`
  - CLI `python -m poc.ocr ENGINE OUT_DIR PDF...`, where ENGINE is `layer` or a key of `ENGINES`. It writes `OUT_DIR/<pdf name>.json` = `{"pages": [str], "seconds": [float]}` or `{"error": str}`, and `OUT_DIR/_meta.json` = `{"engine", "load_seconds", "model_bytes"}`.

- [ ] **Step 1: Download the Tesseract model**

```bash
mkdir -p poc/models/tessdata
curl -fL -o poc/models/tessdata/ces.traineddata https://github.com/tesseract-ocr/tessdata/raw/4.1.0/ces.traineddata
```
Expected: a file of roughly 13 MB.

- [ ] **Step 2: Write the failing test**

`tests/test_ocr.py`:
```python
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from poc.norm import amounts_in
from poc.ocr import ENGINES, has_text, join_boxes, text_layer

ROOT = Path(__file__).resolve().parent.parent
ZA001 = ROOT / "testdata" / "zari" / "ZA001.pdf"
MODULE = {"tesseract": "tesserocr", "rapidocr": "rapidocr", "paddleocr": "paddleocr",
          "doctr": "doctr", "easyocr": "easyocr"}


def sample() -> Image.Image:
    img = Image.new("RGB", (1400, 400), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    for i, line in enumerate(["Faktura - daňový doklad", "IČO: 12345679",
                              "Celkem k úhradě 4 615,24 Kč"]):
        d.text((40, 40 + i * 110), line, fill="black", font=font)
    return img


def test_join_boxes_orders_lines_and_words():
    items = [(300, 100, 400, 130, "b"), (10, 12, 90, 40, "x"),
             (10, 102, 100, 128, "a"), (120, 10, 200, 38, "y")]
    assert join_boxes(items) == "x y\na b"


def test_has_text():
    assert not has_text("  \n ab ")
    assert has_text("Faktura - daňový doklad č. 20269001")


@pytest.mark.skipif(not ZA001.exists(), reason="testdata missing")
def test_text_layer_of_native_pdf():
    assert "12345679" in text_layer(ZA001)[0]


@pytest.mark.parametrize("engine", MODULE)
def test_engine_reads_czech_sample(engine):
    if importlib.util.find_spec(MODULE[engine]) is None:
        pytest.skip(f"{engine} not installed in this venv")
    text = ENGINES[engine]()(sample())
    assert "12345679" in text
    assert 4615.24 in amounts_in(text)


def test_cli_records_error_and_continues(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    out = tmp_path / "out"
    pdfs = [str(bad)] + ([str(ZA001)] if ZA001.exists() else [])
    subprocess.run([sys.executable, "-m", "poc.ocr", "layer", str(out), *pdfs],
                   cwd=ROOT, check=True)
    assert "error" in json.loads((out / "bad.pdf.json").read_text())
    if ZA001.exists():
        assert json.loads((out / "ZA001.pdf.json").read_text())["pages"]
    assert json.loads((out / "_meta.json").read_text())["engine"] == "layer"
```

- [ ] **Step 3: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.ocr'`.

- [ ] **Step 4: Implement**

`poc/ocr.py`. All five engines are defined here. The engines in Tasks 4–6 only need their venv and their test run, because their loaders import lazily.
```python
"""Page text from a PDF: its text layer, or one OCR engine.

One engine per process, so the parent can measure its RSS:
    python -m poc.ocr ENGINE OUT_DIR PDF...
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium

DPI = 300
MODELS = Path(__file__).parent / "models"


def text_layer(pdf: Path) -> list[str]:
    return [page.get_textpage().get_text_bounded() for page in pdfium.PdfDocument(pdf)]


def render(pdf: Path, dpi: int = DPI) -> list:
    return [p.render(scale=dpi / 72).to_pil().convert("RGB") for p in pdfium.PdfDocument(pdf)]


def has_text(s: str) -> bool:
    return len("".join(s.split())) >= 30


def join_boxes(items) -> str:
    """[(x0, y0, x1, y1, text)] -> lines top to bottom, words left to right."""
    lines = []  # [y_center, [(x0, text)]]
    for x0, y0, x1, y1, t in sorted(items, key=lambda i: (i[1] + i[3]) / 2):
        yc = (y0 + y1) / 2
        if lines and abs(yc - lines[-1][0]) < (y1 - y0) / 2:
            lines[-1][1].append((x0, t))
        else:
            lines.append([yc, [(x0, t)]])
    return "\n".join(" ".join(t for _, t in sorted(ws)) for _, ws in lines)


def _quad(box) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _tesseract():
    import tesserocr
    api = tesserocr.PyTessBaseAPI(path=str(MODELS / "tessdata"), lang="ces")

    def run(img):
        api.SetImage(img)
        return api.GetUTF8Text()
    return run


def _rapidocr():
    from rapidocr import RapidOCR
    eng = RapidOCR()

    def run(img):
        r = eng(img)
        if r.boxes is None:
            return ""
        return join_boxes([(*_quad(b), t) for b, t in zip(r.boxes, r.txts)])
    return run


def _paddleocr():
    import numpy as np
    from paddleocr import PaddleOCR
    eng = PaddleOCR(lang="cs")

    def run(img):
        (r,) = eng.predict(np.array(img)[:, :, ::-1])  # RGB -> BGR
        return join_boxes([(*b, t) for b, t in zip(r["rec_boxes"].tolist(), r["rec_texts"])])
    return run


def _doctr():
    import numpy as np
    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True)
    return lambda img: model([np.array(img)]).render()


def _easyocr():
    import easyocr
    import numpy as np
    reader = easyocr.Reader(["cs", "en"], gpu=False)
    return lambda img: join_boxes([(*_quad(b), t) for b, t, _ in reader.readtext(np.array(img))])


ENGINES = {"tesseract": _tesseract, "rapidocr": _rapidocr, "paddleocr": _paddleocr,
           "doctr": _doctr, "easyocr": _easyocr}


def _model_dirs(engine: str) -> list[Path]:
    home = Path.home()
    if engine == "rapidocr":
        return [Path(importlib.util.find_spec("rapidocr").submodule_search_locations[0]) / "models"]
    return {"tesseract": [MODELS / "tessdata"], "paddleocr": [home / ".paddlex" / "official_models"],
            "doctr": [home / ".cache" / "doctr"], "easyocr": [home / ".EasyOCR" / "model"]
            }.get(engine, [])


def model_bytes(engine: str) -> int:
    return sum(f.stat().st_size for d in _model_dirs(engine) if d.exists()
               for f in d.rglob("*") if f.is_file())


def main(engine: str, out_dir: str, pdfs: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    run = None if engine == "layer" else ENGINES[engine]()
    load = time.perf_counter() - t
    for pdf in map(Path, pdfs):
        try:
            if run is None:
                t = time.perf_counter()
                pages = text_layer(pdf)
                secs = [(time.perf_counter() - t) / max(len(pages), 1)] * len(pages)
            else:
                pages, secs = [], []
                for img in render(pdf):
                    t = time.perf_counter()
                    pages.append(run(img))
                    secs.append(time.perf_counter() - t)
            rec = {"pages": pages, "seconds": secs}
        except Exception as e:  # one bad file must not stop the benchmark
            rec = {"error": repr(e)}
        (out / f"{pdf.name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(engine, pdf.name, "error" if "error" in rec else f"{sum(rec['seconds']):.1f}s",
              file=sys.stderr)
    (out / "_meta.json").write_text(json.dumps(
        {"engine": engine, "load_seconds": load, "model_bytes": model_bytes(engine)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
```

- [ ] **Step 5: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_ocr.py -v`
Expected: the join, has_text, text-layer and CLI tests pass, and `tesseract` passes. If `rapidocr` is installed it also runs, and is fixed in Task 4 if it fails. `paddleocr`, `doctr` and `easyocr` are SKIPPED because they are not installed in this venv.

- [ ] **Step 6: Commit**

```bash
git add poc/ocr.py tests/test_ocr.py
git commit -m "PoC: page text via text layer or OCR engine, Tesseract wired"
```

---

### Task 4: RapidOCR

**Files:**
- Test only: `tests/test_ocr.py::test_engine_reads_czech_sample[rapidocr]`. The engine code is already in `poc/ocr.py` (`_rapidocr`).

**Interfaces:**
- Consumes: `ENGINES["rapidocr"]` from Task 3.
- Produces: a working `rapidocr` engine for `python -m poc.ocr rapidocr ...`.

- [ ] **Step 1: Run the engine test**

Run: `.venv/bin/python -m pytest "tests/test_ocr.py::test_engine_reads_czech_sample[rapidocr]" -v`
Expected: PASS. The first run downloads the PP-OCRv6 models into the package's `models/` directory. If the result object lacks `.boxes` or `.txts`, print `dir(r)` for the installed 3.10.0 and adapt `_rapidocr`, and only that function.

- [ ] **Step 2: Smoke-run the CLI on one scan**

Run: `.venv/bin/python -m poc.ocr rapidocr poc/cache/text/rapidocr testdata/zari/ZA044.pdf && head -c 600 poc/cache/text/rapidocr/ZA044.pdf.json`
Expected: the JSON contains readable receipt lines, such as "dm" and amounts.

- [ ] **Step 3: Commit (only if `_rapidocr` changed)**

```bash
git add poc/ocr.py && git commit -m "PoC: fix RapidOCR result handling"
```

---

### Task 5: PaddleOCR in `.venv-paddle`

**Files:**
- Create: `poc/requirements-paddle.txt`

**Interfaces:**
- Consumes: `ENGINES["paddleocr"]` from Task 3.
- Produces: `.venv-paddle/bin/python -m poc.ocr paddleocr ...` works.

- [ ] **Step 1: Write the requirements and install**

`poc/requirements-paddle.txt`:
```
pypdfium2==5.14.0
paddlepaddle==3.3.1
paddleocr==3.7.0
pytest==9.2.0
```
```bash
python3 -m venv .venv-paddle
.venv-paddle/bin/pip install -r poc/requirements-paddle.txt
```

- [ ] **Step 2: Run the engine test**

Run: `.venv-paddle/bin/python -m pytest "tests/test_ocr.py::test_engine_reads_czech_sample[paddleocr]" -v`
Expected: PASS. Models download to `~/.paddlex/official_models`. If `lang="cs"` raises "unsupported language" in 3.7.0, change it to `lang="latin"`. If the result keys differ, print `r.keys()` and adapt `_paddleocr`.

- [ ] **Step 3: Commit**

```bash
git add poc/requirements-paddle.txt poc/ocr.py
git commit -m "PoC: PaddleOCR engine in its own venv"
```

---

### Task 6: docTR and EasyOCR in `.venv-torch`

**Files:**
- Create: `poc/requirements-torch.txt`

**Interfaces:**
- Consumes: `ENGINES["doctr"]` and `ENGINES["easyocr"]` from Task 3.
- Produces: `.venv-torch/bin/python -m poc.ocr {doctr,easyocr} ...` works.

- [ ] **Step 1: Write the requirements and install**

`poc/requirements-torch.txt`:
```
--extra-index-url https://download.pytorch.org/whl/cpu
pypdfium2==5.14.0
torch==2.14.1+cpu
torchvision==0.29.1+cpu
python-doctr==1.1.0
easyocr==1.7.2
pytest==9.2.0
```
```bash
python3 -m venv .venv-torch
.venv-torch/bin/pip install -r poc/requirements-torch.txt
```
docTR pulls in `opencv-python` and EasyOCR pulls in `opencv-python-headless`. Both provide `cv2`, so if `import cv2` breaks, run `.venv-torch/bin/pip uninstall -y opencv-python && .venv-torch/bin/pip install --force-reinstall --no-deps opencv-python-headless`.

- [ ] **Step 2: Run the engine tests**

Run: `.venv-torch/bin/python -m pytest tests/test_ocr.py -k "doctr or easyocr" -v`
Expected: `easyocr` PASS. docTR's default recogniser has a French vocabulary without č, ř or ů, but IČO digits and amounts still read, so `doctr` should PASS too. Losing the diacritics is a measured result, not a bug. Don't tune it.

- [ ] **Step 3: Commit**

```bash
git add poc/requirements-torch.txt poc/ocr.py
git commit -m "PoC: docTR and EasyOCR engines in a CPU torch venv"
```

---

### Task 7: Rules extractor (a) and the extract CLI

**Files:**
- Create: `poc/extract.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `poc.norm.fold`, `amounts_in`, `dates_in`; `poc.checks.ico_valid`; `poc.ocr.has_text`, `render`, `MODELS`.
- Produces:
  - `poc.extract.rules(text: str, client_ico: str | None = None) -> dict`: one document in the label schema (`supplier.name` is `None`, because ARES fills it; `items` is `[]`)
  - `poc.extract.page_texts(name: str, sources: list[Path]) -> list[str]`: for each page, the first source with `has_text`
  - `poc.extract.CLIENT_ICO = "76543218"`
  - CLI `python -m poc.extract EXTRACTOR OUT_DIR SOURCE...`. It writes `OUT_DIR/<pdf name>.json` = `{"documents": [dict], "seconds": float}` or `{"error": str}`, and `OUT_DIR/_meta.json` = `{"extractor", "load_seconds", "model_bytes", "peak_vram_bytes"}`. Results are cached in `poc/cache/extract/<extractor>/<sha256 of input>[:16].json`.
  - `poc.extract.LOADERS: dict[str, Callable[[], Callable[[str], list[dict]]]]`, starting with `"rules"`. Tasks 8 and 9 add to it.

- [ ] **Step 1: Write the failing test**

`tests/test_extract.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

from poc.extract import page_texts, rules

ROOT = Path(__file__).resolve().parent.parent

INVOICE = """Faktura - daňový doklad
Číslo: 20269001
Dodavatel: Kavárna Vzor s.r.o.
IČO: 12345679   DIČ: CZ12345679
Odběratel: Klient Vzor s.r.o. IČO: 76543218 DIČ: CZ76543218
Datum vystavení: 31.07.2026
Datum zdanitelného plnění: 31.07.2026
Datum splatnosti: 14.08.2026
Variabilní symbol: 20269001
Forma úhrady: Převodem
Sazba  Základ  DPH  Celkem
12 %  4 120,75  494,49  4 615,24
Celkem bez DPH 4 120,75
Celkem k úhradě: 4 615,24 Kč
"""

RECEIPT = """Obchod Vzor s.r.o.
IČ: 24680249 DIČ: CZ24680249
Účtenka č. 1234/56
Rohlík 2x 3,90 7,80
DPH 12% 6,96 0,84
CELKEM 7,80
Platba: HOTOVĚ
15.09.2026 14:22
"""


def test_rules_invoice():
    d = rules(INVOICE, client_ico="76543218")
    assert d["type"] == "receivedInvoice" and d["payment"] == "transfer"
    assert d["number"] == "20269001" and d["variabilni_symbol"] == "20269001"
    assert (d["issue_date"], d["duzp"], d["due_date"]) == ("2026-07-31", "2026-07-31", "2026-08-14")
    assert d["supplier"]["ico"] == "12345679" and d["supplier"]["dic"] == "CZ12345679"
    assert d["vat"] == [{"rate": 12, "base": 4120.75, "vat": 494.49}]
    assert d["total"] == 4615.24 and d["to_pay"] == 4615.24 and d["currency"] == "CZK"


def test_rules_never_picks_client_as_supplier():
    client_first = INVOICE.replace("Dodavatel: Kavárna Vzor s.r.o.\nIČO: 12345679   DIČ: CZ12345679\n", "")
    client_first += "Dodavatel: Kavárna Vzor s.r.o. IČO: 12345679 DIČ: CZ12345679\n"
    d = rules(client_first, client_ico="76543218")
    assert d["supplier"]["ico"] == "12345679" and d["supplier"]["dic"] == "CZ12345679"


def test_rules_receipt():
    d = rules(RECEIPT, client_ico="76543218")
    assert d["type"] == "receipt" and d["payment"] == "cash"
    assert d["number"] == "1234/56" and d["supplier"]["ico"] == "24680249"
    assert d["issue_date"] == d["duzp"] == "2026-09-15"
    assert d["vat"] == [{"rate": 12, "base": 6.96, "vat": 0.84}] and d["total"] == 7.80


def test_rules_value_before_label():
    d = rules("Lihovar Vzor s.r.o. 12345679 IČO: CZ12345679 DIČ:", client_ico="76543218")
    assert d["supplier"]["ico"] == "12345679"


def _write(dir_: Path, name: str, rec: dict):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{name}.json").write_text(json.dumps(rec))


def test_page_texts_prefers_first_nonempty_source(tmp_path):
    layer, ocr = tmp_path / "layer", tmp_path / "ocr"
    native = "Faktura - daňový doklad č. 20269001 strana jedna"
    _write(layer, "ZA017.pdf", {"pages": [native, ""], "seconds": [0, 0]})
    _write(ocr, "ZA017.pdf", {"pages": ["ocr page one ............................", "ocr page two with clearly enough text to count"], "seconds": [1, 1]})
    assert page_texts("ZA017.pdf", [layer, ocr]) == [native, "ocr page two with clearly enough text to count"]


def test_cli_records_error_and_continues(tmp_path):
    layer = tmp_path / "layer"
    _write(layer, "bad.pdf", {"error": "boom"})
    _write(layer, "good.pdf", {"pages": [RECEIPT], "seconds": [0]})
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "poc.extract", "rules", str(out), str(layer)],
                   cwd=ROOT, check=True)
    assert "error" in json.loads((out / "bad.pdf.json").read_text())
    good = json.loads((out / "good.pdf.json").read_text())
    assert good["documents"][0]["total"] == 7.80
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.extract'`.

- [ ] **Step 3: Implement**

`poc/extract.py`:
```python
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
```

- [ ] **Step 4: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v`
Expected: 6 passed. If a rules assertion fails, fix the regex in `rules`. Don't weaken the test.

- [ ] **Step 5: Commit**

```bash
git add poc/extract.py tests/test_extract.py
git commit -m "PoC: rules extractor and cached extract CLI"
```

---

### Task 8: Small LLM extractor (b) via llama.cpp

**Files:**
- Modify: `poc/extract.py` (add the schema, the prompt, `parse_documents` and LLM loaders; replace `load_extractor`)
- Modify: `tests/test_extract.py` (append tests)

**Interfaces:**
- Consumes: `LOADERS` and `main` from Task 7; `MODELS`.
- Produces:
  - `poc.extract.SCHEMA: dict` (JSON schema: `{"documents": [doc]}`)
  - `poc.extract.PROMPT: str`
  - `poc.extract.parse_documents(s: str) -> list[dict]`
  - `poc.extract.LLMS: dict[str, Path]`, with keys `qwen3.5-2b` and `qwen3.5-4b`
  - `load_extractor(name)` handles those keys. The meta it returns includes `model_bytes`.

- [ ] **Step 1: Download the models and verify their checksums**

```bash
curl -fL -o poc/models/Qwen3.5-2B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf
curl -fL -o poc/models/Qwen3.5-4B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf
sha256sum -c <<'EOF'
aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223  poc/models/Qwen3.5-2B-Q4_K_M.gguf
00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4  poc/models/Qwen3.5-4B-Q4_K_M.gguf
EOF
```
Expected: both report `OK`.

- [ ] **Step 2: Write the failing test** (append it to `tests/test_extract.py`, moving the imports to the top of the file)

```python
import pytest

from poc.extract import LLMS, SCHEMA, load_extractor, parse_documents


def test_schema_requires_every_label_field():
    doc = SCHEMA["properties"]["documents"]["items"]
    assert set(doc["required"]) == {"type", "payment", "number", "variabilni_symbol", "issue_date",
                                    "duzp", "due_date", "currency", "supplier", "vat", "total",
                                    "to_pay", "items"}


def test_parse_documents():
    assert parse_documents('{"documents": [{"total": 1.5}]}') == [{"total": 1.5}]
    with pytest.raises(ValueError):
        parse_documents('{"documents": []}')
    with pytest.raises(ValueError):
        parse_documents("not json")


@pytest.mark.slow
@pytest.mark.skipif(not LLMS["qwen3.5-2b"].exists(), reason="model not downloaded")
def test_llm_reads_invoice():
    run, meta = load_extractor("qwen3.5-2b")
    (d,) = run(INVOICE)
    assert d["supplier"]["ico"] == "12345679" and abs(d["total"] - 4615.24) < 0.01
    assert meta["model_bytes"] > 1e9
```

- [ ] **Step 3: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v`
Expected: FAIL with `ImportError: cannot import name 'LLMS'`.

- [ ] **Step 4: Implement**

Add this to `poc/extract.py` above `# --- CLI`, and replace the old `load_extractor` with the one below:
```python
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
- ico = IČ/IČO (8 digits). dic = DIČ (e.g. CZ12345678).
- vat: one line per VAT rate from the VAT summary (rekapitulace DPH): rate in %, base without VAT, VAT amount.
  Empty if the supplier is not a VAT payer.
- total = amount including VAT. to_pay = amount to pay (k úhradě) after rounding or deducted advances.
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
    llm = Llama(model_path=str(path), n_ctx=8192, n_threads=len(os.sched_getaffinity(0)), verbose=False)

    def run(text: str) -> list[dict]:
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text[:MAX_CHARS]}],
            response_format={"type": "json_object", "schema": SCHEMA},
            temperature=0, max_tokens=3000)
        return parse_documents(r["choices"][0]["message"]["content"])
    return run


def load_extractor(name: str):
    """-> (fn(input) -> list[dict], meta dict). meta may hold 'finish': () -> dict."""
    if name in LLMS:
        return _load_llm(LLMS[name]), {"model_bytes": LLMS[name].stat().st_size}
    return LOADERS[name](), {}
```
The JSON-schema grammar forces the output to start with `{`, so Qwen3.5 cannot emit a thinking block. No chat-template switch is needed.

- [ ] **Step 5: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v`
Expected: all pass, including `test_llm_reads_invoice`, which takes about 10–60 s on all cores. If llama.cpp rejects the `qwen35` architecture, stop and report it. Don't swap the model silently.

- [ ] **Step 6: Commit**

```bash
git add poc/extract.py tests/test_extract.py
git commit -m "PoC: Qwen3.5 CPU extractor via llama.cpp with JSON schema"
```

---

### Task 9: GPU vision reference (c) via ollama, with peak VRAM

**Files:**
- Modify: `poc/extract.py` (add `_load_vlm`, `vram_sampler`; extend `load_extractor`)
- Modify: `tests/test_extract.py` (append tests)

**Interfaces:**
- Consumes: `SCHEMA`, `PROMPT`, `parse_documents`; `poc.ocr.render`.
- Produces:
  - extractor names `vlm:<ollama model>` that take a PDF `Path`
  - `poc.extract.vram_sampler() -> Callable[[], dict]`: starts sampling, and the returned function stops it and gives `{"peak_vram_bytes": int}`
  - `VRAM_LIMIT = 21 * 2**30`
  - `_meta.json` for VLM runs gains `peak_vram_bytes`, measured as peak − baseline, with the model unloaded before the run.

- [ ] **Step 1: Write the failing test** (append it to `tests/test_extract.py`, moving the imports to the top of the file)

```python
import urllib.request

from poc.extract import VRAM_LIMIT, vram_sampler


def test_vram_sampler_reports_peak():
    stop = vram_sampler()
    assert stop()["peak_vram_bytes"] >= 0


def _ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not (_ollama_up() and (ROOT / "testdata/zari/ZA001.pdf").exists()),
                    reason="needs ollama and testdata")
def test_vlm_reads_invoice_within_vram_limit():
    run, meta = load_extractor("vlm:qwen3.8:27b-q4_K_M")
    (d, *_) = run(ROOT / "testdata/zari/ZA001.pdf")
    assert d["supplier"]["ico"] == "12345679"
    assert meta["finish"]()["peak_vram_bytes"] <= VRAM_LIMIT
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v -k "vram or vlm"`
Expected: FAIL with `ImportError: cannot import name 'VRAM_LIMIT'`.

- [ ] **Step 3: Implement**

Add this to `poc/extract.py` after `_load_llm`, and extend `load_extractor`:
```python
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


def vram_sampler():
    """Sample dGPU VRAM use every 0.2 s; returns stop() -> {"peak_vram_bytes": peak - baseline}."""
    import threading
    used = next((d / "mem_info_vram_used" for d in Path("/sys/class/drm").glob("card*/device")
                 if (d / "mem_info_vram_total").exists()
                 and int((d / "mem_info_vram_total").read_text()) > 16 * 2**30), None)
    read = (lambda: int(used.read_text())) if used else (lambda: 0)
    base, peak, done = read(), [0], threading.Event()

    def loop():
        while not done.wait(0.2):
            peak[0] = max(peak[0], read() - base)
    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def stop():
        done.set()
        th.join()
        return {"peak_vram_bytes": max(peak[0], read() - base, 0)}
    return stop


def _load_vlm(model: str):
    import base64
    import io

    from poc.ocr import render
    _ollama("/api/generate", {"model": model, "keep_alive": 0})  # unload -> clean baseline
    stop = vram_sampler()

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
```
`load_extractor` gets a new first branch:
```python
    if name.startswith("vlm:"):
        return _load_vlm(name[4:])
```

- [ ] **Step 4: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v -k "vram or vlm"`
Expected: 2 passed. If peak VRAM exceeds 21 GB, set `VLM_CTX = 8192` and rerun. If it still doesn't fit, use `vlm:gemma4:26b` (already pulled) and record the switch in the report.

- [ ] **Step 5: Commit**

```bash
git add poc/extract.py tests/test_extract.py
git commit -m "PoC: ollama VLM reference extractor with peak VRAM measurement"
```

---

### Task 10: Benchmark runner

**Files:**
- Create: `poc/bench.py`, `tests/test_bench.py`

**Interfaces:**
- Consumes: the CLIs of `poc.ocr` and `poc.extract`; `ocr.MODELS`.
- Produces:
  - `poc.bench.pdfs(d: Path = TESTDATA) -> list[Path]`
  - `poc.bench.parse_cpus(s: str) -> set[int]`, for strings like `"0"` or `"0-3,8"`
  - `poc.bench.measure(cmd: list, cpus: set[int]) -> dict`, returning `{"wall_s", "peak_rss_mb", "exit"}`
  - `poc.bench.TEXT = poc/cache/text`, `poc.bench.RESULTS = poc/results`
  - CLI `python -m poc.bench RUN {ocr|extract|vlm} [--engines ...] [--extractors ...] [--sources ...] [--model M] [--cpus 0]`. Each child process appends one row to `poc/results/RUN/runs.csv` with the columns `stage,name,cpus,wall_s,peak_rss_mb,exit,load_seconds,model_mb,peak_vram_mb`.

- [ ] **Step 1: Write the failing test**

`tests/test_bench.py`:
```python
import sys

from poc.bench import measure, parse_cpus, pdfs


def test_pdfs_includes_uppercase_extension(tmp_path):
    for n in ["a.pdf", "B.PDF", "c.pdf.expected.json", "d.txt"]:
        (tmp_path / n).write_text("")
    assert [p.name for p in pdfs(tmp_path)] == ["B.PDF", "a.pdf"]


def test_parse_cpus():
    assert parse_cpus("0") == {0}
    assert parse_cpus("0-2,5") == {0, 1, 2, 5}


def test_measure_child_rss_and_affinity():
    code = "import os; b = bytearray(200 * 2**20); assert os.sched_getaffinity(0) == {0}"
    r = measure([sys.executable, "-c", code], {0})
    assert r["exit"] == 0 and r["peak_rss_mb"] >= 200
    small = measure([sys.executable, "-c", "pass"], {0})
    assert small["peak_rss_mb"] < 100  # per child, not the max over all children
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_bench.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.bench'`.

- [ ] **Step 3: Implement**

`poc/bench.py`:
```python
"""Run each OCR engine / extractor in its own process pinned to --cpus, record time and peak RSS.

    .venv/bin/python -m poc.bench RUN ocr     [--engines ...] [--cpus 0]
    .venv/bin/python -m poc.bench RUN extract [--extractors ...] [--sources ...] [--cpus 0]
    .venv/bin/python -m poc.bench RUN vlm     [--model qwen3.8:27b-q4_K_M]
"""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTDATA = ROOT / "testdata" / "zari"
TEXT = Path(__file__).parent / "cache" / "text"
RESULTS = Path(__file__).parent / "results"
VENV = {"layer": ".venv", "tesseract": ".venv", "rapidocr": ".venv", "paddleocr": ".venv-paddle",
        "doctr": ".venv-torch", "easyocr": ".venv-torch"}
OCR_ENGINES = [e for e in VENV if e != "layer"]
COLUMNS = ["stage", "name", "cpus", "wall_s", "peak_rss_mb", "exit", "load_seconds", "model_mb",
           "peak_vram_mb"]


def pdfs(d: Path = TESTDATA) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() == ".pdf")


def parse_cpus(s: str) -> set[int]:
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def measure(cmd: list, cpus: set[int]) -> dict:
    n = str(len(cpus))
    env = {**os.environ, "OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "MKL_NUM_THREADS": n}
    t = time.perf_counter()
    p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env,
                         preexec_fn=lambda: os.sched_setaffinity(0, cpus))
    _, status, ru = os.wait4(p.pid, 0)  # rusage of this child only
    p.returncode = os.waitstatus_to_exitcode(status)
    return {"wall_s": round(time.perf_counter() - t, 1), "peak_rss_mb": ru.ru_maxrss // 1024,
            "exit": p.returncode}


def _record(run: str, stage: str, name: str, cpus: str, m: dict, meta_file: Path) -> None:
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    vram = meta.get("peak_vram_bytes")
    row = {"stage": stage, "name": name, "cpus": cpus, **m,
           "load_seconds": round(meta.get("load_seconds", 0), 1),
           "model_mb": meta.get("model_bytes", 0) // 2**20,
           "peak_vram_mb": vram // 2**20 if vram is not None else ""}
    f = RESULTS / run / "runs.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("stage", choices=["ocr", "extract", "vlm"])
    ap.add_argument("--engines", nargs="+", default=["layer", *OCR_ENGINES])
    ap.add_argument("--extractors", nargs="+", default=["rules", "qwen3.5-2b", "qwen3.5-4b"])
    ap.add_argument("--sources", nargs="+", default=OCR_ENGINES)
    ap.add_argument("--model", default="qwen3.8:27b-q4_K_M")
    ap.add_argument("--cpus", default="0")
    a = ap.parse_args()
    cpus = parse_cpus(a.cpus)
    py = lambda venv: ROOT / venv / "bin" / "python"

    if a.stage == "ocr":
        for e in a.engines:
            m = measure([py(VENV[e]), "-m", "poc.ocr", e, TEXT / e, *pdfs()], cpus)
            _record(a.run, "ocr", e, a.cpus, m, TEXT / e / "_meta.json")
    elif a.stage == "extract":
        for x in a.extractors:
            for s in a.sources:
                out = RESULTS / a.run / f"{x}+{s}"
                srcs = [TEXT / "layer"] + ([TEXT / s] if s != "layer" else [])
                m = measure([py(".venv"), "-m", "poc.extract", x, out, *srcs], cpus)
                _record(a.run, "extract", f"{x}+{s}", a.cpus, m, out / "_meta.json")
    else:
        name = f"vlm:{a.model}"
        out = RESULTS / a.run / name.replace(":", "_")
        m = measure([py(".venv"), "-m", "poc.extract", name, out, *pdfs()], cpus)
        _record(a.run, "vlm", out.name, a.cpus, m, out / "_meta.json")  # name == result dir


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_bench.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/bench.py tests/test_bench.py
git commit -m "PoC: benchmark runner with per-child RSS and CPU pinning"
```

---

### Task 11: Scoring and report

**Files:**
- Create: `poc/score.py`, `tests/test_score.py`

**Interfaces:**
- Consumes:
  - `poc.checks.check` and `ares`
  - `poc.norm.amounts_in`, `dates_in`
  - `poc.ocr.has_text`
  - `poc.bench.TEXT`, `RESULTS`, `TESTDATA`
  - result files `{"documents"|"error", "seconds"}`
  - OCR cache files `{"pages"|"error", "seconds"}`
  - `runs.csv`
- Produces:
  - `poc.score.KEY: list[str]`
  - `poc.score.same(field: str, got, want) -> bool`
  - `poc.score.ocr_recall(text: str, doc: dict) -> tuple[int, int]`, returning `(found, total)`
  - `poc.score.evaluate(result: dict, labels: dict, lookup) -> dict`, a row with one bool per field plus `key_ok`, `pass`, `silent`, `error`, `fails`, `n_docs`, `n_docs_want`, `items_count_ok`, `items_found`
  - `poc.score.report(run: str) -> str`, the markdown
  - CLI `python -m poc.score RUN [--report docs/poc-results.md]`. It writes `poc/results/RUN/scores.csv`, prints the markdown and optionally saves it.

- [ ] **Step 1: Write the failing test**

`tests/test_score.py`:
```python
from poc.score import evaluate, ocr_recall, same

WANT = {"type": "receivedInvoice", "payment": "transfer", "number": "20269001",
        "variabilni_symbol": None, "issue_date": "2026-07-31", "duzp": "2026-07-31",
        "due_date": "2026-08-14", "currency": "CZK",
        "supplier": {"name": "Kavárna Vzor s.r.o.", "ico": "12345679", "dic": "CZ12345679"},
        "vat": [{"rate": 12, "base": 4120.75, "vat": 494.49}], "total": 4615.24,
        "to_pay": 4615.24, "items": [{"text": "koule", "base": 2000.00, "total": None}]}
LABELS = {"file": "ZA001.pdf", "documents": [WANT]}
ARES = {"obchodniJmeno": "Kavárna Vzor s.r.o.", "dic": "CZ12345679", "sidlo": {}}
lookup = lambda ico: ARES if ico == "12345679" else None


def test_same_null_semantics():
    assert same("variabilni_symbol", None, None)
    assert not same("variabilni_symbol", "123", None)
    assert not same("number", None, "20269001")
    assert same("number", "2026 9001", "20269001")
    assert same("total", "4615.24", 4615.24) and not same("total", 4615.0, 4615.24)
    assert same("supplier.name", "Kavarna vzor s. r. o.", "Kavárna Vzor s.r.o.")
    assert same("vat", [{"rate": 12.0, "base": 4120.75, "vat": 494.49}], WANT["vat"])
    assert not same("vat", [], WANT["vat"]) and not same("vat", [{"rate": None}], WANT["vat"])


def test_evaluate_pass_silent_error():
    good = {k: v for k, v in WANT.items() if k != "supplier"} | {"supplier": {"ico": "12345679"}}
    row = evaluate({"documents": [good]}, LABELS, lookup)
    assert row["pass"] and not row["silent"] and row["supplier.name"]  # name filled from ARES

    wrong_date = good | {"due_date": "2026-08-15"}  # checks can't see this
    row = evaluate({"documents": [wrong_date]}, LABELS, lookup)
    assert row["silent"] and not row["pass"]

    caught = good | {"total": 999.0}  # VAT sum check catches it
    row = evaluate({"documents": [caught]}, LABELS, lookup)
    assert not row["pass"] and not row["silent"] and "vat sum != total" in row["fails"]

    row = evaluate({"error": "bad model output"}, LABELS, lookup)
    assert row["error"] and not row["pass"] and not row["silent"]


def test_ocr_recall():
    text = "Faktura 2026 9001 IČO 12345679 vystaveno 31.7.2026 splatnost 14. 8. 2026\n" \
           "12 % 4 120,75 494,49 Celkem 4 615,24"
    assert ocr_recall(text, WANT) == (7, 7)
    assert ocr_recall("nothing", WANT) == (0, 7)
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.score'`.

- [ ] **Step 3: Implement**

`poc/score.py`:
```python
"""Score a benchmark run against the labels and render the markdown report.

    .venv/bin/python -m poc.score RUN [--report docs/poc-results.md]
"""
import argparse
import csv
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path

from poc.bench import RESULTS, TESTDATA, TEXT, pdfs
from poc.checks import ares, check
from poc.norm import amounts_in, dates_in, fold
from poc.ocr import has_text

KEY = ["type", "number", "variabilni_symbol", "issue_date", "duzp", "due_date", "supplier.ico",
       "vat", "total"]
EXTRA = ["payment", "currency", "supplier.dic", "supplier.name", "to_pay"]


def get(doc, path: str):
    for k in path.split("."):
        doc = doc.get(k) if isinstance(doc, dict) else None
    return doc


def _vat(lines) -> list[tuple[float, float, float]]:
    return sorted((float(v["rate"]), float(v["base"]), float(v["vat"])) for v in lines)


def same(field: str, got, want) -> bool:
    if want in (None, "", []):
        return got in (None, "", [])
    if got in (None, ""):
        return False
    try:
        if field == "vat":
            g, w = _vat(got), _vat(want)
            return len(g) == len(w) and all(abs(a - b) <= 0.011 for x, y in zip(g, w) for a, b in zip(x, y))
        if isinstance(want, (int, float)):
            return abs(float(got) - want) <= 0.011
    except (TypeError, ValueError, KeyError):
        return False
    if field.endswith("name"):
        n = lambda s: re.sub(r"[\s.,]", "", fold(str(s)))
        return SequenceMatcher(None, n(got), n(want)).ratio() >= 0.9
    return re.sub(r"\s", "", str(got)) == re.sub(r"\s", "", str(want))


def _items(got: list, want: list) -> tuple[bool, float]:
    amounts = [i.get("base") if i.get("base") is not None else i.get("total") for i in want]
    amounts = [a for a in amounts if a is not None]
    gots = [x for i in got or [] for x in (i.get("base"), i.get("total")) if isinstance(x, (int, float))]
    found = sum(any(abs(g - a) <= 0.011 for g in gots) for a in amounts)
    return len(got or []) == len(want), found / len(amounts) if amounts else 1.0


def evaluate(result: dict, labels: dict, lookup=ares) -> dict:
    wants = labels["documents"]
    docs = result.get("documents") or []
    base = {"error": not docs, "n_docs": len(docs), "n_docs_want": len(wants)}
    if not docs:
        return base | {f: False for f in KEY + EXTRA} | {
            "key_ok": False, "pass": False, "silent": False, "fails": result.get("error", "no output"),
            "items_count_ok": False, "items_found": 0.0}
    got, fails = check(docs[0], lookup)
    scored = [{f: same(f, get(got, f), get(w, f)) for f in KEY + EXTRA} for w in wants]
    i = max(range(len(wants)), key=lambda j: sum(scored[j][f] for f in KEY))
    s = scored[i]
    key_ok = all(s[f] for f in KEY)
    count_ok, found = _items(got.get("items"), wants[i].get("items") or [])
    return base | s | {"key_ok": key_ok, "pass": key_ok and not fails,
                       "silent": not key_ok and not fails, "fails": ";".join(fails),
                       "items_count_ok": count_ok, "items_found": found}


def ocr_recall(text: str, doc: dict) -> tuple[int, int]:
    flat = re.sub(r"\s", "", text)
    amounts = {round(a, 2) for a in amounts_in(text)}
    dates = set(dates_in(text))
    vals = [("s", get(doc, "supplier.ico")), ("s", doc.get("number")),
            *[("d", doc.get(k)) for k in ("issue_date", "duzp", "due_date")],
            ("a", doc.get("total")),
            *[("a", v[k]) for v in doc.get("vat") or [] for k in ("base", "vat")]]
    vals = list({(k, v) for k, v in vals if v not in (None, "")})
    found = sum(re.sub(r"\s", "", v) in flat if k == "s" else v in dates if k == "d"
                else round(v, 2) in amounts for k, v in vals)
    return found, len(vals)


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {"error": "missing"}


def _native(name: str) -> bool:
    pages = _load(TEXT / "layer" / f"{name}.json").get("pages") or []
    return bool(pages) and all(has_text(p) for p in pages)


def _frac(rows, key, native) -> str:
    sel = [r for r in rows if r["native"] == native]
    return f"{sum(bool(r[key]) for r in sel)}/{len(sel)}" if sel else "–"


def report(run: str) -> str:
    labels = {p.name: _load(Path(f"{p}.expected.json")) for p in pdfs(TESTDATA)}
    runs = {}
    if (RESULTS / run / "runs.csv").exists():
        with (RESULTS / run / "runs.csv").open() as fh:
            runs = {r["name"]: r for r in csv.DictReader(fh)}
    out = [f"# Extraction PoC results — run `{run}`", "",
           f"{len(labels)} files, {sum(map(_native, labels))} native / "
           f"{sum(not _native(n) for n in labels)} scans. Timing on CPUs per `cpus` column.", "",
           "## OCR", "", "| engine | value recall native | value recall scans | s/page (median) | "
           "load s | peak RSS MB | model MB |", "|---|---|---|---|---|---|---|"]
    for e in [d.name for d in sorted(TEXT.iterdir()) if d.is_dir()] if TEXT.exists() else []:
        rec = {True: [0, 0], False: [0, 0]}
        secs = []
        for name, lab in labels.items():
            r = _load(TEXT / e / f"{name}.json")
            secs += r.get("seconds", [])
            for doc in lab["documents"]:
                f, t = ocr_recall("\n".join(r.get("pages", [])), doc)
                rec[_native(name)][0] += f
                rec[_native(name)][1] += t
        m = runs.get(e, {})
        pct = lambda a: f"{a[0] / a[1]:.0%}" if a[1] else "–"
        out.append(f"| {e} | {pct(rec[True])} | {pct(rec[False])} | "
                   f"{statistics.median(secs) if secs else 0:.2f} | {m.get('load_seconds', '')} | "
                   f"{m.get('peak_rss_mb', '')} | {m.get('model_mb', '')} |")

    all_rows, per_combo = [], {}
    for d in sorted(p for p in (RESULTS / run).iterdir() if p.is_dir()):
        rows = []
        for name, lab in labels.items():
            r = _load(d / f"{name}.json")
            row = evaluate(r, lab) | {"combo": d.name, "file": name, "native": _native(name),
                                      "seconds": r.get("seconds", 0)}
            rows.append(row)
        per_combo[d.name] = rows
        all_rows += rows
    out += ["", "## Extraction", "", "| combo | auto-pass native | auto-pass scans | silent native | "
            "silent scans | errors | s/doc (median) | load s | peak RSS MB | VRAM MB |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c, rows in per_combo.items():
        m = runs.get(c, {})
        out.append(f"| {c} | {_frac(rows, 'pass', True)} | {_frac(rows, 'pass', False)} | "
                   f"{_frac(rows, 'silent', True)} | {_frac(rows, 'silent', False)} | "
                   f"{sum(r['error'] for r in rows)} | "
                   f"{statistics.median(r['seconds'] for r in rows):.1f} | {m.get('load_seconds', '')} | "
                   f"{m.get('peak_rss_mb', '')} | {m.get('peak_vram_mb', '')} |")
    fields = KEY + EXTRA + ["items_count_ok"]
    out += ["", "## Field accuracy (all files)", "", "| combo | " + " | ".join(fields) + " |",
            "|---" * (len(fields) + 1) + "|"]
    for c, rows in per_combo.items():
        out.append(f"| {c} | " + " | ".join(
            f"{sum(bool(r[f]) for r in rows) / len(rows):.0%}" for f in fields) + " |")

    if all_rows:
        with (RESULTS / run / "scores.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--report")
    a = ap.parse_args()
    md = report(a.run)
    print(md)
    if a.report:
        Path(a.report).write_text(md)
```

- [ ] **Step 4: Run it to see it pass**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/score.py tests/test_score.py
git commit -m "PoC: scoring against labels and markdown report"
```

---

### Task 12: Run the benchmark, write the results and update CLAUDE.md

**Files:**
- Create: `docs/poc-results.md` (generated, plus a hand-written "Findings" section)
- Modify: `CLAUDE.md` (Status and Commands sections)

- [ ] **Step 1: Run the whole test suite in all three venvs**

```bash
.venv/bin/python -m pytest -v
.venv-paddle/bin/python -m pytest tests/test_ocr.py -v
.venv-torch/bin/python -m pytest tests/test_ocr.py -v
```
Expected: everything passes, apart from engine tests skipped in the wrong venv.

- [ ] **Step 2: OCR stage, single core** (in the background; this takes roughly 1–3 h)

Run: `.venv/bin/python -m poc.bench R1 ocr --cpus 0`
Expected: 6 rows in `poc/results/R1/runs.csv`, all with `exit` 0.

- [ ] **Step 3: Read the OCR table and choose the two best sources for the LLMs**

Run: `.venv/bin/python -m poc.score R1`
Use the OCR table to choose the two engines with the highest scan value recall, and break ties by RSS. Note them as `<A>` and `<B>`.

- [ ] **Step 4: Extraction stage, single core** (in the background; the LLMs take several hours)

```bash
.venv/bin/python -m poc.bench R1 extract --extractors rules --cpus 0
.venv/bin/python -m poc.bench R1 extract --extractors qwen3.5-2b qwen3.5-4b --sources <A> <B> --cpus 0
```
Running every extractor on every OCR source is unnecessary. The cache reuses results for native PDFs across sources, so native documents run only once per model.

- [ ] **Step 5: GPU reference**

Run: `.venv/bin/python -m poc.bench R1 vlm`
Expected: `peak_vram_mb` ≤ 21504 in `runs.csv`. If it is higher, follow the fallback in Task 9, Step 4, and rerun.

- [ ] **Step 6: Generate the report and add findings**

Run: `.venv/bin/python -m poc.score R1 --report docs/poc-results.md`
Then append a hand-written `## Findings` section with:
- the pipeline the decision rule selects (or "none — best is X, needs Y MB / Z s");
- the distance from the VLM reference;
- the worst files per combo by file ID and check message, taken from `poc/results/R1/scores.csv`;
- whether the hard cases (ZA044, the one shop's receipts, ZA003, ZA017) failed for every engine, which would trigger round 2 (deskew/crop);
- any new accounting questions, which also go into `docs/otazky-pro-hanu.md`.

Check that the report contains no client data beyond file IDs: `grep -E '[0-9]{8}' docs/poc-results.md` must return nothing.

- [ ] **Step 7: Update CLAUDE.md**

Replace the "Status" section with:
```markdown
## Status

Extraction PoC (`poc/`) done — see `docs/poc-results.md` for the chosen pipeline. No web app yet.

## Commands

- Setup: `python3 -m venv .venv && .venv/bin/pip install -r poc/requirements.txt` (plus `.venv-paddle` / `.venv-torch` from `poc/requirements-{paddle,torch}.txt` for those OCR engines). Models go to `poc/models/` (see the plan in `docs/superpowers/plans/2026-10-28-extraction-poc.md`, Tasks 3 and 8).
- Tests: `.venv/bin/python -m pytest -m "not slow"`; single test: `.venv/bin/python -m pytest tests/test_checks.py::test_ico_checksum`.
- Benchmark: `.venv/bin/python -m poc.bench RUN {ocr|extract|vlm} --cpus 0`, then `.venv/bin/python -m poc.score RUN --report docs/poc-results.md`.
- Labels: `testdata/zari/*.expected.json` (gitignored, schema in `docs/label-schema.md`).
```

- [ ] **Step 8: Commit**

```bash
git add docs/poc-results.md docs/otazky-pro-hanu.md CLAUDE.md
git commit -m "PoC: benchmark results and findings"
```
