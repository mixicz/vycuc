import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from poc.extract import LLMS, SCHEMA, VRAM_LIMIT, load_extractor, page_texts, parse_documents, rules, vram_sampler

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


def test_vram_sampler_reports_zero_for_unloaded_model():
    stop = vram_sampler("no-such-model:latest")
    assert stop()["peak_vram_bytes"] == 0


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
    assert 10 * 2**30 < meta["finish"]()["peak_vram_bytes"] <= VRAM_LIMIT


@pytest.mark.slow
@pytest.mark.skipif(not (_ollama_up() and (ROOT / "testdata/zari/ZA001.pdf").exists()),
                    reason="needs ollama and testdata")
def test_vlm_baseline_excludes_already_loaded_model():
    from poc.extract import _ollama
    _ollama("/api/generate", {"model": "qwen3.8:27b-q4_K_M", "keep_alive": "5m"})  # preload
    run, meta = load_extractor("vlm:qwen3.8:27b-q4_K_M")
    run(ROOT / "testdata/zari/ZA001.pdf")
    assert meta["finish"]()["peak_vram_bytes"] > 10 * 2**30


@pytest.mark.slow
@pytest.mark.skipif(not _ollama_up(), reason="needs ollama")
def test_vram_sampler_counts_only_the_model_not_the_desktop():
    # the card is shared with the desktop; only ollama's own allocation for the model counts
    from poc.extract import _ollama
    model = "qwen3.8:27b-q4_K_M"
    _ollama("/api/generate", {"model": model, "keep_alive": "1m"})
    with urllib.request.urlopen("http://localhost:11434/api/ps") as r:
        expected = next(m["size_vram"] for m in json.load(r)["models"] if m["name"] == model)
    assert vram_sampler(model)()["peak_vram_bytes"] == expected


ROUNDED = INVOICE.replace("Celkem k úhradě: 4 615,24 Kč", "Celkem s DPH: 4 615,24\nZaokrouhlení: -0,24\nCelkem k úhradě: 4 615,00 Kč")


@pytest.mark.slow
@pytest.mark.skipif(not LLMS["qwen3.5-4b"].exists(), reason="model not downloaded")
def test_llm_follows_label_conventions():
    run, _ = load_extractor("qwen3.5-4b")
    (inv,) = run(ROUNDED)
    assert inv["duzp"] == "2026-07-31"  # DUZP, not splatnost
    assert inv["total"] == 4615.24 and inv["to_pay"] == 4615.0  # total before rounding
    (rec,) = run(RECEIPT)
    assert rec["duzp"] == rec["issue_date"] == "2026-09-15"  # receipts: DUZP = issue date
