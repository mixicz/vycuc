import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from poc.ocr import ENGINES, has_text, join_boxes, render, text_layer

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
    assert "615,24" in text  # wiring check; misreads like "1l 615,24" are benchmark results


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


@pytest.mark.skipif(not ZA001.exists(), reason="testdata missing")
def test_render_is_lazy():
    # all pages at 300 DPI held at once would inflate every engine's measured RSS
    pages = render(ZA001)
    assert not isinstance(pages, list)
    assert next(iter(pages)).size[0] > 2000
