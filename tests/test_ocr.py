import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from poc.ocr import (ENGINES, MAX_PIXELS, has_text, join_boxes, layer_words, rapid_words, render,
                     text_layer, text_words)

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
    ico = json.loads(Path(f"{ZA001}.expected.json").read_text())["documents"][0]["supplier"]["ico"]
    assert ico in text_layer(ZA001)[0]


@pytest.mark.parametrize("engine", MODULE)
def test_engine_reads_czech_sample(engine):
    if importlib.util.find_spec(MODULE[engine]) is None:
        pytest.skip(f"{engine} not installed in this venv")
    text, _ = ENGINES[engine]()(sample())
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


@pytest.mark.skipif(not (ZA001.parent / "ZA003.pdf").exists(), reason="testdata missing")
def test_render_caps_oversized_pages_at_a4_300dpi_area():
    # ZA003 is a 1176x6626 pt phone screenshot; 300 DPI would upscale it 4x to 390 MB
    (page,) = render(ZA001.parent / "ZA003.pdf")
    assert page.size[0] * page.size[1] <= MAX_PIXELS * 1.01  # pdfium rounds dims up
    a4 = next(iter(render(ZA001)))
    assert a4.size[0] > 2400  # normal A4 pages keep full 300 DPI


def test_layer_words_split_on_spaces_and_flip_y():
    # pdfium boxes are (left, bottom, right, top) in points, origin bottom-left
    chars = [("I", (10, 90, 14, 100)), ("Č", (14, 90, 20, 100)), (" ", (20, 90, 22, 90)),
             ("1", (22, 90, 26, 100)), ("\r", (26, 90, 26, 90)), ("\n", (26, 90, 26, 90)),
             ("K", (10, 40, 16, 50))]
    assert layer_words(chars, 100, 200) == [[0.1, 0.5, 0.2, 0.55, "IČ", 1.0, 0],
                                            [0.22, 0.5, 0.26, 0.55, "1", 1.0, 0],
                                            [0.1, 0.75, 0.16, 0.8, "K", 1.0, 1]]


def test_rapid_words_normalize_boxes_and_number_lines():
    lines = [(("ACTION", 0.9, [[0, 0], [50, 0], [50, 10], [0, 10]]),),
             (("4", 0.8, [[0, 20], [10, 20], [10, 30], [0, 30]]),
              ("615,24", 0.7, [[12, 21], [40, 20], [40, 30], [12, 31]]))]
    assert rapid_words(lines, 100, 50) == [[0.0, 0.0, 0.5, 0.2, "ACTION", 0.9, 0],
                                           [0.0, 0.4, 0.1, 0.6, "4", 0.8, 1],
                                           [0.12, 0.4, 0.4, 0.62, "615,24", 0.7, 1]]


@pytest.mark.skipif(not ZA001.exists(), reason="testdata missing")
def test_layer_words_match_text_layer():
    (words, *_), (text, *_) = text_words(ZA001), text_layer(ZA001)
    assert " ".join(w[4] for w in words) == " ".join(text.split())
    assert all(0 <= v <= 1 for w in words for v in w[:4])
