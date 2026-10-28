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
MAX_PIXELS = 2480 * 3508  # A4 at 300 DPI; photo-PDFs with huge pages are scaled down to this
MODELS = Path(__file__).parent / "models"


def text_layer(pdf: Path) -> list[str]:
    return [page.get_textpage().get_text_bounded() for page in pdfium.PdfDocument(pdf)]


def render(pdf: Path, dpi: int = DPI):
    """Yield pages one at a time, so peak RSS reflects the engine, not the page count."""
    for p in pdfium.PdfDocument(pdf):
        w, h = p.get_size()
        scale = min(dpi / 72, (MAX_PIXELS / (w * h)) ** 0.5)
        yield p.render(scale=scale).to_pil().convert("RGB")


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
    eng = PaddleOCR(lang="cs", enable_mkldnn=False)  # oneDNN path crashes in paddle 3.3.1 (PIR attr)

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
