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
           "peak_vram_mb", "cpu_s"]


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
            "exit": p.returncode,
            "cpu_s": round(ru.ru_utime + ru.ru_stime, 1)}  # > wall_s x cores = threads escaped pinning


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
