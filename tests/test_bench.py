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


def test_measure_reports_cpu_time_so_escaped_threads_show():
    # a thread that resets its own affinity (as ONNX Runtime does) burns more CPU than wall x cores
    code = """
import hashlib, os, threading, time
data = bytes(2**20)
def spin(core):  # hashlib releases the GIL, so both threads really run
    os.sched_setaffinity(0, {core})
    t = time.time()
    while time.time() - t < 1: hashlib.sha256(data).digest()
ths = [threading.Thread(target=spin, args=(c,)) for c in (1, 2)]
[th.start() for th in ths]; [th.join() for th in ths]
"""
    r = measure([sys.executable, "-c", code], {0})
    assert r["cpu_s"] > 1.5 * r["wall_s"]


def test_ocr_engines_stay_on_their_cores(tmp_path):
    from pathlib import Path
    za040 = Path(__file__).parent.parent / "testdata/zari/ZA040.pdf"
    if not za040.exists():
        import pytest
        pytest.skip("testdata missing")
    for engine in ("tesseract", "rapidocr"):
        r = measure([sys.executable, "-m", "poc.ocr", engine, tmp_path / engine, za040], {0})
        assert r["exit"] == 0 and r["cpu_s"] <= 1.2 * r["wall_s"], (engine, r)
