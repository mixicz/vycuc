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
