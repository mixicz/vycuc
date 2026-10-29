import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / ".githooks" / "leakcheck.py"


def _repo(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", "-C", tmp_path, *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t"), git("config", "user.name", "t")
    data = tmp_path / "testdata" / "set"
    data.mkdir(parents=True)
    (data / "a.pdf.expected.json").write_text(json.dumps({"documents": [{
        "number": "FV-778899", "variabilni_symbol": "778899",
        "supplier": {"name": "Tajná Firma s.r.o.", "ico": "12345679", "dic": "CZ12345679"}}]}))
    (tmp_path / "testdata" / "client_ico").write_text("76543218\n")
    (tmp_path / "testdata" / "forbidden.txt").write_text("Alicia\n")
    (tmp_path / ".gitignore").write_text("testdata/\n")
    return git


def _check(tmp_path, text):
    git = _repo(tmp_path)
    (tmp_path / "x.md").write_text(text)
    git("add", "x.md")
    return subprocess.run([sys.executable, HOOK, "staged"], cwd=tmp_path, capture_output=True, text=True)


def test_blocks_label_values_client_ico_and_extra_terms(tmp_path):
    for leak in ("supplier Tajná Firma s.r.o.", "IČO 12345679", "VS 778899", "client 76543218", "Alicia"):
        r = _check(tmp_path / leak.split()[-1], leak)
        assert r.returncode == 1 and "x.md" in r.stderr, leak


def test_allows_clean_text(tmp_path):
    assert _check(tmp_path, "Faktura č. 20269001, IČO 24680249").returncode == 0


def test_push_scans_unpushed_commits_and_messages(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "x.md").write_text("clean\n")
    git("add", "x.md")
    git("commit", "-q", "--no-verify", "-m", "fix for Tajná Firma s.r.o.")  # leak only in the message
    sha = subprocess.run(["git", "-C", tmp_path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    r = subprocess.run([sys.executable, HOOK, "push"], cwd=tmp_path, capture_output=True, text=True,
                       input=f"refs/heads/main {sha.strip()} refs/heads/main {'0' * 40}\n")
    assert r.returncode == 1 and "message" in r.stderr
