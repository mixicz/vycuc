#!/usr/bin/env python3
"""Block commits/pushes that contain client data from testdata/.

The forbidden values are read at run time from the local, gitignored test set, so the list itself
never enters git: label values (supplier name/IČO/DIČ/street, document number, VS),
testdata/client_ico and any extra terms in testdata/forbidden.txt (one per line).

    leakcheck.py staged   # pre-commit: added lines of the index
    leakcheck.py push     # pre-push: diffs and messages of commits not yet on any remote
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                           check=True).stdout.strip())
DATA = ROOT / "testdata"


def tokens() -> set[str]:
    out = set()
    for f in DATA.rglob("*.expected.json"):
        for d in json.loads(f.read_text()).get("documents", []):
            s = d.get("supplier") or {}
            out |= {str(v).strip() for v in (s.get("name"), s.get("ico"), s.get("dic"), s.get("street"),
                                              d.get("number"), d.get("variabilni_symbol")) if v}
    for f in ("client_ico", "forbidden.txt"):
        if (DATA / f).exists():
            out |= {l.strip() for l in (DATA / f).read_text().splitlines()}
    return {t for t in out if len(t) >= 5}  # ponytail: shorter values (e.g. "Alza") match too much prose


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=ROOT).stdout


def chunks(mode: str):
    """Yield (where, text) pairs to scan."""
    if mode == "staged":
        yield from _added(git("diff", "--cached", "-U0", "--no-color"), "index")
        return
    for line in sys.stdin:  # <local ref> <local sha> <remote ref> <remote sha>
        sha = line.split()[1]
        if set(sha) == {"0"}:
            continue  # branch deletion
        for c in git("rev-list", sha, "--not", "--remotes").split():
            yield f"commit {c[:7]} message", git("log", "-1", "--format=%B", c)
            yield from _added(git("show", "-U0", "--no-color", "--format=", c), f"commit {c[:7]}")


def _added(diff: str, where: str):
    path = "?"
    for l in diff.splitlines():
        if l.startswith("+++ "):
            path = l[6:] if l.startswith("+++ b/") else l[4:]
        elif l.startswith("+"):
            yield f"{where} {path}", l[1:]


def main(mode: str) -> int:
    toks = tokens()
    if not toks:
        return 0  # no local test set: nothing to compare against
    hits = sorted({(w, t) for w, text in chunks(mode) for t in toks if t in text})
    for w, t in hits:
        print(f"leakcheck: {w}: contains test-set value {t[:2]}…{t[-2:]}", file=sys.stderr)
    if hits:
        print("leakcheck: client data must not enter git; use invented values (see CLAUDE.md)",
              file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
