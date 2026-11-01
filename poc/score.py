"""Score a benchmark run against the labels and render the markdown report.

    .venv/bin/python -m poc.score RUN [--report docs/poc-results.md] [--errors poc/results/RUN-errors.md]
"""
import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
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
    rows = ((float(v["rate"]), float(v["base"]), float(v["vat"])) for v in lines)
    return sorted(r for r in rows if r[1] or r[2])  # an all-zero line carries no information


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


def _duzp(d: dict) -> dict:
    return d | {"duzp": d.get("duzp") or d.get("issue_date")}  # export fills it the same way


def evaluate(result: dict, labels: dict, lookup=ares) -> dict:
    wants = labels["documents"]
    docs = result.get("documents") or []
    base = {"error": not docs, "n_docs": len(docs), "n_docs_want": len(wants)}
    if not docs:
        return base | {f: False for f in KEY + EXTRA} | {
            "key_ok": False, "pass": False, "silent": False, "fails": result.get("error", "no output"),
            "items_count_ok": False, "items_found": 0.0}
    got, fails = check(docs[0], lookup)
    scored = [{f: same(f, get(_duzp(got), f), get(_duzp(w), f)) for f in KEY + EXTRA} for w in wants]
    i = max(range(len(wants)), key=lambda j: sum(scored[j][f] for f in KEY))
    s = scored[i]
    key_ok = all(s[f] for f in KEY)
    count_ok, found = _items(got.get("items"), wants[i].get("items") or [])
    return base | s | {"doc_index": i, "key_ok": key_ok, "pass": key_ok and not fails,
                       "silent": not key_ok and not fails, "fails": ";".join(fails),
                       "items_count_ok": count_ok, "items_found": found}


DATES = ("issue_date", "duzp", "due_date")


def _ocr_values(doc: dict) -> list[tuple[str, object]]:
    vals = [("supplier.ico", get(doc, "supplier.ico")), ("number", doc.get("number")),
            *[(k, doc.get(k)) for k in DATES], ("total", doc.get("total")),
            *[(f"vat.{k}", v[k]) for v in doc.get("vat") or [] for k in ("base", "vat")]]
    seen = {}
    for f, v in vals:
        if v not in (None, "") and v not in seen:
            seen[v] = f
    return [(f, v) for v, f in seen.items()]


def ocr_missing(text: str, doc: dict) -> list[tuple[str, str]]:
    """Label values of key fields that do not occur in the OCR text, as (field, value)."""
    flat = re.sub(r"\s", "", text)
    amounts = {round(a, 2) for a in amounts_in(text)}
    dates = set(dates_in(text))
    found = lambda f, v: (v in dates if f in DATES else round(v, 2) in amounts
                          if isinstance(v, (int, float)) else re.sub(r"\s", "", v) in flat)
    return [(f, f"{v:.2f}" if isinstance(v, (int, float)) else v)
            for f, v in _ocr_values(doc) if not found(f, v)]


def ocr_recall(text: str, doc: dict) -> tuple[int, int]:
    n = len(_ocr_values(doc))
    return n - len(ocr_missing(text, doc)), n


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {"error": "missing"}


def _native(name: str) -> bool:
    pages = _load(TEXT / "layer" / f"{name}.json").get("pages") or []
    return bool(pages) and all(has_text(p) for p in pages)


def _frac(rows, key, native) -> str:
    sel = [r for r in rows if r["native"] == native]
    return f"{sum(bool(r[key]) for r in sel)}/{len(sel)}" if sel else "–"


def _labels() -> dict:
    return {p.name: _load(Path(f"{p}.expected.json")) for p in pdfs(TESTDATA)}


def report(run: str) -> str:
    labels = _labels()
    runs = {}
    if (RESULTS / run / "runs.csv").exists():
        with (RESULTS / run / "runs.csv").open() as fh:
            runs = {r["name"]: r for r in csv.DictReader(fh)}
    out = [f"# Extraction PoC results — run `{run}`", "",
           f"{len(labels)} files, {sum(map(_native, labels))} native / "
           f"{sum(not _native(n) for n in labels)} scans. Timing on CPUs per `cpus` column.", "",
           "## OCR", "", "| engine | value recall native | value recall scans | s/page (median) | "
           "load s | peak RSS MB | model MB | file errors | exit |", "|---|---|---|---|---|---|---|---|---|"]
    for e in [d.name for d in sorted(TEXT.iterdir()) if d.is_dir()] if TEXT.exists() else []:
        rec = {True: [0, 0], False: [0, 0]}
        secs, errors = [], 0
        for name, lab in labels.items():
            r = _load(TEXT / e / f"{name}.json")
            secs += r.get("seconds", [])
            errors += "error" in r
            for doc in lab["documents"]:
                f, t = ocr_recall("\n".join(r.get("pages", [])), doc)
                rec[_native(name)][0] += f
                rec[_native(name)][1] += t
        m = runs.get(e, {})
        pct = lambda a: f"{a[0] / a[1]:.0%}" if a[1] else "–"
        out.append(f"| {e} | {pct(rec[True])} | {pct(rec[False])} | "
                   f"{statistics.median(secs) if secs else 0:.2f} | {m.get('load_seconds', '')} | "
                   f"{m.get('peak_rss_mb', '')} | {m.get('model_mb', '')} | {errors} | {m.get('exit', '')} |")

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


def _show(field: str, v) -> str:
    if v in (None, "", []):
        return "–"
    if field == "vat":
        try:
            return "; ".join(f"{r:g}%: {b:.2f} + {t:.2f}" for r, b, t in _vat(v)) or "–"
        except (TypeError, ValueError, KeyError):
            pass
    return str(v).replace("|", "/").replace("\n", " ")


def errors(run: str) -> str:
    """Every wrong value per field and every OCR miss, for weak-point analysis. Contains client data."""
    labels = _labels()
    kind = {n: "native" if _native(n) else "scan" for n in labels}
    out = [f"# Errors — run `{run}`", "",
           "Generated by `poc.score --errors`. Contains client data: keep it out of git.", "",
           "## OCR", "", "Label values of key fields not found in the engine's text. The text layer is "
           "left out for scans.", "", "| file | kind | field | value | missed by |", "|---|---|---|---|---|"]
    engines = [d.name for d in sorted(TEXT.iterdir()) if d.is_dir()] if TEXT.exists() else []
    for name, lab in labels.items():
        missed = {}
        for e in engines:
            if e == "layer" and kind[name] == "scan":
                continue
            r = _load(TEXT / e / f"{name}.json")
            if "error" in r:
                missed.setdefault(("file error", r["error"]), []).append(e)
                continue
            for doc in lab["documents"]:
                for fv in ocr_missing("\n".join(r.get("pages", [])), doc):
                    missed.setdefault(fv, []).append(e)
        out += [f"| {name} | {kind[name]} | {f} | {_show(f, v)} | {', '.join(es)} |"
                for (f, v), es in missed.items()]

    combos = [d for d in sorted((RESULTS / run).iterdir()) if d.is_dir()]
    wrong = {f: {} for f in KEY + EXTRA + ["items"]}  # field -> file -> {"want", "outs": value -> [combo]}
    failed, hard = [], {n: {} for n in labels}
    for d in combos:
        for name, lab in labels.items():
            r = _load(d / f"{name}.json")
            row = evaluate(r, lab, ares)
            if row["error"]:
                failed.append(f"| {name} | {kind[name]} | {d.name} | {_show('', row['fails'])[:200]} |")
                continue
            got = _duzp(check(r["documents"][0], ares)[0])
            want = lab["documents"][row["doc_index"]]
            tag = d.name + ("!" if row["silent"] else "")
            for f in KEY + EXTRA:
                if not row[f]:
                    e = wrong[f].setdefault(name, {"want": _show(f, get(_duzp(want), f)), "outs": {}})
                    e["outs"].setdefault(_show(f, get(got, f)), []).append(tag)
                    hard[name][f] = hard[name].get(f, 0) + 1
            if not row["items_count_ok"]:
                e = wrong["items"].setdefault(name, {"want": f"{len(want.get('items') or [])} items",
                                                     "outs": {}})
                e["outs"].setdefault(f"{len(got.get('items') or [])} items", []).append(tag)
    out += ["", "## Extraction", "", "Wrong values per field, grouped by output. `!` marks a silent error "
            "(all checks passed). Expected values are the matched label document, with a missing DUZP "
            "filled from the issue date on both sides.", ""]
    for f, files in wrong.items():
        out += [f"### {f}", "", f"{len(files)} files.", "", "| file | kind | expected | got: combos |",
                "|---|---|---|---|"]
        for name, e in files.items():
            out.append(f"| {name} | {kind[name]} | {e['want']} | "
                       + "; ".join(f"{v}: {', '.join(cs)}" for v, cs in e["outs"].items()) + " |")
        out.append("")
    out += ["## Failed runs", "", "| file | kind | combo | error |", "|---|---|---|---|", *failed, "",
            "## Hardest files", "", f"Wrong fields, counted over {len(combos)} combos.", "",
            "| file | kind | wrong fields (combos) |", "|---|---|---|"]
    for name, fs in sorted(hard.items(), key=lambda x: -sum(x[1].values())):
        if fs:
            out.append(f"| {name} | {kind[name]} | "
                       + ", ".join(f"{f} {c}" for f, c in sorted(fs.items(), key=lambda x: -x[1])) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--report")
    ap.add_argument("--errors", help="write the error listing here; must be a git-ignored path")
    a = ap.parse_args()
    if a.errors:
        if subprocess.run(["git", "check-ignore", "-q", a.errors]).returncode != 0:
            sys.exit(f"{a.errors} is not git-ignored; the error listing contains client data")
        Path(a.errors).write_text(errors(a.run))
    md = report(a.run)
    print(md)
    if a.report:
        Path(a.report).write_text(md)
