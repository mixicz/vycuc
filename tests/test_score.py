from poc.score import evaluate, ocr_recall, same

WANT = {"type": "receivedInvoice", "payment": "transfer", "number": "20269001",
        "variabilni_symbol": None, "issue_date": "2026-07-31", "duzp": "2026-07-31",
        "due_date": "2026-08-14", "currency": "CZK",
        "supplier": {"name": "Kavárna Vzor s.r.o.", "ico": "12345679", "dic": "CZ12345679"},
        "vat": [{"rate": 12, "base": 4120.75, "vat": 494.49}], "total": 4615.24,
        "to_pay": 4615.24, "items": [{"text": "koule", "base": 2000.00, "total": None}]}
LABELS = {"file": "ZA001.pdf", "documents": [WANT]}
ARES = {"obchodniJmeno": "Kavárna Vzor s.r.o.", "dic": "CZ12345679", "sidlo": {}}
lookup = lambda ico: ARES if ico == "12345679" else None


def test_same_null_semantics():
    assert same("variabilni_symbol", None, None)
    assert not same("variabilni_symbol", "123", None)
    assert not same("number", None, "20269001")
    assert same("number", "2026 9001", "20269001")
    assert same("total", "4615.24", 4615.24) and not same("total", 4615.0, 4615.24)
    assert same("supplier.name", "Kavarna vzor s. r. o.", "Kavárna Vzor s.r.o.")
    assert same("vat", [{"rate": 12.0, "base": 4120.75, "vat": 494.49}], WANT["vat"])
    assert not same("vat", [], WANT["vat"]) and not same("vat", [{"rate": None}], WANT["vat"])


def test_evaluate_pass_silent_error():
    good = {k: v for k, v in WANT.items() if k != "supplier"} | {"supplier": {"ico": "12345679"}}
    row = evaluate({"documents": [good]}, LABELS, lookup)
    assert row["pass"] and not row["silent"] and row["supplier.name"]  # name filled from ARES

    wrong_date = good | {"due_date": "2026-08-15"}  # checks can't see this
    row = evaluate({"documents": [wrong_date]}, LABELS, lookup)
    assert row["silent"] and not row["pass"]

    caught = good | {"total": 999.0}  # VAT sum check catches it
    row = evaluate({"documents": [caught]}, LABELS, lookup)
    assert not row["pass"] and not row["silent"] and "vat sum != total" in row["fails"]

    row = evaluate({"error": "bad model output"}, LABELS, lookup)
    assert row["error"] and not row["pass"] and not row["silent"]


def test_ocr_recall():
    text = "Faktura 2026 9001 IČO 12345679 vystaveno 31.7.2026 splatnost 14. 8. 2026\n" \
           "12 % 4 120,75 494,49 Celkem 4 615,24"
    assert ocr_recall(text, WANT) == (7, 7)
    assert ocr_recall("nothing", WANT) == (0, 7)


def test_zero_vat_line_is_no_information():
    assert same("vat", WANT["vat"] + [{"rate": 0, "base": 0.0, "vat": 0.0}], WANT["vat"])
    assert not same("vat", [{"rate": 0, "base": 8.1, "vat": 0.0}], [])  # non-zero base still counts


def test_missing_duzp_equals_issue_date():
    # export fills a missing DUZP with the issue date, so both spellings mean the same
    good = {k: v for k, v in WANT.items() if k != "supplier"} | {"supplier": {"ico": "12345679"}}
    no_duzp = {"documents": [WANT | {"duzp": None}]}
    assert evaluate({"documents": [good]}, no_duzp, lookup)["duzp"]
    assert evaluate({"documents": [good | {"duzp": None}]}, LABELS, lookup)["duzp"]
    assert not evaluate({"documents": [good | {"duzp": "2026-08-14"}]}, no_duzp, lookup)["duzp"]


def test_report_shows_ocr_errors_and_exit(tmp_path, monkeypatch):
    # a crashed or killed engine must not look like low recall
    import json
    from poc import score
    data, text, results = tmp_path / "data", tmp_path / "text", tmp_path / "results"
    data.mkdir()
    (data / "a.pdf").write_text("")
    (data / "a.pdf.expected.json").write_text(json.dumps(LABELS))
    (text / "layer").mkdir(parents=True)
    (text / "layer" / "a.pdf.json").write_text(json.dumps({"pages": [""], "seconds": [0]}))
    (text / "tesseract").mkdir()
    (text / "tesseract" / "a.pdf.json").write_text(json.dumps({"error": "boom"}))
    (results / "R").mkdir(parents=True)
    (results / "R" / "runs.csv").write_text("stage,name,exit\nocr,tesseract,-9\n")
    monkeypatch.setattr(score, "TESTDATA", data)
    monkeypatch.setattr(score, "TEXT", text)
    monkeypatch.setattr(score, "RESULTS", results)
    row = next(l for l in score.report("R").splitlines() if l.startswith("| tesseract"))
    assert "| 1 | -9 |" in row
