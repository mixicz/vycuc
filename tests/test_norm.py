from poc.norm import amounts_in, dates_in, fold


def test_fold_keeps_length_and_strips_diacritics():
    s = "Datum zdanitelného plnění: Čtvrtek, IČO"
    assert fold(s) == "datum zdanitelneho plneni: ctvrtek, ico"
    assert len(fold(s)) == len(s)


def test_amounts_czech_and_english_formats():
    text = "12 %  4 120,75  494,49  4 615,24 Kč; 1105.00; 1.105,00; -50,00"
    assert amounts_in(text) == [4120.75, 494.49, 4615.24, 1105.00, 1105.00, -50.00]


def test_amounts_ignore_dates_times_and_integers():
    assert amounts_in("15.09.2026 14:22 qty 3 CELKEM 7,80") == [7.80]


def test_dates_common_formats():
    text = "31.07.2026, 1. 8. 2026, 14/08/26, 2026-09-15, 31.02.2026"
    assert dates_in(text) == ["2026-07-31", "2026-08-01", "2026-08-14", "2026-09-15"]


def test_time_after_yearless_date_is_not_a_year():
    assert dates_in("Pondělí 31.08. 14:22") == []


def test_dates_with_month_names():
    text = "4. září 2026 / September 4, 2026 / 4 Sep 2026 / 1. července 2026 / 2. června 2026"
    assert dates_in(text) == ["2026-09-04", "2026-09-04", "2026-09-04", "2026-07-01", "2026-06-02"]
