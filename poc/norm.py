"""Czech amount/date normalisation, shared by extractors and scoring."""
import re
import unicodedata
from datetime import date

# Two decimals required: integers are too ambiguous (quantities, years, codes).
_AMOUNT = re.compile(
    r"(?<![\d.,])-?(?:\d{1,3}(?:[  .,']\d{3})+|\d+)[.,]\d{2}(?!\d|[.,]\d)"
)
_DATE = re.compile(
    r"(?<![\d.])(\d{1,2})\s?[./-]\s?(\d{1,2})\s?[./-]\s?(\d{4}|\d{2})(?![\d.:])"
    r"|(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"
)

# on folded text: "4. zari 2026", "4 Sep 2026", "September 4, 2026"
_NAMED = re.compile(r"(?<!\d)(\d{1,2})\.?\s*([a-z]{3,})\.?\s+(\d{4})(?!\d)"
                    r"|\b([a-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})(?!\d)")
_CS = ["leden ledna", "unor unora", "brezen brezna", "duben dubna", "kveten kvetna", "cerven cervna",
       "cervenec cervence", "srpen srpna", "zari", "rijen rijna", "listopad listopadu",
       "prosinec prosince"]
_EN = "january february march april may june july august september october november december".split()
_MONTHS = {w: i for i, forms in enumerate(_CS, 1) for w in forms.split()} \
    | {w: i for i, m in enumerate(_EN, 1) for w in (m, m[:3])} | {"sept": 9}


def month_of(word: str) -> int | None:
    """Month number from a folded Czech or English month name (whole word, no stems)."""
    return _MONTHS.get(word)


def fold(s: str) -> str:
    """Lower-case without diacritics, one char per char (spans stay valid)."""
    return "".join(unicodedata.normalize("NFKD", c)[0] for c in s).lower()


def amounts_in(s: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(s):
        t = m.group()
        v = int(re.sub(r"\D", "", t)) / 100
        out.append(-v if t.startswith("-") else v)
    return out


def dates_in(s: str) -> list[str]:
    out = []
    for m in _NAMED.finditer(fold(s)):
        d, name, y = m.group(1, 2, 3) if m.group(1) else (m.group(5), m.group(4), m.group(6))
        if mo := month_of(name):
            try:
                out.append((m.start(), date(int(y), mo, int(d)).isoformat()))
            except ValueError:
                pass
    for m in _DATE.finditer(s):
        if m.group(4):
            y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
        else:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y += 2000 if y < 100 else 0
        try:
            out.append((m.start(), date(y, mo, d).isoformat()))
        except ValueError:
            pass
    return [d for _, d in sorted(out)]
