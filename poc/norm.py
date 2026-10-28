"""Czech amount/date normalisation, shared by extractors and scoring."""
import re
import unicodedata
from datetime import date

# Two decimals required: integers are too ambiguous (quantities, years, codes).
_AMOUNT = re.compile(
    r"(?<![\d.,])-?(?:\d{1,3}(?:[  .,']\d{3})+|\d+)[.,]\d{2}(?!\d|[.,]\d)"
)
_DATE = re.compile(
    r"(?<![\d.])(\d{1,2})\s?[./-]\s?(\d{1,2})\s?[./-]\s?(\d{4}|\d{2})(?![\d.])"
    r"|(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"
)


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
    for m in _DATE.finditer(s):
        if m.group(4):
            y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
        else:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y += 2000 if y < 100 else 0
        try:
            out.append(date(y, mo, d).isoformat())
        except ValueError:
            pass
    return out
