"""Spell out integers in English and Hebrew.

Hebrew numbers are gendered. Plain counting, clock times, years and radio
frequencies use the feminine form, which is the default here. Thousands are
counted in the masculine ("אחד עשר אלף"), handled internally.
"""
from __future__ import annotations

# ---------------------------------------------------------------- English
_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _en_below_100(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    tens, ones = divmod(n, 10)
    return _EN_TENS[tens] + ("-" + _EN_ONES[ones] if ones else "")


def _en_below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(_EN_ONES[hundreds] + " hundred")
    if rest or not hundreds:
        parts.append(_en_below_100(rest))
    return " ".join(parts)


def _english(n: int) -> str:
    if n < 1000:
        return _en_below_1000(n)
    parts = []
    for value, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        count, n = divmod(n, value)
        if count:
            parts.append(_en_below_1000(count) + " " + name)
    if n:
        parts.append(_en_below_1000(n))
    return " ".join(parts)


# ---------------------------------------------------------------- Hebrew
_HE_F = ["אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע", "עשר"]
_HE_M = ["אפס", "אחד", "שניים", "שלושה", "ארבעה", "חמישה", "שישה", "שבעה", "שמונה", "תשעה", "עשרה"]
_HE_F_TEENS = ["אחת עשרה", "שתים עשרה", "שלוש עשרה", "ארבע עשרה", "חמש עשרה", "שש עשרה",
               "שבע עשרה", "שמונה עשרה", "תשע עשרה"]
_HE_M_TEENS = ["אחד עשר", "שנים עשר", "שלושה עשר", "ארבעה עשר", "חמישה עשר", "שישה עשר",
               "שבעה עשר", "שמונה עשר", "תשעה עשר"]
_HE_TENS = ["", "", "עשרים", "שלושים", "ארבעים", "חמישים", "שישים", "שבעים", "שמונים", "תשעים"]
_HE_HUNDREDS = ["", "מאה", "מאתיים", "שלוש מאות", "ארבע מאות", "חמש מאות", "שש מאות",
                "שבע מאות", "שמונה מאות", "תשע מאות"]
# 3,000-10,000 use the construct form + plural "אלפים"
_HE_THOUSANDS = ["", "אלף", "אלפיים", "שלושת אלפים", "ארבעת אלפים", "חמשת אלפים", "ששת אלפים",
                 "שבעת אלפים", "שמונת אלפים", "תשעת אלפים", "עשרת אלפים"]


def _he_groups_below_1000(n: int, feminine: bool) -> list[str]:
    """Word groups for 1..999; the caller decides where the 'and' goes."""
    ones_words = _HE_F if feminine else _HE_M
    teens = _HE_F_TEENS if feminine else _HE_M_TEENS
    groups = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        groups.append(_HE_HUNDREDS[hundreds])
    if 11 <= rest <= 19:
        groups.append(teens[rest - 11])
    else:
        tens, ones = divmod(rest, 10)
        if rest == 10:
            groups.append(ones_words[10])
        else:
            if tens:
                groups.append(_HE_TENS[tens])
            if ones:
                groups.append(ones_words[ones])
    return groups


def _he_join(groups: list[str]) -> str:
    """Hebrew puts 'ו' (and) on the last group: מאה עשרים ושלוש."""
    if len(groups) > 1:
        groups = groups[:-1] + ["ו" + groups[-1]]
    return " ".join(groups)


def _hebrew(n: int, feminine: bool) -> str:
    if n == 0:
        return _HE_F[0]
    groups: list[str] = []
    millions, n = divmod(n, 1_000_000)
    thousands, rest = divmod(n, 1000)
    if millions:
        if millions == 1:
            groups.append("מיליון")
        elif millions == 2:
            groups.append("שני מיליון")
        else:
            groups.append(_he_join(_he_groups_below_1000(millions, feminine=False)) + " מיליון")
    if thousands:
        if thousands <= 10:
            groups.append(_HE_THOUSANDS[thousands])
        else:
            groups.append(_he_join(_he_groups_below_1000(thousands, feminine=False)) + " אלף")
    groups.extend(_he_groups_below_1000(rest, feminine))
    return _he_join(groups)


def number_to_words(n: int, lang: str = "en", feminine: bool = True) -> str:
    """Spell a non-negative integer below one billion (Hebrew) / one trillion (English)."""
    n = int(n)
    if n < 0:
        prefix = "minus " if lang == "en" else "מינוס "
        return prefix + number_to_words(-n, lang, feminine)
    if lang == "he":
        if n >= 1_000_000_000:
            raise ValueError("Hebrew numbers above 999,999,999 are not supported")
        return _hebrew(n, feminine)
    return _english(n)


def digits_to_words(digits: str, lang: str = "en") -> str:
    """Read a string digit by digit (decimals, phone-style numbers)."""
    table = _HE_F if lang == "he" else _EN_ONES
    return " ".join(table[int(d)] for d in digits)
