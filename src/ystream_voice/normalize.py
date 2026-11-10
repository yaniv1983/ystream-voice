"""Turn written radio copy into words the model can pronounce.

Handles clock times, decimals (FM frequencies), percentages, years, plain
numbers and a few symbols, in English and Hebrew. Text inside square brackets
(tags) is never touched.
"""
from __future__ import annotations

import re

from .numbers import number_to_words, digits_to_words

_HEBREW_LETTER = re.compile(r"[א-ת]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_TAG = re.compile(r"\[[^\[\]]*\]")

_TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])(?:\s*(am|pm|AM|PM|a\.m\.|p\.m\.)(?![A-Za-z]))?")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_DECIMAL = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.]*\d)")
_GROUPED = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,]|\.\d)")
_ORDINAL_EN = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_INTEGER = re.compile(r"\d+")

_EN_ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}

_SYMBOLS = {
    "en": {"&": " and ", "@": " at ", "+": " plus ", "#": " number "},
    "he": {"&": " ו", "@": " שטרודל ", "+": " פלוס ", "#": " מספר "},
}


def detect_language(text: str) -> str:
    """'he' if the text has more Hebrew letters than Latin ones, else 'en'."""
    plain = _TAG.sub(" ", text)
    hebrew = len(_HEBREW_LETTER.findall(plain))
    latin = len(_LATIN_LETTER.findall(plain))
    return "he" if hebrew > latin else "en"


def _time_words(hour: int, minute: int, meridiem: str | None, lang: str) -> str:
    if lang == "he":
        # 24-hour reading, as on Israeli radio: "עשרים ושלושים", "שמונה וחצי" is left to the script writer.
        words = number_to_words(hour, "he")
        if minute == 0:
            return words
        if minute < 10:
            return f"{words} ואפס {number_to_words(minute, 'he')}"
        return f"{words} ו{number_to_words(minute, 'he')}"
    suffix = ""
    if meridiem:
        suffix = " " + ("A M" if meridiem.lower().startswith("a") else "P M")
    elif hour == 0 or hour > 12:
        suffix = " A M" if hour == 0 else " P M"
    hour12 = hour % 12 or 12
    words = number_to_words(hour12, "en")
    if minute == 0:
        return words + (suffix if suffix else " o'clock")
    if minute < 10:
        return f"{words} oh {number_to_words(minute, 'en')}{suffix}"
    return f"{words} {number_to_words(minute, 'en')}{suffix}"


def _year_words_en(year: int) -> str:
    high, low = divmod(year, 100)
    if low == 0:
        return number_to_words(year, "en") if high % 10 == 0 else number_to_words(high, "en") + " hundred"
    if 2000 <= year <= 2009:
        return number_to_words(year, "en")
    tail = "oh " + number_to_words(low, "en") if low < 10 else number_to_words(low, "en")
    return f"{number_to_words(high, 'en')} {tail}"


def _ordinal_en(n: int) -> str:
    words = number_to_words(n, "en")
    head, sep, last = words.rpartition("-") if "-" in words.split(" ")[-1] else words.rpartition(" ")
    if last in _EN_ORDINALS:
        last = _EN_ORDINALS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def _number_words(value: str, lang: str) -> str:
    whole, _, frac = value.partition(".")
    words = number_to_words(int(whole), lang)
    if frac:
        point = "נקודה" if lang == "he" else "point"
        words += f" {point} {digits_to_words(frac, lang)}"
    return words


def _normalize_plain(text: str, lang: str) -> str:
    # "1,500" is a quantity, never a year: spell it before the year rule can see it.
    text = _GROUPED.sub(lambda m: number_to_words(int(m[0].replace(",", "")), lang), text)
    text = _TIME.sub(lambda m: _time_words(int(m[1]), int(m[2]), m[3], lang), text)

    def percent(m):
        unit = "אחוז" if lang == "he" else "percent"
        return f"{_number_words(m[1], lang)} {unit}"
    text = _PERCENT.sub(percent, text)
    text = _DECIMAL.sub(lambda m: _number_words(m[0], lang), text)

    if lang == "en":
        text = _ORDINAL_EN.sub(lambda m: _ordinal_en(int(m[1])), text)

    def integer(m):
        n = int(m[0])
        # Leading zeros or very long strings are codes/phone numbers: read digit by digit.
        if (len(m[0]) > 1 and m[0][0] == "0") or len(m[0]) > 9:
            return digits_to_words(m[0], lang)
        if lang == "en" and len(m[0]) == 4 and 1100 <= n <= 2099:
            return _year_words_en(n)
        return number_to_words(n, lang)
    text = _INTEGER.sub(integer, text)

    for symbol, spoken in _SYMBOLS[lang].items():
        text = text.replace(symbol, spoken)
    return text


def normalize_text(text: str, lang: str | None = None) -> str:
    """Spell out numbers, times and symbols. Tags in [brackets] pass through untouched."""
    lang = lang or detect_language(text)
    out, pos = [], 0
    for tag in _TAG.finditer(text):
        out.append(_normalize_plain(text[pos:tag.start()], lang))
        out.append(tag.group(0))
        pos = tag.end()
    out.append(_normalize_plain(text[pos:], lang))
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
