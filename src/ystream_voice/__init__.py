"""yStream Voice: radio-hosting layer on top of Chatterbox (English + Hebrew).

Everything in this package is importable without torch, so the text front-end
can be tested on any machine. The model is only touched by `RadioHost`, which
receives an already-loaded Chatterbox model.
"""
from .numbers import number_to_words
from .normalize import normalize_text, detect_language
from .tags import parse_script, Segment, Script, EVENT_TAGS
from .longform import split_into_chunks, RadioHost
from .hebrew import add_niqqud, has_niqqud, HebrewDiacritizerUnavailable

__all__ = [
    "number_to_words", "normalize_text", "detect_language",
    "parse_script", "Segment", "Script", "EVENT_TAGS",
    "split_into_chunks", "RadioHost",
    "add_niqqud", "has_niqqud", "HebrewDiacritizerUnavailable",
]
