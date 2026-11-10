"""Hebrew niqqud (vowel marks) for the multilingual model.

The multilingual T3 was trained on Hebrew text with niqqud. Upstream called
`dicta_onnx.Dicta()` without its required model path, swallowed the error and
fed the model bare consonants, which comes out as gibberish
(resemble-ai/chatterbox#467). This module loads the diacritizer properly and
fails loudly instead of silently airing broken Hebrew.
"""
from __future__ import annotations

import logging
import os
import re
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DICTA_MODEL_URL = (
    "https://github.com/thewh1teagle/dicta-onnx/releases/download/"
    "model-files-v1.0/dicta-1.0.int8.onnx"
)
MODEL_ENV = "YSTREAM_DICTA_MODEL"
ALLOW_BARE_ENV = "YSTREAM_ALLOW_BARE_HEBREW"

_HEBREW_LETTER = re.compile(r"[א-ת]")
_NIQQUD = re.compile(r"[ְ-ׇּׁׂ]")
# {{written|spoken}} lets a script give an exact pronunciation (already with niqqud).
_OVERRIDE = re.compile(r"\{\{([^|{}]*)\|([^|{}]*)\}\}")

_dicta = None
_enabled = True


def set_niqqud(enabled: bool) -> None:
    """Turn automatic niqqud on or off for the whole process (for A/B experiments)."""
    global _enabled
    _enabled = bool(enabled)


def written_form(text: str) -> str:
    """`{{written|spoken}}` -> written. What a listener would transcribe."""
    return _OVERRIDE.sub(lambda m: m.group(1), text)


def spoken_form(text: str) -> str:
    """`{{written|spoken}}` -> spoken. What the model should read."""
    return _OVERRIDE.sub(lambda m: m.group(2), text)


class HebrewDiacritizerUnavailable(RuntimeError):
    pass


def has_niqqud(text: str) -> bool:
    """True when the text already carries vowel marks on a meaningful share of its letters."""
    letters = len(_HEBREW_LETTER.findall(text))
    return letters > 0 and len(_NIQQUD.findall(text)) >= letters * 0.3


def default_model_path() -> Path:
    if os.environ.get(MODEL_ENV):
        return Path(os.environ[MODEL_ENV])
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "ystream-voice" / "dicta-1.0.int8.onnx"


def ensure_model(path: Path | None = None) -> Path:
    """Return the diacritizer model path, downloading it once if missing."""
    path = Path(path) if path else default_model_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Hebrew diacritizer model to %s", path)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(DICTA_MODEL_URL, partial)
    partial.replace(path)
    return path


def _load():
    global _dicta
    if _dicta is None:
        try:
            from dicta_onnx import Dicta
        except ImportError as e:
            raise HebrewDiacritizerUnavailable(
                "Hebrew needs the 'dicta-onnx' package: pip install 'chatterbox-tts[hebrew]'"
            ) from e
        try:
            _dicta = Dicta(str(ensure_model()))
        except Exception as e:
            raise HebrewDiacritizerUnavailable(
                f"Could not load the Hebrew diacritizer ({e}). It needs the model file "
                f"({DICTA_MODEL_URL}; set {MODEL_ENV} to a local copy) and, on first use, "
                f"access to huggingface.co for its tokenizer."
            ) from e
    return _dicta


def _diacritize(text: str) -> str:
    if not _HEBREW_LETTER.search(text) or has_niqqud(text):
        return text
    return _load().add_diacritics(text)


def add_niqqud(text: str, diacritize=None) -> str:
    """Add niqqud to Hebrew text.

    - Text that already has niqqud is kept as written.
    - `{{written|spoken}}` is replaced by `spoken` verbatim (pronunciation override).
    - If the diacritizer is missing this raises, unless YSTREAM_ALLOW_BARE_HEBREW=1.
    `diacritize` can be injected for tests.
    """
    if not _enabled and diacritize is None:
        return spoken_form(text)
    diacritize = diacritize or _diacritize
    try:
        out, pos = [], 0
        for m in _OVERRIDE.finditer(text):
            out.append(diacritize(text[pos:m.start()]))
            out.append(m.group(2))
            pos = m.end()
        out.append(diacritize(text[pos:]))
        return "".join(out)
    except HebrewDiacritizerUnavailable:
        if os.environ.get(ALLOW_BARE_ENV) == "1":
            logger.error("Hebrew diacritizer unavailable; speech will be unintelligible.")
            return _OVERRIDE.sub(lambda m: m.group(2), text)
        raise
