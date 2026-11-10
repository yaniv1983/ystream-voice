"""Catch bad takes before they go to air.

The model sometimes skips words, repeats, or babbles. A speech recogniser
listens to each take; if what it hears is too far from the script, the take is
rejected and `RadioHost` generates another one.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

_NIQQUD = re.compile(r"[֑-ׇ]")
_TAGS = re.compile(r"\[[^\[\]]*\]")
_NOT_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def comparable(text: str) -> list[str]:
    """Words of a text with tags, niqqud, punctuation and case removed."""
    text = _TAGS.sub(" ", text)
    text = _NIQQUD.sub("", unicodedata.normalize("NFC", text))
    text = _NOT_WORD.sub(" ", text.replace("-", " ").replace("־", " ")).lower()
    return text.split()


def word_error_rate(reference: str, heard: str) -> float:
    """Edit distance between word sequences divided by the reference length."""
    ref, hyp = comparable(reference), comparable(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / len(ref)


def char_error_rate(reference: str, heard: str) -> float:
    """Edit distance between the letters (spaces removed) divided by the reference length."""
    ref, hyp = "".join(comparable(reference)), "".join(comparable(heard))
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / len(ref)


def take_error(reference: str, heard: str, lang: str) -> float:
    """How far a transcript is from the script: 0 is perfect, 1 is unrelated.

    The recogniser writes "7:30" and "99.5" as digits while the script has them
    spelled out, so the transcript goes through the same normaliser first.
    Spelling variants (Hebrew full/defective spelling, "okay"/"ok") change whole
    words but few letters, so the letter-level score is allowed to rescue them.
    """
    from .hebrew import written_form
    from .normalize import normalize_text
    reference = written_form(reference)
    heard = written_form(normalize_text(heard, lang))
    return min(word_error_rate(reference, heard), 2.0 * char_error_rate(reference, heard))


def make_check(transcribe: Callable[[np.ndarray, int, str], str], max_seconds_per_word: float = 1.5):
    """Build a `check` function for RadioHost from any transcriber.

    The function returns an error score (see `take_error`); RadioHost redoes
    takes above its `max_error` and keeps the best one. A take far longer than
    the script could need (babbling, a long noisy tail) scores 1.0 unheard.
    """
    def check(wav: np.ndarray, sr: int, text: str, lang: str) -> float:
        from .hebrew import written_form
        words = comparable(written_form(text))
        if words and len(wav) / sr > max(3.0, len(words) * max_seconds_per_word):
            logger.info("Take too long for its text (%.1f s for %d words)", len(wav) / sr, len(words))
            return 1.0
        heard = transcribe(wav, sr, lang)
        error = take_error(text, heard, lang)
        logger.info("check error=%.2f script=%r heard=%r", error, text, heard.strip())
        return error
    return check


def whisper_transcriber(model_size: str = "large-v3-turbo", device: str = "auto", compute_type: str = "default",
                        hebrew_model: str | None = "ivrit-ai/whisper-large-v3-turbo-ct2"):
    """Transcriber backed by faster-whisper (`pip install faster-whisper`).

    Hebrew uses the ivrit.ai fine-tune when `hebrew_model` is set; models load
    on first use. On an 8 GB card run this on the CPU (`device="cpu"`) so it
    does not compete with the voice model for memory.
    """
    models: dict[str, object] = {}

    def transcribe(wav: np.ndarray, sr: int, lang: str) -> str:
        from faster_whisper import WhisperModel
        name = hebrew_model if (lang == "he" and hebrew_model) else model_size
        if name not in models:
            models[name] = WhisperModel(name, device=device, compute_type=compute_type)
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(16000, sr)
            wav = resample_poly(wav, 16000 // g, sr // g)
        segments, _ = models[name].transcribe(wav.astype(np.float32), language=lang, beam_size=5)
        return " ".join(s.text for s in segments)
    return transcribe
