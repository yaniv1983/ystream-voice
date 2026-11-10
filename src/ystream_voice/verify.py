"""Catch bad takes before they go to air.

The model sometimes skips words, repeats, or babbles. A speech recogniser
listens to each take; if what it hears is too far from the script, the take is
rejected and `RadioHost` generates another one.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

import numpy as np

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


def make_check(transcribe: Callable[[np.ndarray, int, str], str], max_wer: float = 0.25,
               max_seconds_per_word: float = 1.5):
    """Build a `check` function for RadioHost from any transcriber.

    Rejects a take when too many words differ, or when it runs far longer than
    the script could need (a sign of babbling or a long noisy tail).
    """
    def check(wav: np.ndarray, sr: int, text: str, lang: str) -> bool:
        words = comparable(text)
        if words and len(wav) / sr > max(3.0, len(words) * max_seconds_per_word):
            return False
        return word_error_rate(text, transcribe(wav, sr, lang)) <= max_wer
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
