"""Long-form rendering: tagged script -> one continuous piece of audio.

Chatterbox degrades on very short inputs (gibberish) and on long ones
(hallucination past ~300 characters), and stops at ~40 s. This module cuts the
script into chunks the model handles well, renders each one, checks it, and
joins the results with exact pauses.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .hebrew import add_niqqud, spoken_form, written_form
from .normalize import detect_language, normalize_text
from .tags import Script, Segment, Style, parse_script

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?…。])[\"')\]]*\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:־–—-])\s+")


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    pieces, current = [], ""
    for clause in _CLAUSE_BREAK.split(sentence):
        while len(clause) > max_chars:  # no punctuation at all: break on a space
            cut = clause.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            head, clause = clause[:cut], clause[cut:].lstrip()
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if current and len(current) + 1 + len(clause) > max_chars:
            pieces.append(current)
            current = clause
        else:
            current = f"{current} {clause}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_into_chunks(text: str, max_chars: int = 220, min_chars: int = 20) -> list[str]:
    """Cut text at sentence ends into chunks of at most `max_chars`.

    Sentences are packed together up to the limit, and fragments shorter than
    `min_chars` ("Hi!", "Yes.") are attached to a neighbour, because the model
    is unreliable on very short inputs.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    pieces = [p for s in sentences for p in _split_long(s, max_chars)]
    chunks: list[str] = []
    for piece in pieces:
        if chunks and (len(chunks[-1]) + 1 + len(piece) <= max_chars):
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    # A short trailing fragment that could not be packed: merge backwards anyway.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}"
    return chunks


def style_to_generate_kwargs(style: Style) -> dict:
    """Map a tag style onto the controls the *pretrained* model has today.

    Until the pace/energy conditioners are trained, pace and energy are steered
    through `exaggeration` and `cfg_weight` (lower cfg_weight = slower, more
    deliberate delivery). This is an approximation, not exact control.
    """
    exaggeration = style.emotion + 0.35 * (style.energy - 1.0)
    cfg_weight = 0.5 + 0.6 * (style.pace - 1.0)
    return {
        "exaggeration": float(min(1.5, max(0.1, exaggeration))),
        "cfg_weight": float(min(0.9, max(0.15, cfg_weight))),
    }


@dataclass
class RenderReport:
    chunks: int = 0
    retries: int = 0
    failed_checks: int = 0
    duration: float = 0.0
    fit_ratio: float | None = None
    generate_seconds: float = 0.0
    check_seconds: float = 0.0
    worst_error: float = 0.0   # error score of the worst chunk that was kept


class RadioHost:
    """Render tagged radio scripts with a loaded Chatterbox multilingual model.

        model = ChatterboxMultilingualTTS.from_pretrained("cuda", t3_model="v3")
        host = RadioHost(model, voice="yaniv_he.wav")
        wav, report = host.render("[excited] בוקר טוב! השעה 07:30. [pause:0.4s] ...")

    `model` only needs `.sr` and `.generate(text, language_id=..., **kwargs)`
    returning audio of shape (1, n) or (n,). `check(wav, sr, text, lang)` scores
    a take (e.g. a Whisper comparison): an error from 0 to 1, or True/False.
    A take above `max_error` is regenerated up to `max_retries` times and the
    best take is kept.
    """

    def __init__(self, model, voice: str | None = None, *, supports_events: bool = False,
                 max_chars: int = 220, max_retries: int = 2,
                 check: Callable[[np.ndarray, int, str, str], bool | float] | None = None,
                 max_error: float = 0.25, sentence_gap: float = 0.12,
                 lead_in: float = 0.0, fade_in: float = 0.008,
                 options: dict | None = None):
        self.model = model
        self.voice = voice
        self.supports_events = supports_events
        self.max_chars = max_chars
        self.max_retries = max_retries
        self.check = check
        self.max_error = max_error
        self.sentence_gap = sentence_gap
        self.lead_in = lead_in      # silence before the first sound of the file
        self.fade_in = fade_in      # ramp at the start of every generated chunk
        self.options = dict(options or {})   # extra generate() settings, e.g. {"temperature": 0.6}
        self.sr = int(model.sr)
        # Analyse the reference voice once, not again for every chunk.
        self._voice_prepared = False
        if voice and hasattr(model, "prepare_conditionals"):
            model.prepare_conditionals(voice)
            self._voice_prepared = True

    # -- text ---------------------------------------------------------------
    def prepare_text(self, segment: Segment, lang: str) -> str:
        """Tags handled and numbers spelled out; still plain text without niqqud."""
        text = segment.text if self.supports_events else segment.text_without_events()
        return normalize_text(text, lang)

    def spoken_text(self, chunk: str, lang: str) -> str:
        """What the model receives: overrides applied, Hebrew with niqqud.

        Done here (not only in the tokenizer) so a missing diacritizer stops the
        render before any audio is produced.
        """
        return add_niqqud(chunk) if lang == "he" else spoken_form(chunk)

    # -- audio --------------------------------------------------------------
    def _silence(self, seconds: float) -> np.ndarray:
        return np.zeros(int(round(seconds * self.sr)), dtype=np.float32)

    def _faded(self, wav: np.ndarray) -> np.ndarray:
        """The model often starts a chunk at full level; a few ms of ramp avoids a hard edge."""
        n = min(len(wav), int(self.fade_in * self.sr))
        if n > 1:
            wav = wav.copy()
            wav[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
        return wav

    def _generate(self, text: str, lang: str, style: Style, report: RenderReport, **overrides) -> np.ndarray:
        kwargs = {**style_to_generate_kwargs(style), **self.options, **overrides}
        if self.voice and not self._voice_prepared:
            kwargs["audio_prompt_path"] = self.voice
        spoken = self.spoken_text(text, lang)
        # Takes are checked against the plain spelling: a recogniser writes Hebrew
        # without niqqud, and with the usual spelling of an overridden name.
        text = written_form(text)
        best, best_error = None, float("inf")
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            out = self.model.generate(spoken, language_id=lang, **kwargs)
            out = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
            wav = out.reshape(-1).astype(np.float32)
            report.generate_seconds += time.perf_counter() - started
            if self.check is None:
                return self._faded(wav)
            started = time.perf_counter()
            result = self.check(wav, self.sr, text, lang)
            report.check_seconds += time.perf_counter() - started
            error = (0.0 if result else 1.0) if isinstance(result, bool) else float(result)
            if error < best_error:
                best, best_error = wav, error
            if error <= self.max_error:
                break
            report.failed_checks += 1
            if attempt < self.max_retries:
                report.retries += 1
        else:
            logger.warning("No take passed the check after %d tries, keeping the best (error %.2f): %r",
                           self.max_retries + 1, best_error, text)
        report.worst_error = max(report.worst_error, best_error)
        return self._faded(best)

    def render(self, script: str | Script, language: str | None = None, **generate_overrides):
        """Return (mono float32 audio at `self.sr`, RenderReport)."""
        parsed = parse_script(script) if isinstance(script, str) else script
        for tag in parsed.unknown_tags:
            logger.warning("Ignoring unknown tag %s", tag)
        report = RenderReport()
        parts: list[np.ndarray] = []
        for segment in parsed.segments:
            if segment.text:
                lang = language or detect_language(segment.text)
                chunks = split_into_chunks(self.prepare_text(segment, lang), self.max_chars)
                for i, chunk in enumerate(chunks):
                    if i:
                        parts.append(self._silence(self.sentence_gap))
                    parts.append(self._generate(chunk, lang, segment.style, report, **generate_overrides))
                    report.chunks += 1
            if segment.pause_after:
                parts.append(self._silence(segment.pause_after))
        if parts and self.lead_in:
            parts.insert(0, self._silence(self.lead_in))
        wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        report.duration = len(wav) / self.sr
        if parsed.fit_seconds and report.duration:
            report.fit_ratio = report.duration / parsed.fit_seconds
        return wav, report
