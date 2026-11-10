"""Long-form rendering: tagged script -> one continuous piece of audio.

Chatterbox degrades on very short inputs (gibberish) and on long ones
(hallucination past ~300 characters), and stops at ~40 s. This module cuts the
script into chunks the model handles well, renders each one, checks it, and
joins the results with exact pauses.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .hebrew import add_niqqud
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


class RadioHost:
    """Render tagged radio scripts with a loaded Chatterbox multilingual model.

        model = ChatterboxMultilingualTTS.from_pretrained("cuda", t3_model="v3")
        host = RadioHost(model, voice="yaniv_he.wav")
        wav, report = host.render("[excited] בוקר טוב! השעה 07:30. [pause:0.4s] ...")

    `model` only needs `.sr` and `.generate(text, language_id=..., **kwargs)`
    returning audio of shape (1, n) or (n,). `check(wav, sr, text, lang)` may
    return False to reject a take (e.g. a Whisper comparison); the chunk is then
    regenerated up to `max_retries` times and the last take is kept.
    """

    def __init__(self, model, voice: str | None = None, *, supports_events: bool = False,
                 max_chars: int = 220, max_retries: int = 2,
                 check: Callable[[np.ndarray, int, str, str], bool] | None = None,
                 sentence_gap: float = 0.12):
        self.model = model
        self.voice = voice
        self.supports_events = supports_events
        self.max_chars = max_chars
        self.max_retries = max_retries
        self.check = check
        self.sentence_gap = sentence_gap
        self.sr = int(model.sr)

    # -- text ---------------------------------------------------------------
    def prepare_text(self, segment: Segment, lang: str) -> str:
        text = segment.text if self.supports_events else segment.text_without_events()
        text = normalize_text(text, lang)
        if lang == "he":
            # Done here (not only in the tokenizer) so a missing diacritizer stops
            # the render before any audio is produced.
            text = add_niqqud(text)
        return text

    # -- audio --------------------------------------------------------------
    def _silence(self, seconds: float) -> np.ndarray:
        return np.zeros(int(round(seconds * self.sr)), dtype=np.float32)

    def _generate(self, text: str, lang: str, style: Style, report: RenderReport, **overrides) -> np.ndarray:
        kwargs = {**style_to_generate_kwargs(style), **overrides}
        if self.voice:
            kwargs["audio_prompt_path"] = self.voice
        wav = None
        for attempt in range(self.max_retries + 1):
            out = self.model.generate(text, language_id=lang, **kwargs)
            out = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
            wav = out.reshape(-1).astype(np.float32)
            if self.check is None or self.check(wav, self.sr, text, lang):
                return wav
            report.failed_checks += 1
            if attempt < self.max_retries:
                report.retries += 1
        logger.warning("Chunk failed its check %d times, keeping last take: %r", self.max_retries + 1, text[:60])
        return wav

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
        wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        report.duration = len(wav) / self.sr
        if parsed.fit_seconds and report.duration:
            report.fit_ratio = report.duration / parsed.fit_seconds
        return wav, report
