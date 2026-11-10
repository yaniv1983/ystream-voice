"""Inline performance tags for radio scripts.

    [excited] Good morning! [laugh] [pause:0.6s] [slow] And now... [/slow]

Three kinds of tag:
  * events   [laugh] [giggle] [breath] ...  a sound at that point
  * styles   [excited] [calm] [fast] [pace:1.2] [energy:high] ...  how to speak;
             stays on until changed, closed with [/name], or reset with [normal]
  * timing   [pause:0.6s] exact silence; [fit:8.5s] total target length (talk-up to a post)

The parser turns a script into segments, each with one fixed style. Events are
kept in the segment text for models trained on them and can be stripped for
models that are not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

EVENT_TAGS = {
    "laugh", "giggle", "chuckle", "breath", "long breath", "sigh", "gasp",
    "cough", "clear throat", "sniff",
}

# name -> style fields it sets. pace/energy are relative to 1.0 = the voice's normal.
STYLE_PRESETS: dict[str, dict[str, float]] = {
    "normal":    {"pace": 1.0, "energy": 1.0, "emotion": 0.5},
    "excited":   {"energy": 1.3, "emotion": 0.8, "pace": 1.08},
    "energetic": {"energy": 1.3, "emotion": 0.75},
    "hype":      {"energy": 1.5, "emotion": 0.95, "pace": 1.12},
    "calm":      {"energy": 0.8, "emotion": 0.35, "pace": 0.95},
    "warm":      {"energy": 0.9, "emotion": 0.45},
    "serious":   {"energy": 0.9, "emotion": 0.3, "pace": 0.95},
    "whisper":   {"energy": 0.5, "emotion": 0.3, "pace": 0.92},
    "fast":      {"pace": 1.2},
    "slow":      {"pace": 0.85},
}
_LEVELS = {"low": 0.75, "medium": 1.0, "normal": 1.0, "high": 1.3, "max": 1.5}
_LIMITS = {"pace": (0.6, 1.6), "energy": (0.4, 1.6), "emotion": (0.0, 1.5)}

_TAG = re.compile(r"\[\s*(/?)\s*([^\[\]:/]+?)\s*(?::\s*([^\[\]]+?)\s*)?\]")
_EVENT_IN_TEXT = re.compile(r"\[(?:%s)\]" % "|".join(sorted(map(re.escape, EVENT_TAGS), key=len, reverse=True)))


@dataclass(frozen=True)
class Style:
    pace: float = 1.0
    energy: float = 1.0
    emotion: float = 0.5


@dataclass
class Segment:
    text: str
    style: Style = field(default_factory=Style)
    pause_after: float = 0.0  # seconds of exact silence after this segment

    def text_without_events(self) -> str:
        return re.sub(r"\s{2,}", " ", _EVENT_IN_TEXT.sub(" ", self.text)).strip()


@dataclass
class Script:
    segments: list[Segment]
    fit_seconds: float | None = None
    unknown_tags: list[str] = field(default_factory=list)


def _seconds(value: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|sec)?", value.strip().lower())
    if not m:
        raise ValueError(f"Bad duration: {value!r}")
    number = float(m[1])
    return number / 1000 if m[2] == "ms" else number


def _level(name: str, value: str) -> float:
    value = value.strip().lower()
    number = _LEVELS[value] if value in _LEVELS else float(value)
    low, high = _LIMITS[name]
    return min(high, max(low, number))


def parse_script(text: str) -> Script:
    """Split a tagged script into styled segments. Unknown tags are dropped and reported."""
    script = Script(segments=[])
    base = Style()
    style = base
    stack: list[tuple[str, Style]] = []  # (tag name, style before it opened)
    buffer: list[str] = []

    def flush(pause: float = 0.0):
        segment_text = re.sub(r"\s+", " ", "".join(buffer)).strip()
        buffer.clear()
        if segment_text:
            script.segments.append(Segment(segment_text, style, pause))
        elif pause and script.segments:
            script.segments[-1].pause_after += pause
        elif pause:
            script.segments.append(Segment("", style, pause))

    pos = 0
    for m in _TAG.finditer(text):
        buffer.append(text[pos:m.start()])
        pos = m.end()
        closing, name, value = bool(m[1]), m[2].lower(), m[3]

        if name in EVENT_TAGS and not closing and value is None:
            buffer.append(f" [{name}] ")
        elif name == "pause" and value:
            flush(_seconds(value))
        elif name == "fit" and value:
            script.fit_seconds = _seconds(value)
        elif closing:
            names = [n for n, _ in stack]
            if name in names:
                flush()
                while stack:
                    opened, previous = stack.pop()
                    style = previous
                    if opened == name:
                        break
            else:
                script.unknown_tags.append(m.group(0))
        elif name in ("pace", "energy", "emotion") and value:
            flush()
            stack.append((name, style))
            style = replace(style, **{name: _level(name, value)})
        elif name in STYLE_PRESETS and value is None:
            flush()
            if name == "normal":
                stack.clear()
                style = base
            else:
                stack.append((name, style))
                style = replace(style, **STYLE_PRESETS[name])
        else:
            script.unknown_tags.append(m.group(0))
    buffer.append(text[pos:])
    flush()
    return script
