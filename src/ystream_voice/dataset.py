"""Turn long voice recordings into training clips with transcripts.

A voice add-on is trained from short clips (3-15 s) that each have the exact
text spoken. This module cuts recordings at natural pauses and writes the
clips plus a metadata file; transcription is plugged in by the caller.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Clip:
    start: float  # seconds
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _frame_db(wav: np.ndarray, sr: int, frame_s: float = 0.02) -> np.ndarray:
    frame = max(1, int(sr * frame_s))
    n = len(wav) // frame
    rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def find_pauses(wav: np.ndarray, sr: int, min_pause: float = 0.25, drop_db: float = 35.0,
                frame_s: float = 0.02) -> list[tuple[float, float]]:
    """Silent stretches of at least `min_pause` seconds, as (start, end) times.

    Silence means `drop_db` below the recording's loud level (95th percentile),
    so it adapts to quiet and loud files alike.
    """
    db = _frame_db(wav, sr, frame_s)
    if len(db) == 0:
        return []
    quiet = db < (np.percentile(db, 95) - drop_db)
    pauses, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * frame_s >= min_pause:
                pauses.append((start * frame_s, i * frame_s))
            start = None
    return pauses


# Tried in order: a stretch with no clear pause is re-examined with shorter,
# shallower pauses (breaths between phrases; compressed audio with little dynamic range).
_PAUSE_LEVELS = [(0.25, 35.0), (0.15, 28.0), (0.10, 22.0), (0.07, 16.0)]


def _cut(wav: np.ndarray, sr: int, offset: float, min_len: float, max_len: float,
         padding: float, level: int) -> list[Clip]:
    total = len(wav) / sr
    min_pause, drop_db = _PAUSE_LEVELS[level]
    pauses = find_pauses(wav, sr, min_pause=min_pause, drop_db=drop_db)
    edges = [0.0] + [t for p in pauses for t in p] + [total]
    runs = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] > 0.05]

    clips: list[Clip] = []
    current: list[tuple[float, float]] = []

    def close():
        if not current:
            return
        start = max(0.0, current[0][0] - padding)
        end = min(total, current[-1][1] + padding)
        current.clear()
        if end - start > max_len:
            # One unbroken stretch that is too long: look for subtler pauses inside it.
            if level + 1 < len(_PAUSE_LEVELS):
                inner = wav[int(start * sr): int(end * sr)]
                clips.extend(_cut(inner, sr, offset + start, min_len, max_len, padding, level + 1))
        elif end - start >= min_len:
            clips.append(Clip(offset + start, offset + end))

    for run in runs:
        if current and run[1] - current[0][0] + 2 * padding > max_len:
            close()
        current.append(run)
        # Prefer ending a clip at a long pause (a sentence end) once it is long enough.
        if run[1] - current[0][0] >= min_len:
            following = next((p for p in pauses if abs(p[0] - run[1]) < 0.03), None)
            if following and following[1] - following[0] >= 0.5:
                close()
    close()
    return clips


def cut_at_pauses(wav: np.ndarray, sr: int, min_len: float = 3.0, max_len: float = 15.0,
                  padding: float = 0.1) -> list[Clip]:
    """Split a recording into clips of `min_len`..`max_len` seconds, cutting only in pauses.

    Each clip keeps up to `padding` seconds of the surrounding silence. A stretch
    with no clear pause for longer than `max_len` is searched again for shorter
    and shallower pauses; only if none exist is it dropped rather than cut
    mid-word. Leftovers shorter than `min_len` are dropped.
    """
    return _cut(wav, sr, 0.0, min_len, max_len, padding, 0)


def _plain(text: str) -> str:
    from .verify import comparable
    return " ".join(comparable(text))


def flag_rows(rows: list[dict]) -> None:
    """Fill each row's `flag` with reasons to leave it out of training (empty = use it).

    - short:    fewer than 3 words (mic checks, "thank you", fillers)
    - stutter:  the same word or phrase repeated back to back inside the clip
    - retake:   a later clip from the same recording says nearly the same thing,
                so this one is an earlier attempt; the last take is kept
    - cut-off:  the transcript ends with "..." (stopped mid-sentence)
    """
    from difflib import SequenceMatcher

    texts = [_plain(r["text"]) for r in rows]
    for i, row in enumerate(rows):
        words = texts[i].split()
        flags = []
        if len(words) < 3:
            flags.append("short")
        if any(words[j:j + n] == words[j + n:j + 2 * n]
               for n in (1, 2, 3) for j in range(len(words) - 2 * n + 1)
               if n > 1 or len(words[j]) > 2):
            flags.append("stutter")
        if row["text"].rstrip().endswith(("...", "…")):
            flags.append("cut-off")
        head = " ".join(words[:8])
        for j in range(i + 1, len(rows)):
            if rows[j]["source"] != row["source"] or len(words) < 3:
                continue
            other = " ".join(texts[j].split()[:8])
            if SequenceMatcher(None, head, other).ratio() > 0.8:
                flags.append("retake")
                break
        row["flag"] = " ".join(flags)


def prepare_dataset(files: list[Path], out_dir: Path, language: str,
                    transcribe: Callable[[np.ndarray, int, str], str] | None = None,
                    min_len: float = 3.0, max_len: float = 15.0) -> list[dict]:
    """Cut `files` into clips under `out_dir/clips` and write `out_dir/metadata.csv`.

    Clips keep the source sample rate (mono, 24-bit). Returns the metadata rows.
    Existing rows for other languages in metadata.csv are preserved. Rows with a
    non-empty `flag` (see `flag_rows`) are meant to be skipped by training.
    """
    import soundfile

    out_dir = Path(out_dir)
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, path in enumerate(sorted(files)):
        wav, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        for n, clip in enumerate(cut_at_pauses(wav, sr, min_len, max_len)):
            audio = wav[int(clip.start * sr): int(clip.end * sr)]
            name = f"{language}_{index:03d}_{n:04d}.wav"
            soundfile.write(str(clip_dir / name), audio, sr, subtype="PCM_24")
            rows.append({
                "clip": f"clips/{name}", "language": language, "source": path.name,
                "start": f"{clip.start:.2f}", "end": f"{clip.end:.2f}", "sample_rate": sr,
                "text": transcribe(audio, sr, language).strip() if transcribe else "",
                "flag": "",
            })
    if transcribe:
        flag_rows(rows)

    meta = out_dir / "metadata.csv"
    kept: list[dict] = []
    if meta.exists():
        with meta.open(encoding="utf-8-sig", newline="") as f:
            kept = [r for r in csv.DictReader(f) if r.get("language") != language]
    with meta.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "language", "source", "start", "end", "sample_rate", "text", "flag"],
                                restval="")
        writer.writeheader()
        writer.writerows(kept + rows)
    return rows


def flag_mismatches(folder, transcribe, max_error: float = 0.15, log=None) -> tuple[int, int]:
    """Flag clips whose audio does not say what their text says.

    For datasets that come with their own captions (crowd or web sources): each
    unflagged clip is transcribed and compared with its text; clips above
    `max_error` get the flag "mismatch" (noise, music, wrong or loose captions).
    `transcribe(wav, sr, lang) -> str`. Returns (checked, flagged). The file is
    rewritten every 200 clips, so an interrupted run can be resumed.
    """
    import soundfile
    from .verify import take_error

    folder = Path(folder)
    meta = folder / "metadata.csv"
    with meta.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    if "checked" not in fields:
        fields.append("checked")

    def write():
        with meta.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)

    checked = flagged = 0
    for row in rows:
        if row.get("flag") or row.get("checked"):
            continue
        wav, sr = soundfile.read(str(folder / row["clip"]), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        error = take_error(row["text"], transcribe(wav, sr, row["language"]), row["language"])
        row["checked"] = f"{error:.2f}"
        if error > max_error:
            row["flag"] = "mismatch"
            flagged += 1
        checked += 1
        if checked % 200 == 0:
            write()
            if log:
                log(f"  checked {checked}, flagged {flagged}")
    write()
    return checked, flagged
