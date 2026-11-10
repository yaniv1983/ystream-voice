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


def cut_at_pauses(wav: np.ndarray, sr: int, min_len: float = 3.0, max_len: float = 15.0,
                  padding: float = 0.1, **pause_options) -> list[Clip]:
    """Split a recording into clips of `min_len`..`max_len` seconds, cutting only in pauses.

    Each clip keeps `padding` seconds of the surrounding silence. Stretches with
    no pause for longer than `max_len` are dropped rather than cut mid-word, and
    so are leftovers shorter than `min_len`.
    """
    total = len(wav) / sr
    pauses = find_pauses(wav, sr, **pause_options)
    # Speech runs between pauses.
    edges = [0.0] + [t for p in pauses for t in p] + [total]
    runs = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] > 0.05]

    clips: list[Clip] = []
    current: list[tuple[float, float]] = []

    def close():
        if current:
            start = max(0.0, current[0][0] - padding)
            end = min(total, current[-1][1] + padding)
            if min_len <= end - start <= max_len:
                clips.append(Clip(start, end))
            current.clear()

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


def prepare_dataset(files: list[Path], out_dir: Path, language: str,
                    transcribe: Callable[[np.ndarray, int, str], str] | None = None,
                    min_len: float = 3.0, max_len: float = 15.0) -> list[dict]:
    """Cut `files` into clips under `out_dir/clips` and write `out_dir/metadata.csv`.

    Clips keep the source sample rate (mono, 24-bit). Returns the metadata rows.
    Existing rows for other languages in metadata.csv are preserved.
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
            })

    meta = out_dir / "metadata.csv"
    kept: list[dict] = []
    if meta.exists():
        with meta.open(encoding="utf-8-sig", newline="") as f:
            kept = [r for r in csv.DictReader(f) if r.get("language") != language]
    with meta.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "language", "source", "start", "end", "sample_rate", "text"])
        writer.writeheader()
        writer.writerows(kept + rows)
    return rows
