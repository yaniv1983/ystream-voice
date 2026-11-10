import csv

import numpy as np
import pytest

from ystream_voice.dataset import cut_at_pauses, find_pauses, prepare_dataset

SR = 16000


def _speech(seconds, seed=0):
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def _silence(seconds):
    return (1e-4 * np.random.default_rng(1).standard_normal(int(seconds * SR))).astype(np.float32)


def _recording(parts):
    """parts: positive = seconds of speech, negative = seconds of silence."""
    return np.concatenate([_speech(p, i) if p > 0 else _silence(-p) for i, p in enumerate(parts)])


def test_find_pauses():
    wav = _recording([2, -0.6, 3, -0.1, 1])
    pauses = find_pauses(wav, SR)
    assert len(pauses) == 1
    assert pauses[0][0] == pytest.approx(2.0, abs=0.05) and pauses[0][1] == pytest.approx(2.6, abs=0.05)


def test_cut_at_pauses_respects_lengths_and_cuts_in_silence():
    wav = _recording([4, -0.8, 2, -0.3, 2.5, -0.8, 1, -0.8, 20, -0.8, 6])
    clips = cut_at_pauses(wav, SR)
    assert [round(c.duration, 1) for c in clips] == [4.1, 5.0, 6.1]   # 1 s leftover and 20 s run dropped
    assert all(3.0 <= c.duration <= 15.0 for c in clips)
    for c in clips:  # both ends sit in silence
        assert np.abs(wav[int(c.start * SR): int(c.start * SR) + 800]).max() < 0.01 or c.start == 0
        assert np.abs(wav[int(c.end * SR) - 800: int(c.end * SR)]).max() < 0.01 or c.end * SR >= len(wav) - 1


def test_prepare_dataset_writes_clips_and_keeps_other_language(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    src = tmp_path / "src"
    src.mkdir()
    soundfile.write(src / "a.wav", np.stack([_recording([4, -0.8, 5])] * 2, axis=1), SR)
    out = tmp_path / "data"
    he = prepare_dataset([src / "a.wav"], out, "he", transcribe=lambda wav, sr, lang: f" {lang} {len(wav) / sr:.0f} ")
    en = prepare_dataset([src / "a.wav"], out, "en")
    prepare_dataset([src / "a.wav"], out, "en")  # re-running replaces, does not duplicate
    assert len(he) == len(en) == 2 and he[0]["text"].startswith("he")
    audio, sr = soundfile.read(out / he[0]["clip"])
    assert sr == SR and audio.ndim == 1
    with (out / "metadata.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["language"] for r in rows] == ["he", "he", "en", "en"]
