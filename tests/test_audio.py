import numpy as np
import pytest

from ystream_voice.master import master, loudness_lufs, true_peak_db
from ystream_voice.verify import word_error_rate, make_check, comparable


def _speechlike(sr, seconds=4.0, level=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    tone = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 900 * t)
    wav = level * envelope * (tone + 0.3 * rng.standard_normal(len(t)))
    wav[sr:sr + 40] += 0.9  # a click, to exercise the limiter
    return wav.astype(np.float32)


@pytest.mark.parametrize("sr", [24000, 48000])
@pytest.mark.parametrize("target", [-16.0, -23.0])
def test_master_hits_loudness_and_peak(sr, target):
    out = master(_speechlike(sr), sr, target_lufs=target)
    assert loudness_lufs(out, sr) == pytest.approx(target, abs=1.0)
    assert true_peak_db(out) <= -1.0 + 0.05
    assert out.dtype == np.float32 and len(out) == 4 * sr


def test_master_handles_short_and_empty():
    assert len(master(np.zeros(0), 48000)) == 0
    short = master(0.99 * np.ones(2000, dtype=np.float32), 48000, highpass_hz=None)
    assert true_peak_db(short) <= -1.0 + 0.05


def test_word_error_rate():
    assert word_error_rate("Good morning, Tel Aviv!", "good morning tel aviv") == 0
    assert word_error_rate("one two three four", "one three four") == 0.25
    assert word_error_rate("[excited] שָׁלוֹם, עוֹלָם!", "שלום עולם") == 0
    assert comparable("drum-and-bass") == ["drum", "and", "bass"]
    assert word_error_rate("", "") == 0 and word_error_rate("", "hello") == 1


def test_take_error_forgives_digits_and_spelling():
    from ystream_voice.verify import take_error
    script = "It's seven thirty, and you're listening to ninety-nine point five FM."
    assert take_error(script, "It's 7:30 and you're listening to 99.5 FM", "en") == 0
    assert take_error("השעה שבע ושלושים", "השעה 7:30", "he") == 0
    assert take_error("שלום לכולם", "שלום לכלם", "he") < 0.25      # spelling variant
    assert take_error("Stay tuned for more music.", "Something else entirely here", "en") > 0.6


def test_make_check_scores_wrong_words_and_runaway_length():
    check = make_check(lambda wav, sr, lang: "stay tuned for more music")
    good = np.zeros(24000 * 2, dtype=np.float32)
    assert check(good, 24000, "Stay tuned for more music.", "en") == 0
    assert check(good, 24000, "Something else entirely here.", "en") > 0.6
    assert check(np.zeros(24000 * 30, dtype=np.float32), 24000, "Stay tuned for more music.", "en") == 1.0


def test_clean_start_drops_a_click_before_the_speech_but_keeps_soft_onsets():
    import numpy as np

    from ystream_voice.longform import clean_start

    sr = 24000
    t = np.arange(sr) / sr
    speech = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    click = (0.3 * np.sin(2 * np.pi * 900 * t[: int(0.04 * sr)])).astype(np.float32)

    out = clean_start(np.concatenate([silence, click, silence, speech]), sr)
    assert abs(len(out) - (len(speech) + int(0.03 * sr))) <= sr // 100     # click and silences gone

    plain = clean_start(np.concatenate([silence, speech]), sr)
    assert abs(len(plain) - (len(speech) + int(0.03 * sr))) <= sr // 100

    # A soft consonant (20 dB below the vowel) right before the vowel is speech, not a click.
    soft = (0.03 * np.sin(2 * np.pi * 3000 * t[: int(0.08 * sr)])).astype(np.float32)
    kept = clean_start(np.concatenate([silence, soft, speech]), sr)
    assert len(kept) >= len(soft) + len(speech)
