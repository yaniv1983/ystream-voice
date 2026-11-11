import numpy as np
import pytest

from ystream_voice import (number_to_words, normalize_text, detect_language, parse_script,
                           split_into_chunks, RadioHost, add_niqqud, has_niqqud)
from ystream_voice.longform import style_to_generate_kwargs
from ystream_voice.tags import Style


@pytest.mark.parametrize("n,expected", [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (21, "twenty-one"), (100, "one hundred"),
    (138, "one hundred thirty-eight"), (1000, "one thousand"), (12045, "twelve thousand forty-five"),
    (2000000, "two million"),
])
def test_english_numbers(n, expected):
    assert number_to_words(n, "en") == expected


@pytest.mark.parametrize("n,expected", [
    (0, "אפס"), (1, "אחת"), (2, "שתיים"), (10, "עשר"), (11, "אחת עשרה"), (12, "שתים עשרה"),
    (20, "עשרים"), (21, "עשרים ואחת"), (99, "תשעים ותשע"), (100, "מאה"), (101, "מאה ואחת"),
    (112, "מאה ושתים עשרה"), (123, "מאה עשרים ושלוש"), (200, "מאתיים"), (350, "שלוש מאות וחמישים"),
    (1000, "אלף"), (1999, "אלף תשע מאות תשעים ותשע"), (2000, "אלפיים"), (2026, "אלפיים עשרים ושש"),
    (3000, "שלושת אלפים"), (10000, "עשרת אלפים"), (11000, "אחד עשר אלף"),
    (21500, "עשרים ואחד אלף וחמש מאות"), (1000000, "מיליון"),
])
def test_hebrew_numbers(n, expected):
    assert number_to_words(n, "he") == expected


def test_hebrew_masculine():
    assert number_to_words(3, "he", feminine=False) == "שלושה"
    assert number_to_words(15, "he", feminine=False) == "חמישה עשר"


def test_detect_language():
    assert detect_language("Good morning Tel Aviv") == "en"
    assert detect_language("בוקר טוב עם Armin van Buuren") == "en"  # more Latin letters
    assert detect_language("בוקר טוב לכל המאזינים עם Tiesto") == "he"
    assert detect_language("[excited] שלום לכולם") == "he"


@pytest.mark.parametrize("text,expected", [
    ("It's 7:30 on 99.5 FM", "It's seven thirty on ninety-nine point five {{FM|eff em}}"),
    ("It's 20:05", "It's eight oh five P M"),
    ("Back at 9:00", "Back at nine o'clock"),
    ("Wake up at 6:15 am", "Wake up at six fifteen A M"),
    ("A classic from 1999", "A classic from nineteen ninety-nine"),
    ("Released in 2005", "Released in two thousand five"),
    ("138 BPM", "one hundred thirty-eight {{BPM|bee pee em}}"),
    ("50% off", "fifty percent off"),
    ("The 3rd track, 21st hour", "The third track, twenty-first hour"),
    ("1,500 listeners", "one thousand five hundred listeners"),
    ("Drum & bass", "Drum and bass"),
    ("Call 054", "Call zero five four"),
])
def test_normalize_english(text, expected):
    assert normalize_text(text, "en") == expected


@pytest.mark.parametrize("text,expected", [
    ("השעה 20:30", "השעה עשרים ושלושים"),
    ("השעה 8:00", "השעה שמונה"),
    ("השעה 7:05", "השעה שבע ואפס חמש"),
    ("רדיו 99.5", "רדיו תשעים ותשע נקודה חמש"),
    ("שנת 2026", "שנת אלפיים עשרים ושש"),
    ("50% הנחה", "חמישים אחוז הנחה"),
])
def test_normalize_hebrew(text, expected):
    assert normalize_text(text, "he") == expected


def test_normalize_leaves_tags_alone():
    assert normalize_text("[pause:0.5s] Top 40 [pace:1.2] now", "en") == "[pause:0.5s] Top forty [pace:1.2] now"


def test_parse_events_and_pause():
    script = parse_script("Hello [laugh] there. [pause:600ms] Next up")
    assert [s.text for s in script.segments] == ["Hello [laugh] there.", "Next up"]
    assert script.segments[0].pause_after == pytest.approx(0.6)
    assert script.segments[0].text_without_events() == "Hello there."


def test_parse_styles_nest_and_close():
    script = parse_script("One. [excited] Two! [fast] Three! [/fast] Four! [/excited] Five.")
    paces = [s.style.pace for s in script.segments]
    energies = [s.style.energy for s in script.segments]
    assert [s.text for s in script.segments] == ["One.", "Two!", "Three!", "Four!", "Five."]
    assert paces == [1.0, 1.08, 1.2, 1.08, 1.0]
    assert energies == [1.0, 1.3, 1.3, 1.3, 1.0]


def test_parse_values_fit_normal_and_unknown():
    script = parse_script("[fit:8.5s][pace:1.3][energy:high] Go! [normal] Done [banana] ok [long breath]")
    assert script.fit_seconds == 8.5
    assert script.segments[0].style == Style(pace=1.3, energy=1.3)
    assert script.segments[1].style == Style()
    assert script.segments[1].text == "Done ok [long breath]"
    assert script.unknown_tags == ["[banana]"]
    assert parse_script("[pace:9] x").segments[0].style.pace == 1.6  # clamped


def test_chunks_respect_limit_and_merge_short():
    text = "Hi! " + "This is a fairly long sentence about the next track on the show. " * 6 + "Yes."
    chunks = split_into_chunks(text, max_chars=120)
    assert all(len(c) <= 120 + 20 for c in chunks)
    assert all(len(c) >= 20 for c in chunks)
    assert " ".join(chunks) == " ".join(text.split())
    assert split_into_chunks("Hi! Yes. Okay.") == ["Hi! Yes. Okay."]


def test_chunks_split_unpunctuated_run():
    chunks = split_into_chunks("word " * 100, max_chars=80)
    assert all(len(c) <= 80 + 20 for c in chunks) and len(chunks) > 5


def test_style_mapping_is_monotonic():
    slow, normal, fast = (style_to_generate_kwargs(Style(pace=p))["cfg_weight"] for p in (0.8, 1.0, 1.3))
    assert slow < normal < fast and normal == pytest.approx(0.5)
    assert style_to_generate_kwargs(Style(energy=1.5))["exaggeration"] > style_to_generate_kwargs(Style())["exaggeration"]


def test_niqqud_helpers():
    marked = "שָׁלוֹם"
    assert has_niqqud(marked) and not has_niqqud("שלום")
    fake = lambda t: t.replace("שלום", marked)
    assert add_niqqud("שלום {{Tiesto|טִיֶסְטוֹ}} שלום", diacritize=fake) == f"{marked} טִיֶסְטוֹ {marked}"


class FakeModel:
    sr = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, language_id=None, **kwargs):
        self.calls.append((text, language_id, kwargs))
        return np.ones((1, self.sr), dtype=np.float32)  # one second per chunk


def test_radio_host_renders_segments_pauses_and_retries():
    model = FakeModel()
    takes = iter([False, True, True])
    host = RadioHost(model, voice="me.wav", check=lambda wav, sr, text, lang: next(takes))
    wav, report = host.render("[fit:4s][excited] It's 7:30! [laugh] [pause:0.5s] [calm] Stay tuned for more music.")
    assert [c[0] for c in model.calls] == ["It's seven thirty!"] * 2 + ["Stay tuned for more music."]
    assert all(c[1] == "en" and c[2]["audio_prompt_path"] == "me.wav" for c in model.calls)
    assert model.calls[0][2]["exaggeration"] > model.calls[2][2]["exaggeration"]
    assert len(wav) == int(2.5 * 24000) and np.all(wav[24000:36000] == 0)
    assert (report.chunks, report.retries, report.failed_checks) == (2, 1, 1)
    assert report.fit_ratio == pytest.approx(2.5 / 4)


def test_radio_host_keeps_events_when_model_supports_them():
    model = FakeModel()
    RadioHost(model, supports_events=True).render("Hello [laugh] there.")
    assert model.calls[0][0] == "Hello [laugh] there."


def test_radio_host_speaks_override_but_checks_written_form():
    model = FakeModel()
    seen = []
    host = RadioHost(model, check=lambda wav, sr, text, lang: seen.append(text) or True)
    host.render("Next up, {{Tiësto|Tee-esto}} live at 9:00.")
    assert model.calls[0][0] == "Next up, Tee-esto live at nine o'clock."
    assert seen == ["Next up, Tiësto live at nine o'clock."]


def test_radio_host_keeps_best_take_and_prepares_voice_once():
    class Model(FakeModel):
        prepared = 0

        def prepare_conditionals(self, path):
            self.prepared += 1

        def generate(self, text, language_id=None, **kwargs):
            super().generate(text, language_id, **kwargs)
            return np.full((1, 100), float(len(self.calls)), dtype=np.float32)

    model = Model()
    errors = iter([0.6, 0.3, 0.9])
    host = RadioHost(model, voice="me.wav", check=lambda *a: next(errors), max_error=0.25)
    wav, report = host.render("Stay tuned for more music.")
    assert model.prepared == 1 and "audio_prompt_path" not in model.calls[0][2]
    assert len(model.calls) == 3 and wav[-1] == 2.0         # second take had the lowest error
    assert report.worst_error == pytest.approx(0.3) and report.failed_checks == 3


def test_niqqud_can_be_switched_off():
    from ystream_voice import hebrew
    hebrew.set_niqqud(False)
    try:
        assert add_niqqud("שלום {{Tiesto|טִיֶסְטוֹ}}") == "שלום טִיֶסְטוֹ"
    finally:
        hebrew.set_niqqud(True)


def test_acronyms_are_spelled_as_letter_names():
    from ystream_voice.hebrew import spoken_form, written_form
    from ystream_voice.normalize import normalize_text

    out = normalize_text("You're on 99.5 FM with DJ Dana, 128 BPM", "en")
    assert spoken_form(out) == ("You're on ninety-nine point five eff em with dee jay Dana, "
                                "one hundred twenty-eight bee pee em")
    assert "FM" in written_form(out) and "DJ" in written_form(out)
    assert "אֶף אֶם" in spoken_form(normalize_text("אתם על 99.5 FM", "he"))


def test_acronym_rule_leaves_ordinary_words_and_overrides_alone():
    from ystream_voice.hebrew import spoken_form
    from ystream_voice.normalize import normalize_text

    assert normalize_text("I AM here, OK", "en") == "I AM here, OK"
    assert spoken_form(normalize_text("1080 AM", "en")).endswith("ay em")
    assert normalize_text("{{MP3|em pee three}} and 2 [pause:0.5s]", "en") == "{{MP3|em pee three}} and two [pause:0.5s]"
    assert spoken_form(normalize_text("MP3", "en")) == "em pee three"


def test_vav_prefix_is_read_u_before_shva_and_bet_mem_pe():
    from ystream_voice.hebrew import add_niqqud, fix_vav

    assert fix_vav("שֶׁבַע וְשְׁלוֹשִׁים") == "שֶׁבַע וּשְׁלוֹשִׁים"      # ushloshim
    assert fix_vav("וְמוּזִיקָה") == "וּמוּזִיקָה"                    # umuzika
    assert fix_vav("וְעַכְשָׁיו") == "וְעַכְשָׁיו"                    # ve'achshav stays
    assert fix_vav("תִּקְוְמוּ") == "תִּקְוְמוּ"                      # only a prefix, never inside a word
    assert add_niqqud("x", diacritize=lambda t: "וְשְׁלוֹשִׁים") == "וּשְׁלוֹשִׁים"
