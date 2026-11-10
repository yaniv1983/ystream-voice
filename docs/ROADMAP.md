# Roadmap

Goal: a Chatterbox that a station owner can clone their voice into (10-30 minutes of audio, or under a minute for instant cloning) and use as a radio host in English and Hebrew, on their own GPU.

| Phase | What | Needs training | State |
|---|---|---|---|
| 0 | Fork, tests, baseline recordings of how upstream sounds | No | In progress |
| 1 | Hebrew niqqud fix, radio text normaliser, script tags, long-form rendering | No | Code written; needs listening tests on a GPU |
| 1b | Automatic bad-take detection (Whisper) and loudness mastering | No | Code written; needs listening tests on a GPU |
| 1c | Half-precision mode for 8 GB cards | No | Next |
| 2 | Data pipeline and training harness; 48 kHz vocoder replacing the 24 kHz one | Yes | Planned |
| 3 | Hebrew quality and mixed Hebrew/English fine-tune; per-voice add-on from 10-30 minutes | Yes | Planned |
| 4 | Real `[laugh]`/`[breath]` sounds; exact pace, energy and pitch control; `[fit]` to a target length | Yes | Planned |
| 5 | Streaming for live use, HTTP service | No | Planned |
| 6 | CPU-only mode for PCs without a GPU: compressed (8-bit) model and the two-step decoder for offline rendering, then a small Hebrew + English model distilled from the full one for near-real-time speech | Yes (small model) | Planned |

## Known limits today

- Output is still 24 kHz (the upstream vocoder). Native 48 kHz, with 44.1 kHz as an option, arrives in phase 2.
- English names inside a Hebrew sentence are read with Hebrew phonetics. Use `{{written|spoken}}` as a workaround.
- `[pace]` and `[energy]` steer the existing model's expressiveness controls and are approximate.
- `[fit:8.5s]` only reports how far the result is from the target length.
- Hebrew number wording uses feminine forms and 24-hour times; nouns that need masculine numbers should be written out in words.

## Principles

- The audio watermark from upstream stays on.
- Voices are cloned only from people who agreed to it.
- Fork-specific code lives in `src/ystream_voice/`; changes to upstream files are kept small.
