# Task 010: word starts, wider adapter, delivery settings

Thanks for 009, clean work. Pull `radio-foundation` at 3265802 (95 tests expected: run pytest first and report the count). Please also commit your `tests/radio_scripts/en_sentences.txt` in your report as a diff or paste its 8 lines (no personal data in it).

Work unattended, in order. If a step fails, note it and go on. Report to `reports/010.md`.

## A. Clipped word starts (new fix to verify)
The decoder zeroes its first 20 ms and fades the next 20 ms. `generate()` now has `lead_silence_tokens=3` (default on) which puts silence tokens in front.
1. With the base v3 model and the English reference, generate these 4 lines twice each, once with `lead_silence_tokens=0` and once with 3 (same seed for each pair: `torch.manual_seed(n)` before each call): "Good morning everyone.", "Perfect timing for the weekend.", "Tonight we go back to the classics.", "Big tunes coming up next."
2. For each file report: time from file start to first sample above -40 dBFS, and what Whisper heard. Tell me if the token 4299 really decodes to silence (level of the first 100 ms with 3 tokens).
3. If the lead silence is noisy or not silent, say so and set it to 0 for the rest of the task (`RadioHost` does not pass it, so edit the default and report the diff).

## B. "FM" check
The normaliser now sends "FM" to the model as "eff em" (and Hebrew letter names in Hebrew). Generate "You're listening to 99.5 FM, the home of dance music." 3 takes through `try_voice.py --check --verbose` style (normalised) and 3 takes with the raw text straight into `model.generate`. Report what Whisper heard for each. Same for the Hebrew line "אתם מאזינים ל-99.5 FM" (3 takes, normalised only).

## C. Wider adapter (attention + feed-forward layers)
`train_voice.py` has `--targets all`. Memory will be higher: start with `--batch-tokens 1000`, drop to 800 on out-of-memory.
1. `adapters\joint_v2`: `--data data\fleurs_he data\yaniv --multi-speaker fleurs_he --repeat yaniv=4 --targets all --rank 32 --steps 2500 --eval-every 125`
2. `adapters\yaniv_wide_on_joint`: `--data data\yaniv --start-from adapters\joint_v2 --targets all --rank 16 --steps 300 --eval-every 50 --batch-tokens 800`
Report the per-dataset held-out loss lines as before, plus time and peak memory.

## D. Delivery numbers (new script)
`scripts/prosody_stats.py NAME=PATH ...` prints pitch spread, pitch range, pauses and loudness spread. Run it on:
- `real_he` = folder of the Hebrew training clips of the host (unflagged if easy; whole folder is fine), `real_en` likewise
- every he_basic_*.wav and en_basic_*.wav from Set 5
Paste the table. This tells us whether generated speech is flatter than the real recordings.

## E. Settings sweep on the best adapter
Pick the adapter with the best host-voice held-out loss among joint_v1, joint_v2, yaniv_wide_on_joint (call it BEST). With `try_voice.py --adapter BEST --check` render he_basic.txt:
1. default
2. `--temperature 0.6`
3. default, but with the liveliest Hebrew radio clip you have as `--voice` (pick the unflagged Hebrew clip of 8-15 s with the highest pitch spread from prosody_stats; report only its clip number, not its text)
4. as 3 with the whole script wrapped in `[excited]` (energy preset)
And en_basic.txt: default, and with the liveliest English DJ clip as voice.
Run prosody_stats on these too.

## F. Listening page
Add a new Set 6 at the top with at most 6 clips, plain labels, in this order: Hebrew 1-4 from step E, English 2 from step E. Question for the listener: "Which Hebrew version is closest to how you would read it on air? Are the starts of words still clipped in English?" Keep older sets below.

## Report
Numbers and tables for A-E, diffs for any code you changed, anything that failed. End with "Task 010 done".
