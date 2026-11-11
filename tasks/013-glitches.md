# Task 013: glitches at the start of phrases (the owner's top defect)

Thanks for relaying feedback-002. Do this as soon as task 012's training steps allow; if 012 is in a long GPU run, finish 012 first, then this. Pull `radio-foundation` at f60cc9e (97 tests). Report to `reports/013.md`.

What changed:
- `RadioHost` now runs `clean_start` on every take (`trim_start=True`): it removes short bursts (up to 120 ms, followed by a gap of 50 ms or more) and the silence before the first real speech, keeping 30 ms ahead of it.
- Hebrew: after niqqud, the prefix vav before a shva consonant or before bet/mem/pe is turned into "u" (ushloshim).

## A. Find the glitch in the files the owner heard (no GPU needed; can run during 012 training)
For the Set 7 files (base, joint_v2, joint_v3 he_basic and the short line), print the level in 10 ms frames (dB relative to file peak) for the 400 ms before and 300 ms after each place where speech starts after a silence of 150 ms or more. I want to see what the glitch is: a short burst then a gap, a click glued to the speech, or noise during the "silence" tokens. Put one or two typical traces in the report and describe the rest in a sentence. If you kept per-chunk takes, also say whether the burst is inside the take (model output) or at a join.

## B. Measure the fix (GPU, after 012)
With joint_v2 and the Hebrew reference, generate 12 Hebrew chunk-openers straight through `model.generate` (use the 8 lines of he_sentences.txt plus "נשארים איתנו, תכף חוזרים.", "בשעה הקרובה נשמע את הלהיטים הגדולים.", "תודה.", "ועכשיו, מוזיקה."), 2 seeds each, for `lead_silence_tokens` 0 and 3. For each of the 48 takes report in a table: number of leading bursts that `clean_start` removes, total time removed, and Whisper's text after `clean_start`. Summary line: share of takes with a leading burst for lead 0 vs lead 3, and whether any take lost its first word or first consonant after cleaning (compare Whisper before/after).
If bursts are clearly more common with lead 3 than lead 0, set the `lead_silence_tokens` default to 0 in `mtl_tts.py` for part C and report it.

## C. Listening page: new Set 9 on top, plain labels
All joint_v2, `try_voice.py --check`:
1. he_basic.txt, new code
2. he_short.txt, new code
3. en_basic.txt with the original English reference (not the lively DJ clip), new code
Question: "Are the glitches before phrases gone? Is 'ושלושים' now 'ushloshim'? Are any word starts cut?"
Also report the niqqud text the model received for "השעה 7:30" (the vav of ושלושים should carry a shuruk).

End with "Task 013 done".
