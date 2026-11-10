# Task 009: teach the model Hebrew from native speech, then add the host's voice

Thanks for report 008. It showed the training pipeline is sound (step-0 loss 4.01,
held-out loss down to 3.65, then overfitting on 14 minutes of clips) and that
intelligibility and voice similarity barely move. The host's complaint is accent and
machine-like delivery, and 14 minutes of one voice cannot teach the model Hebrew.
This task trains on hours of native Hebrew read speech, then compares.

You may fix bugs locally in `scripts/fetch_fleurs.py`, `scripts/train_voice.py`,
`src/ystream_voice/training.py` and `scripts/hebrew_ab.py`; report every change as a
unified diff. `fetch_fleurs.py` was written without access to Hugging Face, so expect
to adapt it (file layout, TSV columns). Do not change other files.

## 1. Update
`git pull` (expect e065535 or later), `pytest -q` (expect 91 passed).

## 2. Hebrew data: FLEURS
    python scripts\fetch_fleurs.py --lang he --out data\fleurs_he
FLEURS (`google/fleurs`, config `he_il`) is CC BY 4.0. Before using it, open the
dataset card and confirm the licence in your report (quote the licence line). If the
script's assumptions are wrong, fix it: the goal is `data\fleurs_he\clips\*.wav` plus
`metadata.csv` with columns clip, language, source, start, end, sample_rate, text, flag
(same as `data\yaniv\metadata.csv`), clips of 3-15 s, text = the transcription WITH
punctuation. Report clips and hours. Listen to nothing; just check 3 random rows have
sensible Hebrew text and the audio duration matches.

Optional second source, only if all of these hold: it is on Hugging Face, NOT gated
(no login or terms to accept), its card states a licence that allows training models
including commercial use, and it is clean read speech with transcripts: e.g.
`ivrit-ai/crowd-recital`. If so, bring up to 20 hours of it into `data\recital_he` in
the same format and quote its licence line. If any condition fails or is unclear,
skip it and say why. Do not use SASPEECH, ILSpeech or other non-commercial data.

## 3. Train a Hebrew adapter (many voices)
    python scripts\train_voice.py --data data\fleurs_he --multi-speaker fleurs_he --languages he --out adapters\hebrew_v1 --rank 32 --steps 4000 --eval-every 250 --batch-tokens 1200
(add `data\recital_he --multi-speaker recital_he` if you fetched it). Feature
conversion runs once and is cached. Save output to `out\train_hebrew_v1.log`.
If free card memory is under 5 GB use `--batch-tokens 800`; on out-of-memory halve it.
Report: the "voices found / dropped" line, clips and minutes, every step line, the
final line, peak memory, minutes. If validation loss is still falling at step 4000,
run again with `--steps 8000 --out adapters\hebrew_v2` and report that too.

## 4. Add the host's voice on top
    python scripts\train_voice.py --data data\yaniv --languages he en --start-from adapters\hebrew_v1 --out adapters\yaniv_on_hebrew_v1 --steps 400 --eval-every 50 --batch-tokens 800
(use hebrew_v2 if that was better). Report its step lines.

Also one joint run, as the alternative recipe:
    python scripts\train_voice.py --data data\fleurs_he data\yaniv --multi-speaker fleurs_he --repeat yaniv=8 --out adapters\joint_v1 --rank 32 --steps 4000 --eval-every 250 --batch-tokens 1200
Its log prints held-out loss per dataset (`yaniv ...`, `fleurs_he ...`): report those lines.

## 5. Compare everything, same sentences as before
    python scripts\hebrew_ab.py --voice voices\yaniv_he_ref.wav --out out\hebrew_ab_v2 --settings v3_default --takes 3 --adapter yaniv_v1=adapters\yaniv_v1 --adapter hebrew_v1=adapters\hebrew_v1 --adapter yaniv_on_hebrew=adapters\yaniv_on_hebrew_v1 --adapter joint_v1=adapters\joint_v1
Report the table exactly. Then English, to check Hebrew training did not damage it:
    python scripts\hebrew_ab.py --lang en --file tests\radio_scripts\en_sentences.txt --voice voices\yaniv_en_ref2.wav --out out\english_ab_v2 --settings v3_default --takes 2 --adapter hebrew_v1=adapters\hebrew_v1 --adapter yaniv_on_hebrew=adapters\yaniv_on_hebrew_v1 --adapter joint_v1=adapters\joint_v1

## 6. Listening files for the host
For each of: base (no adapter), hebrew_v1, yaniv_on_hebrew_v1, joint_v1:
    python scripts\try_voice.py --voice voices\yaniv_he_ref.wav --file tests\radio_scripts\he_basic.txt --out out\he_basic_<name>.wav [--adapter adapters\<name>] --check
and for yaniv_on_hebrew_v1 and joint_v1 also the English `en_basic.txt` with
`voices\yaniv_en_ref2.wav`. Put them on the host's listening page as a new set, each
labelled in plain words (e.g. "Base model", "Taught Hebrew from native speakers",
"Hebrew + your voice", "Hebrew and your voice together"), with one question at the
top: which Hebrew version sounds most like a native Israeli presenter, and does it
still sound like you?

## 7. Report
`reports/009.md`: licence lines, data sizes, all training lines, both tables, the
listening-file lines, diffs, errors. Keep going through the steps even if one fails
(say which and why). Then poll for task 010.
