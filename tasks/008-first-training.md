# Task 008: test the training code, train a first add-on on the prepared clips, compare

Thanks for report 007: clear and complete, and the channel works. Findings I took
from it: niqqud is essential, v2 and v3 are equivalent, intelligibility is already
good (error about 0.03), so the remaining Hebrew problem is accent and naturalness,
which the recogniser cannot see. This task trains a small add-on (LoRA) on the
host's own clips, which is the first thing that can move accent and voice.

The training code is new and has NOT been run anywhere (the cloud machine has no
torch yet). You are its first tester. For this task you may fix bugs in
`src/ystream_voice/training.py`, `scripts/train_voice.py`, `scripts/hebrew_ab.py`
and `tests/test_training.py` locally; put every change in the report as a unified
diff so I can apply it. Do not change other files.

## 1. Update and test
`git pull` on radio-foundation (expect the commit "Lead-in and fade-in for chunk
starts..." or later), then `pytest -q`. Expect 82 old tests plus 6 new ones in
`tests/test_training.py` (they build a tiny random model; no downloads). If the
training tests fail, fix what is wrong when the cause is clear from the traceback
(wrong attribute names, shapes, API differences in transformers 5.2 / torch 2.11),
and report tracebacks and diffs. The test
`test_logits_match_upstream_forward_layout` matters most: it proves the training
sequence layout is the one the pretrained model uses. Do not weaken it to make it pass;
if it cannot pass, stop the task and report.

## 2. Train
    python scripts\train_voice.py --data data\yaniv --out adapters\yaniv_v1 --steps 1200 --eval-every 100
Save console output to `out\train_yaniv_v1.log`. Check free card memory first; if
less than 5 GB is free, add `--batch-tokens 800`. If it runs out of memory, halve
`--batch-tokens` and retry (down to 400). Report: number of training / validation
clips, every "step N: train ... val ..." line, the final "Held-out speech loss"
line, peak GPU memory, minutes taken. `adapters/` is not git-ignored yet: do not
add it to git.

Sanity checks to report: (a) is the step-0 validation loss a plausible number for
a working pretrained model (roughly 2 to 5), or absurdly high (above 8), which
would mean the layout or features are wrong; (b) does validation loss go down,
flatten, or go up (overfitting) over the run.

## 3. Compare base against trained
    python scripts\hebrew_ab.py --voice voices\yaniv_he_ref.wav --out out\hebrew_ab_v1 --settings v3_default --adapter yaniv_v1=adapters\yaniv_v1 --takes 3
Report the table exactly (it now has a "voice sim" column: similarity of each take
to the reference voice). Then the same for English:
    python scripts\hebrew_ab.py --lang en --file tests\radio_scripts\en_sentences.txt --voice voices\yaniv_en_ref2.wav --out out\english_ab_v1 --settings v3_default --adapter yaniv_v1=adapters\yaniv_v1 --takes 3
For the English sentence file, create `tests\radio_scripts\en_sentences.txt` yourself
with 8 short radio-style English sentences (one per line) and include it in the report.

## 4. Listening files for Yaniv
    python scripts\try_voice.py --voice voices\yaniv_he_ref.wav --file tests\radio_scripts\he_basic.txt --out out\he_basic_trained_v1.wav --adapter adapters\yaniv_v1 --check
    python scripts\try_voice.py --voice voices\yaniv_he_ref.wav --file tests\radio_scripts\he_basic.txt --out out\he_basic_base_v2.wav --check
    python scripts\try_voice.py --voice voices\yaniv_en_ref2.wav --file tests\radio_scripts\en_basic.txt --out out\en_basic_trained_v1.wav --adapter adapters\yaniv_v1 --check
(These now have a 0.15 s lead-in and a short fade-in at chunk starts.) Report their
Saved / Time / Takes rejected lines. Add them to Yaniv's listening page, labelled
"base model" and "trained on your clips", so he can compare in the morning.

## 5. Report
Write `reports/008.md` and push it. Then poll for task 009.
