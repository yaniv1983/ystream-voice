# Task 011: far more natural Hebrew speech

Thanks for 010. Pull `radio-foundation` at 0352bda (95 tests; `en_sentences.txt` is now in the repo, drop your local copy if git complains). Work unattended; if a step fails, note it and go on. Report to `reports/011.md`.

Why: FLEURS is 8 hours of people reading sentences. The loss on Hebrew kept falling with more capacity, so the next lever is more and more natural Hebrew. The host's own clips overfit within 50 steps, so they stay a small part of a joint run.

## A. Find a larger open Hebrew source (time box: 45 minutes of looking)
Candidates, in order. Use the first that is **not gated** and whose licence allows commercial use (CC0, CC BY, CC BY-SA are fine; anything "NC", "research only" or a custom licence is not). Report the licence text line and URL for whatever you use.
1. `espnet/yodas` on Hugging Face, Hebrew manual-caption subset (`he000`; CC BY 3.0, YouTube speech with uploader captions). Inspect the repo layout yourself (audio tars + text json per shard).
2. `espnet/yodas2` Hebrew, same idea.
3. Mozilla Common Voice Hebrew (CC0), only if you can download it without logging in.
Do not use SASPEECH, ILSpeech, or ivrit.ai sets (gated / non-commercial).

## B. Build `data\web_he` (write `scripts/fetch_yodas.py` or similar, modelled on `fetch_fleurs.py`; send me the script in the report)
- Cap: about 40 hours of audio or 25 GB of download, whichever first. Check free disk before starting and keep 30 GB free.
- Keep utterances of 3-15 s. Merge neighbouring captions of the same video when each is shorter than 3 s and the gap is under 0.3 s, if that is easy; otherwise just drop the short ones.
- Write 16 kHz mono WAV clips and `metadata.csv` with columns `clip,language,source,start,end,sample_rate,text,flag` (`source` = video id; no titles or channel names).
- Drop text containing Latin letters or digits-only lines (the normaliser handles digits, but captions with English are code-switching we do not train yet), and text with brackets like [music].
- Then run `python scripts/filter_dataset.py --data data\web_he --max-error 0.15 --device cuda` (new). It transcribes every clip with the Hebrew Whisper and flags clips whose audio does not say their caption. It is resumable. If it would take more than 3 hours, cut the dataset down rather than wait.
- Report: hours downloaded, clips kept, hours kept after the filter, share flagged.

## C. Train
1. `adapters\joint_v3`: `--data data\fleurs_he data\web_he data\yaniv --multi-speaker fleurs_he --multi-speaker web_he --repeat yaniv=6 --targets all --rank 32 --steps 6000 --eval-every 250 --batch-tokens 1000`
   - Feature building for web_he is one-time and may take a while; that is expected.
   - If the voice grouping finds very few clips per voice and drops more than 30% of web_he, tell me the numbers and continue.
2. If A/B produced nothing usable, instead train `adapters\joint_v3` on fleurs + yaniv with `--targets all --rank 64 --steps 4000 --repeat yaniv=4` so the night is not wasted.
Report per-dataset held-out loss lines, time, peak memory.

## D. Compare
- `hebrew_ab.py` Hebrew (3 takes x 8 sentences): v3_default, joint_v2, joint_v3.
- `hebrew_ab.py --lang en` (2 takes): v3_default, joint_v2, joint_v3.
- `prosody_stats.py` on the listening files below plus real_he / real_en.

## E. Listening page: new Set 7 on top, at most 6 clips, plain labels
Hebrew, script he_basic.txt, all with `--check`:
1. base model
2. joint_v2
3. joint_v3
Hebrew, script he_short.txt: 4. joint_v3
English, en_basic.txt with the lively English clip (en_004_0000) as voice: 5. joint_v2, 6. joint_v3
Question for the listener: "Rank Hebrew 1-3 by how native and natural they sound. Is the accent problem smaller in any of them?"

## Report
Tables and numbers for B-D, the fetch script, diffs for anything else you changed, anything that failed. End with "Task 011 done".
