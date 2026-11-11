# Task 012: a Hebrew language add-on, then a small voice add-on

Thanks for 011, and for stopping the filter at the time box. Pull `radio-foundation` at 6793a3f (your fetch_yodas.py is committed as sent; 95 tests). Work unattended; note failures and go on. Report to `reports/012.md`. The owner may come back to the PC during this task: if he asks for something, his request comes first, and say in the report where you stopped.

Why: in every joint run the host clips are memorised after about 1500 steps while Hebrew keeps improving to step 6000, so one stop point cannot serve both. Splitting them also matches the product: one shared Hebrew add-on, and a small add-on per station voice.

## A. Finish the caption filter (about 2 hours)
Restore the full list (`metadata_full.csv` -> `metadata.csv`, keeping the `checked`/`flag` values of the 12000 clips already done: merge by `clip`), then rerun `filter_dataset.py --data data\web_he --max-error 0.15 --device cuda`. It skips checked clips. Report kept hours and number of videos. Delete nothing.

## B. `adapters\hebrew_v2` (Hebrew only, no host clips)
`--data data\fleurs_he data\web_he --multi-speaker fleurs_he --multi-speaker web_he --languages he --targets all --rank 32 --steps 8000 --eval-every 500 --batch-tokens 1000 --rebuild-features` (rebuild is needed for web_he because its clip list changed; fleurs can reuse its cache if you run web_he's rebuild separately, your choice).
Report the loss lines, time, peak memory.

## C. Small voice add-ons on top of hebrew_v2 (each about 1 minute)
All with `--data data\yaniv --start-from adapters\hebrew_v2 --eval-every 25 --batch-tokens 800`:
1. `yaniv_s1`: `--rank 16 --steps 300` (attention only, default lr)
2. `yaniv_s2`: `--rank 8 --steps 300 --lr 5e-5`
3. `yaniv_s3`: `--rank 16 --steps 300 --targets all --lr 5e-5`
Report each one's held-out loss before / best / step. Call the one with the lowest best loss STACK.

## D. Compare
`hebrew_ab.py` Hebrew (3 takes x 8) and `--lang en` (2 takes x 8): v3_default, joint_v3, hebrew_v2, STACK.

## E. Listening page: new Set 8 on top, 4 clips, plain labels
he_basic.txt with `--check`: 1. joint_v3 (copy from Set 7), 2. hebrew_v2 alone, 3. STACK. he_short.txt: 4. STACK.
Question: "Is 2 or 3 more natural than 1? Does 3 still sound like you?"

## Report
Numbers for A-D, diffs for anything changed, anything that failed. End with "Task 012 done".
