# Task 007: confirm this channel, then report Tasks 5 and 6

1. Check that you can push: create `reports/007.md` on branch `pc-reports` as
   described in MAILBOX.md, containing the items below, and push it. If the push
   fails, follow the PUSH FAILED fallback.

2. In the report, give the results of the overnight job (Tasks 5 and 6) once it
   has finished; if it is still running, wait for it:
   - `en_basic_ref2`: the Saved / Time / Takes rejected lines.
   - Spill test: run or skipped; if run, the real-time factors and peak shared memory.
   - Hebrew A/B: the final table printed by `scripts/hebrew_ab.py`, exactly.
   - Onsets for `en_basic_raw.wav` versus `en_basic.wav`: per chunk, ms of silence
     before the first sound and the level of the first 10 ms relative to 50-100 ms
     later. State plainly whether chunks start abruptly at full level.
   - Dataset v2 (Task 5 step 2-3): summary lines per language, usable minutes after
     flags, and per-file covered/uncovered seconds as numbers only (refer to files
     by `Script NN` / `DjN` / `news`, no transcript text).
   - Any errors or tracebacks, in full.

3. Then start polling for task 008.
