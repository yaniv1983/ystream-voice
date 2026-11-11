# Task 014: resume after the power cut

The PC lost power on 11 Nov around 09:15 UTC, during the `hebrew_v2` training of task 012.
Do this first, before anything else in 012 or 013.

## 1. See what survived (report briefly)
- `git pull` on `radio-foundation` (latest is f60cc9e or newer), run the tests.
- Is `data/web_he` intact, and how far did the caption filter get (rows with `checked`)?
- Is there a saved best checkpoint for `hebrew_v2`? At which step, with which held-out loss?
- Check that no dataset CSV or adapter file is truncated (open each, count rows / load once).

## 2. New order of work
Glitches at phrase starts are the owner's top complaint, and task 013 is short. So:
1. **Task 013 first** (glitch traces, lead silence 0 vs 3, Set 9 with `joint_v2`). Push `reports/013.md`.
2. **Then task 012.** If a `hebrew_v2` checkpoint from step 3000 or later exists, do not retrain:
   use it as it is for the voice add-ons and say so in the report. Otherwise restart the
   training from the beginning.

## 3. For the future
If the training script cannot resume from a checkpoint after an interruption, say so in the report
and I will add it. Do not write that code yourself during this task.

Same ground rules as always. Reports go to `pc-reports` only.
