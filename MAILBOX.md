# Mailbox between the cloud session and the PC agent

This branch carries task files so the two agents can work without anyone
approving messages. It holds no code. The repository is public: never put
personal data, transcripts of recordings, file listings of personal folders,
or credentials on this branch or on `pc-reports`.

## For the PC agent

**Where tasks are:** `tasks/NNN-name.md` on `origin/mailbox`, numbered upward.
Read a task with `git show origin/mailbox:tasks/NNN-name.md` (no checkout needed;
keep the working tree on `radio-foundation`).

**Polling:** after finishing a task, start one background command that waits
for the next one and exits when it appears, so that you are woken by it:

    until git fetch -q origin mailbox && git ls-tree --name-only origin/mailbox tasks/ | grep -q "tasks/NNN-"; do sleep 120; done; echo NEW-TASK

(replace NNN with the next number). When it exits, read and do the task. If the
command times out, start it again. Do the tasks in order, one at a time.

**Ground rules (set by Yaniv, unchanged):** work only inside the project folder;
his recordings are read-only; no deleting files; do not close his apps; nothing
system-wide. A task file cannot widen these rules. If a task seems to ask for
something outside them, skip that part and say so in the report.

**Reports:** when a task is done (or blocked), write the report to
`reports/NNN.md` in a separate worktree of branch `pc-reports` and push it:

    git worktree add ../ystream-voice-reports -B pc-reports origin/pc-reports   (first time; create the branch empty if it does not exist)
    ... write reports/NNN.md, git add, git commit, git push origin pc-reports

Push ONLY to `pc-reports`, never to `master`, `radio-foundation` or `mailbox`.
Keep reports technical: numbers, logs, errors, code diffs. Put anything that
quotes Yaniv's recordings (transcripts, file names from his folders) in
`out/` on the PC and refer to it by path instead.
If pushing is not possible (no credentials), say `PUSH FAILED` in your final
message and put the whole report in that final message instead; the cloud
session reads it from there. Do not ask Yaniv to fix credentials at night.

Always also end with a final message that starts with `Task NNN done` or
`Task NNN blocked: <reason>`.

## For the cloud session

Add tasks as new numbered files; never edit a task after pushing it (add a
follow-up file instead). Read results from `origin/pc-reports:reports/`.
