<!--
This PR template is auto-applied to every pull request on this repo.

Before opening: pick ONE card from
  docs/superpowers/plans/2026-05-10-pcvr-roadmap-v3-architecture.md
Implement the single lever described in that card. Do NOT bundle.
The discipline this template enforces is exactly what prevented (and what's
needed to prevent more of) the v3/v4/v5 regression pattern.
-->

## Card

- v3 roadmap card ID (e.g., B1, B2, C3):
- Card title:
- Anchor at submission time: pcvr v0 = AUC 0.81144

## What this PR changes

(One paragraph. Describe the SINGLE lever changed and which file(s) you modified. If you're changing more than one thing, this PR is bundling — split it.)

## Discipline checklist (from roadmap v3 §"Picking up a card")

- [ ] I read all 12 fields of the card, not just **Direction** and **Why**.
- [ ] I am changing exactly **ONE lever** (no bundling B1+B2, encoder+loss, data+model, etc.).
- [ ] I wrote the card's **Test contract** as a real pytest test in `pcvr/tests/`.
- [ ] The test **passes locally** on the conda env (`taac`).
- [ ] I diffed the card's **Integration** field's files against my changes and confirmed I did **NOT** touch them.
- [ ] I have a clear **Rollback** — flipping the new config flag returns to the prior anchor with no side effects.

If any box above is unchecked, do NOT merge this PR yet. The most common skipped box historically has been #3 (writing the test contract as a real pytest test) — that's the v4-style regression surface.

## Test contract — output

Paste the pytest invocation and the PASSED line for your new test:

```
$ /c/Users/84447/anaconda3/envs/taac/python.exe -m pytest pcvr/tests/test_<your_test_file>.py::test_<your_test_name> -v
... PASSED ...
```

If the test couldn't be made local (e.g., it depends on the platform's full data), say so here and link to the platform log line that exercised it.

## Submission status

Pick one:

- [ ] **Code-only PR.** No leaderboard submission yet. The zip rebuilds will happen later, against the merged state.
- [ ] **Submission PR.** I built and archived the submission artifacts:
  - [ ] Step-1 zip built via `pcvr/build_step1_submission.py`.
  - [ ] Step-3 zip built via `pcvr/build_step3_submission.py`.
  - [ ] Both archived under `submissions/YYYY-MM-DD_<tag>/`.
  - [ ] `submissions/YYYY-MM-DD_<tag>/NOTES.md` written (which card, which lever, test-contract status, config file used).

If already graded on the leaderboard:

- Leaderboard AUC:
- Δ vs anchor 0.81144:
- Submission slot used: <date> + slot index (1/2/3 for that day)

## Notes / blockers

(Optional. Anything that didn't go to plan, anything the reviewer should know, anything you'd flag for the next person picking up an adjacent card.)
