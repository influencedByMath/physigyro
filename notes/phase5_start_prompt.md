# Start prompt for Phase 5 (new session, from 2026-10-22)

Written for: Claude starting a new session, and the author. Copy everything between the two lines into the new session.

**Names.** "Task 5" in older notes (D-40, D-41, `notes/task4_results.md`, the handoff) means **Phase 5, the ML tuner**. They are the same work. From now on it is called Phase 5, with its own plan `notes/phase5_plan.md` and its own tasks (Phase 5 Task 1, Task 2, ...).

---

Start Phase 5 (ML tuner) of my thesis. This session writes **the Phase 5 plan only**, for my approval. Do not write code or train models before I approve the plan.

**Context.** Repository `D:\research&thesis`, branch `phase4` (from `phase3`; neither merged into `main`). Phases 0-4 are done ✅. Decisions D-01 to D-43 are recorded; the next ID is **D-44**. GitHub: remote `origin` = `influencedByMath/physigyro` (public, D-43); `phase3` and `phase4` are pushed; GitHub's `main` keeps its own 'Initial commit' (MIT `LICENSE`) until I merge. Read these first, in order:
1. `notes/session_handoff.md` (where things stand, standing rules, environment notes).
2. THESIS.md: Section 0; Section 2, rows D-36 to D-43; Section 6 (targets); 7.3 (limits); 7.6 (training set); **7.7 (ML tuner)**; **7.8 (safety check)**; 7.9 (explainability); Section 8 (the experiments and figures of Phase 5: Table 3, Figs. 5-7); Section 10, Phase 5 checklist; Section 17, open VERIFY items.
3. `notes/task4_results.md` and `data/labels/train_v1_meta.json` (the training set: 10,000 generator cases; D-41: Mode A on the 9,976 and Mode B on the 9,999 labelled cases).
4. `notes/task4_plan.md` items T4-1, T4-3 and T4-4; `notes/task3_results.md` (the real-noise and real-gain cases, both halves); `notes/decision_log.md` Sections C, E and F.
5. The code you will reuse: `src/optimizer.py`, `src/robust.py` (or wherever the D-36 worst-case check lives), `experiments/phase4/rq1_run.py`, `experiments/phase4/dataset_gen.py`, `src/priors.py` (`require_fit_half`, `require_fit_rows`).

**The plan must cover, each with evidence from the files and your recommendation:**
- Features and targets (7.7): exactly which record fields, the log-PSD bands, peaks and floor; continuous outputs as log(f); the harmonic count as a class; one model per mode.
- Data split. ⚠️ 7.7 says 70/15/15 grouped by drone id; T4-3 suggests grouping by the generator's source vehicle so a real vehicle's noise shape is not in both train and test. Show the numbers and ask me.
- Validation and test sets: the synthetic test set; the 520 fit real-noise cases as a separate validation set (T4-1 (a)); the test-half real set in its three parts (real noise, real gains, Scenario B with RPM), used only once, for the final Table 3.
- Models B0-B3, M1, M2 and the small hyperparameter search (at most 30 configurations, GroupKFold 5).
- Learning curve at 1,000 / 2,500 / 5,000 / 10,000 cases and the threshold that triggers extending the set to 20,000 (T4-4). ⚠️ I decide the threshold.
- `safety_check.py` as in 7.8, and ⚠️ the open fallback question: for a real drone whose flown setting is "reference unsafe", the repair line ends at an unsafe reference; should a known-stable fallback replace it (this changes 7.8)? Give the data and options.
- Metrics: regret median and 95th percentile; all three goals; violation rate before and after the check, with the rule-of-three / Clopper-Pearson bound; prediction time compared with optimizer time.
- Explainability (Figs. 6, 7) and what Phase 6 needs from Phase 5 (exported models; the open TBD targets E10, E11, E13 and the overshoot limit stay for Phase 6).
- Tests to write first, mutation checks, expected run times, and the order of tasks.

Ask me every open decision with AskUserQuestion, with your recommended option first. Then write `notes/phase5_plan.md`, check every number in it against the files, record my answers as D-44 onward in THESIS Section 2 and `notes/decision_log.md`, update `notes/session_handoff.md`, run the full suite, commit locally, and **stop** for my approval. Push to GitHub only if I ask.

**Stop and report immediately** on any landmine: a contradiction between THESIS and the code or data, a test-half file needed for a choice, a count that does not match the meta file, or anything not covered by the decisions. If anything is unclear, ask me with your recommended options. Do not guess.

**Report format:** mark every item ✅ done, ⬜ open, ⚠️ needs my decision.

**Constraints (unchanged):** tests first and see them fail; full suite before each commit; every decision in THESIS Section 2 and `notes/decision_log.md`; the fit half for every choice, the test half only for evaluation; no merge; no push unless I ask; never run mutation checks during a long run.

**Environment notes:** Windows; Python is `.venv/Scripts/python`. Shell heredocs and `python -c` mangle backslashes and quotes, so write scripts with the Write tool. Markdown table cells must not contain a bare `|`. Stop background jobs with TaskStop. A full suite takes about 4-7 min (448 tests).

---
