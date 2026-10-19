# Start prompt for Task 4 (new session, 2026-10-20)

Copy everything between the two lines into the new session. Before you paste it, make sure Windows sleep is really off (Settings > System > Power & battery).

---

Start Task 4 (`dataset_gen.py`, the training set) of my thesis now. I have switched off Windows sleep, and the laptop is free for you all day.

**Context.** Repository `D:\research&thesis`, branch `phase4` (from `phase3`; neither merged; no push). Phase 4 Tasks 1-3 are done. Decisions D-01 to D-40 are recorded; the next ID is **D-41**. Task 4 is approved as D-40 and NOT started. Read these first, in order:
1. `notes/session_handoff.md` (where things stand, standing rules, environment notes).
2. `notes/task4_plan.md`: the approved plan; follow it exactly.
3. THESIS.md Section 2, rows D-36 to D-40; Section 7.6 (training set); Section 7.7 (the features Task 5 needs, so every record must hold them).
4. `notes/task3_results.md` Sections 5, 6 and 8, and `experiments/phase4/rq1_run.py`: the Task 3 machinery you reuse (`synthetic_drones`, `_synthetic_work`, `solve`, `run_part`, `build_generator`, `generator_draw`, `git_hash`, `SEED = 20261018`), plus `src/tier3.py` (`sample_drones`) and `src/optimizer.py` (`optimize_checked`).

**My decisions (D-38, D-40), all approved:**
- 10,000 cases, labelled with exactly the D-37 optimizer (Modes A and B; nominal limits and the D-36 worst-case limits on the 8 corners; fast coarse grid, full-grid check and refinement; reference-seeded). No fast-grid-only labels and no greedy notch search.
- All 10,000 are generator cases: Tier 3 synthetic drones with generator spectra (generator fitted on the fit half only). No test-half data of any kind. Scenario B at its natural share (~10 %). Flagged drones are kept and flagged.
- The 1,000 RQ1 synthetic cases (`experiments/phase4/rq1_synthetic.json`, code `1e9c561`) are the first 1,000, reused behind the guard: re-solve 20 random RQ1 cases with the current code; every setting and metric must match to 1e-9.
- T4-1 (a): the 520 fit-half real-noise cases of Task 3 are NOT in the 10,000 (they become a separate validation set in Task 5).
- T4-2 (a): commit `data/labels/train_v1.jsonl.gz` and `data/labels/train_v1_meta.json`. Keep `data/labels/train_v1_spectra.npz` local and git-ignored, store its SHA-256 in the meta, and add a test that regenerates it bit-exactly from the seed. Git-ignore the partial file too.
- Run guards: the partial file starts with a header (seed, n, git hash, config hash), and a restart under a different version or config refuses to resume; a run refuses to start on a dirty working tree; no mutation check or any other run while the run goes (workers import source files from disk).
- Joblib with 6 workers, chunks of 48, resumable (as `run_part`). Expected: about 12 min sampling, then about 8.5 h labelling the 9,000 new cases (3.4 s per case, measured).
- Meta: seed, n, git hash, RQ1 commit and reuse-check result, limits, uncertainty width and delay step, FAST and FULL grids, f_hi, chip delays, SHA-256 of all inputs and outputs, Python/NumPy/SciPy/joblib versions, n_jobs, start and end times, counts (draws, in window, kept, Scenario B, flagged).

**Order of work:**
1. Write the tests first (plan Section 7), watch each fail, then implement `experiments/phase4/dataset_gen.py` (reuse the Task 3 code; do not change the label code).
2. Mutation check of the new code, then the full suite, then a local commit of the code. The run needs a clean tree.
3. Run: sample, then the reuse check, then the smoke chunk (48 new cases; check format and limits automatically), then label the rest in chunks, then merge, then write the meta. Run it in the background and watch it with a monitor that reports progress AND every failure (Traceback, Error, guard refusal, process exit), not only success.
4. After the run: verify the merged file (10,000 unique ids, RQ1 records unchanged, every optimum within the limits, Scenario B share, spectra SHA-256 matches), run the full suite, and commit the records and meta.
5. Record any new decision or ruling in THESIS Section 2 (from D-41) and `notes/decision_log.md`; update THESIS Sections 7.6, 10 (Phase 4 checklist) and 17 (session log), `notes/session_handoff.md` and the plan ledger.

**Stop and report immediately** on any landmine: a reuse-check mismatch, a chunk failure, a guard trip, a test-half path opened, a result outside the limits, a count far from the plan (for example a Scenario B share far from ~10 % or a kept share far from ~87 %), or anything that contradicts the plan or Task 3. If anything is unclear or not covered by the plan, ask me with your recommended options. Do not guess.

**Report after completion:**
- final counts (draws, in window, kept, Scenario B, flagged);
- any anomalies (chunk failures, reuse-check results, restarts);
- the meta file contents (seed, hashes, versions);
- the first 10 rows of `train_v1.jsonl.gz` (in a readable short form);
- the full suite result and the commits made.

Then stop. Task 5 (ML tuner) starts only after my review.

**Constraints (unchanged):** tests first; full suite before each commit; record every decision in THESIS Section 2 and `notes/decision_log.md`; stop on any landmine; commit locally only, no merge, no push; update `notes/session_handoff.md` before reporting back. Work carefully: every decision must rest on data, and if reality contradicts your expectation, ask me.

**Environment notes:** Windows; Python is `.venv/Scripts/python`. Shell heredocs and `python -c` mangle backslashes and quotes, so write scripts with the Write tool. Markdown table cells must not contain a bare `|`. Stop background jobs with TaskStop. A full suite takes about 4 min.

---
