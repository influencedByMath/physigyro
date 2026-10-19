# Task 4 plan: `dataset_gen.py`, the training set (D-38)

Written 2026-10-19. **Approved 2026-10-19 as D-40** with T4-1 (a), T4-2 (a), T4-3 noted for Task 5, T4-4 accepted (threshold set in Task 5), T4-5: run on 2026-10-20 in a new session, after the author has switched off Windows sleep. **Task 4 is not started.** Start prompt: `notes/task4_start_prompt.md`.

All timings below are measured in Task 3, not estimated: `experiments/phase4/rq1_synthetic.json` and a scratch timing run of the sampler.

## 1. What the 10,000 cases are (the split)

- **All 10,000 are generator cases.** Each case is one Tier 3 synthetic drone, drawn by the D-36 sampler `tier3.sample_drones`, plus one spectrum from the Phase 3 generator.
  - The generator is fitted on the fit half only: `build_generator` reads only `fit_*` files, through `priors.fit_path`, which refuses `test_*` paths.
  - Labels: Modes A and B of exactly the D-37 optimizer (D-38 (1)).
- **No test-half data of any kind** (D-38 (6)): no test spectra, and no test real-noise or real-gain cases. A test checks that no `test_` path is opened while the set is built.
- **Scenario B at its natural share** (D-38 (2)). The generator draws its source vehicle uniformly, and 5 of the 50 sources carry motor harmonics, so about 10 % of cases are Scenario B (101 of the 1,000 RQ1 cases). The share is not forced.
- **Flagged drones are kept and flagged** (D-37 (6)): the PX4 defaults pass the nominal limits but fail the worst case. This is about 4.4 % of cases.
- **The test half in Task 5 needs no new run.** The 430 test real-noise cases and the 28 test real-gain cases already carry labels from the same optimizer (Task 3). They are the real held-out test set of THESIS 7.7, parts 1-3.
- **T4-1 (question): what happens to the 520 fit-half real-noise cases of Task 3?** They are legal (fit half), but they are not generator cases.
  - (a) **Recommended:** keep them out of the 10,000 and use them as a separate real-spectrum validation set in Task 5. Models can then be chosen on real spectra without touching the test half.
  - (b) Add them to training (10,520 cases; each real fit spectrum would appear 10 times).

## 2. Reuse of the 1,000 RQ1 synthetic cases (D-38 (3))

- **The same cases come back first.** The sampler is sequential, so with seed 20261018, `sample_drones(n=10,000)` returns the 1,000 RQ1 drones as its first 1,000.
  - Checked in a scratch run: the first 200 of 200 match, including parameters, chip, flag and spectrum source/scale.
  - A test will check all 1,000.
- **Their labels are taken from `rq1_synthetic.json`** (code `1e9c561`).
  - The label code has not changed since then: `src/` is unchanged, and the only change in `rq1_run.py` is the Mode C front path, which labels do not use.
- **Guard before the run:** re-solve 20 randomly chosen RQ1 cases with the current code.
  - Every setting and metric must match to 1e-9.
  - Any difference is a landmine: I stop and report, and reuse nothing.
- **Fields the RQ1 records lack** (PID gains, the default's robust metrics) come from the re-sampled drone, which the check above shows to be identical.

## 3. Output files (T4-2)

| File | Content | Size (est., scaled from RQ1) |
|---|---|---|
| `data/labels/train_v1.jsonl.gz` | One record per case: id; drone ($f_c$, $\tau_m$, $T_d$, nominal $T_d$, delay step, $r$, rates, PID gains, chip, chip delay, motor frequencies, flag); spectrum source and scale; metrics and worst-corner metrics of the PX4 default; for Modes A and B: feasible, settings $\theta^*$, metrics, worst-corner metrics, solver seconds | about 25 MB raw, about 5 MB gzipped |
| `data/labels/train_v1_spectra.npz` | Input spectra (float32, 1,521 points from 20 to 780 Hz at 0.5 Hz) in the same order as the records. Needed for the Task 5 features (THESIS 7.7); RQ1 did not store spectra | about 60 MB |
| `data/labels/train_v1_meta.json` | Seed, hashes, versions, counts (Section 6) | small |

**T4-2 (question): what goes into git?**
- (a) **Recommended:** commit the records and the meta. Keep the spectra file out of git, store its SHA-256 in the meta, and add a script, with a test, that regenerates it bit-exactly from the seed.
- (b) Commit everything (about 65 MB; GitHub warns above 50 MB per file).
- (c) Commit only the meta.

## 4. Resumable chunks and joblib

- **Reused from Task 3:** `rq1_run.run_part` (already tested).
  - Chunks of 48 cases, run with `joblib.Parallel(n_jobs=6)`; the machine has 8 logical cores, so 2 stay free for the system.
  - Each finished chunk is appended to `train_v1.partial.jsonl`, and a restart skips the ids already done. A shutdown or crash therefore loses at most one chunk (about 3 min).
- **New guards (tests first):**
  1. The partial file starts with a header line (seed, n, git hash, config hash). A restart under a different code version or config refuses to resume. This is the lesson of the stale partial files in Task 3.
  2. A run refuses to start on a dirty working tree, so the stored hash is the code that ran. The code is committed before the run, and the data after it.
  3. No mutation check and no other run while it runs, because the workers import source files from disk.
- **Order of the run:**
  1. Sample all 10,000 drones (sequential, about 12 min) and write the spectra file.
  2. Run the reuse check (Section 2).
  3. Label cases 1,000-9,999 in chunks.
  4. Merge with the 1,000 RQ1 records into one file and write the meta with the output hashes.
- **Smoke test:** the first chunk (48 new cases, about 3 min) is checked automatically for format and limits. The run then continues from the same partial file. I stop only on a landmine.
- **Your action:** switch off Windows sleep for the night of the run (Settings > System > Power & battery). A restart is safe, but sleep pauses the run.

## 5. Expected wall time on 6 workers

| Step | Basis (measured) | Time |
|---|---|---|
| Sample 10,000 drones | 0.07 s per kept drone, one process | about 12 min |
| Reuse check, 20 cases | about 18 s per case, 6 workers | about 1-2 min |
| Labels for 9,000 new cases | 3.4 s per case of wall time (Task 3: 18,733 s on one worker per 1,000 cases; mean 19.5 s per Scenario A case, 11.9 s per Scenario B case) | **about 8.5 h** |
| **Total** | | **about 9 h (one night)** |
| Fallback of 5,000 cases (4,000 new) | same | about 3.8 h |

## 6. Seed and git hash

- **Seed:** `SEED = 20261018`, the RQ1 seed (needed for the reuse).
  - Design draws come from `default_rng(seed)` and spectrum draws from `default_rng([seed, 1])`, as in Task 3.
- **Stored in `train_v1_meta.json`:**
  - seed and n;
  - git hash of HEAD (clean tree required), plus the RQ1 commit `1e9c561` for the first 1,000 and the result of the reuse check;
  - limits (nominal and worst case), uncertainty width and delay step, `FAST` and `FULL` grids, $f_{hi}$, chip delays;
  - SHA-256 of every input (`model_fits.json`, `generator_check.json`, `priors_from_logs.json`, `fit_windows.csv`, `fit_psd_native.npz`) and of every output file;
  - Python, NumPy, SciPy and joblib versions; n_jobs; start and end time;
  - counts: draws, in window, kept, Scenario B, flagged.
- **Review Focus item 4** (`notes/phase4_plan.md`), as a test: the same seed gives the same first 100 cases (drones and spectra).

## 7. Tests first (planned)

1. Same seed gives the same first 100 drones and spectra; the first 1,000 drones are the RQ1 drones.
2. No test-half path is opened (leakage guard).
3. Resume: stopping after a chunk and restarting gives the same file; a header mismatch (git hash or config) refuses.
4. A dirty working tree refuses to start.
5. The record schema holds every field Task 5 needs (the THESIS 7.7 features, labels A and B, id, source).
6. Merge: 10,000 unique ids, and the RQ1 records are unchanged.
7. The spectra file regenerates bit-exactly (only with T4-2 (a)).

After the tests: a mutation check of the new code, then the full suite and a commit, then the run.

## 8. Questions for you

- **T4-1 Fit-half real-noise cases (520):** (a) separate real-spectrum validation set (recommended); (b) add them to training.
- **T4-2 Git:** (a) records and meta in git, spectra regenerable (recommended); (b) everything; (c) meta only.
- **T4-3 ML split grouping (for information; decided in Task 5).**
  - THESIS 7.7 groups the split by drone id, but every drone is unique. Meanwhile the spectra of one generator source vehicle appear in about 200 cases.
  - My recommendation for Task 5 is to group by generator source (50 groups), so the synthetic test set holds unseen source vehicles.
  - The records store the source, so nothing in Task 4 depends on this choice.
- **T4-4 Extension policy (why 10,000 and not 15,000 or 20,000 now).**
  - Task 5 runs a learning curve (1,000 / 2,500 / 5,000 / 10,000 cases).
  - If the validation regret still falls clearly from 5,000 to 10,000, the set is extended to 20,000 by appending cases: another about 9.5 h, with nothing recomputed. You set the threshold then.
- **T4-5 When to run:** the run needs the laptop for about 9 h. Options: start right after the code is committed, or at a time you choose (for example overnight).
