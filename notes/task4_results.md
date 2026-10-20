# Task 4 results: the training set `train_v1` (2026-10-20)

Written for: the author. Plan `notes/task4_plan.md` (D-38, D-40). **STOP after Task 4: Task 5 starts only after your review.**
Every number below comes from `data/labels/train_v1_meta.json` or from the records (`dataset_gen.py verify` and an independent read-back script).

## 1. What was made

| File | In git | Content |
|---|---|---|
| `data/labels/train_v1.jsonl.gz` (6.5 MB) | yes (`df6b02e`) | 10,000 records `syn0000`-`syn9999`, one JSON line each: the RQ1 record format (`rq1_run._synthetic_work`) plus `gains` (Kp, Ki, Kd) and `origin` ("rq1" for the first 1,000, "task4" for the rest) |
| `data/labels/train_v1_meta.json` | yes | seed, n, git hash, config hash, RQ1 commit and reuse check, limits, set, grids, f_hi, chip delays, SHA-256 of all inputs and outputs, versions, n_jobs, chunk, start/end, counts, label summaries |
| `data/labels/train_v1_spectra.npz` (61 MB) | no (git-ignored, T4-2 (a)) | `S` float32 10,000 × 1,521 (20-780 Hz, 0.5 Hz), `f`, `ids` in record order; SHA-256 `576b045d…923e` in the meta |

- **Code:** `experiments/phase4/dataset_gen.py` (commit `9324a5f`; the labels use `rq1_run._synthetic_work` unchanged, i.e. exactly the D-37 optimizer).
- **Tests:** `tests/test_dataset_gen.py`, 43 tests written first and each seen red against a stub. Mutation check `experiments/phase4/mutation_check_task4.py`: 30 of 30 mutants caught, none by timeout, file restored byte-identically.
- **Regenerate the spectra:** `.venv/Scripts/python experiments/phase4/dataset_gen.py check-spectra` (about 9 min; result after the run: **MATCH**).

## 2. The run

- **One invocation, no restart, no guard trip, no chunk failure.** Commit `9324a5f`, clean tree; start 11:42, end 21:04 (9 h 22 min), 6 workers, chunks of 48.
- **Order:** sampling (525 s), count check, spectra file, reuse check (84 s), smoke chunk, 187 more chunks, merge, verification, meta.
- **Reuse check:** 20 random RQ1 cases (seed 20261024: syn0020, syn0039, …, syn0887) re-solved with the current code; **20 of 20 match, largest difference 0** (bit-identical). The first 1,000 drones of the sampler equal the RQ1 drones in every field.
- **Smoke chunk** (syn1000-syn1047): format and limits checked automatically; 48 of 48 feasible in both modes. Every later chunk was checked the same way.
- **Speed:** 3.2 s per case until about 18:30, then up to 3.7 s. Other programs (a video meeting app, Chrome, OneDrive) used most of the CPU in the evening, and some chunks waited for one slow case. The labels do not depend on this; only the per-case `seconds` of the late cases are inflated.

## 3. Counts

| Quantity | Task 4 (10,000) | RQ1 (first 1,000) |
|---|---|---|
| Draws | 21,095 | 2,223 |
| In the PM window | 11,539 (54.7 % of draws) | 1,152 (51.8 %) |
| Kept at the PX4 defaults | 10,000 (86.7 % of in-window) | 1,000 (86.8 %) |
| Scenario B | 973 (9.7 %) | 101 (10.1 %) |
| Flagged (fail only the worst case at the defaults) | 420 (4.2 %) | 44 (4.4 %) |

All shares lie within 2.6 combined binomial sd of the RQ1 shares (the run stops at 5 sd). Crossover bands (unfiltered f_c): 3-4 Hz 3,444; 4-5 Hz 2,576; 5-6.5 Hz 2,602; 6.5-8 Hz 1,175; 8-10 Hz 203 (2.0 %).

## 4. Checks after the run

- `dataset_gen.py verify`: OK. 10,000 unique ids in order; the 1,000 RQ1 records unchanged (byte-equal JSON after removing `gains` and `origin`); every optimum inside the limits; spectra SHA-256 and ids match.
- Independent read-back (limits recoded separately): 0 optima outside the limits; worst-corner M_s of the optima at most 3.0000, nominal M_s at most 1.9994.
- `check-spectra`: all 10,000 spectra regenerated from the seed, SHA-256 equal.
- Full suite: **437 passed** (the stored-spectra test now runs).

**Labels agree with RQ1** (median [p10, p90] over the feasible cases):

| Part | Mode A feasible | ΔPM at equal noise [deg] | Mode B feasible | Mode B N_rel |
|---|---|---|---|---|
| all 10,000 | 9,976 | 3.20 [0.50, 7.78] | 9,999 | 0.572 [0.299, 0.941] |
| RQ1 (first 1,000) | 1,000 | 3.33 [0.48, 8.13] | 1,000 | 0.569 [0.294, 0.939] |
| new 9,000 | 8,976 | 3.18 [0.51, 7.76] | 8,999 | 0.573 [0.299, 0.942] |

## 5. Finding: 24 cases without a Mode A label, 1 without a Mode B label (question for you)

All 24 are **flagged** drones (24 of 420 flagged, 5.7 %; 16 Scenario A, 8 Scenario B). The PX4 default fails the worst case, so it cannot seed the search. The RQ1 sample had none (0 of its 44 flagged drones). No limit is broken; these are the outputs of exactly the D-37 optimizer.

| Kind | Cases | What it means |
|---|---|---|
| Mode A infeasible on the coarse grid | 21 | Every nominal-feasible coarse candidate was checked and failed on the full grid. The Mode B optimum (when it exists) is noisier than the default (N_rel 1.0002-1.43): every safe setting found is noisier than the default, as for the flagged real drone of Fig. 4 |
| Mode A search gap | 2 (syn6473, syn7266, both Scenario B) | The Mode B optimum (N_rel 0.993 and 0.988) meets every Mode A constraint, so a Mode A setting exists that the coarse-grid search did not reach |
| Search cap of 3,000 candidates reached | syn8559 (Mode A: 3,000 of 3,149 checked); syn2012 (Mode B: 3,000 of 15,395) | Not proven infeasible. In RQ1 the cap never bound (at most 966) |

The records hold `feasible`, `n_candidates` and `n_feasible_coarse` per mode, so every group can be selected in Task 5 without changing the data.

## 6. Rulings made during Task 4 (each with its cost if wrong)

1. **Fields added to the RQ1 record format: only `gains` and `origin`.** The sampler's default metrics are bit-identical to the record's `reference_robust` (checked at the merge for all 10,000), so they are not stored twice. Cost: none.
2. **The chunk loop is new code (`dataset_gen.label_chunks`), with the logic of `rq1_run.run_part`** plus the header line, an fsync per chunk, the format/limits check of every chunk and the source check. `rq1_run.py` is unchanged, so RQ1 stays reproducible. Cost: none.
3. **Guard 3 (no other run during the run) is enforced in code:** after every chunk the SHA-256 over all tracked `.py` files must be unchanged, or the run stops. Cost: none.
4. **"Dirty tree" includes untracked files.** Cost: none.
5. **Count check:** a share (in window, kept, Scenario B, flagged) more than 5 combined binomial sd from the RQ1 share stops the run before labelling. Cost: a false stop at most.
6. **Reuse check:** picks drawn with seed 20261024 (the brute-force check used 20261023); absolute tolerance 1e-9; every field except the solver seconds compared.
7. **A torn last line of the partial file** (crash during a write) is dropped and its cases relabelled; any other broken line refuses. Not needed in this run.
8. **Bit-exact spectra:** the suite compares the first 200 rows and the SHA-256 in the meta; the full regeneration is the `check-spectra` command (9 min, too slow for every suite run). npz uncompressed (NumPy writes a fixed zip time stamp); records gzip with mtime 0, so both files are deterministic.
9. **Infeasible labels are kept as produced** (Section 5) and reported; how Task 5 uses them is your decision.

## 7. Process notes

- Tests first: all 43 tests were run red against a stub (one vacuous pass was found and fixed before the implementation). The mutation check ran on the implementation before the `run()` fix of the next point; `run()` itself is covered by an end-to-end dry run.
- **End-to-end dry run before the commit** (N = 1,010 into the scratchpad: run, restart, verify, check-spectra): it found that `run()` relied on argument defaults bound at import; fixed (explicit sizes) before the code commit.
