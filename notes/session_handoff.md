# Session handoff (read at the start of a new session)

Last update: 2026-10-19, Phase 4: Tasks 1-3 done (Task 3 = RQ1, D-37); decisions D-28 to D-39 recorded; Task 4 plan written, STOP for the author's approval. Written for: Claude starting a new session, and the author.

## Where things stand

- **Branch `phase4`** (created from the tip of `phase3`): everything is committed locally. **`phase3` and `phase4` are NOT merged** into `main`; the author decides the merge. **No GitHub push** until the author asks.
- **Tests:** `.venv/Scripts/python -m pytest -q` (394 pass). Run the full suite before every commit.
- **Decided:**
  - **D-28** Tier 1 priors (evidence `notes/tier1_priors.md`); the ranges of f_c, τ_m and T_d are revised by D-36;
  - **D-29** Fig. 1 with two panels: (a, main) PX4 default with the ESC bank off, (b) bank on. Caption computed (`experiments/phase4/figures/fig1_caption.md`); crossover band 3-10 Hz since D-36;
  - **D-30** PM floor 30°; **D-31** $M_{s,max}$ 2.0 (nominal), GM ≥ 6 dB;
  - **D-32** separate worst-case limits; **D-33** ICM chip delay by the fit-error rule (lower bound as nominal T_d shift, rest in the delay step; real identified plants: FIFO -> control correction only; scope confirmed in D-36);
  - **D-34** all earlier unasked rulings accepted; **D-35** the 31 dropout logs on the cut list;
  - **D-36** A4 and F1-F4 final (parts (1)-(6)):
    - (1) ±20 % on b and τ_m, delay step 1 controller sample, $M_{s,wc}$ 3.0; D-14 identification settings kept, gates frozen;
    - (2) flagged real drones ("reference unsafe") kept in the test set and reported;
    - (3) τ_m 15-60 ms, T_d 0.3-8 ms (net of the chip filter); identified plants only as a sensitivity check;
    - (4) PM window 60-95°, integral corner 0.03;
    - (5) f_c 3-10 Hz (real drones above 10 Hz reported separately); f_lo = max(5 f_c, 20 Hz);
    - (6) Task 3 real-gain cases need an identification and a steady window (fit half: 26 cases, 15 vehicles).
- **THESIS Section 11, item 10:** the reviewer's 10 declined points, the Scenario B limitation and the training crossover range, one sentence each.
- **Evidence:** `notes/evidence_pack_A2_A4.md`, `notes/followup_A4_F1_F4.md` (tables `experiments/phase4/evidence_tables.md`, `followup_tables.md`).
- **D-37** Task 3 design (T3-1..T3-6 option (a)): optimizer limits = nominal + D-36 worst case on the 8 corners; both halves (test last); 1,000 synthetic + 950 real-noise cases; fast coarse grid + full-grid refinement; 10 Hz by the unfiltered crossover; flagged synthetic drones kept; "reference unsafe" always with both numbers.
- **Task 3 done (2026-10-19): `notes/task3_results.md`** (Table 2, findings, A5 recommendation, rulings). Tables `experiments/phase4/rq1_tables.md`; figures `figures/fig3_pm_gain`, `fig4_pareto`. Code: `src/tier3.py`, `optimizer.optimize_checked`, `experiments/phase4/rq1_cases.py` (S2), `rq1_run.py` (S3; resumable parts), `rq1_report.py` (S4). **Next: the author reviews Task 3 and decides A5** (recommendation 10,000 cases) and the 4 test cases below 3 Hz. Task 4 is NOT started.
- **D-38** A5: 10,000 cases labelled with exactly the D-37 optimizer; Scenario B at its natural share (~10 %, 5 sources); the 1,000 RQ1 synthetic cases are the first 1,000; fallback 5,000, not 30,000; no fast-grid-only labels or greedy notch search without a new check and the author's decision; no test spectra in training.
- **D-39** test-half reporting: the 4 test cases below 3 Hz are reported separately (Table 2 column, "outside the 3-10 Hz training prior" in the Table 2 footnote, band tables and Table S1; kept in all totals); the test-half observations are findings in THESIS 11 (safe references only: +2.1°). New THESIS 7.3.1 holds Table 2 and the band tables, copied verbatim from `rq1_tables.md`.
- **Next: the author reviews `notes/task4_plan.md`** (questions T4-1 to T4-5). **Task 4 is NOT started; do not start it before the author approves the plan.** Measured inputs: 3.4 s per case on 6 workers; sampler 0.07 s per kept drone; seed 20261018 reproduces the RQ1 drones first (200 of 200 checked).
- **Independent review (2026-10-18, author request):** no critical findings; 6 important ones fixed test-first. Deferred minor: ESC RPM averaged from first take-off to last landing in multi-flight logs.
- Plan ledger (git-ignored): `.superpowers/sdd/phase4_plan/progress.md` (rulings, process notes).

## Read first (in this order)

1. `THESIS.md`: Section 0; Section 2 (D-01 to D-39); 7.1-7.1.1, 7.2, 7.8; 7.4.7 and 7.5.1; Section 10 Phase 4; Section 17 (VERIFY items F1-F5).
2. `notes/task4_plan.md` (waits for the author), `notes/task3_results.md`, then `notes/followup_A4_F1_F4.md` and `notes/evidence_pack_A2_A4.md` (the evidence behind D-36).
3. `notes/phase4_plan.md`: the approved plan (Tasks 3-4 next).
4. `notes/tier1_priors.md`: Task 1 evidence.
5. `notes/decision_log.md`: decisions, rulings, corrections. Keep it updated.
6. `experiments/part2/REPORT.md` Sections 13-16 and `experiments/feasibility/REPORT.md`.

## The author's standing rules (keep them)

- **Decisions:**
  - No serious decision without evidence and the author's permission; ask with AskUserQuestion and a recommendation.
  - Every decision goes into THESIS Section 2 (next ID: **D-40**) **and** `notes/decision_log.md`.
- **Stopping:** stop and report immediately on any new landmine (unexpected data or code problem).
- **Testing:**
  - Write tests first and see them fail.
  - Mutation checks: `experiments/phase4/mutation_check_*.py`.
    - `main(SRC, MUTANTS, TEST)` in `mutation_check_priors.py` is reusable.
    - It has a 180 s timeout per mutant: a mutant that makes a test loop forever counts as caught.
  - Verify every number before reporting it (self-checks caught 39/99 -> 42/102, 16 -> 15, 60 -> 68 % before they reached the author).
- **Leakage:**
  - The fit half is used for every choice; the test half only for evaluation.
  - `src/priors.py` `require_fit_half` / `require_fit_rows` refuse test-half inputs: use them in every new script.
- **Git:** commit locally per task with a clear message. No push, no merge without asking.
- **Environment:**
  - Windows with Git Bash. Python is `.venv/Scripts/python`.
  - raw.githubusercontent.com is blocked; read PX4 source through the GitHub API (`curl -H "Accept: application/vnd.github.raw" https://api.github.com/repos/PX4/PX4-Autopilot/contents/<path>?ref=v1.16.0`).
  - The arXiv tool rate-limits often (HTTP 429, wait 60 s); WebSearch/WebFetch work; `pdftotext` exists in Git Bash.
  - Avoid shell heredocs with backslashes or nested quotes; write Python edit scripts with the Write tool.
  - Long runs (`plant_variation.py` about 15-19 min, `robust_limits.py` about 8 min) belong in the background.

## Key facts for Tasks 3-4

- **Controller rate:**
  - Use the measured rate (median interval of `vehicle_angular_velocity`, high-rate or sysid logger profile only, `priors.full_rate_logged`).
  - 42 of 102 fit logs differ from `IMU_GYRO_RATEMAX` by more than 5 %.
- **Synthetic drones (D-36; code `src/tier3.py`, the Task 3 sampler):** f_c 3-10 Hz, τ_m 15-60 ms, T_d 0.3-8 ms (log-uniform; T_d net of the chip filter); PM window 60-95°, integral corner 0.03, r in [0.5, 3]; (FIFO rate, controller rate) pairs from the 37 same-IMU full-rate fit logs (`priors_from_logs.json` `rate_rows`). Chip delays (D-33, fit-error rule, `priors_from_logs.json` `chip_delay[chip]['rms']`): ICM-42688-P 0.91-1.37 ms, ICM-42686-P 2.00-3.02 ms, ICM-45686 0.25-0.89 ms, IIM-42652 1.14-1.16 ms; the chip comes from the generator's source vehicle. D-36 lives in `src/tier3.py` (gains designed on the plant including the D-33 chip lower bound); `robust_limits` / `f_options` keep the D-28 defaults only to reproduce the evidence pack.
- **Hover f_m of synthetic drones:** the generator's source vehicle RPM/60 times the sampled frequency scale (`noise_generator.sample` meta `source`, `scale`).
- **Real-gain cases:**
  - plants from `sysid.piece_spectra` + `fit_frf` with the D-14 instrument rule, up to 8 held skipped publications, 1.28 s segments (the D-14 6-frequency rule depends on this resolution) and the gates of `plant_variation.accept`, **frozen in D-36** (never change them after seeing test-half results);
  - gains include `MC_*RATE_K`;
  - filter rate = the control IMU's raw rate (`robust_limits.control_rate`; needs `vehicle_imu_status`, which `noise_data.TOPICS` does not load);
  - identified T_d corrected for FIFO ≠ control IMU (`robust_limits.chip_td_correction`); rows with loop parameters changed in flight excluded.
- **Robust check:** `src/robust.py` (`robust_metrics`, `passes` with same or separate worst-case limits). It reproduces the feasibility worst case exactly.
- **ICM chip delay (D-25, A4):**
  - identified plants already contain the FIFO chip's filter: only the FIFO → control chip correction for real-gain drones;
  - if synthetic T_d priors are ever taken from identified plants: no chip increment on top;
  - synthetic drones whose chip has a fitted low-pass term get it (D-33: ICM-42688-P, ICM-42686-P, ICM-45686, IIM-42652; 54 % of the generator entries);
  - never both (Review Focus 2).
- **Phase delay with notches:** valid only below the first notch; above it, sum the factor phases (`fig1_bode.factors`).
- **Limits (D-30..D-32):** nominal PM ≥ 30°, $M_s$ ≤ 2.0, GM ≥ 6 dB; worst case over the set by $M_{s,wc}$ with its implied PM / GM (`a4_followup.separate_limits`, `robust.passes(..., wc=robust_limits.wc_limits(Ms_wc))`).
- **Noise band (D-36):** f_lo = max(5 f_c, 20 Hz) to 780 Hz (`LoopCase`, `loop_model.F_LO_MIN`).
- **Optimizer (D-37):** `optimizer.optimize_checked(case, base, options_fast, mode, robust=Robust(...), coarse=fast_case, options_full=..., reference=...)`: coarse search on `FAST`, candidates checked and refined on the full case incl. the 8 corners; a feasible reference bounds the optimum; `brute_force` check; `LIMITS` = D-30/D-31. Run time (risk R4): a two-peak Scenario A case (57 notch options) takes about 80 s on one core with the full grid, about 12 s with the feasibility fast grid (500 points, 1 Hz, 24 x 18); the fast optimum broke the noise limit on the full grid in 1 of 4 planning cases, so refine and report on the full grid.
- **Real-gain cases need both an identification and a library spectrum (D-36 (6); built by `rq1_cases.py`, `fit_`/`test_rq1_cases.json`):** b79d2e3d has no steady window (Q10). Fit half: 26 cases from 15 vehicles; 9 flagged (5 nominal, 4 worst case only); 2 above 10 Hz (unfiltered); 4 with f_lo clamped at 20 Hz; Scenario B: b27d09e3, 5cecf734.
- **Scenario B:** only 5 generator entries carry motor harmonics (51603946 has valid RPM in 1 of 3 windows).
- **Fig. 1:** `fig1_bode.make_figure` writes the two-panel figure and the caption (crossover band 3-10 Hz, noise band from max(5 f_c, 20 Hz)); the script takes an optional output folder and prints UTF-8. At 2 kHz the curves change by up to 0.16 ms (panel a) / 0.19 ms (panel b) in the crossover band.
- **Phase 3 limitations still apply** (THESIS 7.5.1).
