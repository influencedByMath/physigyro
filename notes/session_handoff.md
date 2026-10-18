# Session handoff (read at the start of a new session)

Last update: 2026-10-18, Phase 4: Tasks 1-2 done, evidence pack done, author's decisions D-29..D-35 recorded, follow-up questions answered. Written for: Claude starting a new session, and the author.

## Where things stand

- **Branch `phase4`** (created from the tip of `phase3`): everything is committed locally. **`phase3` and `phase4` are NOT merged** into `main`; the author decides the merge. **No GitHub push** until the author asks.
- **Tests:** `.venv/Scripts/python -m pytest -q` (341 pass). Run the full suite before every commit.
- **Decided:**
  - A1 = **D-28** (Tier 1 priors; evidence `notes/tier1_priors.md`);
  - **D-29** Fig. 1 with two panels: (a, main) PX4 default with the ESC bank off, (b) bank on. Caption `experiments/phase4/figures/fig1_caption.md` is computed; the bank adds 1.8 ms in the crossover band;
  - **D-30** A2: PM floor 30°;
  - **D-31** A3: $M_{s,max}$ 2.0 (nominal);
  - **D-32** separate worst-case limits (concept): every plant of the set stable with $M_s \le M_{s,wc}$. **Value open**;
  - **D-33** ICM chip delay by the fit-error rule (lower bound as nominal T_d shift, rest in the delay step; real identified plants: FIFO -> control correction only);
  - **D-34** all earlier unasked rulings accepted; identification gates frozen with A4;
  - **D-35** the 31 dropout logs on the cut list.
- **Evidence:** `notes/evidence_pack_A2_A4.md` (tables `experiments/phase4/evidence_tables.md`).
- **Follow-up answers (the author's questions 1-4 on A4, F1-F4 options, A5 estimate, the reviewer's 10 declined points): `notes/followup_A4_F1_F4.md`.**
  - Tables: `experiments/phase4/followup_tables.md`.
  - Runners: `a4_followup.py` (about 6 min), `f_options.py` (about 4 min; needs `a4_followup.json`), `followup_tables.py`.
- **Open decisions (the author), before Task 3:**
  - **A4 rest:** set width, delay step and the value of $M_{s,wc}$. Recommendation: ±20 %, 1 controller sample, $M_{s,wc}$ 3.0; keep the D-14 identification settings; report the 5 flagged real settings.
  - **F1-F4** (recommendations in the follow-up, Section 7):
    - F1: τ_m 15-60 ms, T_d 0.3-8 ms;
    - F2: PM window 60-95°, integral corner 0.03;
    - F3: sample f_c in 4-10 Hz, or 3-10 Hz with F4;
    - F4: f_lo = max(5 f_c, 20 Hz), which follows from D-23 Q5, optionally with f_c from 3 Hz.
- **Before Task 4:** A5 (training-set size, Scenario B share). **Tasks 3 and 4 are NOT started** (author's instruction).
- **Independent review (2026-10-18, author request):** no critical findings; 6 important ones fixed test-first. Deferred minor: ESC RPM averaged from first take-off to last landing in multi-flight logs.
- Plan ledger (git-ignored): `.superpowers/sdd/phase4_plan/progress.md` (rulings, process notes).

## Read first (in this order)

1. `THESIS.md`: Section 0; Section 2 (D-01 to D-35); 7.1-7.1.1, 7.2, 7.8; 7.4.7 and 7.5.1; Section 10 Phase 4; Section 17 (VERIFY items F1-F5).
2. `notes/followup_A4_F1_F4.md` (the latest answers and recommendations), then `notes/evidence_pack_A2_A4.md` (the evidence).
3. `notes/phase4_plan.md`: the approved plan (Tasks 3-4 next).
4. `notes/tier1_priors.md`: Task 1 evidence.
5. `notes/decision_log.md`: decisions, rulings, corrections. Keep it updated.
6. `experiments/part2/REPORT.md` Sections 13-16 and `experiments/feasibility/REPORT.md`.

## The author's standing rules (keep them)

- **Decisions:**
  - No serious decision without evidence and the author's permission; ask with AskUserQuestion and a recommendation.
  - Every decision goes into THESIS Section 2 (next ID: **D-36**) **and** `notes/decision_log.md`.
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
- **Synthetic drones (D-28):** f_c 4-15 Hz, τ_m 20-60 ms, T_d 0.5-5 ms (log-uniform); (FIFO rate, controller rate) pairs from the 37 same-IMU full-rate fit logs (`priors_from_logs.json` `rate_rows`). F1-F3 may change this.
- **Hover f_m of synthetic drones:** the generator's source vehicle RPM/60 times the sampled frequency scale (`noise_generator.sample` meta `source`, `scale`).
- **Real-gain cases:**
  - plants from `sysid.piece_spectra` + `fit_frf` with the D-14 instrument rule, up to 8 held skipped publications, 1.28 s segments (the D-14 6-frequency rule depends on this resolution) and the (provisional) gates of `plant_variation.accept`; the gates must be fixed before the test half is touched;
  - gains include `MC_*RATE_K`;
  - filter rate = the control IMU's raw rate (`robust_limits.control_rate`; needs `vehicle_imu_status`, which `noise_data.TOPICS` does not load);
  - identified T_d corrected for FIFO ≠ control IMU (`robust_limits.chip_td_correction`); rows with loop parameters changed in flight excluded.
- **Robust check:** `src/robust.py` (`robust_metrics`, `passes` with same or separate worst-case limits). It reproduces the feasibility worst case exactly.
- **ICM chip delay (D-25, A4):**
  - identified plants already contain the FIFO chip's filter: only the FIFO → control chip correction for real-gain drones;
  - if synthetic T_d priors are ever taken from identified plants: no chip increment on top;
  - synthetic ICM-4268x drones get it;
  - never both (Review Focus 2).
- **Phase delay with notches:** valid only below the first notch; above it, sum the factor phases (`fig1_bode.factors`).
- **Limits (D-30..D-32):** nominal PM ≥ 30°, $M_s$ ≤ 2.0, GM ≥ 6 dB; worst case over the set by $M_{s,wc}$ with its implied PM / GM (`a4_followup.separate_limits`, `robust.passes(..., wc=robust_limits.wc_limits(Ms_wc))`).
- **Noise band:** the code still uses f_lo = 5 f_c (`LoopCase`); if the author takes F4 (d), use max(5 f_c, 20 Hz) (D-23 Q5 band 20-780 Hz).
- **Real-gain cases need both an identification and a library spectrum:** b79d2e3d has no steady window (Q10).
- **Scenario B:** only 5 generator entries carry motor harmonics (51603946 has valid RPM in 1 of 3 windows).
- **Fig. 1:** `fig1_bode.make_figure` writes the two-panel figure and the caption; the old comparison files are gone.
- **Phase 3 limitations still apply** (THESIS 7.5.1).
