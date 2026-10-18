# Session handoff (read at the start of a new session)

Last update: 2026-10-18, Phase 4 Tasks 1-2 done, evidence pack for A2-A4 done. Written for: Claude starting a new session, and the author.

## Where things stand

- **Branch `phase4`** (created from the tip of `phase3`): Tasks 1-2 of `notes/phase4_plan.md` and the evidence pack for A2-A4 are committed locally. **`phase3` and `phase4` are NOT merged** into `main`; the author decides the merge. **No GitHub push** until the author asks.
- **Tests:** `.venv/Scripts/python -m pytest -q` (301 pass). Run the full suite before every commit.
- **Decided:** A1 = **D-28** (Tier 1 priors). Evidence: `notes/tier1_priors.md`.
- **Evidence pack for A2-A4: `notes/evidence_pack_A2_A4.md`** (generated tables: `experiments/phase4/evidence_tables.md`).
- **Open decisions (the author), before Task 3:**
  - A2 (PM floor), A3 ($M_{s,max}$), A4 (uncertainty set, worst-case limits, ICM chip delay): recommendations in Section 6 of the pack.
  - **F1-F4** (Section 5 of the pack):
    - F1: D-28 plant priors slower than 9 of 25 identified plants;
    - F2: Tier 3a gain design unlike the real drones;
    - F3: no synthetic drone with f_c ≥ 10 Hz is safe at the PX4 defaults;
    - F4: f_lo rule for real crossovers below 4 Hz.
  - Fig. 1 variant: ESC-RPM bank on or off (`experiments/phase4/figures/fig1_compare_esc_off_on.png`).
- **Before Task 4:** A5 (training-set size, Scenario B share). **Tasks 3 and 4 are NOT started** (author's instruction).
- **Independent review:** requested by the author on 2026-10-18 (code and method review of the whole `phase4` branch); see the session log in THESIS Section 17 for its outcome.
- Plan ledger (git-ignored): `.superpowers/sdd/phase4_plan/progress.md` (rulings, process notes).

## Read first (in this order)

1. `THESIS.md`: Section 0; Section 2 (D-01 to D-28); 7.1-7.1.1, 7.2, 7.8; 7.4.7 and 7.5.1; Section 10 Phase 4; Section 17 (VERIFY items F1-F5).
2. `notes/evidence_pack_A2_A4.md`: the evidence and the open decisions.
3. `notes/phase4_plan.md`: the approved plan (Tasks 3-4 next).
4. `notes/tier1_priors.md`: Task 1 evidence.
5. `notes/decision_log.md`: decisions, rulings, corrections. Keep it updated.
6. `experiments/part2/REPORT.md` Sections 13-16 and `experiments/feasibility/REPORT.md`.

## The author's standing rules (keep them)

- **Decisions:**
  - No serious decision without evidence and the author's permission; ask with AskUserQuestion and a recommendation.
  - Every decision goes into THESIS Section 2 (next ID: **D-29**) **and** `notes/decision_log.md`.
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
  - plants from `sysid.piece_spectra` + `fit_frf` with the D-14 instrument rule and the (provisional) gates of `plant_variation.accept`; the gates must be fixed before the test half is touched;
  - gains include `MC_*RATE_K`;
  - filter rate = the control IMU's raw rate (`robust.control_imu_raw_rate`).
- **Robust check:** `src/robust.py` (`robust_metrics`, `passes` with same or separate worst-case limits). It reproduces the feasibility worst case exactly.
- **ICM chip delay (D-25, A4):**
  - identified plants already contain the FIFO chip's filter: no increment for real-gain drones;
  - synthetic ICM-4268x drones get it;
  - never both (Review Focus 2).
- **Phase delay with notches:** valid only below the first notch; above it, sum the factor phases (`fig1_bode.factors`).
- **Phase 3 limitations still apply** (THESIS 7.5.1).
