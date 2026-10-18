# Session handoff (read at the start of a new session)

Last update: 2026-10-18, Phase 4 Tasks 1-2 done. Written for: Claude starting a new session, and the author.

## Where things stand

- **Branch `phase4`** (created from the tip of `phase3`): Tasks 1-2 of `notes/phase4_plan.md` done and committed locally. **`phase3` and `phase4` are NOT merged** into `main`; the author decides the merge. **No GitHub push** until the author asks (after Phase 4 or 5).
- **Tests:** `.venv/Scripts/python -m pytest -q` (244 pass). Run the full suite before every commit.
- **Decided:** A1 = **D-28** (Tier 1 priors). Evidence: `notes/tier1_priors.md`.
- **Next (needs the author):**
  - A2 (PM floor), A3 ($M_{s,max}$), A4 (uncertainty set, including the ICM chip delay) **before Task 3** (RQ1 results).
  - A5 (training-set size, Scenario B share) **before Task 4** (`dataset_gen.py`).
  - Prepare an evidence pack for A2-A4 first (feasibility REPORT 5 and 10.4: with ±20 % / +1 sample, the pilot's good setting fails PM 30° and $M_s$ 2.0 in the worst case; THESIS 17 VERIFY "Safety-check calibration").
- Plan ledger (git-ignored): `.superpowers/sdd/phase4_plan/progress.md` (rulings of Tasks 1-2).

## Read first (in this order)

1. `THESIS.md`: Section 0; Section 2 (D-01 to D-28); 7.1-7.1.1 (now with the D-28 priors), 7.2, 7.8; 7.4.7 and 7.5.1; Section 10 Phase 4; Section 17.
2. `notes/phase4_plan.md`: the approved plan (Tasks 3-4 next).
3. `notes/tier1_priors.md`: Task 1 evidence, the ICM chip-delay table (input to A4), the A1 proposals.
4. `notes/decision_log.md`: decisions, rulings, corrections. Keep it updated.
5. `experiments/part2/REPORT.md` Sections 13-16 and `experiments/feasibility/REPORT.md` (optimizer, loop model, safety-check warning).

## The author's standing rules (keep them)

- **Decisions:**
  - No serious decision without evidence and the author's permission; ask with AskUserQuestion and a recommendation.
  - Every decision goes into THESIS Section 2 (next ID: **D-29**) **and** `notes/decision_log.md`.
- **Stopping:** stop and report immediately on any new landmine (unexpected data or code problem).
- **Testing:**
  - Write tests first and see them fail.
  - Mutation checks: `experiments/phase4/mutation_check_priors.py` (17/17), `mutation_check_fig1.py` (9/9); `main(SRC, MUTANTS, TEST)` is reusable for new modules.
  - Verify every number before reporting it (a self-check in Task 2 corrected "39 of 99" to "42 of 102"; decision log D).
- **Leakage:**
  - The fit half is used for every choice; the test half only for evaluation.
  - `src/priors.py` `require_fit_half` / `require_fit_rows` refuse test-half inputs: use them in every new script that makes a choice.
- **Git:** commit locally per task with a clear message. No push, no merge without asking.
- **Environment:**
  - Windows with Git Bash. Python is `.venv/Scripts/python`.
  - raw.githubusercontent.com is blocked; read PX4 source through the GitHub API (`curl -H "Accept: application/vnd.github.raw" https://api.github.com/repos/PX4/PX4-Autopilot/contents/<path>?ref=v1.16.0`).
  - The arXiv tool rate-limits often (HTTP 429, wait 60 s); WebSearch/WebFetch work. `pdftotext` exists in Git Bash.
  - Avoid shell heredocs containing backslashes (`\t`, `\n`, LaTeX); write Python edit scripts with the Write tool instead.

## Key facts for Tasks 3-4

- **Controller rate:** use the **measured** rate (median interval of `vehicle_angular_velocity`, only logs with the high-rate or sysid logger profile, `priors.full_rate_logged`), never `IMU_GYRO_RATEMAX` alone: 42 of 102 fit logs differ by more than 5 % (e.g. 667 instead of 800 Hz). For real-gain test-half cases, measure it per log the same way.
- **Synthetic drones (D-28):** f_c 4-15 Hz, τ_m 20-60 ms, T_d 0.5-5 ms, log-uniform; (FIFO rate, controller rate) as empirical pairs from fit-half logs whose FIFO IMU is the control IMU (38 of 111) and that are logged at full rate.
- **Hover f_m of synthetic drones:** the generator's source vehicle RPM/60 times the sampled frequency scale (`noise_generator.sample` meta: `source`, `scale`); only 6 fit vehicles have valid RPM, so Scenario B synthetic cases can use only those sources (input to A5).
- **ICM chip delay (D-25, A4):** ICM-42688-P 0.91-1.37 ms (1.5 % fit-error rule) or 0.99-1.04 ms (BIC rule, too narrow); ICM-42686-P at least 2.0 ms, not bounded above by the data. Never count it twice (Review Focus 2).
- **Phase delay with notches:** `filters.phase_delay` / `np.unwrap` is valid only below the first notch; above it, sum the factor phases (`fig1_bode.factors`). Loop-model margins are proven unaffected (`test_margins_do_not_depend_on_hitting_a_notch_centre`).
- **Phase 3 limitations still apply** (THESIS 7.5.1): +4.4 dB generator bias at 20-100 Hz (non-ICM chips), about 3 dB motor-band misfit, frequency jitter from 4 drones, small chip groups.
