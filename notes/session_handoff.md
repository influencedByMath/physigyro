# Session handoff (read at the start of a new session)

Last update: 2026-10-18, end of Phase 3. Written for: Claude starting a new session, and the author.

## Where things stand

- **Branch `phase3`:** Phase 3 complete, about 50 local commits, NOT merged into `main`. The author decides the merge after Phase 4. **No GitHub push** until the author asks (after Phase 4 or 5).
- **Tests:** `.venv/Scripts/python -m pytest -q` (200 pass). Run the full suite before every commit.
- **Phase 4:** the plan `notes/phase4_plan.md` was approved by the author on 2026-10-18.
  - Start with Task 1 (Tier 1 priors) and Task 2 (Fig. 1).
  - Stop at the marked STOP points for decisions A1-A5.
  - Work on a branch `phase4` created from the tip of `phase3`.

## Read first (in this order)

1. `THESIS.md`, in this order:
   - Section 0 (how to use the file);
   - Section 2 (decisions D-01 to D-27);
   - Sections 7.1-7.1.1 (plant model and tiers), 7.2 (metrics), 7.8 (safety check);
   - Sections 7.4.7 (Table 1) and 7.5.1 (Phase 3 results);
   - Section 10 Phase 4;
   - Section 17 (status, VERIFY list, session log).
2. `notes/phase4_plan.md`: the approved plan.
3. `notes/decision_log.md`: the decision history, changes, rulings and corrections. Keep it updated.
4. `experiments/part2/REPORT.md` Sections 13-16: final Phase 3 results and the code-review findings.
5. `experiments/feasibility/REPORT.md`: the optimizer and loop model that Phase 4 uses.

## The author's standing rules (keep them)

- **Decisions:**
  - No serious decision without evidence and the author's permission; ask with AskUserQuestion and a recommendation.
  - Every decision goes into THESIS Section 2 (next ID: **D-28**) **and** `notes/decision_log.md`.
- **Stopping:** stop and report immediately on any new landmine (unexpected data or code problem).
- **Testing:**
  - Write tests first and see them fail.
  - The author values mutation checks: break the code on purpose, show the tests catch it, restore it byte-identical.
  - Verify every number before reporting it: earlier overclaims were caught by an independent review.
- **Leakage:**
  - The fit half is used for every choice.
  - The test half is used only for evaluation, never for tuning (this includes the Tier 1 priors: fit half only).
- **Git:** commit locally per task with a clear message. No push, no merge without asking.
- **Environment:**
  - Windows with Git Bash. Python is `.venv/Scripts/python`.
  - raw.githubusercontent.com is blocked; read PX4 source through the GitHub API (`curl -H "Accept: application/vnd.github.raw" https://api.github.com/repos/PX4/PX4-Autopilot/contents/<path>?ref=v1.16.0`).
  - Avoid shell heredocs containing `\n` or tricky quotes; write Python edit scripts with the Write tool instead.

## Key Phase 3 facts Phase 4 needs

- **Real noise library:** `data/spectra/{fit,test}_*` (windows CSV, native and grid PSD npz).
- **Generator:** `src/noise_generator.py`, fitted to `experiments/part2/model_fits.json` (fit half). Jitter is measured in `experiments/part2/generator_check.py`.
- **ICM-42688-P/42686-P drones:** an on-chip low-pass of about 3 decades above 500 Hz. Its delay is unknown (D-25) and must enter the T_d uncertainty for these drones; it must not be counted twice.
- **Limitations** (THESIS 7.5.1):
  - +4.4 dB generator bias at 20-100 Hz (non-ICM chips);
  - about 3 dB motor-band misfit;
  - frequency jitter from only 4 drones;
  - small chip groups.
