# Phase 4 Plan (DRAFT, waiting for the author's review): plant priors, Fig. 1, then optimizer results and training data

> Status: **draft 2026-10-16, not started.** Nothing in this plan is executed before the author approves it.
> Executor: native, test-first, full suite before every commit (author's standing instructions).
> Stop and report on any new landmine.

**Goal:**
1. Give every plant parameter of THESIS 7.1.1 a sourced Tier 1 prior range.
2. Produce Fig. 1, the Bode plot of the PX4 filter stack (experiment E1).
3. Then, after a second approval: the RQ1 results (E3) and the training set (`dataset_gen.py`), built from the Phase 3 generator.

**Schedule:** Phase 4 is planned for Days 24-26 (Oct 21-23, D-18). Phase 3 finished on Day 19, so there are about 4 days of buffer.

**Spec:**
- THESIS 7.1 / 7.1.1 (plant model, three tiers), 7.2 (metrics), 7.8 (safety check), Section 8 (E1, E3) and the Phase 4 checklist.
- Decisions D-09, D-12, D-14, D-15, D-25 (the ICM chip delay goes into the $T_d$ uncertainty) and D-26/D-27 (generator).

## Branch question (for the author)

`phase3` is not merged into `main` (author: merge later). Phase 4 needs the Phase 3 code (generator, library). Proposal: start a branch **`phase4` from the tip of `phase3`**. When `phase3` is merged later, `phase4` follows without conflicts. The alternative is to wait for the merge.

## Decisions needed (asked at the points marked STOP)

| # | Decision | Candidate (THESIS) | When |
|---|---|---|---|
| A1 | Tier 1 prior ranges for $f_c$, $\tau_m$, $T_d$, $f_m$, control rate and FIFO rate (large PX4 drone; FPV extension) | From Task 1's evidence table | After Task 1 (STOP) |
| A2 | PM floor $PM_{min}$ | 30° | Before Task 3 |
| A3 | $M_{s,max}$ | 2.0 (6 dB) | Before Task 3 |
| A4 | Uncertainty set: $\delta_b$, $\delta_\tau$, $\Delta_d$, plus the ICM chip-delay range (D-25) | 20 %, 20 %, 1 control sample; chip delay from Task 1 step 4 | Before Task 3 |
| A5 | Training-set size and the share of Scenario B (RPM) cases | 1,000 debug, then 10,000-30,000 | Before Task 4 |

---

## Task 1: Tier 1 plant priors (THESIS 7.1.1 table, "Source / status" column)

**Files:**
- Create `notes/tier1_priors.md`: one row per parameter with the source, the exact place in the source (page, table or figure), the value as printed, how we convert it, and the resulting range.
- Create `experiments/phase4/priors_from_logs.py`.
- Modify THESIS 7.1.1 (after A1).

**Rules:**
- **No number without a source that was read.** Literature is searched with the arXiv tools and the existing `paper/refs.bib` / `notes/literature.md` (e.g. Bangura & Mahony 2017 for $\tau_m$, the PX4 docs for control latency).
- A value that cannot be found stays **VERIFY**, with the reason. It is never filled from memory.
- **Our own log evidence uses the fit half only** (no test-half leakage into the priors).

Steps:
1. **Literature:** for $\tau_m$, $T_d$ (ESC protocol: PWM, OneShot, DShot; PX4 control-path latency), typical crossover $f_c$ of tuned multirotors, and the FPV ranges.
2. **Fit-half log evidence** (`priors_from_logs.py`, reads `data/spectra/fit_*` and `fit_logs.json` only):
   - control-loop rate (`IMU_GYRO_RATEMAX` and the logged `vehicle_angular_velocity` rate);
   - FIFO rate;
   - hover motor frequency from valid ESC RPM;
   - real PID gains (inputs to Tier 3b).
3. **Existing Tier 2 result** (feasibility test, log b27d09e3): $b$ ≈ 431-474, $\tau_m$ ≈ 29-34 ms. It is a single vehicle and is reported as such.
4. **ICM chip delay (D-25):** for every chip low-pass term within a BIC gain of 10 of the chosen one (the set of indistinguishable fits), compute the group delay at 20-100 Hz. The min-max range becomes the proposed $T_d$ increment for ICM-4268x drones (input to A4).
5. **STOP:** show the evidence table; the author decides A1 (recorded as a new Section 2 decision).

Tests: `priors_from_logs.py` refuses `test_*` inputs; the group-delay function matches the analytic low-frequency group delay of a Butterworth of order n (THESIS 4.4).

## Task 2: Fig. 1, Bode plot of the PX4 filter stack (E1)

**Files:**
- Create `experiments/phase4/fig1_bode.py` and `experiments/phase4/figures/fig1_filter_stack.{png,pdf}`.
- Test: `tests/test_fig1_bode.py`.

**Content:**
- **Filters:** the PX4 v1.16 multicopter defaults (D-09):
  - gyro low-pass `IMU_GYRO_CUTOFF` (2nd-order Butterworth);
  - D-term PT1 `IMU_DGYRO_CUTOFF`;
  - the ESC-RPM notch bank at a typical hover motor frequency (Task 1);
  - optionally a static notch.
- **Sample rate:** the filters run at the FIFO rate (verified in the feasibility test).
- **Panels:**
  - (a) magnitude in dB of each filter and of the total, for the gyro path and the D-term path;
  - (b) phase or group delay in ms.
- **Marked:** the crossover range $f_c$ (from A1) and the noise band from $5 f_c$ to 780 Hz.
- **No dual axis:** magnitude and delay are two stacked panels sharing the frequency axis. The palette is checked with the dataviz validator, and the rendered figure is inspected.

Steps (test first):
1. **Tests:** the plotted responses equal `scipy.signal.freqz` of the same coefficients from `src/filters.py` (within 1e-9); the low-frequency group delay of the 2nd-order low-pass equals the analytic value; the default parameters are read from one table (v1.16 defaults, as in `loop_model.PX4_V116_MC_DEFAULT`).
2. **Implement the figure.**
3. **Commit.** Fig. 1 needs only the $f_c$ marker from A1; without A1 it is drawn with the THESIS candidate range, labelled "candidate".

## Task 3: RQ1 results (E3: Fig. 3, Fig. 4, Table 2). Needs A2-A4

- **Synthetic noise from the Phase 3 generator:** default vs optimal settings, modes A/B/C, reporting $\tilde N$, $\tau_F$ and PM together.
- **Real test-half spectra:** default vs the pilot's actual setting vs optimal, with the real gains (D-15).
- **Rule:** the test half is used only for evaluation (Phase 3 rule).
- **Outputs:** the PM-gain histogram (Fig. 3), Pareto fronts (Fig. 4) and Table 2.
- **Limitations carried along:** the ICM chip delay range (D-25) and the Phase 3 generator limitations (THESIS 7.5.1).

## Task 4: `dataset_gen.py` (training set). Needs A5

- **Pipeline:** Tier 3 drones (7.1.1) + generator spectra → optimizer labels for modes A and B.
- **Run:** 1,000 cases to debug, then the final size.
- **Run conditions:** parallel `joblib`; seed and git hash stored; resumable in chunks, so a laptop shutdown loses nothing.

## Optional: Tier 2 system identification (E12; on the cut list)

- **Data:** fit-half logs with profile bit 3 or 4.
- **Method:** D-14 instrument rules.
- **Output:** it would replace the Tier 1 ranges. It is proposed only if Tasks 1-4 finish early.

## Review Focus (to be pinned by tests)

1. **Priors must never read test-half data:** a test that the priors script raises on `test_*` paths.
2. **The ICM chip delay must not be counted twice** (as both a chip term and a $T_d$ increment): Task 3 uses one or the other, never both.
3. **Fig. 1 must show exactly the filter chain the loop model uses** (same functions, same rate): tested against `scipy.signal`.
4. **The training set is resumable and reproducible:** the same seed gives the same first 100 cases.
