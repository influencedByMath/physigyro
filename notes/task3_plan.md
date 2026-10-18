# Task 3 plan: RQ1 results (E3: Fig. 3, Fig. 4, Table 2)

**Status: DRAFT, waiting for the author's approval. Task 3 is not started.**
Written 2026-10-18 for the author, after the decisions D-29 to D-36.

**What the numbers here are:**
- Fit half only. No test-half result has been opened.
- They are planning estimates. Scratch scripts produced them, calling the repo's tested functions (listed in Section 8).
- Task 3 recomputes every number with tested code.

---

## 1. What Task 3 answers

**RQ1:** how much stability margin do the default and the pilot-chosen filter settings waste, for realistic drones and real noise?

**Three settings are compared for every case:**
- **default:**
  - synthetic drones: the PX4 v1.16 default (D-09);
  - real drones: the log's own default (D-09).
- **actual:** the pilot's flown setting. Real drones only; it is the reference (D-15).
- **optimal:**
  - Mode A: the best PM at equal noise;
  - Mode B: the least noise at equal PM;
  - Mode C: the Pareto front.

**Reported together for every setting** (D-08): $\tilde N$, $\tau_F$ and $\tau_D$, PM, GM, $M_s$, and the worst case over the uncertainty set (D-36).

**Fixed by the decisions:**
- **Limits:**
  - nominal: PM ≥ 30°, $M_s$ ≤ 2.0, GM ≥ 6 dB (D-30, D-31);
  - worst case: every plant of the set (±20 % on $b$ and $\tau_m$, +1 controller sample) is stable, with $M_s$ ≤ 3.0 (D-32, D-36).
- **Noise band:** $f_{lo} = \max(5 f_c, 20\ \mathrm{Hz})$ to 780 Hz.
- **Chip delay:** D-33, rule and treatment.

## 2. Real drones that qualify (D-36 (6): an identification and a steady window from the same log)

### Fit half: 26 cases from 15 vehicles

b79d2e3d (2 cases) is excluded: it is identified, but it has no steady window.

| Vehicle | Axes | Controller | Chip | Unfiltered $f_c$ [Hz] (roll / pitch) | Flown setting vs the check (D-36) | Notes |
|---|---|---|---|---|---|---|
| 215afd8c | roll, pitch | 399 Hz | BMI088 | 6.7 / 6.0 | passes | |
| 225bab55 | roll, pitch | 400 Hz | ICM20689 | 3.6 / 4.2 | passes | roll: $f_{lo}$ clamped at 20 Hz |
| 4e73d30d | pitch | 400 Hz | ICM45686 | 8.5 | worst case fails | |
| 5cecf734 | roll, pitch | 801 Hz | ICM45686 | 3.2 / 3.3 | passes | Scenario B (RPM; bank off as flown); $f_{lo}$ clamped; control IMU unknown |
| 7e8f0064 | roll, pitch | 800 Hz | ICM45686 | 4.0 / 3.0 | passes | FFT notch not modelled; pitch: $f_{lo}$ clamped |
| 852b8368 | roll | 807 Hz | BMI055 | 6.3 | worst case fails | control IMU unknown |
| 94bd7879 | roll, pitch | 400 Hz | ICM45686 | 4.5 / 8.3 | roll passes; pitch: worst case fails | |
| 9ca62be3 | roll, pitch | 664 Hz | BMI055 | 5.0 / 4.8 | passes | control IMU unknown |
| ab1aecd2 | pitch | 998 Hz | BMI088 | 15.1 | nominal fails | above 10 Hz; FFT notch not modelled |
| b27d09e3 | roll, pitch | 800 Hz | ICM45686 | 10.1 / 9.4 | roll: worst case fails; pitch passes | Scenario B (bank on as flown); roll above 10 Hz (pitch: 10.06 Hz flown) |
| c901f9c1 | roll, pitch | 401 Hz | ICM20689 | 9.3 / 5.5 | nominal fails (both) | doubtful identification (F5) |
| d32cc24c | roll, pitch | 804 Hz | IIM42653 | 8.3 / 6.3 | nominal fails (both) | FPV airframe; FFT notch; ESC notch on without valid RPM |
| df9a5d9e | pitch | 810 Hz | ICM42688P | 4.7 | passes | |
| f96af168 | roll, pitch | 819 Hz | IIM42652 | 4.2 / 4.3 | passes | |
| fcdef3a8 | roll, pitch | 667 Hz | BMI055 | 5.7 / 4.0 | passes | |

**Counts:**
- **Flagged "reference unsafe": 9 of 26 (35 %).** They are kept and reported (D-36 (2)).
  - 5 fail nominally.
  - 4 fail only in the worst case: 4 of the 21 nominal passes (19 %). The 22 % of the follow-up was 5 of 23; b79d2e3d is no longer counted.
- **Above 10 Hz** (reported separately, D-36 (5)):
  - by the unfiltered crossover: 2 cases (ab1aecd2 pitch, b27d09e3 roll);
  - by the flown crossover: 3 (b27d09e3 pitch as well).
- **$f_{lo}$ clamped at 20 Hz:** 4 cases (flown $f_c$ 3.1-3.6 Hz).
- **Scenario B:** 4 cases from 2 vehicles.

### Test half

It is not identified yet, and nothing of it has been opened.
- **Estimate from the fit half's rate:** 15 of the 52 usable fit vehicles qualify (29 %). That gives about **12 of the 43 usable test vehicles** (binomial 90 % range 8-17), and **about 21 cases**.
- **The exact list comes from step S2:** the frozen identification pipeline (D-14 settings, D-36 gates), run before any RQ1 number is computed on the test half.
- No gate or setting changes after the test half is seen.

## 3. Synthetic drones with the D-36 priors

**Planning run:**
- 1,000 designs, seed 20261018, rate pairs of D-28.
- Priors and design: $\tau_m$ 15-60 ms, $T_d$ 0.3-8 ms, $f_c$ 3-10 Hz, PM window 60-95°, integral corner 0.03.
- The chip delay of D-33 is applied per chip. Results are weighted by the generator's chip shares.

| Quantity | Value |
|---|---|
| Draws needed per design (unfiltered PM window 60-95°) | 1.85 (acceptance 54 %) |
| Kept at the PX4 defaults (Tier 3 step 3, nominal limits) | **85 %** (by chip: 77 % ICM-42686-P, 82-86 % ICM-42688-P / IIM-42652 / ICM-45686, 88 % chips without a low-pass term) |
| Kept, but failing the worst case at the defaults | 4 % of the kept (3-8 % by chip) |
| Crossover of the kept drones | 3-4 Hz 33 %, 4-5 Hz 27 %, 5-6.5 Hz 25 %, 6.5-8 Hz 13 %, **8-10 Hz 1.7 %** |
| Real qualifying cases, unfiltered $f_c$ (for comparison) | 4 / 8 / 6 / 1 / 5 of 26 in the same bins, plus 2 above 10 Hz |

**The 8-10 Hz bin is thin.**
- 98 of the 1,000 designs lie there, but only about 15 pass at the PX4 defaults.
- The real drones in that bin fly with raised cutoffs or are flagged themselves.

**Proposal:**
- **1,000 kept synthetic drones**, each with one generator spectrum. About 1,170 designs and 2,170 draws are needed (seconds).
  - **Scenario B** for the drones whose source vehicle has motor harmonics: about 10 %, so about 100 drones from 5 sources.
  - **Scenario A** for the rest: static notches at the two largest peaks found in the spectrum. The detector (prominence 0.5 decades, D-26) is the same for synthetic and real spectra.
- **Real noise with synthetic drones:** each of the 95 vehicle spectra (52 fit, 43 test) with 10 synthetic drones, 950 cases. The two halves are reported separately.

## 4. Your decisions before Task 3 starts (recommended option first)

| # | Question | Options and impact |
|---|---|---|
| T3-1 | Do the optimizer's limits $\mathcal{S}$ include the worst-case limits? | **(a) Nominal limits for the nominal plant and the D-36 worst-case limits for the 8 corners (recommended).** Every "optimal" setting then passes the safety check of Section 7.8 by construction. It is checked lazily: grid points are tested against the 8 corners in the order of the objective (about 0.02 s per point), and the refinement uses all 9 plants. Run-time cost: about 5-10 %. **(b) Nominal only, as `optimizer.py` does now.** The optima may fail the worst case, so "optimal" would not be safe in the thesis's own sense. |
| T3-2 | Which half for the real parts? | **(a) Both halves, reported separately (recommended).** The fit half is in-sample for the A2-A4 choices. The test half runs last, with frozen code, and is the held-out result. **(b) Test half only,** as the Phase 4 plan said: about 12 vehicles. **(c) Fit half only:** the test half stays untouched until Phase 5, and there is no held-out RQ1 result. |
| T3-3 | Synthetic sample size | **(a) 1,000 kept drones, plus 950 real-noise cases (recommended).** **(b) 500 plus 475:** half the run time, wider intervals per crossover bin. **(c) Stratified:** at least 200 kept drones per crossover bin. The population result is then re-weighted. |
| T3-4 | Optimizer grid | **(a) Coarse search on the feasibility test's fast grid, then the final refinement and all reported metrics on the full grid (recommended).** The fast grid is 500 stability points, a 1 Hz noise grid and 24 × 18. Plus the brute-force check of THESIS 7.3 on 20 cases. About 6.7× faster. In 4 planning cases the fast optimum was within 0.2° PM and 1 % noise of the full-grid optimum. But in one case it broke the noise limit on the full grid ($\tilde N$ = 1.023), so the full-grid refinement is required. **(b) Full grid everywhere:** about 7× the run time. |
| T3-5 | "Real drones above 10 Hz" (D-36 (5)) | **(a) By the unfiltered crossover (recommended).** This is the definition the synthetic prior uses. 2 fit-half cases. **(b) By the flown crossover:** 3 cases. Both counts are reported either way. |
| T3-6 | Synthetic drones that pass nominally but fail the worst case at the defaults (about 4 % of the kept) | **(a) Keep and flag, as for the real drones (recommended).** Tier 3 step 3 asks only for the nominal limits, and D-36 (2) keeps flagged real drones. **(b) Drop them.** |

## 5. Figures and tables

- **Fig. 3, PM gain at equal noise** (Mode A: PM(θ*) − PM(θ_ref)).
  - One histogram per population: synthetic, real noise, real gain (fit), real gain (test).
  - A second row shows the change of $\tau_F$ and $\tau_D$.
  - Flagged drones and drones above 10 Hz are marked.
  - Small multiples, no dual axis.
- **Fig. 4, Pareto fronts** (Mode C, $\tilde N_{max}$ 0.25-2): the best PM against $\tilde N$ for 4 drones:
  - the median synthetic drone;
  - a synthetic drone at 8-10 Hz;
  - b27d09e3 roll (Scenario B);
  - one flagged real drone.

  Each panel marks the default, the actual setting and the Mode A / B optima.
- **Table 2, one row per population and scenario:**
  - n;
  - reference unsafe (nominal / worst case);
  - the share where Mode A is feasible;
  - median [p10, p90] of ΔPM, Δτ_F and Δτ_D at $\tilde N$ ≤ 1;
  - median [p10, p90] of Mode B $\tilde N$ at PM ≥ PM_ref;
  - the worst-case $M_s$ of the optimum.

  Scenario B rows are separate.
- **Table S1 (supplement):** every real-gain case: default / actual / optimal metrics and flags.
- **Results per crossover bin**, because the kept distribution is skewed.

## 6. Steps (test-first; full suite before each commit; local commits only)

1. **S1, D-36 in code:**
   - a Tier 3 sampler with the D-36 constants and the D-33 chip delays;
   - the `LoopCase` noise band $\max(5 f_c, 20\ \mathrm{Hz})$;
   - the optimizer limits from D-30 / D-31, plus the worst case if T3-1 (a), plus the grid of T3-4.

   **Tests:**
   - constants pinned to D-36;
   - the sampler reproduces the planning run of Section 3;
   - the f_lo rule;
   - the robust optimum passes all 9 plants;
   - the brute-force check.
2. **S2, real-gain cases:**
   - fit half: from the stored identifications and the library windows (26 cases);
   - test half (if T3-2 a or b): header eligibility first, then the frozen identification pipeline, then the same case builder.
3. **S3, RQ1 runner:**
   - synthetic, real-noise and real-gain parts; Modes A, B and C;
   - resumable chunks, with the seed and the git hash stored;
   - joblib with 6 workers;
   - the test-half part runs last.
4. **S4, outputs:**
   - Fig. 3, Fig. 4, Table 2 and Table S1;
   - THESIS 7.x, 8 and 17 updated;
   - a mutation check of the new tests;
   - the full suite, then a commit.
   - **STOP and report to you.**

## 7. Expected run time (this laptop, 8 logical CPUs, 6 workers)

**Measured on one core:**

| Case | Notch options | Full grid | Fast grid |
|---|---|---|---|
| Scenario A, two peaks | 57 | 77-80 s | 11.5-11.9 s |
| Scenario B | 19 | about 23 s | about 3 s (feasibility test: 1.9-2.4 s) |

A lazy robust check costs about 0.02 s per candidate point. Six workers should be 4-6 times faster than one core (not measured).

| Part | Cases | Fast grid (T3-4 a) | Full grid (T3-4 b) |
|---|---|---|---|
| Synthetic drones, generator spectra | 1,000 | 30-45 min | 3.5-5.5 h |
| Synthetic drones, real spectra | 950 | 30-45 min | 3.5-5 h |
| Real-gain cases (fit 26 + test about 21) | about 47 | 3-5 min, plus about 1 min of tracking-notch spectra per RPM case | 15-25 min |
| Test-half identification (S2) | about 43 logs | 6-8 min (the fit half took 15-19 min for 102 logs) | same |
| Brute-force check | 20 | about 5 min | same |
| **Total compute** | | **about 1.5-2 h** | **about 8-11 h** |

Writing the code and tests for S1-S4 takes about one working day.

**Input for A5 (risk R4):**
- Even on the fast grid, 10,000 two-peak training cases take about 5-8 h on 6 workers, and 30,000 take 16-24 h.
- The R4 target in THESIS was at most 3 s per case. The fast grid takes about 12 s for a two-peak Scenario A case.

## 8. Risks, and what I will not do

- **The test half:** nothing of it is opened before you approve this plan. Nothing is tuned after the test half is seen.
- **Mode A for flagged drones** can be infeasible: no setting at equal noise may meet $\mathcal{S}$. This is reported as a count. Mode B is usually feasible (higher cutoffs).
- **Scenario B is small:**
  - synthetic: about 100 drones from 5 sources;
  - real: 2 fit vehicles and about 2 test vehicles.

  It is reported separately (THESIS Section 11).
- **The kept synthetic population leans to low crossovers** (8-10 Hz: 1.7 %), so results are also reported per bin.
- **The Phase 5 question stays open:** a fallback for flagged real drones (THESIS 17).

**Planning scripts** (session scratchpad, not in the repo):
- Sections 2 and 3: `C:\Users\RAFI\AppData\Local\Temp\claude\d--research-thesis\af1a907e-f265-4915-8bfd-7c46fbe929fd\scratchpad\task3_numbers.py`
- Section 7, cost components: `C:\Users\RAFI\AppData\Local\Temp\claude\d--research-thesis\af1a907e-f265-4915-8bfd-7c46fbe929fd\scratchpad\task3_timing2.py`
- Section 7, fast vs full grid: `C:\Users\RAFI\AppData\Local\Temp\claude\d--research-thesis\af1a907e-f265-4915-8bfd-7c46fbe929fd\scratchpad\task3_timing3.py`
