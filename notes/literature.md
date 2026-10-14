# Literature notes

Made 2026-10-14 (Day 17, Phase 1). Keys in `[brackets]` are the BibTeX keys in `paper/refs.bib`; numbers like (#26) are the numbers in THESIS.md Section 16.

**How these notes were made.** Every entry was checked to exist (DOI from Crossref, or the arXiv page, or the URL). The notes come from the abstract, the publisher page and well-known content of the paper ("pass 1" of Keshav's 3-pass method). Entries marked **★ read deeper** are the ones the paper's Introduction, Related Work or Method will lean on. For those, do a "pass 2" before Phase 7: read the introduction, the figures and the conclusion (about 30-45 min each). Write anything new under "Your notes".

**Import into Zotero:** Zotero → File → Import… → `paper/refs.bib` → tick "Place imported collections and items into new collection". That gives you 60 references at once (Phase 1 target: at least 15).

---

## 0. One-page map: where our work sits

| Group | What they do | What they do NOT do (= our gap) |
|---|---|---|
| A. Noise-aware PID / filter co-design (process control) | Optimize PID + **one** measurement filter for robustness ($M_s$) and noise | No notch filters, no RPM-tracking banks, no multirotor, no real vibration spectra, no ML |
| B. Multirotor dynamics, vibration, notch filters | Show vibration comes from motors/propellers; notches cost phase margin | Hand-tuned single notches; no optimization of the whole filter stack |
| C. System identification | Closed-loop identification theory; quadrotor models from flight data | Not used to tune filters |
| D. Controller autotuning (BO, RL, metaheuristics) | Tune **PID gains** by many experiments or simulations per vehicle | Do not tune the filter stack; need many evaluations per drone; few guarantees |
| E. Amortized optimization / learned controllers with guarantees | Train a model to output an optimizer's answer instantly; verify statistically | Not applied to flight-controller filter tuning |
| F. TinyML | Run small models on microcontrollers | Tools only |
| G. Firmware docs and hobby tools | Practical rules; FPVTune uses a neural net on human tunes | No formal model, no stability guarantee |

**Gap sentence (for the Introduction, matches THESIS 5.2):** no published work models the complete multirotor gyro filter stack (lowpass, static notches, RPM-tracking notch bank) inside the attitude loop, optimizes it for stability margin under a motor-noise limit using **real** flight noise, and amortizes that optimization into a small, verifiable ML model.

**Closest competitors a reviewer may name:** Soltesz 2017 / Garpinger 2015 (group A: same *kind* of optimization, but one filter, no notches, no drone); FPVTune (group G: ML tuner, but human labels and no guarantee); Torres-García 2026 (group A: newest co-design paper, one lowpass only); Berkenkamp safe BO (group D: safe tuning, but gains and many experiments).

---

## A. Noise-aware PID design and controller-filter co-design

### ★ [soltesz2017simultaneous] Soltesz, Grimholt, Skogestad 2017, IET CTA (#27)
- **What:** Designs the PID gains **and** the measurement filter together by optimization: best load-disturbance performance subject to robustness ($M_s$, $M_t$) and a limit on noise sensitivity.
- **We take:** The same problem *shape* as our Mode A/B: performance vs. noise with $M_s$ as a hard limit. Cite it as the closest formulation.
- **We differ:** One filter of fixed type; no notches; process-control plants; no ML; no real noise spectra. Our gains are fixed and only the filter stack is tuned (PX4 practice).

### ★ [garpinger2015software] Garpinger, Hägglund 2015, J. Process Control (#28)
- **What:** Software tool for optimal PID design with robustness ($M_s$) and **noise-sensitivity** constraints.
- **We take:** Justifies using $M_s$ together with a noise measure as design constraints (Section 7.2, 7.3).
- **We differ:** Same as above; we add notches, real drone noise and the ML layer.

### [romerosegovia2014noise] Romero Segovia, Hägglund, Åström 2014, J. Process Control (#26)
- **What:** How the measurement-noise filter changes PID performance; measures noise as the variance of the control signal caused by sensor noise ($\sigma_u$).
- **We take:** Our noise metric $N^2 = \int |C_{fb}/(1+L)|^2 S_n\,df$ is exactly the control-signal noise variance idea. Cite it in Section 7.2.

### [isaksson2002derivative] Isaksson, Graebe 2002, IEE Proc. CTA (#29)
- **What:** The derivative filter must be designed **with** the PID, not added afterwards; it changes the margins.
- **We take:** One-line motivation: "filters are part of the loop design, not post-processing". Our D-term PT1 result (largest delay in the loop, REPORT Section 6) is an example.

### [kristiansson2006robust] Kristiansson, Lennartson 2006, IEEE CSM (#30)
- **What:** Robust PI/PID tuning that keeps derivative action useful despite sensor noise; uses high-frequency gain limits.
- **We take:** Supports the trade-off "D-term helps margins, but amplifies noise", which is why the D-term filter matters most.

### [abramovitch2002disk] Abramovitch, Franklin 2002, IEEE CSM (#31)
- **What:** History of hard-disk servo control: notch filters suppress mechanical resonances but cost phase margin; typical PM targets around 30-35°.
- **We take:** Source for the PM floor candidate (30°) and for "notches vs phase margin" in another industry with the same problem.

### ★ [torresgarcia2026codesign] Torres-García, Michiels 2026, arXiv (#1)
- **What:** Optimizes PID gains **and** a derivative low-pass filter together for systems with time delay, by minimizing the spectral abscissa of the filtered closed loop. Main message: the filter "cannot be treated as a post-processing step".
- **We take:** Newest evidence that controller-filter co-design is an active topic; cite in Related Work.
- **We differ:** One lowpass only; no notch or RPM filters; no noise spectra from real drones; no ML; generic plants.

### [astrom2008feedback] Åström, Murray, *Feedback Systems* (textbook)
- Reference for PM, GM, $M_s$, Nyquist criterion. (Per your learning style: use the Brian Douglas videos to learn; cite the book in the paper.)

---

## B. Multirotor dynamics, vibration and notch filters

### [mahony2012multirotor] Mahony, Kumar, Corke 2012, IEEE RAM (new)
- **What:** Standard tutorial on quadrotor modelling and control.
- **We take:** Citation for the rigid-body/rotor model behind $P(s) = \frac{b}{s}\cdot\frac{1}{\tau_m s+1}\cdot e^{-sT_d}$ (Section 7.1).

### ★ [bangura2017thrust] Bangura, Mahony 2017, IEEE T-RO (#35)
- **What:** Rotor/thrust dynamics and thrust control of multirotors; motor speed response.
- **We take:** Tier 1 prior for the motor time constant $\tau_m$ (Section 7.1.1). **Phase 2 action:** check whether the paper reports a measured motor/rotor time constant (pass 2); if yes, write it in the "Source / status" column, if not, use our own Tier 2 value (τ_m ≈ 29-34 ms, feasibility test) and NeuroBEM rotor-speed data.

### ★ [chen2023quadcopter] Chen, Zhang, Quan 2023, Nonlinear Dynamics (#32)
- **What:** Vibration from propeller imbalance degrades quadcopter attitude control; they add a notch filter at the vibration frequency inside a dynamic-inversion controller; flight tests.
- **We take:** Peer-reviewed proof that (1) vibration is at motor-related frequencies and (2) notching it helps attitude control.
- **We differ:** One hand-placed notch; no optimization of the whole stack; no stability-margin vs noise trade-off; no ML.

### ★ [xu2019tailsitter] Xu, Gu, Qing, Lin, Zhang 2019, ICUAS (new)
- **What:** Quadrotor tail-sitter with a structural resonance near 14 Hz; they add a notch instead of stiffening the airframe and note that the notch reduces phase margin and must be tuned with care.
- **We take:** A direct, peer-reviewed statement of our core trade-off ("notch removes vibration but eats phase margin"). Good for the Introduction.

### [ghalamchi2020vibration] Ghalamchi, Jia, Mueller 2020, IEEE/ASME T-Mech (new)
- **What:** Real-time propeller-imbalance diagnosis from onboard accelerometer vibration; vibration power scales with motor speed.
- **We take:** Supports the noise model "motor-harmonic peaks whose size depends on the operating point" (Section 7.5) and the motivation for in-flight retuning (damaged propeller → noise changes).

### [smeur2016adaptive] Smeur, Chu, de Croon 2016, JGCD (#33) and [tal2021accurate] Tal, Karaman 2021, IEEE TCST (#34)
- **What:** INDI attitude control. Both rely on **filtered** angular acceleration and stress that the filter delay must be matched/synchronized.
- **We take:** Evidence that filter delay is a first-order design issue even for advanced controllers. One sentence in Related Work or Discussion ("our method also applies to INDI-style controllers" is **future work**, do not claim).

---

## C. System identification (Tier 2, decision D-14)

### ★ [forssell1999closedloop] Forssell, Ljung 1999, Automatica (new)
- **What:** The standard survey of closed-loop identification: direct, indirect and joint input-output methods; explains why direct least squares on closed-loop data is biased when noise and input are correlated through feedback.
- **We take:** The theory behind what the feasibility test saw (least squares failed, held-out VAF negative) and behind D-14 (use an external reference signal as the instrument: $G = S_{r\omega}/S_{ru}$ is the "indirect / joint input-output" idea). **Cite this when you explain D-14.**

### [bergamasco2014identification] Bergamasco, Lovera 2014, IEEE TCST (new)
- **What:** Identifies linear models of a hovering quadrotor's attitude dynamics from flight data, in closed loop.
- **We take:** Precedent that quadrotor attitude dynamics can be identified from closed-loop flight data; compare their model structure and parameter ranges with our $b, \tau_m, T_d$ (Tier 1 priors).

### [tischler2012aircraft] Tischler, Remple 2012, AIAA book (new)
- **What:** The frequency-response method for aircraft/rotorcraft identification (CIFER): frequency sweeps, coherence, fitting transfer functions.
- **We take:** Source for the coherence rule (coherence ≥ 0.5 in D-14 and the ≥ 6-frequency rule). Cite, do not read in full.

---

## D. Controller autotuning (competing approaches)

### ★ [berkenkamp2016safe] Berkenkamp, Schoellig, Krause 2016, ICRA (#36) and [berkenkamp2023bayesian] Machine Learning 2023 (new)
- **What:** SafeOpt: Bayesian optimization that only tries controller parameters that are safe with high probability; tuned on real quadrotors.
- **We take:** Main "safe tuning" competitor. Our difference: (1) we tune the **filter stack**, not gains; (2) we need **no flight experiments per drone** — the physics model plus a measured noise spectrum gives the answer in one forward pass; (3) safety comes from a physics check (Nyquist + robust margins), not from a GP's confidence bounds.

### [holzapfel2023eventtriggered] Holzapfel, Brunzema, Trimpe 2023, arXiv (new)
- **What:** Safe BO that detects when the system changes over time (wear, operation changes) and restarts safely; quadcopter hardware experiments.
- **We take:** Supports our "noise changes during a flight / over the drone's life" motivation for **instant** retuning. Our method reacts in one prediction instead of re-exploring.

### [loquercio2022autotune] Loquercio, Saviolo, Scaramuzza 2022, IEEE RA-L (#37)
- **What:** Sampling-based controller tuning (Metropolis-Hastings) for high-speed quadrotor flight, mostly in simulation.
- **We take:** Another "search per vehicle" tuner. Same difference as BO: many evaluations, gains not filters.

### [vaiuso2025multiobjective] Vaiuso et al. 2025, arXiv (#11)
- **What:** Compares metaheuristics, Bayesian optimization and deep RL for tuning quadrotor PID gains in simulation, with a composite cost including **acoustic noise** and power. Grey Wolf Optimization did best; RL did not beat the baseline.
- **We take:** Recent evidence that (1) black-box search per case is the usual approach, (2) RL is not an easy win (supports D-05: no RL). Note: their "noise" is **acoustic**, ours is **motor-command noise from gyro vibration** — do not confuse them in the paper.

### [sonmez2025rl] Sönmez et al. 2025, Drones (new; preprint arXiv:2502.04552)
- **What:** Deep RL (DDPG) trained offline on a PX4 quadrotor attitude PID, adjusting gains online; simulation and flight tests.
- **We take:** Closest "ML predicts tuning parameters for PX4" work. Our differences: supervised amortization of a physics optimizer (labels are provably optimal within the model), filter stack instead of gains, physics safety check, real noise from public logs. Also shows MDPI *Drones* publishes this kind of work (venue fit).

### [kavas2024insitu] Kavas et al. 2024, arXiv (#12)
- **Warning:** this paper is about a **laser powder-bed-fusion** process, not drones. Keep it only as a general "BO autotuning" example, or drop it; Berkenkamp/Holzapfel are better drone examples.

### [fpvtune2025] FPVTune blog (#10)
- **What:** Commercial tool; a neural network trained on human before/after tunes from Betaflight Blackbox logs.
- **We differ:** human labels (copies human mistakes), closed source, no stability check, anecdotal validation. Our labels come from a physics optimizer and every prediction is verified.

---

## E. Amortized optimization, learned controllers with guarantees, physics-guided ML

### ★ [amos2023tutorial] Amos 2023, Foundations and Trends in ML (#38)
- **What:** The reference tutorial on amortized optimization: learn a model $\hat\theta(x) \approx \arg\min_\theta f(\theta; x)$ so that solving a new instance costs one forward pass. Distinguishes **regression-based** amortization (learn from optimizer labels — what we do) and **objective-based** amortization (train through the objective).
- **We take:** The name and framing of our ML part (contribution C3). Our ML is "regression-based amortization with optimizer labels plus a verification layer".

### [chen2022learning] Chen et al. 2022, JMLR (new)
- **What:** Survey/benchmark of "learning to optimize" (L2O). Key warning: learned optimizers do poorly **out of distribution**.
- **We take:** Motivation for our synthetic-to-real test (RQ2) and for the safety check: the ML may fail on unusual spectra, so the physics check must catch it.

### [withoft2026amortized] Withöft et al. 2026, arXiv (#13)
- **What:** Amortized neural optimization for signal-integrity design: a policy maps a context to near-optimal parameters in one forward pass; about **10 % optimality loss for 3-4 orders of magnitude speed-up**.
- **We take:** A recent number to compare with our regret target (median ≤ 2° PM) and speed-up (optimizer ~2 s vs. ML microseconds). Different domain.

### [osti2481183] Neural network approaches for parameterized optimal control (#14)
- **What:** NNs that output solutions of parameterized optimal-control problems.
- **Action:** the OSTI record has no authors in our bib yet; complete it from the OSTI page before Phase 7, or drop it (Amos 2023 covers the same idea).

### ★ [hertneck2018learning] Hertneck, Köhler, Trimpe, Allgöwer 2018, IEEE L-CSS (#39)
- **What:** Learns an approximate MPC with a neural network and gives a **statistical guarantee** of closed-loop safety by validating on many sampled cases (Hoeffding-type bound).
- **We take:** The template for Section 7.8, step 7: "0 violations in $n$ held-out cases ⇒ violation rate < 3/$n$ with 95 % confidence". Cite it together with [clopper1934confidence].

### [karg2020efficient] Karg, Lucia 2020, IEEE T-Cyb (#40)
- **What:** Deep networks that approximate MPC control laws compactly, for embedded use.
- **We take:** Precedent for "learn an expensive optimizer, run the cheap network on embedded hardware".

### [willard2022integrating] Willard et al. 2022, ACM Computing Surveys (#41)
- **What:** Survey of physics-guided / science-guided ML.
- **We take:** Justifies the term "physics-guided ML" in the title. In their taxonomy our method uses physics (1) to **generate training labels** and (2) as a **hard check on the output**. Use their wording.

### [brunke2022safe] Brunke et al. 2022, Annual Review (#42)
- **What:** Survey of safe learning in robotics; separates "learning + safety filter/certificate" approaches.
- **We take:** Our safety check is a "safety filter" in their language (a learned proposal + a model-based verification and projection).

### [clopper1934confidence] Clopper, Pearson 1934, Biometrika (new)
- **What:** The exact binomial confidence interval.
- **We take:** The violation-rate upper bound when $k > 0$ violations (Section 7.8, step 7; `scipy.stats.binomtest(k, n).proportion_ci()`).

---

## F. TinyML (Phase 6)

| Key | One line | Use |
|---|---|---|
| [lai2018cmsisnn] (#43) | Optimized NN kernels for Arm Cortex-M | Per-MAC cost for the Cortex-M7 estimate (Section 7.10) |
| [banbury2021mlperftiny] (#44) | Standard benchmark suite for tiny ML | Context: how embedded ML is benchmarked |
| [david2021tflm] (new) | TensorFlow Lite Micro inference framework | Alternative to our hand-written C forward pass; cite as "standard toolchain we did not need" |
| [jacob2018quantization] (new) | Integer-only (int8) quantization scheme | Method behind the optional int8 version of the MLP |

---

## G. Data, firmware and software sources

| Key | What it is | Why we cite it |
|---|---|---|
| [meier2015px4] (new) | The PX4 architecture paper (ICRA 2015) | The standard academic citation for PX4 |
| [px4docs_filtertuning] (#2) | PX4 filter tuning guide | Today's practice = trial and error with FFT plots (see `notes/px4_notes.md`) |
| [px4src_imugyro], [px4src_logger] (#3, #4, #19) | PX4 source files | Exact filter structure, defaults, logging profiles |
| [px4_flightreview], [px4_dataset_licence], [px4docs_statistical] (#5, #20, #21) | Public log database, its CC-BY licence | Data source and licence statement |
| [pyulog] (#6) | ULog parser | Software citation |
| [betaflight_src], [betaflight_rpm], [liang_bffiltering] (#7, #8, #17) | Betaflight sources and guides | Optional Betaflight extension; hobby practice |
| [ardupilot_notch] (#9) | ArduPilot harmonic notch docs | The third big firmware uses the same filter structure → generality |
| [lusk_adaptivegyro] (#15) | Adaptive notch research code | "Filtering method, not tuning of the whole stack" |
| [roboto2025insights] (#18) | Statistics from 120k PX4 logs | Context on public log data |
| [le2023rflymad], [puchalski2024padre], [padre_github], [delmerico2019uzhfpv], [bauersfeld2021neurobem] (#16, #22-25) | Other drone datasets | Why we use Flight Review (only it has raw high-rate gyro + PX4 parameters at scale); NeuroBEM also gives measured rotor speeds (Tier 1 prior for $f_m$) |
| [welch1967psd] (new) | Welch's PSD method | Section 7.4.4 (all spectra) |
| [pedregosa2011sklearn] (new) | scikit-learn | Software citation for all ML |

---

## H. Must-read list for pass 2 (in this order, about 5 h total)

1. [soltesz2017simultaneous] — closest problem formulation (45 min)
2. [amos2023tutorial] — only Sections 1-2 and the "regression-based amortization" part (45 min)
3. [hertneck2018learning] — short letter; the statistical-guarantee idea (30 min)
4. [forssell1999closedloop] — only the introduction and the part on why direct methods are biased (30 min)
5. [chen2023quadcopter] — notch filter on a real quadcopter (30 min)
6. [berkenkamp2016safe] — main competitor (30 min)
7. [sonmez2025rl] — PX4 + ML tuning in our target journal (30 min)
8. [torresgarcia2026codesign] — newest co-design paper (30 min)

Video alternatives (learning style): Brian Douglas "Control Systems Lectures" (Nyquist, margins), Steve Brunton "Control Bootcamp" (sensitivity, robustness). For the papers themselves there is no video; reading the introduction and figures is enough.

## I. Searches done (Section 5.3 search terms) and what they found

| Search (2026-10-14) | Result |
|---|---|
| quadrotor gyro vibration notch filter phase lag | Chen 2023 (already had), Xu 2019 tail-sitter (added), ArduPilot docs |
| Bayesian optimization PID quadrotor safe | Berkenkamp 2023 journal version, Holzapfel 2023 (added) |
| quadrotor control latency / delay margin | Mostly theses and simulation papers; nothing better than our existing references. Delay-margin theory comes from Åström & Murray |
| learning to optimize survey | Chen et al. 2022 JMLR (added) |
| neural network PID gains quadrotor | Sönmez 2025 (added); others are simulation-only RL, not cited |
| multirotor vibration spectrum | Ghalamchi 2020 (added) |
| TinyML / quantization | TFLite Micro 2021, Jacob 2018 (added) |
| PX4 framework / ULog | Meier 2015 (added) |
| closed-loop identification (for D-14) | Forssell & Ljung 1999, Bergamasco & Lovera 2014, Tischler & Remple 2012 (added) |

**No paper found that tunes the full PX4/Betaflight gyro filter stack by optimization or ML.** The gap of THESIS 5.2 still holds as of 2026-10-14. Re-run the searches "gyro filter tuning optimization drone" and "notch filter tuning machine learning quadrotor" once more in Phase 7 before submission (new preprints appear every month).

---

## Your notes

*(Add notes here after pass-2 reading.)*
