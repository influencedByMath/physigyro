# Math roadmap for the thesis

Made 2026-10-10 (Day 13) from THESIS.md. Every concept has an ID (A1, C9, F10 ...). To get an explanation, ask Claude: "explain C9".

**Legend:** ★ = core: understand it well, your professor and reviewers can ask about it. ○ = know the idea: numpy, scipy or sklearn does the work.

No books here. Only videos, interactive pages and short web pages. Where no good short source exists, the entry says "ask me".

---

## A. Foundations (used everywhere)

| ID | Concept | Where you use it |
|---|---|---|
| A1 ★ | Complex numbers: real and imaginary part, magnitude, angle (phase), polar form | every H(e^jω) and L(f) |
| A2 ★ | Euler's formula e^(jθ) = cos θ + j sin θ; the unit circle | z = e^(j2πfT_s) |
| A3 ★ | Multiplying complex numbers: magnitudes multiply, angles add | filter stack, notch bank |
| A4 ★ | Radians vs degrees; ω = 2πf | everywhere |
| A5 ★ | Logarithms (log10, ln), log-spaced grids, log-uniform sampling | 7.1.1, 7.2, 7.5 |
| A6 ★ | Decibels: 20·log10 for amplitude, 10·log10 for power | GM 6 dB, M_s, PSD errors in dB |
| A7 ○ | Trig details: tan, atan2, phase unwrapping | Butterworth K = tan(πf_cT_s), phase calculation |
| A8 ○ | Small-value approximations (Taylor: tan x ≈ x, arctan x ≈ x) | delay rules of thumb (4.4) |
| A9 ○ | Interpolation in log space: exp((1−λ)·log a + λ·log b) | safety-check repair (7.8) |

**Learn fast (about 2 h):**

- 3Blue1Brown (YouTube): "What is Euler's formula actually saying?"
- BetterExplained.com: "A Visual, Intuitive Guide to Imaginary Numbers" and "Intuitive Understanding of Euler's Formula"
- Khan Academy: "Complex numbers" unit (polar form, multiplying in polar form). Only if you are rusty.
- A6-A9: ask me, about 10 minutes each.

---

## B. Calculus and difference equations

| ID | Concept | Where you use it |
|---|---|---|
| B1 ★ | Definite integral = area under a curve; the integral of a spectrum = power in a band | noise metric N² (7.2) |
| B2 ★ | Trapezoidal rule on a non-uniform grid | `numpy.trapezoid` |
| B3 ★ | Difference equations: compute y_n from y_(n−1) (recursion) | PT1, biquads, sysid model, time simulation |
| B4 ○ | Derivative and finite difference (ω_k − ω_(k−1)) / T_c | D term, sysid |
| B5 ○ | First-order differential equation, exponential response e^(−t/τ), time constant | motor lag τ_m |

**Learn fast (about 1.5 h):**

- 3Blue1Brown, "Essence of calculus" playlist: ch. 1, ch. 5 (Euler's number e), ch. 8 (integration), ch. 11 (Taylor series, also covers A8)
- B2, B3: ask me. They are short, and you learn them best in code.

---

## C. Signals and spectra (Phase 3, needed now)

| ID | Concept | Where you use it |
|---|---|---|
| C1 ★ | Sampling, sample rate f_s, sample period T_s | all logs, gate Q5 |
| C2 ★ | Nyquist frequency, aliasing, folding | gate Q5, E10 |
| C3 ★ | The Fourier idea: any signal is a sum of sine waves | foundation of everything below |
| C4 ★ | DFT and FFT, frequency bins, resolution Δf = f_s / N | PSD, peaks |
| C5 ★ | Spectral leakage and window functions (Hann) | Welch |
| C6 ★ | Power spectral density (PSD) and its units | the whole noise pipeline |
| C7 ★ | Welch's method: segments, overlap, averaging; `nperseg` trades resolution against noise | 7.4.4, gate Q13 |
| C8 ★ | Parseval's theorem: area under the PSD = signal variance | band power, noise metric |
| C9 ★ | PSD through a filter: S_out(f) = \|H(f)\|² · S_in(f) | noise metric, E10, E11. **The most important formula of the thesis** |
| C10 ★ | Fundamental and harmonics (h · f_m, with f_m = RPM / 60) | notch bank, noise model |
| C11 ○ | White-noise floor and the Lorentzian peak shape A / (1 + ((f − f0)/γ)²) | noise model (7.5) |
| C12 ○ | Peak detection and prominence | `find_peaks` |
| C13 ○ | Mean removal, detrending, high-pass, resampling and decimation | sysid, 7.11 |
| C14 ○ | Making a time signal from a PSD (random-phase inverse FFT) | time-domain simulation (7.11) |
| C15 ○ | Stationarity (noise statistics stay constant inside a window) | assumption in 7.4 and 7.11 |
| C16 ○ | Convolution and impulse response (what a filter does in time) | background |

**Learn fast (about 4 h):**

- 3Blue1Brown: "But what is the Fourier Transform? A visual introduction" (20 min)
- Jack Schaedler, "Seeing Circles, Sines and Signals" (interactive): https://jackschaedler.github.io/circles-sines-signals/ . Covers sampling, aliasing, DFT and leakage.
- Iain Explains Signals, Systems, and Digital Comms (YouTube): "What is Power Spectral Density (PSD)?"
- Steve Brunton (YouTube), "Fourier Analysis" playlist: the FFT videos and "Denoising Data with FFT [Python]"
- C7-C9, C11, C14: ask me. Then try them on one pilot log with `scipy.signal.welch`.

---

## D. Transforms and LTI systems (Phase 2)

| ID | Concept | Where you use it |
|---|---|---|
| D1 ★ | LTI system (linear + time-invariant): why one frequency response describes it | the whole frequency-domain model |
| D2 ★ | Laplace transform, the s-domain, transfer function H(s) | plant P(s), PID |
| D3 ★ | Poles and zeros; stability = all poles in the left half-plane | stability, unit tests |
| D4 ★ | Z-transform; z⁻¹ = one-sample delay; H(z) from a difference equation | all digital filters |
| D5 ★ | Frequency response: put s = j2πf, or z = e^(j2πfT_s) | every metric |
| D6 ★ | Discrete-time stability: all poles inside the unit circle | filter checks |
| D7 ★ | Blocks in series multiply: magnitudes multiply (dB add), phases add | filter stack, notch bank, L(f) |
| D8 ★ | Pure delay e^(−sT): gain 1, phase −2πfT | T_d, ZOH, delay margin |
| D9 ★ | Integrator b/s and first-order lag 1/(τs + 1) | plant P(s) |
| D10 ★ | Bilinear (Tustin) transform, frequency warping, pre-warping with tan | Butterworth and notch formulas (4.4) |
| D11 ★ | Zero-order hold (ZOH) and why it is about half a sample of delay | loop model (7.1) |
| D12 ★ | Phase lag, phase delay τ = φ/(2πf), and group delay −dφ/dω (know the difference) | τ_F, τ_D (7.2) |
| D13 ○ | Multi-rate systems (filters at the FIFO rate, controller at 800 Hz) | VERIFY item, E10 |

**Learn fast (about 3 h):**

- Zach Star (YouTube): "The intuition behind Fourier and Laplace transforms I was never taught in school"
- Steve Brunton, Control Bootcamp: "Laplace Transforms and the Transfer Function"
- Brian Douglas (YouTube), "Discrete control" series #1-#6. #2 (discretize), #3 (zero-order hold), #5 (bilinear transform) and #6 (z-plane warping) cover D4, D10 and D11 directly.
- D7, D8, D12, D13: ask me.

---

## E. Digital filters (Phase 2, `filters.py`)

| ID | Concept | Where you use it |
|---|---|---|
| E1 ★ | FIR vs IIR (recursive) filters | why PX4 filters are IIR |
| E2 ★ | First-order lowpass (PT1, the same as an EMA), cutoff, −3 dB point | PT1 (4.4) |
| E3 ○ | PT2/PT3 cascades and the cutoff correction factor c_n | Betaflight extension |
| E4 ★ | Second-order section (biquad): coefficients b0, b1, b2, a1, a2 | Butterworth, notch |
| E5 ★ | Butterworth lowpass: maximally flat, Q = 1/√2 (where the √2 comes from) | PX4 gyro and D-term filters |
| E6 ★ | Notch filter: centre f0, bandwidth BW, Q = f0 / BW, phase near the notch | static and RPM notches |
| E7 ★ | Deriving the delay rules: 1/(2πf_c), √2/(2πf_c), BW/(2πf0²) | unit tests, paper |
| E8 ○ | State-variable filters | Betaflight extension only |

**Learn fast (about 2 h):**

- Brian Douglas, "Discrete control" #5-#6 (already listed in D)
- EarLevel Engineering, "Biquad calculator" (earlevel.com, interactive): move the cutoff and Q of a lowpass or notch and watch magnitude and phase change
- W3C "Audio EQ Cookbook": one web page with the lowpass and notch formulas (already in THESIS.md)
- Then plot your own filters with `scipy.signal.freqz`. This is where it sticks.
- E5, E7: ask me (derivations, about 20 minutes).

---

## F. Feedback control (Phase 2, the heart of contribution C1)

| ID | Concept | Where you use it |
|---|---|---|
| F1 ★ | Block diagrams; open loop L = C_fb · ZOH · P; closed loop | 7.1 |
| F2 ★ | Characteristic equation 1 + L = 0 | why −1 is the "danger point" |
| F3 ★ | Sensitivity S = 1/(1+L), complementary sensitivity T = L/(1+L), noise-to-control C/(1+L) | M_s, noise metric N |
| F4 ★ | PID in the frequency domain; D on measurement; D-term filter; why feed-forward does not change L | 7.1, PX4 controller |
| F5 ★ | Bode plot (magnitude in dB and phase vs log frequency) | Fig. 1, debugging |
| F6 ★ | Gain crossover f_c and phase crossover f_180 | 7.2 |
| F7 ★ | Phase margin and gain margin | all results |
| F8 ★ | Delay margin DM = PM/(2πf_c) and the phase cost of delay ΔPM = 360 · f_c · τ | 4.4, 7.0 |
| F9 ★ | Nyquist plot and the Nyquist stability criterion (encirclements of −1) | safety check, step 2 |
| F10 ★ | Peak sensitivity M_s = 1 / (shortest distance from the Nyquist curve to −1); disk margin | safety check, all tables |
| F11 ★ | Several crossovers caused by notches: always take the worst | 7.2 |
| F12 ★ | Robustness: plant uncertainty, worst case over corner plants | uncertainty set, 7.8 |
| F13 ○ | Crossover design: solve K_p so that \|L(f_c)\| = 1 | Tier 3a |
| F14 ○ | Closed-loop poles as a cross-check (`python-control`) | unit tests |
| F15 ○ | Step response: rise time, overshoot, settling time, RMS error | 7.11, E8 |
| F16 ○ | Argument principle (the math behind Nyquist) | only if a reviewer asks |

**Learn fast (about 5 h):**

- Brian Douglas (YouTube), "Control System Lectures" playlist: "Transfer Functions", "Bode Plots by Hand" (first 2-3 videos), "Nyquist Stability Criterion" parts 1-2, "Gain and Phase Margins Explained!"
- MATLAB Tech Talks, "Understanding PID Control" Parts 1-3 (Part 3 is about the derivative and its filter)
- MATLAB Tech Talks, "Robust Control, Part 2: Understanding Disk Margin" (14 min). This is the picture behind M_s.
- Steve Brunton, Control Bootcamp: "Sensitivity and Robustness", "Loop Shaping", "Limitations on Robustness"
- Practice: `control.margin()` and `control.nyquist_plot()` from python-control on a simple loop

---

## G. Linear algebra

| ID | Concept | Where you use it |
|---|---|---|
| G1 ★ | Vectors, matrices, matrix × vector | MLP forward pass, MAC count |
| G2 ★ | Linear least squares Ax ≈ b, residuals | Tier 2 system identification (`lstsq`) |
| G3 ○ | Rotation matrices and reference frames (board → body FRD) | gate Q7 |
| G4 ★ | Covariance matrix, eigenvectors and eigenvalues | PCA |
| G5 ○ | SVD (how PCA is actually computed) | PCA |
| G6 ★ | Euclidean distance | kNN, k-means |
| G7 ○ | Standardization (z-score) | `StandardScaler` |

**Learn fast (about 3 h):**

- 3Blue1Brown, "Essence of linear algebra" playlist: ch. 1-5 (vectors, matrices, 3-D transforms = rotations), ch. 9 (dot product), ch. 14 (eigenvectors)
- StatQuest: "Linear Regression, Clearly Explained" (the least-squares idea)
- Steve Brunton, SVD playlist: the "Linear Regression" and "Principal Component Analysis (PCA)" videos
- G3 for PX4: ask me when you reach gate Q7.

---

## H. Probability and statistics

| ID | Concept | Where you use it |
|---|---|---|
| H1 ★ | Mean, variance, standard deviation, median, percentiles, histograms | regret (median, 95th percentile), all reports |
| H2 ★ | Distributions: uniform, log-uniform, Gaussian, multivariate Gaussian | Tier 3 sampling, noise generator |
| H3 ★ | Random sampling, random seeds, Monte Carlo | `dataset_gen.py` |
| H4 ★ | Chi-squared distribution and degrees of freedom → confidence bound of a PSD | gate Q13, S_up in the safety check |
| H5 ★ | Binomial distribution; confidence interval of a proportion; Clopper-Pearson; rule of three | violation-rate bound (7.8 step 7), audit 7.7 % CI |
| H6 ★ | Variance accounted for (VAF), similar to R² | sysid quality gate |
| H7 ○ | Correlation, cross-tabulation | gate Q12 |
| H8 ○ | Autocorrelation and Wiener-Khinchin (PSD = Fourier transform of the autocorrelation) | deeper understanding only |
| H9 ★ | Data leakage, independence, grouped splits (an idea, not a formula) | vehicle split, GroupKFold |

**Learn fast (about 2.5 h):**

- Seeing Theory (interactive): https://seeingtheory.brown.edu . Chapters "Basic Probability", "Probability Distributions", "Frequentist Inference" (confidence intervals)
- StatQuest: "Histograms", "The Normal Distribution", "Quantiles and Percentiles", "R-squared"
- Khan Academy: "Binomial distribution" and "Chi-square distribution introduction"
- Wikipedia: "Rule of three (statistics)" (5 min)
- H4 and H5 for your exact case: ask me.

---

## I. Optimization (Phase 4)

| ID | Concept | Where you use it |
|---|---|---|
| I1 ★ | Objective, decision variables, constraints, feasible set | 7.3 |
| I2 ★ | Constrained optimization (Modes A and B) | 7.3 |
| I3 ★ | Grid search (brute force) with vectorization | solver step 1 |
| I4 ○ | Local vs global optimum; gradient | why grid first, then refine |
| I5 ○ | Nelder-Mead (derivative-free simplex) | refinement |
| I6 ○ | SLSQP and Lagrange multipliers (idea only) | refinement with constraints |
| I7 ★ | Root finding: bisection and Brent's method | K_p design (`brentq`), safety-check repair |
| I8 ★ | Multi-objective optimization: Pareto front, dominance, ε-constraint method | Mode C |
| I9 ★ | Nonlinear least squares; fitting in log space | noise-model fit (`least_squares`) |
| I10 ○ | Mixed discrete + continuous variables | harmonic count, notch on/off |
| I11 ★ | Amortized optimization: train a model to output the optimizer's answer | the core ML idea, contribution C3 |
| I12 ○ | Projection (clipping to bounds) | safety check, step 1 |

**Learn fast (about 1.5 h):**

- SciPy docs, "Optimization and root finding" tutorial (docs.scipy.org): only the `minimize`, `least_squares` and root-finding parts
- Wikipedia: "Nelder-Mead method" (the animation is enough) and "Pareto front"
- Khan Academy: "Lagrange multipliers, introduction" (optional)
- Brandon Amos, "Tutorial on amortized optimization" (ref. 38 in THESIS.md): read only the introduction and Figure 1. You must cite it anyway.
- I8 and I11: ask me.

---

## J. Machine learning (Phases 3 and 5)

| ID | Concept | Where you use it |
|---|---|---|
| J1 ★ | Supervised learning: features, targets; regression vs classification | 7.7 |
| J2 ★ | Loss functions MSE and MAE; why predict log(f) | 7.7 |
| J3 ★ | Train / validation / test; cross-validation; GroupKFold | training protocol |
| J4 ★ | Overfitting, bias-variance, regularization | model choice |
| J5 ★ | k-nearest neighbours regression | baseline B2 |
| J6 ★ | Ridge regression = least squares + L2 penalty | baseline B3 |
| J7 ★ | Regression trees (split to reduce squared error) | inside gradient boosting |
| J8 ★ | Gradient boosting: each new tree fits the remaining error; learning rate | main model M1 |
| J9 ★ | MLP: neuron, weights, bias, ReLU, forward pass | main model M2, C export |
| J10 ○ | Backpropagation, gradient descent, Adam | how the MLP trains |
| J11 ○ | Multi-output regression; classification output (softmax, cross-entropy) | several θ outputs, harmonic count |
| J12 ○ | Hyperparameters and random search | `RandomizedSearchCV` |
| J13 ★ | PCA: explained variance ratio, keep 95 % | 7.5 |
| J14 ★ | k-means: centroids, inertia, Lloyd's algorithm | 7.5 |
| J15 ★ | Silhouette score | choosing k |
| J16 ★ | Permutation importance | 7.9, Fig. 6 |
| J17 ★ | Partial dependence | 7.9, Fig. 7 |
| J18 ★ | Your own metrics: regret, violation rate | Table 3 |
| J19 ○ | Distribution shift (the synthetic-to-real gap) | RQ2 |

**Learn fast (about 6 h):**

- StatQuest (Josh Starmer, YouTube): "Cross Validation", "Bias and Variance", "K-nearest neighbors", "Regularization Part 1: Ridge (L2) Regression", "Regression Trees", "Gradient Boost Part 1 and Part 2 (Regression)", "Neural Networks Pt. 1: Inside the Black Box", "Gradient Descent, Step-by-Step", "Backpropagation Main Ideas", "PCA Step-by-Step", "K-means clustering"
- 3Blue1Brown, "Neural networks" playlist: ch. 1-3
- MLU-Explain (interactive): https://mlu-explain.github.io . Pages "Train, Test, and Validation Sets", "Cross-Validation", "Bias-Variance Tradeoff", "Decision Trees"
- setosa.io: "Principal Component Analysis explained visually"
- scikit-learn example: "Selecting the number of clusters with silhouette analysis on KMeans clustering"
- Kaggle Learn, "Machine Learning Explainability" (free mini-course): only the lessons on permutation importance and partial plots
- J18, J19: ask me.

---

## K. Embedded / TinyML (Phase 6)

| ID | Concept | Where you use it |
|---|---|---|
| K1 ★ | Counting MACs: Σ n_(ℓ−1) · n_ℓ | Table 4 |
| K2 ○ | float32 vs int8; quantization with a scale and zero-point; quantization error | int8 export |
| K3 ○ | Model size = number of parameters × bytes per parameter | 20 KB target |

**Learn fast (about 1 h):** K1 and K3: ask me (5 minutes each). K2: Maarten Grootendorst, "A Visual Guide to Quantization" (blog); read only the int8 / scale and zero-point part.

---

## L. Drone physics (light math)

| ID | Concept | Where you use it |
|---|---|---|
| L1 ★ | Angular rate and angular acceleration; torque = inertia × angular acceleration, so the plant is b/s | P(s) in 7.1 |
| L2 ★ | RPM → Hz (÷ 60); motor harmonics; blade-pass frequency | notches, noise model, E13 |
| L3 ○ | Motor and propeller lag as a first-order system | τ_m |

**Learn fast:** your Betaflight experience covers most of this. Ask me to connect it to the equations.

---

## Order to learn (matched to the THESIS.md phases)

Day 13 = 2026-10-10. The tracker in THESIS.md shows Phase 2 not ticked yet, so groups D-F are already due.

1. **A + B** (about ½ day): everything else uses complex numbers, dB and difference equations.
2. **C + H4** (1 day): pilot study and PSDs (Phase 3, now).
3. **D, E, F** (2 days): `filters.py` and `loop_model.py` (Phase 2 backlog).
4. **G4-G6, J13-J15** (½ day): clustering (end of Phase 3).
5. **I, G2, H5, H6** (½ day): optimizer and system identification (Phase 4).
6. **J** (1-1.5 days): ML tuner (Phase 5).
7. **K** (1-2 h): deployment (Phase 6).

Total: about 30 hours of videos and interactive pages (rough estimate).

**Rule that saves the most time:** after each video, reproduce the idea in about 10 lines of Python, for example plot a notch with `scipy.signal.freqz` or a PSD with `scipy.signal.welch`. Doing this teaches you faster than watching more videos.
