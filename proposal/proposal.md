# Thesis proposal

## Physics-Guided Machine Learning for Instant, Stability-Safe Gyro Filter Tuning in Multirotor Flight Controllers

**Student:** [Your name] · **Proposed supervisor and co-author:** [Professor's name] · **Date:** 14 October 2026

**Problem.** Drone flight controllers remove motor vibration from the gyroscope signal with a stack of digital filters: lowpass filters, static notches, and notch banks that follow each motor's speed. Every filter adds delay, and delay reduces the stability margin of the attitude loop. Too little filtering sends noise to the motors, which heats them and wastes battery. Today the filters are tuned by trial and error: the official PX4 guide asks for one test flight per setting, judged by eye on a spectrum plot, with no stability check.

**Gap.** Earlier work co-designs a PID controller with a single measurement filter for process plants (Soltesz et al. 2017; Garpinger and Hägglund 2015), or tunes PID *gains* of drones with many experiments per vehicle (Bayesian optimization, Berkenkamp et al. 2016; reinforcement learning, Sönmez et al. 2025). A commercial neural-network tuner (FPVTune) learns from human tunes and has no stability guarantee. We found no work that models and optimizes the *complete* multirotor filter stack inside the control loop on *real* flight noise and turns that optimization into a small, verifiable ML model.

**Research questions.** (RQ1) How much stability margin do default and pilot-chosen filter settings waste on real drone noise? (RQ2) Can a small ML model predict near-optimal settings instantly from a noise spectrum, and does it still work on real noise if it was trained on synthetic noise? (RQ3) Is it small and fast enough for a flight-controller microcontroller?

**Method.**
1. **Physics model.** An exact frequency-domain model of the PX4 filter stack in the attitude loop gives phase and gain margin, peak sensitivity $M_s$, filter delay and motor-command noise.
2. **Optimizer.** Maximize phase margin without more noise, or minimize noise without less margin (vectorized grid plus local refinement).
3. **Real data.** Public PX4 flight logs (CC-BY) with raw high-rate gyro (about 7.7 % of real multirotor logs), 16 data-quality gates, split by vehicle into fit and test halves. The logs also supply real filter settings, PID gains and ESC RPM.
4. **ML (amortized optimization).** k-NN, ridge, gradient boosting and a small MLP learn the optimizer's answers on synthetic drones with realistic noise; they are tested on the held-out real logs.
5. **Safety check.** Every prediction passes a Nyquist test and robust margin checks over a set of plant models, or is repaired by bisection toward a known-safe setting; violation rates get a statistical bound.
6. **Deployment and validation.** Export to C (size, time per inference); time-domain simulation.

**Preliminary results** (feasibility test, 10–14 Oct 2026, public PX4 logs). These are early, single-log results:
- On a real PX4 log, the model predicted the firmware's own filtered gyro, D-term input and controller output within 0.9, 0.8 and 1.8 dB (wrong model variants: 12–36 dB off). The plant was identified from the same log.
- For six synthetic drones with real noise, the optimizer gained a median 3.5° phase margin at equal noise (range 2.0–4.8°), or cut motor noise to a median of 0.58 of the default at equal margin. On the real drone it gained 2.5° over the pilot's own setting.
- A time-domain simulation agreed with the stability model in 158 of 160 random cases, with no disagreements (the other 2 were undecided at the stability limit).

**Plan (Days 17–47, to 13 November 2026).**

| Dates | Work |
|---|---|
| 14–15 Oct | Literature (60 references collected), proposal |
| 16–20 Oct | Real-noise library: log screening, vehicle split, pilot study, data-quality gates |
| 21–23 Oct | RQ1 results; training data (10,000–30,000 optimizer runs) |
| 24–29 Oct | ML models, baselines, safety check, explainability |
| 30 Oct–1 Nov | C export, time-domain and real-log validation |
| 2–10 Nov | Paper writing; your review and revision |
| 11–13 Nov | Open-source release (GitHub, Zenodo DOI), arXiv preprint, journal submission (e.g. MDPI *Drones* or IEEE *Access*) |

**Resources.** Laptop CPU only, free software, and public data. No flight tests are needed.

**Request.** I would be grateful if you would supervise this thesis, co-author the paper, and endorse the arXiv submission if needed. The full plan (THESIS.md) is available on request.
