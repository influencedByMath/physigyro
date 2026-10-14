"""Independent checks of every pre-ML pipeline stage (audit, Day 17, 2026-10-14).

run_feasibility.py shows that the pipeline runs and what it finds. This script
asks whether each stage is RIGHT, with tests the pipeline itself does not use:

  1 data      axes/rotation and FIFO time base against the firmware's own output;
              controller rate; PSD repeatability between two windows of the same flight
  3 plant     repeatability of the Tier 2 fit on sub-windows (first data for the
              uncertainty set of the safety check)
  4 model     time-domain simulation of the multi-rate loop (filters at the FIFO
              rate, PID at the control rate, ZOH, delay) against the Nyquist verdict
              of loop_model.margins, on random settings including unstable ones
    noise     part of the motor-command noise above f_max = 800 Hz that the metric ignores
    optimizer does the Mode A optimum stay "not noisier than the pilot" on another
              window of the same flight? Is it as robust over the 8 plant corners?
              Step response of default / pilot / optimum in the time domain

Run after run_feasibility.py (reads results.json):
    .venv/Scripts/python experiments/feasibility/pipeline_checks.py
Writes checks.json next to this file.
"""
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import noise_data as nd                     # noqa: E402
import optimizer as opt                     # noqa: E402
import sysid                                # noqa: E402
from filters import (esc_notch_frequencies, lowpass2_coeffs, notch_coeffs, pt1_alpha)   # noqa: E402
from loop_model import FilterSettings, Gains, LoopCase, Plant, Rates   # noqa: E402

R = json.load(open(os.path.join(HERE, "results.json")))
LOGS = {"b27d09e3": "b27d09e3-4542-4496-a18b-2c3c5317024e.ulg",
        "a2571040": "a2571040-29ef-4b1a-b7cf-7776f1a78a1f.ulg"}
OUT = {}
RNG = np.random.default_rng(20261014)


def log(*a):
    print(*a, flush=True)


def settings(d):
    d = dict(d)
    d["notches"] = tuple(tuple(n) for n in d["notches"])
    return FilterSettings(**d)


def option_settings(base: FilterSettings, mrow):
    """FilterSettings of an optimizer result row (Scenario B labels)."""
    lab = mrow["option"]
    if "off" in lab:
        return replace(base, f_gyro=mrow["f_gyro"], f_dterm=mrow["f_dterm"], dnf_esc=False)
    bw, h = float(lab.split("bw ")[1].split(" Hz")[0]), int(lab.split("x")[-1])
    return replace(base, f_gyro=mrow["f_gyro"], f_dterm=mrow["f_dterm"], dnf_esc=True, dnf_bw=bw, dnf_hmc=h)


# ------------------------------------------------------------------ stage 1
def check_data():
    log("\n=== 1. data: axes, time base, controller rate, PSD repeatability")
    res, ctx = {}, {}
    for key, fn in LOGS.items():
        u = nd.load(os.path.join(ROOT, "data", "raw_logs", fn))
        gs = nd.gyro_stream(u)
        r = {}
        # (a) axes and time base: FIFO body rate (smoothed) vs the firmware's filtered rate
        try:
            v = u.get_dataset("vehicle_angular_velocity").data
        except Exception:
            v = None
        if v is None:
            r["axis_correlation_xyz"] = "vehicle_angular_velocity not logged"
            res[key], ctx[key] = r, dict(ulog=u, gs=gs, w0=None, w1=None)
            continue
        tv = nd._t(u, v["timestamp_sample"])
        a0, b0 = max(gs.segments, key=lambda s: s[1] - s[0])
        seg_t = gs.t[a0:b0]
        m = (tv > seg_t[0] + 1) & (tv < seg_t[-1] - 1)
        k = max(1, int(round(gs.fs / 40)))                     # 40 Hz moving average: flight motion only
        corr, lag_ms = [], []
        for ax in range(3):
            sm = np.convolve(gs.gyro[a0:b0, ax], np.ones(k) / k, mode="same")
            logged = v[f"xyz[{ax}]"][m].astype(float)
            mine = np.interp(tv[m], seg_t, sm)
            corr.append(round(float(np.corrcoef(mine, logged)[0, 1]), 4))
            # time offset of best agreement (FIFO time base vs firmware timestamp_sample)
            shifts = np.arange(-10, 11) * 0.5e-3
            errs = [np.mean((np.interp(tv[m] + s, seg_t, sm) - logged) ** 2) for s in shifts]
            lag_ms.append(float(shifts[int(np.argmin(errs))] * 1e3))
        r["axis_correlation_xyz"] = corr
        r["best_time_shift_ms_xyz"] = lag_ms
        r["rotation_used"] = gs.rotation
        # (b) controller rate: publication rates vs IMU_GYRO_RATEMAX
        rates = {}
        for topic in ("vehicle_angular_velocity", "vehicle_torque_setpoint"):
            try:
                d = u.get_dataset(topic).data
                tt = nd._t(u, d["timestamp"])
                tt = tt[(tt > 0)]
                rates[topic] = round(float(1.0 / np.median(np.diff(tt))), 1)
            except Exception:
                rates[topic] = None
        r["logged_rates_hz"] = rates
        r["IMU_GYRO_RATEMAX"] = float(u.initial_parameters.get("IMU_GYRO_RATEMAX", 400))
        # (c) PSD repeatability: quietest window vs the next quiet window that does not overlap
        wins = nd.steady_windows(u)
        w0 = wins[0]
        w1 = next((w for w in wins[1:] if w[0] >= w0[1] or w[1] <= w0[0]), None)
        f, P0, n0, _ = nd.window_psd(gs, w0[0], w0[1])
        r["window_main"] = [round(w0[0], 2), round(w0[1], 2)]
        if w1 is not None:
            _, P1, n1, _ = nd.window_psd(gs, w1[0], w1[1])
            bands = [(20, 60), (60, 90), (90, 130), (130, 180), (180, 240), (240, 320), (320, 400), (400, 800)]
            err = []
            for lo, hi in bands:
                mm = (f >= lo) & (f < hi)
                err.append(round(float(10 * np.log10(np.trapezoid(P1[mm, 0], f[mm]) / np.trapezoid(P0[mm, 0], f[mm]))), 2))
            r["window_second"] = [round(w1[0], 2), round(w1[1], 2)]
            r["second_vs_main_roll_band_db"] = dict(zip([f"{lo}-{hi}" for lo, hi in bands], err))
        res[key] = r
        ctx[key] = dict(ulog=u, gs=gs, w0=w0, w1=w1)
        log(f"   {key}: {json.dumps(r)}")
    OUT["data"] = res
    return ctx


# ------------------------------------------------------------------ stage 3
def check_plant(ctx):
    log("\n=== 3. plant: repeatability on sub-windows (first data for the uncertainty set)")
    u, gs = ctx["b27d09e3"]["ulog"], ctx["b27d09e3"]["gs"]
    res = {}
    for name, axis, (t0, t1), parts in [("roll", 0, (24.0, 53.0), 2), ("pitch", 1, (8.5, 22.9), 2)]:
        edges = np.linspace(t0, t1, parts + 1)
        fits = []
        for a, b in zip(edges[:-1], edges[1:]):
            try:
                r = sysid.identify(sysid.prepare(u, gs, a, b, axis=axis), nperseg=512)
                fits.append(dict(window=[round(a, 1), round(b, 1)], b=round(r["b"], 1), tau_m=round(r["tau_m"], 4),
                                 Td=round(r["Td"], 5), n_freq=r["n_freq"]))
            except ValueError as e:
                fits.append(dict(window=[round(a, 1), round(b, 1)], error=str(e)))
        ok = [x for x in fits if "b" in x]
        full = R["plant"][name]
        spread = None
        if len(ok) >= 2:
            bs, ts = np.array([x["b"] for x in ok]), np.array([x["tau_m"] for x in ok])
            spread = dict(b_rel_range=round(float((bs.max() - bs.min()) / full["b"]), 3),
                          tau_rel_range=round(float((ts.max() - ts.min()) / full["tau_m"]), 3),
                          Td_range_ms=round(float(np.ptp([x["Td"] for x in ok]) * 1e3), 2))
        res[name] = dict(full=dict(b=full["b"], tau_m=full["tau_m"], Td=full["Td"]),
                         att_sp_instrument=full.get("att_sp_instrument"), sub_windows=fits, spread=spread)
        log(f"   {name}: {json.dumps(res[name])}")
    OUT["plant"] = res


# ------------------------------------------------------------------ stage 4: time-domain loop
class _BQ:
    """Vectorized direct-form-I biquads, one per case (coefficient arrays of shape (K,))."""

    def __init__(self, b, a):
        self.b0, self.b1, self.b2 = (np.asarray(x, float) for x in b)
        self.a1, self.a2 = (np.asarray(x, float) for x in a)
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def __call__(self, x):
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1, self.y2, self.y1 = self.x1, x, self.y1, y
        return y


def simulate(cases, fs, f_ctrl, T=4.0, kick=0.2, step=None):
    """Linear multi-rate simulation of PX4 rate loops, vectorized over cases.

    Every case: dict(plant, gains, s (FilterSettings), motor_hz). Gyro filters
    and the derivative run at fs; P, I, D and the output (ZOH) at f_ctrl; plant
    b/s/(tau s+1) integrated exactly over each fs step; delay Td rounded to 1/fs.
    Initial condition: body rate = kick (or a setpoint step of size ``step``).
    Returns the body rate (n_steps, K)."""
    K = len(cases)
    dt = 1.0 / fs
    dec = int(round(fs / f_ctrl))
    b = np.array([c["plant"].b for c in cases])
    tau = np.array([c["plant"].tau_m for c in cases])
    dly = np.array([int(round(c["plant"].Td * fs)) for c in cases])
    Kp, Ki, Kd = (np.array([getattr(c["gains"], k) for c in cases]) for k in ("Kp", "Ki", "Kd"))
    e1 = np.exp(-dt / tau)
    # notch stack: pad every case to the same number of biquads (identity = pass-through)
    stacks = []
    for c in cases:
        s, cs = c["s"], []
        cs += [notch_coeffs(f0, bw, fs) for f0, bw in s.notches]
        if s.dnf_esc:
            cs += [notch_coeffs(f0, s.dnf_bw, fs) for f0 in esc_notch_frequencies(c["motor_hz"], s.dnf_hmc,
                                                                                  s.dnf_bw, s.dnf_min)]
        stacks.append(cs)
    n_nf = max(len(x) for x in stacks)
    ident = ((1.0, 0.0, 0.0), (0.0, 0.0))
    notches = []
    for j in range(n_nf):
        co = [x[j] if j < len(x) else ident for x in stacks]
        notches.append(_BQ([[float(c[0][i]) for c in co] for i in range(3)], [[float(c[1][i]) for c in co] for i in range(2)]))
    lp = lowpass2_coeffs(np.array([c["s"].f_gyro for c in cases]), fs)
    lpf = _BQ(lp[0], lp[1])
    alpha = pt1_alpha(np.array([c["s"].f_dterm for c in cases]), fs)
    n = int(T * fs)
    buf = np.zeros((int(dly.max()) + 2, K))           # delay line of the motor command
    w = np.full(K, kick if step is None else 0.0)
    m = np.zeros(K)
    u_hold = np.zeros(K)
    integ = np.zeros(K)
    prev = None
    dstate = np.zeros(K)
    r = 0.0 if step is None else step
    out = np.empty((n, K))
    cols = np.arange(K)
    np.seterr(over="ignore", invalid="ignore")
    for i in range(n):
        x = w.copy()
        for nf in notches:
            x = nf(x)
        xf = lpf(x)
        if prev is None:
            prev = xf.copy()
        acc = (xf - prev) * fs
        prev = xf
        dstate = dstate + alpha * (acc - dstate)
        if i % dec == 0:                               # rate controller (PX4: I updated after use)
            e = r - xf
            u_hold = Kp * e + integ - Kd * dstate
            integ = integ + Ki * e / f_ctrl
        buf = np.roll(buf, 1, axis=0)
        buf[0] = u_hold
        ud = buf[dly, cols]
        # exact integration of m' = (ud - m)/tau, w' = b m over one step with ud constant
        m_new = ud + (m - ud) * e1
        w = w + b * (ud * dt + (m - ud) * tau * (1 - e1))
        m = m_new
        out[i] = w
    return out


def check_model_time_domain(ctx):
    log("\n=== 4a. loop model: Nyquist verdict vs time-domain multi-rate simulation")
    info = R["data"]["b27d09e3"]
    fs, f_ctrl = info["fs_nominal"], info["f_ctrl"]
    rates = Rates(fs=fs, f_ctrl=f_ctrl)
    p = R["plant"]["roll"]
    plant0 = Plant(p["b"], p["tau_m"], p["Td"])
    g0 = Gains(**info["gains_roll"])
    motor = np.array(info["rpm_hz"])
    f_dummy = np.linspace(0, fs / 2, 2049)
    psd_dummy = np.ones_like(f_dummy)
    cases, verdict = [], []
    for _ in range(160):
        kg = float(np.exp(RNG.uniform(np.log(0.5), np.log(4.0))))          # gain scale: many unstable cases
        plant = Plant(p["b"] * float(np.exp(RNG.uniform(-0.4, 0.4))), p["tau_m"] * float(np.exp(RNG.uniform(-0.4, 0.4))),
                      float(RNG.uniform(0.0, 0.006)))
        g = Gains(g0.Kp * kg, g0.Ki * kg, g0.Kd * kg * float(np.exp(RNG.uniform(-0.7, 0.7))))
        s = FilterSettings(f_gyro=float(np.exp(RNG.uniform(np.log(15), np.log(300)))),
                           f_dterm=float(np.exp(RNG.uniform(np.log(10), np.log(200)))),
                           notches=(((float(RNG.uniform(60, 250)), float(RNG.uniform(5, 100))),) if RNG.random() < 0.4 else ()),
                           dnf_esc=bool(RNG.random() < 0.5), dnf_bw=float(RNG.choice([5, 10, 20, 30])),
                           dnf_hmc=int(RNG.integers(1, 4)), dnf_min=25.0)
        try:
            case = LoopCase(plant, g, rates, f_dummy, psd_dummy, motor_hz=motor, f_hi=800.0, reference=s, f_lo=50.0)
        except ValueError:
            continue
        m = case.ref_metrics
        cases.append(dict(plant=plant, gains=g, s=s, motor_hz=motor))
        verdict.append((bool(m["stable"]), float(m["GM"]), float(m["PM"]), float(m["Ms"])))
    t = time.perf_counter()
    y = simulate(cases, fs, f_ctrl, T=4.0)
    t_sim = time.perf_counter() - t
    n = y.shape[0]
    early = np.sqrt(np.mean(y[int(0.25 * n / 4):int(1.0 * n / 4)] ** 2, axis=0))
    late = np.sqrt(np.mean(y[int(3.5 * n / 4):] ** 2, axis=0))
    with np.errstate(all="ignore"):
        growth = late / np.maximum(early, 1e-300)
    growth = np.where(np.isfinite(late) & np.isfinite(early), growth, np.inf)    # overflow = clearly unstable
    sim_stable = np.where(growth < 0.2, True, np.where(growth > 1.0, False, None))
    agree = disagree = unclear = 0
    bad = []
    for i, (st, gm, pm, ms) in enumerate(verdict):
        if sim_stable[i] is None:
            unclear += 1
        elif bool(sim_stable[i]) == st:
            agree += 1
        else:
            disagree += 1
            bad.append(dict(model_stable=st, GM=round(gm, 2), PM=round(pm, 2), Ms=round(ms, 2), growth=float(growth[i])))
    near = [abs(v[3]) for v in verdict]
    res = dict(n_cases=len(cases), model_stable=int(sum(v[0] for v in verdict)), agree=agree, disagree=disagree,
               unclear_in_4s=unclear, disagreements=bad[:10], sim_seconds=round(t_sim, 1))
    # unclear cases should be the lightly damped ones (Ms large)
    if unclear:
        res["Ms_of_unclear_cases"] = sorted(round(verdict[i][3], 1) for i in range(len(verdict)) if sim_stable[i] is None)
    OUT["time_domain_vs_nyquist"] = res
    log(f"   {json.dumps(res)}")


def real_case(psd_f, psd, ref_S_eff, plant=None, **kw):
    info = R["data"]["b27d09e3"]
    rates = Rates(fs=info["fs_nominal"], f_ctrl=info["f_ctrl"])
    p = R["plant"]["roll"]
    plant = plant or Plant(p["b"], p["tau_m"], p["Td"])
    return LoopCase(plant, Gains(**info["gains_roll"]), rates, psd_f, psd, motor_hz=np.array(info["rpm_hz"]),
                    reference=settings(info["actual"]), reference_S_eff=ref_S_eff, **kw)


def check_optimizer(ctx):
    log("\n=== 4b. noise above 800 Hz, other window, plant corners, step response")
    c = ctx["b27d09e3"]
    u, gs, (t0, t1, _) = c["ulog"], c["gs"], c["w0"]
    info = R["data"]["b27d09e3"]
    act = settings(info["actual"])
    dflt = settings(info["default"])
    rg = R["optimizer"]["real_gain"]
    sA, sB = option_settings(act, rg["mode_A"]), option_settings(act, rg["mode_B"])
    named = {"PX4 default": dflt, "pilot": act, "mode A": sA, "mode B": sB}

    def tracked(t0_, t1_, s):
        if not s.dnf_esc:
            return None
        f, P, _ = nd.tracking_notch_psd(u, gs, t0_, t1_, s.dnf_bw, s.dnf_hmc, s.dnf_min, axes=(0,))
        return f, P[:, 0]

    f, P, _, _ = nd.window_psd(gs, t0, t1)
    res = {}
    # (a) noise above f_max: integrate to 0.499 fs instead of 800 Hz
    for f_hi in (800.0, 0.499 * gs.fs):
        case = real_case(f, P[:, 0], tracked(t0, t1, act), f_hi=f_hi)
        row = {}
        for k, s in named.items():
            tr = tracked(t0, t1, s)
            row[k] = round(float(case.evaluate(s, S_eff=None if tr is None else case.psd_on_noise_grid(*tr))["N_rel"]), 3)
        res[f"N_rel_f_hi_{f_hi:.0f}"] = row
    # (b) same settings on a second window of the same flight
    w1 = c["w1"]
    if w1 is not None:
        f1, P1, _, _ = nd.window_psd(gs, w1[0], w1[1])
        case = real_case(f1, P1[:, 0], tracked(w1[0], w1[1], act))
        row = {}
        for k, s in named.items():
            tr = tracked(w1[0], w1[1], s)
            m = case.evaluate(s, S_eff=None if tr is None else case.psd_on_noise_grid(*tr))
            row[k] = dict(N_rel=round(float(m["N_rel"]), 3), PM=round(float(m["PM"]), 2))
        res["second_window"] = dict(window=[round(w1[0], 2), round(w1[1], 2)], metrics=row)
    # (c) robustness over the 8 plant corners (+-20 % b, tau_m; +1 control sample)
    p = R["plant"]["roll"]
    worst = {}
    for k, s in named.items():
        w = dict(PM=1e9, GM=1e9, Ms=0.0)
        for db in (0.8, 1.2):
            for dtau in (0.8, 1.2):
                for dd in (0.0, 1.0 / info["f_ctrl"]):
                    pl = Plant(p["b"] * db, p["tau_m"] * dtau, p["Td"] + dd)
                    case = real_case(f, P[:, 0], None, plant=pl, n_stab=800)
                    m = case.evaluate(s, _noise=False)
                    w = dict(PM=min(w["PM"], m["PM"]), GM=min(w["GM"], m["GM"]), Ms=max(w["Ms"], m["Ms"]))
        worst[k] = {kk: round(float(v), 2) for kk, v in w.items()}
    res["worst_of_8_corners"] = worst
    # (d) step response (setpoint step 1 rad/s), nominal plant, time domain
    pl = Plant(p["b"], p["tau_m"], p["Td"])
    g = Gains(**info["gains_roll"])
    cases = [dict(plant=pl, gains=g, s=s, motor_hz=np.array(info["rpm_hz"])) for s in named.values()]
    y = simulate(cases, info["fs_nominal"], info["f_ctrl"], T=1.0, kick=0.0, step=1.0)
    tt = np.arange(y.shape[0]) / info["fs_nominal"]
    step = {}
    for j, k in enumerate(named):
        yy = y[:, j]
        t_rise = float(tt[np.argmax(yy >= 0.9)]) if (yy >= 0.9).any() else None
        step[k] = dict(overshoot_pct=round(float((yy.max() - 1) * 100), 1),
                       rise_ms=None if t_rise is None else round(t_rise * 1e3, 1),
                       ringing_rms_after_150ms=round(float(np.std(yy[tt > 0.15])), 4))
    res["step_response"] = step
    OUT["optimizer"] = res
    log(f"   {json.dumps(res)}")


def main():
    t = time.perf_counter()
    ctx = check_data()
    check_plant(ctx)
    check_model_time_domain(ctx)
    check_optimizer(ctx)
    OUT["meta"] = dict(date=time.strftime("%Y-%m-%d"), seconds=round(time.perf_counter() - t, 1),
                       results_json_date=R["meta"]["date"])
    with open(os.path.join(HERE, "checks.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"\nDone in {OUT['meta']['seconds']} s")


if __name__ == "__main__":
    main()
