"""Single-axis rate-control loop of a PX4 multicopter (THESIS.md Sections 7.1-7.2).

Signal path, copied from PX4 v1.16 (VehicleAngularVelocity.cpp, rate_control.cpp):

    raw gyro --notches--> --gyro LPF2 (IMU_GYRO_CUTOFF)--> filtered rate  -> P, I
                                                     +--> d/dt --PT1 (IMU_DGYRO_CUTOFF)--> D
    All gyro filters and the derivative run at the FIFO rate fs.
    P, I, D are combined at the controller rate f_ctrl (IMU_GYRO_RATEMAX):
        u = Kp e + I - Kd * d(omega_f)/dt,   I_k+1 = I_k + Ki e Tc  (I uses the old sum)

Open loop L(f) = C_fb(f) * ZOH(f) * P(f) with plant P(s) = b/s * 1/(tau_m s + 1) * exp(-s Td).

Modelling assumptions (state them in the paper):
- linear, one axis, small signals (no saturation, no I-term attenuation);
- the decimation from fs to f_ctrl is ignored in L (|L| is tiny above f_ctrl/2);
- the raw gyro spectrum is treated as sensor noise entering the loop.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from filters import (derivative_response, esc_notch_frequencies, lowpass2_response,
                     notch_bank_response, notch_response, pt1_response)


# ----------------------------------------------------------------- data types

@dataclass(frozen=True)
class Plant:
    b: float = 1.0        # control effectiveness [rad/s^2 per unit torque command]
    tau_m: float = 0.05   # motor + propeller time constant [s]
    Td: float = 0.002     # extra delay: ESC protocol, output, computation [s]


@dataclass(frozen=True)
class Gains:
    Kp: float
    Ki: float
    Kd: float

    @classmethod
    def from_px4(cls, P, I, D, K=1.0):
        """PX4 multiplies P, I and D by MC_*RATE_K (MulticopterRateControl.cpp)."""
        return cls(K * P, K * I, K * D)


@dataclass(frozen=True)
class Rates:
    fs: float = 6400.0     # gyro FIFO rate = rate of all gyro filters [Hz]
    f_ctrl: float = 800.0  # IMU_GYRO_RATEMAX = rate-controller rate [Hz]


@dataclass(frozen=True)
class FilterSettings:
    f_gyro: float = 40.0          # IMU_GYRO_CUTOFF [Hz]
    f_dterm: float = 30.0         # IMU_DGYRO_CUTOFF [Hz]
    notches: tuple = ()           # static notches ((f0, bw), ...): IMU_GYRO_NF0/NF1
    dnf_esc: bool = False         # IMU_GYRO_DNF_EN bit 0 (ESC RPM notch bank)
    dnf_bw: float = 15.0          # IMU_GYRO_DNF_BW [Hz]
    dnf_hmc: int = 3              # IMU_GYRO_DNF_HMC
    dnf_min: float = 25.0         # IMU_GYRO_DNF_MIN [Hz]


# PX4 v1.16 multicopter defaults (decision D-09). ESC notch bank is off by default.
PX4_V116_MC_DEFAULT = FilterSettings(f_gyro=40.0, f_dterm=30.0)
F_LO_MIN = 20.0          # lower edge of the frozen noise band (D-23 Q5); f_lo = max(5 fc, F_LO_MIN) (D-36 (5))


# ------------------------------------------------------------ building blocks

def notch_stack(f, s: FilterSettings, fs, motor_hz=None):
    """All notches of the gyro path (static + ESC-RPM bank)."""
    H = np.ones(np.shape(f), dtype=complex)
    for f0, bw in s.notches:
        H = H * notch_response(f, f0, bw, fs)
    if s.dnf_esc and motor_hz is not None:
        H = H * notch_bank_response(f, esc_notch_frequencies(motor_hz, s.dnf_hmc, s.dnf_bw, s.dnf_min),
                                    s.dnf_bw, fs)
    return H


def gyro_stack(f, s: FilterSettings, fs, motor_hz=None):
    """F_g: notches then the 2nd-order gyro lowpass."""
    return notch_stack(f, s, fs, motor_hz) * lowpass2_response(f, s.f_gyro, fs)


def integrator_response(f, f_ctrl):
    """I(z) = Tc z^-1 / (1 - z^-1): PX4 adds Ki e dt to the sum after using it."""
    zi = np.exp(-2j * np.pi * np.asarray(f) / f_ctrl)
    return zi / (1.0 - zi) / f_ctrl


def zoh_response(f, f_ctrl):
    """Sampler + zero-order hold at the controller rate: (1 - e^-sT) / (sT)."""
    s = 2j * np.pi * np.asarray(f)
    return (1.0 - np.exp(-s / f_ctrl)) / (s / f_ctrl)


def plant_response(f, p: Plant):
    s = 2j * np.pi * np.asarray(f)
    return p.b / s / (p.tau_m * s + 1.0) * np.exp(-s * p.Td)


def n_integrators(g: Gains):
    """Free integrators in L: one in the plant, one more if the I gain is on."""
    return 2 if g.Ki > 0 else 1


# -------------------------------------------------------------------- metrics

def margins(f, L, n_int=2):
    """Stability metrics of open loop(s) L (..., F) on an increasing grid f (F,).

    PM and GM are checked at EVERY crossover and the smallest one is returned.
    Closed-loop stability uses the Nyquist argument principle (THESIS 7.8 step 2):
    with n_int free integrators and no other unstable open-loop pole, the closed
    loop is stable iff arg(1 + L) rises by n_int * 90 deg from f -> 0 to the
    end of the grid (where |L| must be small). Each net -180 deg means one
    closed-loop pole in the right half plane.
    """
    f = np.asarray(f, dtype=float)
    L = np.asarray(L)
    logf = np.log(f)
    lm = np.log(np.abs(L))
    ph = np.unwrap(np.angle(L), axis=-1)

    # gain crossovers |L| = 1
    a, b = lm[..., :-1], lm[..., 1:]
    cross = (a > 0) != (b > 0)
    t = np.where(cross, a / np.where(cross, a - b, 1.0), 0.0)
    ph_c = ph[..., :-1] + t * (ph[..., 1:] - ph[..., :-1])
    pm_all = np.where(cross, np.degrees(np.angle(np.exp(1j * (ph_c + np.pi)))), np.inf)
    k = np.argmin(pm_all, axis=-1)
    PM = np.take_along_axis(pm_all, k[..., None], -1)[..., 0]
    tk = np.take_along_axis(t, k[..., None], -1)[..., 0]
    fc = np.exp(logf[k] + tk * (logf[k + 1] - logf[k]))
    fc = np.where(np.isfinite(PM), fc, np.nan)

    # phase crossovers (phase = -180 deg + n 360 deg) with |L| < 1 -> upper gain margin
    u = np.floor((ph + np.pi) / (2 * np.pi))
    pc = u[..., :-1] != u[..., 1:]
    v = 2 * np.pi * np.maximum(u[..., :-1], u[..., 1:]) - np.pi
    dph = ph[..., 1:] - ph[..., :-1]
    tp = np.where(pc, (v - ph[..., :-1]) / np.where(pc, dph, 1.0), 0.0)
    lm_c = a + tp * (b - a)
    gm_all = np.where(pc & (lm_c < 0), -20.0 / np.log(10.0) * lm_c, np.inf)
    GM = gm_all.min(axis=-1)

    # peak sensitivity
    Ms = (1.0 / np.abs(1.0 + L)).max(axis=-1)

    # Nyquist winding test
    r = np.unwrap(np.angle(1.0 + L), axis=-1)
    start_target = -n_int * np.pi / 2
    shift = 2 * np.pi * np.round((start_target - r[..., 0]) / (2 * np.pi))
    end = r[..., -1] + shift
    n_rhp = np.round(-end / np.pi).astype(int)
    tail_ok = np.abs(L[..., -1]) < 0.5

    DM = np.radians(PM) / (2 * np.pi * fc)
    return dict(PM=PM, GM=GM, Ms=Ms, fc=fc, DM=DM, n_rhp=n_rhp,
                stable=(n_rhp == 0) & tail_ok & np.isfinite(PM))


def phase_lag_at(f_grid, H, f_eval):
    """Phase lag of H (..., F) at f_eval (...,) in degrees and as delay in seconds."""
    lag = -np.unwrap(np.angle(H), axis=-1)
    f_eval = np.asarray(f_eval, dtype=float)
    if lag.ndim == 1:
        deg = np.degrees(np.interp(f_eval, f_grid, lag))
    else:
        flat = lag.reshape(-1, lag.shape[-1])
        fe = np.broadcast_to(f_eval, lag.shape[:-1]).ravel()
        deg = np.degrees(np.array([np.interp(x, f_grid, row) for x, row in zip(fe, flat)])).reshape(lag.shape[:-1])
    return deg, np.radians(deg) / (2 * np.pi * f_eval)


# ------------------------------------------------------------- PID design (Tier 3a)

def design_gains(f_cross, r, plant: Plant, rates: Rates, i_ratio=0.1):
    """Crossover design of THESIS 7.1.1 Tier 3a.

    C(s) = Kp (1 + wi/s + s/wd), wi = i_ratio 2 pi f_cross, wd = r 2 pi f_cross.
    Kp is chosen so that |L(f_cross)| = 1 WITHOUT filters (|L| is linear in Kp).
    Returns the gains and the unfiltered phase margin."""
    wi, wd = i_ratio * 2 * np.pi * f_cross, r * 2 * np.pi * f_cross
    f = _stability_grid(rates)
    PZ = zoh_response(f, rates.f_ctrl) * plant_response(f, plant)
    C1 = 1.0 + wi * integrator_response(f, rates.f_ctrl) + derivative_response(f, rates.fs) / wd
    L1c = np.interp(np.log(f_cross), np.log(f), np.abs(C1 * PZ))
    Kp = 1.0 / L1c
    g = Gains(Kp, Kp * wi, Kp / wd)
    m = margins(f, Kp * C1 * PZ, n_integrators(g))
    return g, m


def _stability_grid(rates: Rates, n=1500, f_min=0.01):
    return np.geomspace(f_min, 0.499 * rates.f_ctrl, n)


# ------------------------------------------------------------------ evaluator

class LoopCase:
    """One drone (plant, gains, rates) + one noise spectrum on fixed frequency grids.

    The noise metric (THESIS 7.2) is the motor-command noise power
        N^2 = integral over [f_lo, f_hi] of |C_fb / (1 + L)|^2 S_n df,
    reported relative to a reference setting on the same drone and spectrum.
    """

    def __init__(self, plant: Plant, gains: Gains, rates: Rates, psd_f, psd, motor_hz=None,
                 f_hi=800.0, f_lo=None, reference: FilterSettings = PX4_V116_MC_DEFAULT,
                 noise_df=0.5, n_stab=1500, reference_S_eff=None):
        self.plant, self.gains, self.rates = plant, gains, rates
        self.motor_hz = None if motor_hz is None else np.asarray(motor_hz, dtype=float)
        self.n_int = n_integrators(gains)
        self.reference = reference
        if f_hi > 0.5 * rates.fs:
            raise ValueError("f_hi above the Nyquist frequency of the gyro data")
        self.f_hi = f_hi
        self.f_stab = _stability_grid(rates, n_stab)
        self._psd_f, self._psd = np.asarray(psd_f), np.asarray(psd)
        # reference crossover sets the lower noise limit f_lo = max(5 fc, 20 Hz) (D-36 (5); fixed per drone)
        self.ref_metrics = self.evaluate(reference, _noise=False)
        if f_lo is None and not np.isfinite(self.ref_metrics["fc"]):
            raise ValueError("reference setting has no gain crossover: f_lo = 5 fc is undefined")
        self.f_lo = max(5.0 * self.ref_metrics["fc"], F_LO_MIN) if f_lo is None else f_lo
        self.f_noise = np.arange(self.f_lo, f_hi + 1e-9, noise_df)
        lp = np.interp(self.f_noise, self._psd_f, np.log(np.maximum(self._psd, 1e-30)))
        self.S = np.exp(lp)
        self.N_ref = None
        # reference_S_eff (f, psd): tracking-notch spectrum when the reference uses the ESC-RPM bank
        ref_S = None if reference_S_eff is None else self.psd_on_noise_grid(*reference_S_eff)
        self.ref_metrics = self.evaluate(reference, S_eff=ref_S)
        self.N_ref = self.ref_metrics["N"]
        self.ref_metrics["N_rel"] = 1.0

    # parts that do not depend on the filter settings
    def _parts(self, f):
        r = self.rates
        A = self.gains.Kp + self.gains.Ki * integrator_response(f, r.f_ctrl)
        Dd = self.gains.Kd * derivative_response(f, r.fs)
        PZ = zoh_response(f, r.f_ctrl) * plant_response(f, self.plant)
        return A, Dd, PZ

    def psd_on_noise_grid(self, f, S):
        """Interpolate a PSD (log-linear) onto the noise grid."""
        return np.exp(np.interp(self.f_noise, f, np.log(np.maximum(S, 1e-30))))

    def evaluate(self, s: FilterSettings, S_eff=None, _noise=True):
        """All metrics of one setting: PM, GM, Ms, fc, DM, stable, tau_F, tau_D, N, N_rel.

        S_eff (on the noise grid): noise spectrum with the ESC-RPM notch bank
        already applied in the time domain (tracking notch, see
        noise_data.tracking_notch_psd). Then the noise path leaves that bank
        out; the stability path keeps it, placed at the mean motor frequencies."""
        fs = self.rates.fs
        f = self.f_stab
        A, Dd, PZ = self._parts(f)
        Fg = gyro_stack(f, s, fs, self.motor_hz)
        Fd = pt1_response(f, s.f_dterm, fs)
        L = Fg * (A + Dd * Fd) * PZ
        m = {k: (v.item() if np.ndim(v) == 0 else v) for k, v in margins(f, L, self.n_int).items()}
        fc = m["fc"] if np.isfinite(m["fc"]) else 10.0
        _, m["tau_F"] = phase_lag_at(f, Fg, fc)
        _, m["tau_D"] = phase_lag_at(f, Fg * Fd * derivative_response(f, fs) / (2j * np.pi * f), fc)
        m["tau_F"], m["tau_D"] = float(m["tau_F"]), float(m["tau_D"])
        if _noise:
            fn = self.f_noise
            A, Dd, PZ = self._parts(fn)
            sn = s if S_eff is None else replace(s, dnf_esc=False)
            Fn = gyro_stack(fn, sn, fs, self.motor_hz)
            Fl = gyro_stack(fn, s, fs, self.motor_hz)
            K = A + Dd * pt1_response(fn, s.f_dterm, fs)
            S = self.S if S_eff is None else S_eff
            m["N"] = float(np.sqrt(np.trapezoid(np.abs(Fn * K / (1.0 + Fl * K * PZ))**2 * S, fn)))
            m["N_rel"] = m["N"] / self.N_ref if self.N_ref else 1.0
        return m

    def evaluate_grid(self, f_gyro, f_dterm, notch_fn, S_eff=None, noise_notch_fn=None):
        """Vectorized metrics for all (f_gyro x f_dterm) pairs with one notch configuration.

        notch_fn(f) returns the notch-stack response on grid f (used for
        stability). With S_eff, the noise path uses noise_notch_fn (default:
        no notch) and the spectrum S_eff, as in evaluate().
        Returns a dict of (G, D) arrays: PM, GM, Ms, fc, DM, n_rhp, stable, N_rel."""
        fs = self.rates.fs
        fg = np.asarray(f_gyro, dtype=float)[:, None, None]
        fd = np.asarray(f_dterm, dtype=float)[None, :, None]
        out = {}
        f = self.f_stab
        A, Dd, PZ = self._parts(f)
        Fg = notch_fn(f) * lowpass2_response(f, fg, fs)
        L = Fg * (A + Dd * pt1_response(f, fd, fs)) * PZ
        out.update(margins(f, L, self.n_int))
        fn = self.f_noise
        A, Dd, PZ = self._parts(fn)
        K = lowpass2_response(fn, fg, fs) * (A + Dd * pt1_response(fn, fd, fs))
        Nl = notch_fn(fn)
        if S_eff is None:
            Nn, S = Nl, self.S
        else:
            Nn = noise_notch_fn(fn) if noise_notch_fn else 1.0
            S = S_eff
        N = np.sqrt(np.trapezoid(np.abs(Nn * K / (1.0 + Nl * K * PZ))**2 * S, fn, axis=-1))
        out["N_rel"] = N / self.N_ref
        return out
