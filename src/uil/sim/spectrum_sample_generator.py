"""Physics-based upstream spectrum generator (native 8x200 linear format).

Generators are inlined verbatim from:
  netsys_aiml/modelZoo/notebooks/docsis_pnm_us_impairment_cnn/data_generator.py
No cross-repo import is used.

Severity is always explicit — internal randomisation branches removed so
every call with the same inputs produces the same output.

Output: numpy float32 arrays of shape (T_SNAPS=8, NUM_BINS=200) in
linear power units (floor ≈ 8, clip [0, 32767]).  This is exactly the
format the v1 CNN was trained on, so CnnClassifier.classify_snapshots()
can consume the output directly with no conversion.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum_sample import (
    DeviceSpecification,
    GroupSpectrumResult,
    ImpairmentType,
    SpectrumSample,
)

# ── Physical constants (inlined from data_generator.py) ───────────────────────
T_SNAPS    = 8
NUM_BINS   = 200
F_START_HZ = 5_000_000
BIN_HZ     = 400_000
MAX_VAL    = 32767.0
FLOOR_VAL  = 8.0

# ── ImpairmentLabel → ImpairmentType bridge table ─────────────────────────────
# Covers all 9 ImpairmentLabel values so T1/T4 CNN-path can always look up.
_LABEL_TO_IMPAIRMENT_TYPE: dict[ImpairmentLabel, ImpairmentType] = {
    ImpairmentLabel.Clean:                  ImpairmentType.clean,
    ImpairmentLabel.CPD:                    ImpairmentType.cpd,
    ImpairmentLabel.Ingress:               ImpairmentType.ingress_burst,
    ImpairmentLabel.ImpulseNoise:          ImpairmentType.impulse_noise,
    ImpairmentLabel.WidebandNoise:         ImpairmentType.ingress_burst,
    ImpairmentLabel.NarrowbandInterference: ImpairmentType.ingress_narrowband,
    ImpairmentLabel.Ripple:                ImpairmentType.micro_reflection,
    ImpairmentLabel.Suckout:              ImpairmentType.amplitude_tilt,
    ImpairmentLabel.UnknownImpairment:    ImpairmentType.clean,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Private helpers (inlined from data_generator.py) ──────────────────────────

def _floor(rng: np.random.Generator) -> np.ndarray:
    """Variable noise floor: N(8, 1.8) per bin/snapshot, clipped [2, 20]."""
    return np.clip(
        rng.normal(FLOOR_VAL, 1.8, (T_SNAPS, NUM_BINS)),
        2.0, 20.0,
    ).astype(np.float32)


def _bin(hz: float) -> int:
    """Convert frequency in Hz to the nearest bin index [0, NUM_BINS-1]."""
    return int(np.clip((hz - F_START_HZ) / BIN_HZ, 0, NUM_BINS - 1))


# ── Generators (inlined verbatim, severity always explicit) ────────────────────

def _gen_normal(rng: np.random.Generator, severity: float) -> np.ndarray:
    """Clean spectrum: floor ≈ 8 units with small Gaussian jitter."""
    return _floor(rng)


def _gen_ingress_narrowband(rng: np.random.Generator, severity: float) -> np.ndarray:
    """1–3 narrowband RF sources at random frequencies (5–85 MHz)."""
    data     = _floor(rng)
    n_peaks  = int(rng.integers(1, 4))
    mode     = rng.choice(
        ["persistent", "intermittent", "drifting", "am_sideband"],
        p=[0.55, 0.20, 0.15, 0.10],
    )
    bins_arr = np.arange(NUM_BINS)
    for p_idx in range(n_peaks):
        cf_hz = rng.uniform(5.0e6, 85.0e6)
        cb    = _bin(cf_hz)
        peak  = severity * rng.uniform(300, 2500)
        sigma = rng.uniform(1.2, 5.0)
        if mode == "intermittent":
            n_active = int(rng.integers(2, T_SNAPS))
            snaps    = rng.choice(T_SNAPS, size=n_active, replace=False)
        else:
            snaps = np.arange(T_SNAPS)
        drift_rate = rng.uniform(0.3, 2.0) * rng.choice([-1, 1]) if mode == "drifting" else 0.0
        for t in snaps:
            c = int(np.clip(cb + drift_rate * t, 0, NUM_BINS - 1)) \
                if mode == "drifting" else cb
            gauss = peak * np.exp(-0.5 * ((bins_arr - c) / sigma) ** 2)
            if mode == "am_sideband" and p_idx == 0:
                sb_off = int(rng.integers(1, 5))
                sb_amp = peak * rng.uniform(0.3, 0.7)
                for sb in [c - sb_off, c + sb_off]:
                    if 0 <= sb < NUM_BINS:
                        gauss += sb_amp * np.exp(
                            -0.5 * ((bins_arr - sb) / (sigma * 0.6)) ** 2
                        )
            jitter  = rng.normal(0, max(peak * 0.03, 5), NUM_BINS)
            data[t] = np.maximum(data[t], gauss + jitter)
    return np.clip(data, 0, MAX_VAL).astype(np.float32)


def _gen_ingress_burst(rng: np.random.Generator, severity: float) -> np.ndarray:
    """Wideband burst over a portion of the upstream band (1–7 of 8 snapshots)."""
    data        = _floor(rng)
    n_burst     = int(rng.integers(1, T_SNAPS))
    burst_snaps = rng.choice(T_SNAPS, size=n_burst, replace=False)
    elev        = severity * rng.uniform(20, 300)
    mode        = rng.choice(
        ["standard", "upper_only", "notched", "decaying"],
        p=[0.55, 0.15, 0.15, 0.15],
    )
    if mode == "upper_only":
        s_bin = _bin(rng.uniform(35e6, 50e6))
        e_bin = _bin(rng.uniform(60e6, 85e6))
        for t in burst_snaps:
            data[t, s_bin:e_bin] += rng.uniform(elev * 0.75, elev * 1.25, e_bin - s_bin)
    elif mode == "notched":
        e_bin   = _bin(rng.uniform(12e6, 70e6))
        notch_s = int(rng.integers(5, max(6, e_bin - 15)))
        notch_e = notch_s + int(rng.integers(5, 16))
        for t in burst_snaps:
            arr = rng.uniform(elev * 0.75, elev * 1.25, e_bin)
            arr[notch_s:notch_e] = 0.0
            data[t, :e_bin] += arr
    elif mode == "decaying":
        e_bin = _bin(rng.uniform(12e6, 70e6))
        taper = np.linspace(elev, elev * 0.05, e_bin)
        for t in burst_snaps:
            data[t, :e_bin] += taper + rng.normal(0, elev * 0.1, e_bin)
    else:  # standard
        e_bin = _bin(rng.uniform(12e6, 70e6))
        for t in burst_snaps:
            data[t, :e_bin] += rng.uniform(elev * 0.75, elev * 1.25, e_bin)
    return np.clip(data, 0, MAX_VAL).astype(np.float32)


def _gen_impulse_noise(rng: np.random.Generator, severity: float) -> np.ndarray:
    """Sparse high-amplitude spikes: 0–34 per snapshot, 1–4 bins wide."""
    data = _floor(rng)
    mode = rng.choice(
        ["standard", "very_sparse", "periodic", "low_freq", "burst_clustered"],
        p=[0.52, 0.12, 0.09, 0.12, 0.15],
    )
    if mode == "very_sparse":
        n_total = int(rng.integers(1, 6))
        for _ in range(n_total):
            t = int(rng.integers(T_SNAPS))
            b = int(rng.integers(NUM_BINS))
            data[t, b] += severity * rng.uniform(200, 3000)
    elif mode == "periodic":
        spacing = int(rng.integers(5, 25))
        start   = int(rng.integers(0, spacing))
        for t in range(T_SNAPS):
            if rng.random() < 0.75:
                for b in range(start, NUM_BINS, spacing):
                    data[t, b] += severity * rng.uniform(50, 800)
    elif mode == "low_freq":
        lf_end = _bin(35e6)
        for t in range(T_SNAPS):
            n_spk = int(rng.integers(0, 20))
            for _ in range(n_spk):
                b = int(rng.integers(0, lf_end))
                data[t, b] += severity * rng.uniform(50, 3000)
    elif mode == "burst_clustered":
        n_active = int(rng.integers(2, 5))
        t_start  = int(rng.integers(0, T_SNAPS - n_active + 1))
        for t in range(t_start, t_start + n_active):
            n_spk = int(rng.integers(5, 25))
            bins  = rng.integers(0, NUM_BINS, n_spk)
            amps  = severity * rng.uniform(100, 3000, n_spk)
            wids  = rng.integers(1, 5, n_spk)
            for b, a, w in zip(bins, amps, wids):
                for db in range(int(w)):
                    if b + db < NUM_BINS:
                        data[t, b + db] += float(a) * np.exp(-0.5 * db ** 2)
    else:  # standard
        for t in range(T_SNAPS):
            n_spk = int(rng.integers(0, 35))
            if n_spk == 0:
                continue
            bins = rng.integers(0, NUM_BINS, n_spk)
            amps = severity * rng.uniform(50, 3000, n_spk)
            wids = rng.integers(1, 5, n_spk)
            for b, a, w in zip(bins, amps, wids):
                for db in range(int(w)):
                    if b + db < NUM_BINS:
                        data[t, b + db] += float(a) * np.exp(-0.5 * db ** 2)
    return np.clip(data, 0, MAX_VAL).astype(np.float32)


def _gen_micro_reflection(rng: np.random.Generator, severity: float) -> np.ndarray:
    """Sinusoidal frequency-domain ripple from a cable-stub reflection."""
    data = _floor(rng)
    mode = rng.choice(
        ["single_stub", "dual_stub", "amplitude_varying",
         "short_delay", "long_delay_alias", "phase_drift"],
        p=[0.33, 0.20, 0.12, 0.10, 0.13, 0.12],
    )

    def _ripple(delay_ns: float, amp: float, phase: float) -> np.ndarray:
        period_bins = 1.0 / (delay_ns * 1e-9 * BIN_HZ)
        return amp * np.cos(2 * np.pi * np.arange(NUM_BINS) / period_bins + phase)

    if mode == "dual_stub":
        delays  = [rng.uniform(100, 700), rng.uniform(800, 2200)]
        ripples = [
            _ripple(d, severity * rng.uniform(1.0, 9.0), rng.uniform(0, 2 * np.pi))
            for d in delays
        ]
        composite = ripples[0] + ripples[1]
        for t in range(T_SNAPS):
            data[t] += composite + rng.normal(0, 0.3, NUM_BINS)
    elif mode == "amplitude_varying":
        delay_ns = rng.uniform(100, 2000)
        base_amp = severity * rng.uniform(1.5, 14.0)
        phase    = rng.uniform(0, 2 * np.pi)
        for t in range(T_SNAPS):
            amp_t    = base_amp * rng.uniform(0.6, 1.4)
            data[t] += _ripple(delay_ns, amp_t, phase) + rng.normal(0, 0.3, NUM_BINS)
    elif mode == "short_delay":
        delay_ns = rng.uniform(30, 120)
        amp      = severity * rng.uniform(1.5, 14.0)
        phase    = rng.uniform(0, 2 * np.pi)
        ripple   = _ripple(delay_ns, amp, phase)
        for t in range(T_SNAPS):
            data[t] += ripple + rng.normal(0, 0.3, NUM_BINS)
    elif mode == "long_delay_alias":
        delay_ns = rng.uniform(2500, 8000)
        amp      = severity * rng.uniform(2.0, 18.0)
        phase    = rng.uniform(0, 2 * np.pi)
        ripple   = _ripple(delay_ns, amp, phase)
        for t in range(T_SNAPS):
            data[t] += ripple + rng.normal(0, 0.3, NUM_BINS)
    elif mode == "phase_drift":
        delay_ns    = rng.uniform(100, 2000)
        amp         = severity * rng.uniform(1.5, 14.0)
        phase_start = rng.uniform(0, 2 * np.pi)
        drift_rate  = rng.uniform(0.05, 0.4) * rng.choice([-1.0, 1.0])
        for t in range(T_SNAPS):
            data[t] += _ripple(delay_ns, amp, phase_start + t * drift_rate) \
                       + rng.normal(0, 0.3, NUM_BINS)
    else:  # single_stub
        delay_ns    = rng.uniform(100, 2000)
        period_bins = 1.0 / (delay_ns * 1e-9 * BIN_HZ)
        amp         = severity * rng.uniform(1.5, 14.0)
        phase       = rng.uniform(0, 2 * np.pi)
        ripple      = amp * np.cos(2 * np.pi * np.arange(NUM_BINS) / period_bins + phase)
        for t in range(T_SNAPS):
            data[t] += ripple + rng.normal(0, 0.3, NUM_BINS)
    return np.clip(data, 0, MAX_VAL).astype(np.float32)


def _gen_amplitude_tilt(rng: np.random.Generator, severity: float) -> np.ndarray:
    """Slope across the upstream band from amplifier equalization failure."""
    data = _floor(rng)
    mag  = severity * rng.uniform(5, 45)
    sign = rng.choice([-1.0, 1.0])
    mode = rng.choice(
        ["linear", "curved", "piecewise", "drifting", "with_ripple"],
        p=[0.43, 0.17, 0.13, 0.13, 0.14],
    )
    if mode == "curved":
        x    = np.linspace(0, 1, NUM_BINS)
        tilt = sign * mag * (1.0 - x ** 2)
        for t in range(T_SNAPS):
            data[t] += tilt + rng.normal(0, 0.4, NUM_BINS)
    elif mode == "piecewise":
        mid      = int(rng.integers(80, 120))
        lo_end   = sign * mag * rng.uniform(0.7, 1.0)
        hi_start = -sign * mag * rng.uniform(0.1, 0.4)
        hi_end   = -sign * mag * rng.uniform(0.2, 0.5)
        tilt     = np.empty(NUM_BINS)
        tilt[:mid] = np.linspace(lo_end, hi_start, mid)
        tilt[mid:] = np.linspace(hi_start, hi_end, NUM_BINS - mid)
        for t in range(T_SNAPS):
            data[t] += tilt + rng.normal(0, 0.4, NUM_BINS)
    elif mode == "drifting":
        base_tilt = np.linspace(sign * mag, -sign * mag * 0.2, NUM_BINS)
        for t in range(T_SNAPS):
            scale    = rng.uniform(0.8, 1.2)
            data[t] += base_tilt * scale + rng.normal(0, 0.4, NUM_BINS)
    elif mode == "with_ripple":
        tilt            = np.linspace(sign * mag, -sign * mag * 0.2, NUM_BINS)
        ripple_delay_ns = rng.uniform(50, 300)
        ripple_period   = 1.0 / (ripple_delay_ns * 1e-9 * BIN_HZ)
        ripple_amp      = severity * rng.uniform(0.3, 2.5)
        ripple_phase    = rng.uniform(0, 2 * np.pi)
        ripple          = ripple_amp * np.cos(
            2 * np.pi * np.arange(NUM_BINS) / ripple_period + ripple_phase
        )
        for t in range(T_SNAPS):
            data[t] += tilt + ripple + rng.normal(0, 0.4, NUM_BINS)
    else:  # linear
        tilt = np.linspace(sign * mag, -sign * mag * 0.2, NUM_BINS)
        for t in range(T_SNAPS):
            data[t] += tilt + rng.normal(0, 0.4, NUM_BINS)
    return np.clip(data, 0, MAX_VAL).astype(np.float32)


def _gen_cpd(rng: np.random.Generator, severity: float) -> np.ndarray:
    """Common Path Distortion: raised wideband floor + harmonic beat products."""
    data = _floor(rng)
    mode = rng.choice(
        ["single_source", "dual_source", "harmonic_only",
         "decaying_harm", "intermittent"],
        p=[0.38, 0.10, 0.10, 0.30, 0.12],
    )

    def _add_harmonics(spacing_hz: float, beat_scale: float,
                       decay: bool = False, snaps=None) -> None:
        if snaps is None:
            snaps = range(T_SNAPS)
        for n in range(2, int(80e6 / spacing_hz) + 1):
            cb   = _bin(n * spacing_hz)
            beat = beat_scale * rng.uniform(4, 30)
            if decay:
                beat *= float(n) ** (-1.5)
            for t in snaps:
                for db in (-1, 0, 1):
                    b = cb + db
                    if 0 <= b < NUM_BINS:
                        data[t, b] += beat * np.exp(-0.5 * float(db) ** 2)

    if mode == "harmonic_only":
        _add_harmonics(rng.uniform(5.5e6, 7.5e6), severity, decay=False)
    elif mode == "dual_source":
        floor_raise = severity * rng.uniform(5, 40)
        data += rng.uniform(floor_raise * 0.8, floor_raise * 1.2, (T_SNAPS, NUM_BINS))
        for sp in [rng.uniform(5.5e6, 6.3e6), rng.uniform(6.5e6, 7.5e6)]:
            _add_harmonics(sp, severity * 0.6, decay=True)
    elif mode == "decaying_harm":
        floor_raise = severity * rng.uniform(5, 70)
        data += rng.uniform(floor_raise * 0.8, floor_raise * 1.2, (T_SNAPS, NUM_BINS))
        _add_harmonics(rng.uniform(5.5e6, 7.5e6), severity, decay=True)
    elif mode == "intermittent":
        n_active     = int(rng.integers(3, T_SNAPS))
        active_snaps = list(rng.choice(T_SNAPS, size=n_active, replace=False))
        floor_raise  = severity * rng.uniform(5, 70)
        for t in active_snaps:
            data[t] += rng.uniform(floor_raise * 0.8, floor_raise * 1.2, NUM_BINS)
        if rng.random() > 0.30:
            _add_harmonics(rng.uniform(5.5e6, 7.5e6), severity,
                           decay=bool(rng.random() > 0.5), snaps=active_snaps)
    else:  # single_source
        floor_raise = severity * rng.uniform(5, 70)
        data += rng.uniform(floor_raise * 0.8, floor_raise * 1.2, (T_SNAPS, NUM_BINS))
        if rng.random() > 0.30:
            _add_harmonics(rng.uniform(5.5e6, 7.5e6), severity, decay=False)
    return np.clip(data, 0, MAX_VAL).astype(np.float32)


# ── Generator registry ─────────────────────────────────────────────────────────
_GENERATORS: dict[ImpairmentType, Callable] = {
    ImpairmentType.clean:              _gen_normal,
    ImpairmentType.cpd:                _gen_cpd,
    ImpairmentType.ingress_narrowband: _gen_ingress_narrowband,
    ImpairmentType.ingress_burst:      _gen_ingress_burst,
    ImpairmentType.impulse_noise:      _gen_impulse_noise,
    ImpairmentType.micro_reflection:   _gen_micro_reflection,
    ImpairmentType.amplitude_tilt:     _gen_amplitude_tilt,
}


class SpectrumSampleGenerator:
    """Stateless physics-based upstream spectrum generator.

    Produces 8-snapshot × 200-bin linear-power arrays in the exact format
    the v1 CNN was trained on.  All calls are deterministic for given inputs.
    """

    def generate(
        self,
        spec: DeviceSpecification,
        run_severity: float = 1.0,
    ) -> SpectrumSample:
        """Generate one spectrum sample for a device specification.

        Parameters
        ----------
        spec:
            Device specification including impairment types and optional
            per-device severity override.
        run_severity:
            Run-level default severity from Scenario.severity.  Used when
            spec.severity is None.
        """
        eff_severity = spec.severity if spec.severity is not None else run_severity
        rng    = self._rng_for(spec.deviceId, spec.impairments, eff_severity)
        matrix = self._synthesize(spec.impairments, eff_severity, rng)
        return SpectrumSample(
            deviceId=spec.deviceId,
            deviceType=spec.deviceType,
            impairments=[i.value for i in spec.impairments],
            severity=eff_severity,
            snapshots=matrix.tolist(),
            timestamp=_now(),
        )

    def generate_group(
        self,
        specs: list[DeviceSpecification],
        run_severity: float = 1.0,
        max_devices: int = 50,
    ) -> GroupSpectrumResult:
        """Generate spectrum samples for a list of devices.

        Raises ValueError if len(specs) > max_devices.
        """
        if len(specs) > max_devices:
            raise ValueError(
                f"Device count {len(specs)} exceeds limit {max_devices}"
            )
        samples = [self.generate(s, run_severity) for s in specs]
        return GroupSpectrumResult(
            runSeverity=run_severity,
            deviceCount=len(samples),
            samples=samples,
            generatedAt=_now(),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _rng_for(
        device_id: str,
        impairments: list[ImpairmentType],
        severity: float,
    ) -> np.random.Generator:
        """Deterministic RNG seed from (device_id, impairments, severity)."""
        key  = f"{device_id}:{'|'.join(sorted(i.value for i in impairments))}:{severity:.4f}"
        seed = int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "big")
        return np.random.default_rng(seed)

    def _synthesize(
        self,
        impairments: list[ImpairmentType],
        severity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Combine generators using data_generator.py combine_impairments formula.

        Formula: base + Σ(gen_i - FLOOR_VAL) for each additional impairment.
        Subtracting the constant FLOOR_VAL (not the random floor array) avoids
        the double-floor issue while matching the training data distribution.
        """
        active = [imp for imp in impairments if imp != ImpairmentType.clean]
        if not active:
            # Pure clean: just the noise floor
            return _gen_normal(rng, severity)
        # First impairment is the base (contains its own floor + signal)
        combined = _GENERATORS[active[0]](rng, severity).astype(np.float64)
        # Each additional impairment: add signal contribution, subtract constant floor
        # to avoid doubling the floor for each extra generator (matches combine_impairments)
        for imp in active[1:]:
            combined = combined + _GENERATORS[imp](rng, severity).astype(np.float64) - FLOOR_VAL
        return np.clip(combined, 0.0, MAX_VAL).astype(np.float32)
