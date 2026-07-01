# Spectrum Sample Generation

## Overview

`SpectrumSampleGenerator` (`src/uil/sim/spectrum_sample_generator.py`) is the
physics-based synthetic data engine for the upstream impairment tool chain.  It
produces realistic 8-snapshot × 200-bin linear-power spectra covering the
traditional upstream band (5–85 MHz) — the **native training format** of the v1
CNN classifier.

This replaces the original `SpectrumSimulator` (256-bin dBmV, 8 impairment
labels stubbed to generic random noise) with physically-motivated models for
all 7 impairment types.

---

## Signal Format

| Parameter | Value | Notes |
|-----------|-------|-------|
| Time snapshots | **8** | ~30-second observation window |
| Frequency bins | **200** | 5.0–85.0 MHz at 400 kHz/bin |
| Bin 0 center | 5.0 MHz | `F_START_HZ = 5_000_000` |
| Bin spacing | 400 kHz | `BIN_HZ = 400_000` |
| Bin 199 center | 84.6 MHz | `5e6 + 199 × 400 kHz` |
| Power units | **linear** (arbitrary) | Floor ≈ 8 units; clip max = 32 767 |
| Noise floor | N(8, 1.8) clipped [2, 20] | Per-bin, per-snapshot |
| Array dtype | `float32` | Shape: `(8, 200)` |

This format matches `data_generator.py` exactly so `CnnClassifier.classify_snapshots()`
can consume the output **with no conversion step**.

---

## Impairment Types

Seven types are implemented, corresponding to the CNN's 6 training classes
plus `clean`:

| `ImpairmentType` | Physical cause | Key signature |
|------------------|---------------|---------------|
| `clean` | Nominal upstream plant | Floor ≈ 8 units, low variance |
| `cpd` | Oxidized passive (tap/splitter/connector) acting as non-linear junction | Raised wideband floor + 5.5–7.5 MHz harmonic beat products |
| `ingress_narrowband` | Damaged cable shielding; external RF leakage (LTE, shortwave, CB) | 1–3 Gaussian peaks at random frequencies; temporal modes: persistent / intermittent / drifting / AM-sideband |
| `ingress_burst` | Intermittent electrical interference (appliance, HVAC, power tool) | Wideband floor elevation on 1–7 of 8 snapshots; spatial modes: standard / upper_only / notched / decaying |
| `impulse_noise` | Electrical arcing, switching PSU, motor start | Sparse high-amplitude spikes (0–34/snapshot, 1–4 bins wide); modes: standard / very_sparse / periodic / low_freq / burst_clustered |
| `micro_reflection` | Impedance mismatch (unterminated tap, water-damaged end-cap) | Sinusoidal frequency-domain ripple; delay 30–8000 ns; modes: single_stub / dual_stub / amplitude_varying / short_delay / long_delay_alias / phase_drift |
| `amplitude_tilt` | Degraded amplifier equalization or failing diplex filter | Linear slope across 5–85 MHz; modes: linear / curved / piecewise / drifting / with_ripple |

### Mode Diversity

Each generator randomly selects among 4–6 structural **modes** on every call.
The mode probabilities are tuned to match observed HFC field distributions.
This diversity prevents the CNN from overfitting to a single signature shape.

---

## Severity Parameter

`severity` is a float in `[0.0, 1.0]` that scales the signal amplitude of any
impairment (floor elevation, peak height, ripple depth, spike magnitude, etc.).

| Value | Effect |
|-------|--------|
| `0.0` | Impairment signal is effectively zero — indistinguishable from clean |
| `0.1` | Weak, near detection threshold |
| `0.5` | Moderate — clearly detectable |
| `1.0` | Full-strength impairment |

**Severity is always explicit** — generators never randomise it internally.
This means every call with the same `(device_id, impairments, severity)` always
produces the same output (deterministic).

### Run-level vs per-device severity

```
Scenario.severity = 0.8          ← run-level default (fixed for a run's lifetime)

DeviceSpecification(severity=None)  → uses Scenario.severity (0.8)
DeviceSpecification(severity=0.3)   → uses 0.3 (per-device override)
```

---

## Determinism

The RNG seed is derived from `(device_id, impairments, severity)`:

```python
key  = f"{device_id}:{'|'.join(sorted(i.value for i in impairments))}:{severity:.4f}"
seed = int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "big")
rng  = np.random.default_rng(seed)
```

The same inputs always produce the **same** 8×200 array. Different
`device_id` or `severity` values produce different arrays.

---

## Multi-label synthesis

When multiple impairments are specified for one device, signals are combined
using additive floor-subtracted deltas to avoid doubling the noise floor:

```
combined = floor + Σ (generator_i(rng, severity) − floor)
         = floor + signal_A + signal_B + …
```

Each `generator_i` returns `floor + signal_i`, so subtracting the floor
extracts the pure signal delta before summing.  The result is clipped to
`[0, MAX_VAL=32767]`.

---

## ImpairmentLabel → ImpairmentType bridge

The existing 9-value `ImpairmentLabel` enum (CamelCase, used throughout the
MCP tool chain) maps to the 7-value `ImpairmentType` generator enum
(snake_case) via `_LABEL_TO_IMPAIRMENT_TYPE`:

| `ImpairmentLabel` (CamelCase) | → `ImpairmentType` (generator) | Notes |
|-------------------------------|--------------------------------|-------|
| `Clean` | `clean` | |
| `CPD` | `cpd` | |
| `Ingress` | `ingress_burst` | Wideband, best match |
| `ImpulseNoise` | `impulse_noise` | |
| `WidebandNoise` | `ingress_burst` | Raised floor, best match |
| `NarrowbandInterference` | `ingress_narrowband` | |
| `Ripple` | `micro_reflection` | |
| `Suckout` | `amplitude_tilt` | Closest available generator |
| `UnknownImpairment` | `clean` | Safe fallback |

This bridge is used in the `use_cnn_path=True` server path (T1/T4) to convert
`Scenario.rpd_label` and `Scenario.amps[*].label` into generator types.

---

## Device naming convention (T7/T8)

When calling `getDeviceSpectrumSamples` (T7) or `getSignalMetrics` (T8), the
`deviceId` prefix determines the device type:

| Prefix | `deviceType` | Example IDs |
|--------|-------------|-------------|
| `rpd-` | `RPD` | `rpd-001`, `rpd-north-042`, `rpd-site-a` |
| `amp-` | `AMP` | `amp-001`, `amp-north-042`, `amp-site-a-lvl2` |

The `device_type` is inferred automatically; callers do not specify it separately.
IDs that do not match either prefix return an `INVALID_DEVICE_ID` error.

---

## Integration into the tool chain

```
use_cnn_path=False (default — legacy)
  T1/T4: SpectrumSimulator  → RawSpectrum (256-bin, dBmV, 3 traces)
  T2/T5: RuleClassifier     → Classification via threshold heuristic

use_cnn_path=True (CNN path — new)
  T1/T4: SpectrumSampleGenerator → snapshots (8×200, linear) stored in HandleStore
  T2/T5: CnnClassifier.classify_snapshots() → Classification via CNN inference
           ↑ No dBmV conversion, no resampling — native format, no domain gap

T7 getDeviceSpectrumSamples (always via SpectrumSampleGenerator)
  Input:  devices list with ImpairmentLabel CamelCase values
  Bridge: ImpairmentLabel → ImpairmentType internally
  Output: sampleSetRef (handle) — snapshots never on reference surface

T8 getSignalMetrics (always via SpectrumSampleGenerator)
  Input:  modemId (rpd-* or amp-*), windowSec
  Output: rawArtifactRef (handle) + scalar RF metrics (TX power, SNR, uncorr)
```

---

## Example output (clean vs CPD)

```
clean  — floor ≈ 8 units, low variance
  mean=8.03  min=2.00  max=14.42   (all 8 snapshots similar)

cpd (severity=0.8) — raised floor + harmonic beats
  mean=63.38  min=2.00  max=80     (floor elevated ~8× + periodic comb)

ingress_narrowband (severity=0.75)
  mean=122.20  min=2.00  max=1747  (sharp Gaussian peak at one frequency)

impulse_noise (severity=0.5)
  mean=50.59  min=2.00  max=2482   (sparse high spikes across band)
```
