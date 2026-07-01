# CNN Impairment Classifier

## Overview

`CnnClassifier` (`src/uil/classifier/cnn_classifier.py`) wraps the v1 retrained
multi-label CNN (`models/us_impairment_cnn_v1.pt`).  It replaces the rule-based
bootstrap `RuleClassifier` and exposes the same `classify_spectrum()` interface
plus a new `classify_snapshots()` method for native 8×200 linear input.

---

## Model checkpoint

| Property | Value |
|----------|-------|
| **File** | `models/us_impairment_cnn_v1.pt` |
| **Architecture** | `v1_baseline` |
| **Parameters** | 722,534 |
| **Best validation macro-F1** | **0.9322** (epoch 38) |
| **Training data** | Synthetic spectra from `data_generator.py` (8×200 linear, 5–85 MHz) |
| **Classes** | 6 (multi-label) |
| **Per-class thresholds** | `[0.55, 0.75, 0.45, 0.55, 0.45, 0.55]` |
| **Training mean** | 0.2586 |
| **Training std** | 0.1152 |
| **LOG\_MAX** | log10(32767) = 4.5154 |

Thresholds, mean, and std are stored inside the checkpoint and loaded
automatically at model-load time.

---

## CNN architecture (v1_baseline)

```
Input: (batch=1, channel=1, time=8, freq=200)

Block 1 ─── ConvBNReLU(1  →  32, kernel=(3,7), pad=(1,3))
         ── ConvBNReLU(32 →  32, kernel=(3,7), pad=(1,3))
         ── MaxPool2d(kernel=(1,2))
         Output: (1, 32, 8, 100)

Block 2 ─── ConvBNReLU(32 →  64, kernel=(3,5), pad=(1,2))
         ── ConvBNReLU(64 →  64, kernel=(3,5), pad=(1,2))
         ── MaxPool2d(kernel=(1,2))
         Output: (1, 64, 8, 50)

Block 3 ─── ConvBNReLU(64  → 128, kernel=(3,5), pad=(1,2))
         ── ConvBNReLU(128 → 128, kernel=(3,3), pad=(1,1))
         ── MaxPool2d(kernel=(2,2))
         Output: (1, 128, 4, 25)

Block 4 ─── ConvBNReLU(128 → 256, kernel=(3,3), pad=(1,1))
         ── Dropout2d(p=0.30)
         ── MaxPool2d(kernel=(2,5))
         Output: (1, 256, 2, 5)

Global Average Pooling → (1, 256, 1, 1) → flatten → (1, 256)

FC Head ─── Linear(256 → 128) + ReLU + Dropout(0.40)
         ── Linear(128 →  64) + ReLU + Dropout(0.30)
         ── Linear( 64 →   6)              ← raw logits, one per class

Output: sigmoid(logits) → probabilities in [0, 1] for each of 6 classes
```

`_ConvBNReLU` is `Conv2d(bias=False) → BatchNorm2d → ReLU(inplace)` fused as
a single `nn.Sequential`.

Wider frequency kernels `(3,7)` and `(3,5)` in early blocks capture broad
spectral features (floor elevation, tilt, burst).  Narrower `(3,3)` kernels in
deeper blocks learn localised patterns (harmonic beats, spike clusters, ripple
phase).

---

## 6 output classes

| Bit | Training name | `ImpairmentLabel` | Threshold | Physical signature |
|-----|--------------|------------------|-----------|--------------------|
| 0 | `ingress_narrowband` | `NarrowbandInterference` | 0.55 | Gaussian RF peak from external leakage |
| 1 | `ingress_burst` | `Ingress` | **0.75** | Wideband floor burst on subset of snapshots |
| 2 | `impulse_noise` | `ImpulseNoise` | 0.45 | Sparse broadband spikes |
| 3 | `micro_reflection` | `Ripple` | 0.55 | Sinusoidal frequency-domain ripple |
| 4 | `amplitude_tilt` | `WidebandNoise` | 0.45 | Linear slope across upstream band |
| 5 | `cpd` | `CPD` | 0.55 | Raised floor + harmonic beat products |

`ingress_burst` has a higher threshold (0.75) because its wideband floor
elevation overlaps with `cpd` at low severity; conservative detection reduces
false positives.  `impulse_noise` and `amplitude_tilt` have lower thresholds
(0.45) because these classes are harder to detect at weak severity.

The model is **multi-label**: multiple bits can fire simultaneously (e.g., CPD
+ amplitude\_tilt is a common co-occurring fault pair in degraded plant).

---

## Classification pipeline

### Path A — `classify_spectrum(RawSpectrum)` — legacy dBmV input

Used when T1/T4 data comes from `SpectrumSimulator` (`use_cnn_path=False`).

```
RawSpectrum.traces.average   (256 values, dBmV)
    │
    ▼ step 1: dBmV → pseudo-linear
    val = 10 ^ ((dBmV + 45) / 10)
    # Maps nominal -40 dBmV clean floor → 10^(5/10) ≈ 3.2 linear units
    │
    ▼ step 2: resample 256 → 200 bins
    np.interp(dst_hz_200, src_hz_256, val)
    │
    ▼ step 3: replicate single snapshot × 8
    mat = np.tile(resampled, (8, 1))   # shape (8, 200)
    │
    ▼ step 4: log10 normalise
    log_mat = log10(max(mat, 1.0)) / log10(2000)   # _LOG_MAX = 3.301
    │
    ▼ step 5: z-score
    z = (log_mat − 0.2586) / 0.1152
    │
    ▼ CNN forward pass → sigmoid → probabilities [6]
    │
    ▼ per-class threshold comparison → active classes
    │
    ▼ dominant label (highest probability) → Classification
```

> **Note**: `_LOG_MAX = log10(2000)` in this path differs from the training
> `LOG_MAX = log10(32767)`. This path was tuned for the dBmV→linear conversion
> and its use is maintained for backward compatibility with the legacy simulator.

---

### Path B — `classify_snapshots(ndarray)` — native 8×200 linear input

Used when T1/T4 data comes from `SpectrumSampleGenerator` (`use_cnn_path=True`)
or when T7 samples are classified externally.

```
snapshots: ndarray shape (8, 200), linear power units
    │   (same format as training data — no conversion needed)
    │
    ▼ step 1: log10 normalise   ← starts here; steps 1-3 of Path A are skipped
    log_mat = log10(max(snapshots, 1.0)) / log10(32767)
    #                                      ▲ _LOG_MAX_NATIVE = 4.5154
    #                                        matches data_generator.py LOG_MAX exactly
    │
    ▼ step 2: z-score
    z = (log_mat − 0.2586) / 0.1152
    │
    ▼ CNN forward pass → sigmoid → probabilities [6]
    │
    ▼ per-class threshold comparison → active classes
    │
    ▼ dominant label (highest probability) → Classification
```

**This path has zero domain gap**: the normalisation matches the training
pipeline exactly, so the model operates on inputs from the same distribution it
was trained on.

---

## How the model arrives at an impairment label

```
proba = sigmoid(CNN(z))     → float[6]  each in [0, 1]

active = [(i, proba[i]) for i in range(6)
          if proba[i] >= threshold[i]]

if not active:
    label      = ImpairmentLabel.Clean
    confidence = 1.0 − max(proba)       # inversion: low max prob → high clean confidence
else:
    sort active descending by probability
    dominant class  = active[0]
    label           = _CNN_TO_LABEL[dominant class]
    confidence      = proba[dominant class]
    # all active classes logged in Classification.observations
```

### Multi-label case

If multiple classes fire (e.g., CPD and amplitude\_tilt both above threshold):
- `klass` = label of the **highest-probability** active class (dominant)
- `observations` = list of all active classes with their probabilities
- Downstream: `analyzeSpectrumMeasurements` reports all active labels in
  `classesPresent`; `GraphLocalizer` checks for conflicting labels across
  devices and may escalate to `low_confidence`

---

## How the model detects each impairment

### CPD (bit 5, threshold 0.55)
The raised wideband floor creates a persistent high-energy region across all
8 snapshots — the temporal dimension shows no variance.  The 5.5–7.5 MHz
harmonic beat products appear as a regular comb in the frequency dimension.
Early conv blocks `(3,7)` detect the broad floor; later `(3,3)` blocks detect
the comb periodicity.

### Ingress burst (bit 1, threshold 0.75)
High temporal variance: a subset of snapshots show elevated energy while others
are clean.  The conv blocks along the time axis `(3,*)` detect this on/off
pattern.  The conservative threshold (0.75) avoids confusing CPD intermittent
mode (which also shows snapshot variation) with true ingress burst.

### Ingress narrowband (bit 0, threshold 0.55)
A localised Gaussian peak at one frequency bin — high energy concentrated in
1–5 adjacent bins.  The frequency kernels detect this concentration; the
temporal blocks confirm persistence or intermittency.

### Impulse noise (bit 2, threshold 0.45)
Sparse random spikes: very few bins elevated, scattered across both time and
frequency.  The pattern is non-coherent (no fixed frequency spacing, no
temporal correlation).  Lower threshold compensates for weak, sparse signals.

### Micro-reflection (bit 3, threshold 0.55)
Sinusoidal ripple across the frequency axis — coherent and persistent across
all 8 snapshots.  Ripple period depends on delay: wide period (short delay,
30–120 ns) looks like a slope; narrow period (long delay, >2500 ns) aliases
into apparent noise but remains coherent.  The model was trained on all 6
structural modes.

### Amplitude tilt (bit 4, threshold 0.45)
A monotonic slope from low-band to high-band (or reverse) — globally elevated
energy on one side.  This is the hardest class to separate from micro-reflection
`with_ripple` mode.  The lower threshold ensures weak tilts are not missed.

---

## Confidence interpretation

| Range | Interpretation |
|-------|---------------|
| 0.45–0.60 | Marginal detection; check for co-occurring classes |
| 0.60–0.80 | Clear detection |
| 0.80–0.99 | Strong detection; high confidence in dominant class |

For `Clean`: confidence = 1 − max(proba), so a very low max activation
(all classes well below threshold) gives high clean confidence.

---

## Lazy loading

The model is loaded on the **first** call to `classify_spectrum()` or
`classify_snapshots()`. Subsequent calls reuse the loaded model.

The checkpoint path is resolved in this priority order:
1. `UIL_CNN_MODEL_PATH` environment variable
2. Constructor `model_path` argument
3. `<repo-root>/models/us_impairment_cnn_v1.pt` (default)

The device is `cuda` if a GPU is available, otherwise `cpu`.

---

## Switching from RuleClassifier to CnnClassifier

### Standalone injection (any path)
```python
from uil.classifier.cnn_classifier import CnnClassifier
server = MockMcpServer(scenario, classifier=CnnClassifier())
```

### Full CNN path (SpectrumSampleGenerator + CnnClassifier, no conversion)
```python
server = MockMcpServer(scenario, use_cnn_path=True)
# T1/T4 generate 8×200 linear via SpectrumSampleGenerator
# T2/T5 classify via classify_snapshots() — no dBmV conversion
```

### Guard
```python
# Raises ValueError:
MockMcpServer(scenario, use_cnn_path=True, classifier=RuleClassifier())
```

---

## Observed classification accuracy (synthetic data, E2E)

| Injected impairment | CNN label | Status |
|--------------------|-----------|--------|
| CPD | CPD | localized (confidence 0.80) |
| Ripple (micro\_reflection) | Ripple | localized (confidence 0.80) |
| NarrowbandInterference | NarrowbandInterference | localized (confidence 0.80) |

All three classify correctly end-to-end through the 6-step localization chain
with `use_cnn_path=True` and `severity=0.8`.

Validation macro-F1 on the held-out synthetic test set: **0.9322**.
OOD evaluation (out-of-distribution real-world proxies): 92.5% / 87.5% / 100%
depending on impairment type.
