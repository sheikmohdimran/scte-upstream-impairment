"""CNN-based upstream impairment classifier using the v1 retrained model.

Wraps ``us_impairment_cnn_v1.pt`` (6-class multi-label CNN, 722 534 params) and
exposes the same ``classify_spectrum`` interface as :class:`RuleClassifier`.

Input conversion
----------------
The MCP surface delivers ``RawSpectrum`` traces in dBmV over 5–85 MHz (256 bins).
The CNN was trained on *linear-power* synthetic data (8 time snapshots × 200 bins).

Conversion pipeline:
  1. Take the ``average`` trace (dBmV, 256 bins).
  2. Convert dBmV → pseudo-linear units: ``val = 10 ** ((dbmv + 45) / 10)``.
     This maps the nominal -40 dBmV clean floor to ~3.2 — matching the synthetic
     training floor range (3–9 arbitrary linear units).
  3. Interpolate 256 bins → 200 bins (same 5–85 MHz band).
  4. Replicate the single snapshot 8 times → shape (8, 200).
  5. Apply training normalisation: ``(log10(max(val,1)) / LOG_MAX - μ) / σ``.

Label mapping (CNN class → ImpairmentLabel)
--------------------------------------------
  0 ingress_narrowband → NarrowbandInterference
  1 ingress_burst      → Ingress
  2 impulse_noise      → ImpulseNoise
  3 micro_reflection   → Ripple
  4 amplitude_tilt     → WidebandNoise  (closest; no exact SCTE-265 label)
  5 cpd               → CPD

The classifier is multi-label; the dominant (highest-probability) active label
is returned as ``klass``; all active labels appear in ``observations``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from uil.domain.classification import Classification, Observation
from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum import RawSpectrum

# ---------------------------------------------------------------------------
# Default model location: ``models/us_impairment_cnn_v1.pt`` inside the repo
# root, or overridden by the ``UIL_CNN_MODEL_PATH`` env var.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/uil/classifier → repo root
_DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "us_impairment_cnn_v1.pt"

# CNN training constants (embedded to avoid dependency on the training code)
_LOG_MAX: float = math.log10(2000.0)          # kept for classify_spectrum() backward compat
_LOG_MAX_NATIVE: float = math.log10(32767.0)  # matches data_generator.py LOG_MAX exactly
_NUM_BINS_CNN: int = 200
_NUM_SNAPS: int = 8
_DBMV_SHIFT: float = 45.0  # shift so -40 dBmV clean floor → ~3.2 linear

# 6 CNN output classes in training order
_CNN_CLASS_NAMES: list[str] = [
    "ingress_narrowband",
    "ingress_burst",
    "impulse_noise",
    "micro_reflection",
    "amplitude_tilt",
    "cpd",
]

# CNN class index → ImpairmentLabel
_CNN_TO_LABEL: dict[int, ImpairmentLabel] = {
    0: ImpairmentLabel.NarrowbandInterference,
    1: ImpairmentLabel.Ingress,
    2: ImpairmentLabel.ImpulseNoise,
    3: ImpairmentLabel.Ripple,
    4: ImpairmentLabel.WidebandNoise,
    5: ImpairmentLabel.CPD,
}

# Fall-back thresholds (tuned during training; overridden by checkpoint)
_DEFAULT_THRESHOLDS: list[float] = [0.55, 0.75, 0.45, 0.55, 0.45, 0.55]


class CnnClassifier:
    """Multi-label CNN classifier backed by ``us_impairment_cnn_v1.pt``.

    Parameters
    ----------
    model_path:
        Explicit path to the ``.pt`` checkpoint file.  Falls back to the
        ``UIL_CNN_MODEL_PATH`` environment variable, then to
        ``<repo-root>/models/us_impairment_cnn_v1.pt``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, …).  Defaults to
        ``"cuda"`` when a GPU is available, otherwise ``"cpu"``.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        env_path = os.environ.get("UIL_CNN_MODEL_PATH", "")
        self._model_path = Path(env_path) if env_path else Path(model_path or _DEFAULT_MODEL_PATH)
        self._device_str = device  # resolved lazily
        self._model = None  # lazy-loaded on first call
        self._torch = None
        # Normalisation stats — overridden by checkpoint values at load time
        self._mean: float = 0.2586
        self._std: float = 0.1152
        self._thresholds: list[float] = list(_DEFAULT_THRESHOLDS)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for CnnClassifier. "
                "Install it with:  pip install torch"
            ) from exc

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"CNN model checkpoint not found: {self._model_path}\n"
                "Set the UIL_CNN_MODEL_PATH environment variable or place the "
                "model at the path above."
            )

        if self._device_str is None:
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(self._device_str)

        ckpt = torch.load(self._model_path, map_location=device, weights_only=False)

        # Override normalisation stats and thresholds from checkpoint
        if "train_mean" in ckpt:
            self._mean = float(ckpt["train_mean"])
        if "train_std" in ckpt:
            self._std = float(ckpt["train_std"])
        if "best_thresholds" in ckpt:
            self._thresholds = [float(t) for t in ckpt["best_thresholds"]]

        self._model = self._build_model()
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        self._device = device

    @staticmethod
    def _build_model():
        """Reconstruct the v1 CNN architecture (must match train_v1_baseline.py exactly)."""
        import torch.nn as nn

        class _ConvBNReLU(nn.Module):
            def __init__(self, ci, co, k, p):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(ci, co, k, padding=p, bias=False),
                    nn.BatchNorm2d(co),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.seq(x)

        class US_Impairment_CNN_V1(nn.Module):
            def __init__(self, num_classes=6, drop_cnn=0.3, drop_fc=0.4):
                super().__init__()
                self.block1 = nn.Sequential(
                    _ConvBNReLU(1,   32,  (3, 7), (1, 3)),
                    _ConvBNReLU(32,  32,  (3, 7), (1, 3)),
                    nn.MaxPool2d((1, 2)),
                )
                self.block2 = nn.Sequential(
                    _ConvBNReLU(32,  64,  (3, 5), (1, 2)),
                    _ConvBNReLU(64,  64,  (3, 5), (1, 2)),
                    nn.MaxPool2d((1, 2)),
                )
                self.block3 = nn.Sequential(
                    _ConvBNReLU(64,  128, (3, 5), (1, 2)),
                    _ConvBNReLU(128, 128, (3, 3), (1, 1)),
                    nn.MaxPool2d((2, 2)),
                )
                self.block4 = nn.Sequential(
                    _ConvBNReLU(128, 256, (3, 3), (1, 1)),
                    nn.Dropout2d(p=drop_cnn),
                    nn.MaxPool2d((2, 5)),
                )
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(
                    nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(drop_fc),
                    nn.Linear(128,  64), nn.ReLU(inplace=True), nn.Dropout(drop_fc * 0.75),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                return self.head(self.gap(x).flatten(1))

        return US_Impairment_CNN_V1()

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, spectrum: RawSpectrum):
        avg_dbmv = np.array(spectrum.traces.average, dtype=np.float32)

        # 1. dBmV → pseudo-linear: -40 dBmV clean floor → 10^((−40+45)/10) ≈ 3.2
        linear = (10.0 ** ((avg_dbmv + _DBMV_SHIFT) / 10.0)).astype(np.float32)

        # 2. Resample: numBins (256) → _NUM_BINS_CNN (200)
        src_hz = np.linspace(spectrum.startFrequencyHz, spectrum.stopFrequencyHz, spectrum.numBins)
        dst_hz = np.linspace(spectrum.startFrequencyHz, spectrum.stopFrequencyHz, _NUM_BINS_CNN)
        resampled = np.interp(dst_hz, src_hz, linear).astype(np.float32)

        # 3. Replicate single snapshot → (8, 200)
        mat = np.tile(resampled, (_NUM_SNAPS, 1))

        # 4. log10 normalise, then z-score
        log_mat = np.log10(np.maximum(mat, 1.0)) / _LOG_MAX
        z = ((log_mat - self._mean) / (self._std + 1e-8)).astype(np.float32)

        # Shape: (1, 1, 8, 200) — batch × channel × time × freq
        return self._torch.from_numpy(z[np.newaxis, np.newaxis]).to(self._device)

    # ------------------------------------------------------------------
    # Public API (same signature as RuleClassifier)
    # ------------------------------------------------------------------

    def classify_spectrum(
        self,
        spectrum: RawSpectrum,
        *,
        device_type: str,
        measurement_id: str,
        rpd_id: Optional[str] = None,
        port_id: Optional[str] = None,
        amp_id: Optional[str] = None,
    ) -> Classification:
        """Classify one upstream spectrum using the v1 CNN.

        Parameters mirror :meth:`RuleClassifier.classify_spectrum` exactly.
        """
        self._load()

        x = self._preprocess(spectrum)
        with self._torch.no_grad():
            proba: np.ndarray = self._torch.sigmoid(self._model(x)).cpu().numpy()[0]

        # Active classes: probability ≥ per-class threshold
        active = [
            (cls_idx, float(proba[cls_idx]))
            for cls_idx in range(6)
            if proba[cls_idx] >= self._thresholds[cls_idx]
        ]

        observations: list[Observation] = []

        if not active:
            label = ImpairmentLabel.Clean
            confidence = float(1.0 - float(proba.max()))
            observations.append(
                Observation(
                    finding="All class probabilities below detection threshold",
                    supports=["Clean"],
                )
            )
        else:
            # Sort descending by probability; dominant = primary label
            active.sort(key=lambda t: t[1], reverse=True)
            dominant_cls, dominant_prob = active[0]
            label = _CNN_TO_LABEL[dominant_cls]
            confidence = dominant_prob
            for cls_idx, prob in active:
                observations.append(
                    Observation(
                        finding=f"{_CNN_CLASS_NAMES[cls_idx]}: p={prob:.3f}",
                        supports=[_CNN_TO_LABEL[cls_idx].value],
                        confidence=prob,
                    )
                )

        status = "clean" if label == ImpairmentLabel.Clean else "impaired"
        return Classification(
            deviceType=device_type,  # type: ignore[arg-type]
            measurementId=measurement_id,
            status=status,  # type: ignore[arg-type]
            **{"class": label},
            confidence=confidence,
            rpdId=rpd_id,
            portId=port_id,
            ampId=amp_id,
            observations=observations,
        )

    def classify_snapshots(
        self,
        snapshots: np.ndarray,
        *,
        device_type: str,
        measurement_id: str,
        rpd_id: Optional[str] = None,
        port_id: Optional[str] = None,
        amp_id: Optional[str] = None,
    ) -> Classification:
        """Classify native 8x200 linear-power snapshots (CNN training format).

        Skips the dBmV->linear, resample 256->200, and replicate x8 steps
        that classify_spectrum() performs. Uses _LOG_MAX_NATIVE = log10(32767.0)
        to match the training normalisation in data_generator.py exactly.

        Parameters
        ----------
        snapshots:
            numpy array of shape (8, 200), linear power units, same format
            produced by SpectrumSampleGenerator.
        """
        self._load()

        # Validate input shape before feeding to the CNN
        if snapshots.shape != (_NUM_SNAPS, _NUM_BINS_CNN):
            raise ValueError(
                f"classify_snapshots expects shape ({_NUM_SNAPS}, {_NUM_BINS_CNN}), "
                f"got {snapshots.shape}. Input must be 8 time-snapshots × 200 frequency bins "
                "in linear power units (output of SpectrumSampleGenerator)."
            )

        # Steps 1-3 of _preprocess() are skipped (no conversion needed).
        # Start directly at log10 normalise -> z-score using training LOG_MAX.
        log_mat = np.log10(np.maximum(snapshots, 1.0)) / _LOG_MAX_NATIVE
        z = ((log_mat - self._mean) / (self._std + 1e-8)).astype(np.float32)
        x = self._torch.from_numpy(z[np.newaxis, np.newaxis]).to(self._device)

        with self._torch.no_grad():
            proba: np.ndarray = self._torch.sigmoid(self._model(x)).cpu().numpy()[0]

        active = [
            (cls_idx, float(proba[cls_idx]))
            for cls_idx in range(6)
            if proba[cls_idx] >= self._thresholds[cls_idx]
        ]

        observations: list[Observation] = []

        if not active:
            label = ImpairmentLabel.Clean
            confidence = float(1.0 - float(proba.max()))
            observations.append(
                Observation(
                    finding="All class probabilities below detection threshold",
                    supports=["Clean"],
                )
            )
        else:
            active.sort(key=lambda t: t[1], reverse=True)
            dominant_cls, dominant_prob = active[0]
            label = _CNN_TO_LABEL[dominant_cls]
            confidence = dominant_prob
            for cls_idx, prob in active:
                observations.append(
                    Observation(
                        finding=f"{_CNN_CLASS_NAMES[cls_idx]}: p={prob:.3f}",
                        supports=[_CNN_TO_LABEL[cls_idx].value],
                        confidence=prob,
                    )
                )

        status = "clean" if label == ImpairmentLabel.Clean else "impaired"
        return Classification(
            deviceType=device_type,  # type: ignore[arg-type]
            measurementId=measurement_id,
            status=status,  # type: ignore[arg-type]
            **{"class": label},
            confidence=confidence,
            rpdId=rpd_id,
            portId=port_id,
            ampId=amp_id,
            observations=observations,
        )
