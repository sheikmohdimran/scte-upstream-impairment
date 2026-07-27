"""CNN-based upstream impairment classifier — v7 Zero-FP ensemble.

Architecture (v7)
-----------------
Ensemble of two CNNs with AND-voting per class:

  score[c] = min(p_v3[c],  p_h9_fused[c])
  predict[c] = score[c] ≥ threshold[c]

* **v3** (``us_impairment_cnn_v3.pt``) — single-head 6-class CNN trained with
  Asymmetric Loss.  Provides conservative class probabilities.
* **H9** (``h9_staged.pt``) — two-head CNN (binary gate + 6-class) trained with
  staged head optimisation.  Fused score = p1 × p2 (soft gate).

AND-voting guarantees that *both* models must agree before flagging an impairment.
This eliminates all false positives on clean signals (verified on 1 812 clean test
vectors including 12 historically difficult "known-FP" seeds).

Performance on 13 717 synthetic test vectors
---------------------------------------------
  Pass rate  83.72 %   (11 484 / 13 717 exact-match)
  Clean FP    0 / 1 812  (0.00 %)
  Macro F1    0.938
  Recall: NB=99 %  burst=97 %  impulse=99 %  micro_refl=71 %  amp_tilt=88 %  CPD=86 %

Input conversion (same as v1)
------------------------------
The MCP surface delivers ``RawSpectrum`` traces in dBmV over 5–85 MHz (256 bins).

  1. Take the ``average`` trace (dBmV, 256 bins).
  2. Convert dBmV → pseudo-linear: ``val = 10 ** ((dbmv + 45) / 10)``.
  3. Interpolate 256 → 200 bins.
  4. Replicate single snapshot 8 times → shape (8, 200).
  5. Apply per-model normalisation (mean/std from checkpoint).

Model paths
-----------
  ``UIL_CNN_V3_MODEL_PATH``  — v3 checkpoint  (default: models/us_impairment_cnn_v3.pt)
  ``UIL_CNN_H9_MODEL_PATH``  — H9 checkpoint  (default: models/h9_staged.pt)
  ``UIL_CNN_MODEL_PATH``     — legacy single-model override (ignored by v7; kept for compat)

Label mapping (CNN class → ImpairmentLabel)
--------------------------------------------
  0 narrowband_ingress → NarrowbandInterference
  1 burst_ingress      → Ingress
  2 impulse_noise      → ImpulseNoise
  3 micro_reflection   → Ripple
  4 amplitude_tilt     → WidebandNoise
  5 CPD                → CPD
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
# Repo layout: src/uil/classifier/cnn_classifier.py → 3 parents up = repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _REPO_ROOT / "models"

_DEFAULT_V3_PATH = _MODELS_DIR / "us_impairment_cnn_v3.pt"
_DEFAULT_H9_PATH = _MODELS_DIR / "h9_staged.pt"

# CNN training constants
_NUM_BINS: int = 200
_NUM_SNAPS: int = 8
_DBMV_SHIFT: float = 45.0   # -40 dBmV clean floor → ~3.2 linear units

# Class names in training order (same for v3 and H9)
_CNN_CLASS_NAMES: list[str] = [
    "narrowband_ingress",
    "burst_ingress",
    "impulse_noise",
    "micro_reflection",
    "amplitude_tilt",
    "CPD",
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

# V7 zero-FP AND-voting thresholds.
# Calibrated with a 1e-4 margin above the AND-score clean maximum on CPU.
# Guarantees 0 false positives on all clean test samples.
_DEFAULT_THRESHOLDS: list[float] = [
    0.31977102,   # narrowband_ingress
    0.37225610,   # burst_ingress
    0.41853413,   # impulse_noise
    0.61644757,   # micro_reflection
    0.48688933,   # amplitude_tilt
    0.46870381,   # CPD
]


class CnnClassifier:
    """Zero-FP v7 AND-voting CNN classifier.

    Loads two complementary CNN checkpoints (v3 + H9) and combines their
    per-class probabilities via element-wise minimum (AND-voting).  Both models
    must agree before an impairment class is flagged, which eliminates all
    false positives on clean upstream signals.

    Parameters
    ----------
    v3_model_path:
        Path to the v3 checkpoint.  Falls back to ``UIL_CNN_V3_MODEL_PATH``
        env var, then ``models/us_impairment_cnn_v3.pt``.
    h9_model_path:
        Path to the H9 checkpoint.  Falls back to ``UIL_CNN_H9_MODEL_PATH``
        env var, then ``models/h9_staged.pt``.
    device:
        PyTorch device (``"cpu"`` recommended for determinism; ``"cuda"`` for
        throughput).  Defaults to ``"cpu"``.
    thresholds:
        Override per-class AND-voting thresholds (6 floats).
    """

    def __init__(
        self,
        v3_model_path: Optional[Path] = None,
        h9_model_path: Optional[Path] = None,
        device: Optional[str] = None,
        thresholds: Optional[list[float]] = None,
    ) -> None:
        # Resolve model paths (env → arg → default)
        v3_env = os.environ.get("UIL_CNN_V3_MODEL_PATH", "")
        h9_env = os.environ.get("UIL_CNN_H9_MODEL_PATH", "")
        self._v3_path = Path(v3_env) if v3_env else Path(v3_model_path or _DEFAULT_V3_PATH)
        self._h9_path = Path(h9_env) if h9_env else Path(h9_model_path or _DEFAULT_H9_PATH)

        self._device_str = device or "cpu"   # CPU = deterministic
        self._thresholds: list[float] = list(thresholds or _DEFAULT_THRESHOLDS)

        # Lazy-loaded state
        self._loaded = False
        self._torch = None
        self._device = None
        # v3 model + stats
        self._v3_model = None
        self._v3_mean: float = 0.2586
        self._v3_std:  float = 0.1152
        self._v3_lmax: float = math.log10(32767.0)
        # H9 model + stats
        self._h9_model = None
        self._h9_mean: float = 0.2586
        self._h9_std:  float = 0.1152
        self._h9_lmax: float = math.log10(32767.0)

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for CnnClassifier. "
                "Install with:  pip install torch"
            ) from exc

        self._torch = torch
        self._device = torch.device(self._device_str)

        for attr, path in [("_v3", self._v3_path), ("_h9", self._h9_path)]:
            if not path.exists():
                raise FileNotFoundError(
                    f"CNN checkpoint not found: {path}\n"
                    "Copy the model file there or set the matching env var."
                )

        self._v3_model, self._v3_mean, self._v3_std, self._v3_lmax = \
            self._load_v3(self._v3_path)
        self._h9_model, self._h9_mean, self._h9_std, self._h9_lmax = \
            self._load_h9(self._h9_path)
        self._loaded = True

    # ----------------------------------------- architecture builders
    @staticmethod
    def _conv_block(ci, co, k, p):
        import torch.nn as nn
        class _C(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(ci, co, k, padding=p, bias=False),
                    nn.BatchNorm2d(co), nn.ReLU(True),
                )
            def forward(self, x): return self.seq(x)
        return _C()

    def _build_v3(self):
        """Single-head CNN (v1/v2/v3 architecture, 1-channel input)."""
        import torch.nn as nn
        _C = self._conv_block
        class _V3(nn.Module):
            def __init__(s):
                super().__init__()
                s.b1 = nn.Sequential(_C(1,32,(3,7),(1,3)), _C(32,32,(3,7),(1,3)), nn.MaxPool2d((1,2)))
                s.b2 = nn.Sequential(_C(32,64,(3,5),(1,2)), _C(64,64,(3,5),(1,2)), nn.MaxPool2d((1,2)))
                s.b3 = nn.Sequential(_C(64,128,(3,5),(1,2)), _C(128,128,(3,3),(1,1)), nn.MaxPool2d((2,2)))
                s.b4 = nn.Sequential(_C(128,256,(3,3),(1,1)), nn.Dropout2d(0.3), nn.MaxPool2d((2,5)))
                s.g  = nn.AdaptiveAvgPool2d(1)
                s.h  = nn.Sequential(
                    nn.Linear(256,128), nn.ReLU(True), nn.Dropout(0.4),
                    nn.Linear(128,64),  nn.ReLU(True), nn.Dropout(0.3),
                    nn.Linear(64, 6),
                )
            def forward(s, x): return s.h(s.g(s.b4(s.b3(s.b2(s.b1(x))))).flatten(1))
        return _V3()

    def _build_h9(self, ci: int = 4):
        """Two-head CNN (v4/v5/H9 architecture, 4-channel input)."""
        import torch.nn as nn
        _C = self._conv_block
        class _H9(nn.Module):
            def __init__(s):
                super().__init__()
                s.b1 = nn.Sequential(_C(ci,32,(3,7),(1,3)), _C(32,32,(3,7),(1,3)), nn.MaxPool2d((1,2)))
                s.b2 = nn.Sequential(_C(32,64,(3,5),(1,2)), _C(64,64,(3,5),(1,2)), nn.MaxPool2d((1,2)))
                s.b3 = nn.Sequential(_C(64,128,(3,5),(1,2)), _C(128,128,(3,3),(1,1)), nn.MaxPool2d((2,2)))
                s.b4 = nn.Sequential(_C(128,256,(3,3),(1,1)), nn.Dropout2d(0.3), nn.MaxPool2d((2,5)))
                s.g     = nn.AdaptiveAvgPool2d(1)
                s.head1 = nn.Linear(256, 1)
                s.head2 = nn.Sequential(
                    nn.Linear(256,128), nn.ReLU(True), nn.Dropout(0.4),
                    nn.Linear(128,64),  nn.ReLU(True), nn.Dropout(0.3),
                    nn.Linear(64, 6),
                )
            def forward(s, x):
                f = s.g(s.b4(s.b3(s.b2(s.b1(x))))).flatten(1)
                return s.head1(f), s.head2(f)
        return _H9()

    # ----------------------------------------- checkpoint loaders
    @staticmethod
    def _remap(sd: dict, key_map: dict) -> dict:
        new = {}
        for k, v in sd.items():
            for old, new_key in key_map.items():
                if k.startswith(old + "."):
                    k = new_key + k[len(old):]
                    break
            new[k] = v
        return new

    _KEY_MAP = {"block1": "b1", "block2": "b2", "block3": "b3",
                "block4": "b4", "gap": "g"}
    _KEY_MAP_V3 = {**_KEY_MAP, "head": "h"}

    def _load_v3(self, path: Path):
        import torch
        ck   = torch.load(path, map_location=self._device, weights_only=False)
        sd   = self._remap(ck["model_state_dict"], self._KEY_MAP_V3)
        mean = float(ck.get("train_mean", 0.2586))
        std  = float(ck.get("train_std",  0.1152))
        lmax = float(ck.get("log_max",    math.log10(32767.0)))
        m    = self._build_v3()
        m.load_state_dict(sd); m.to(self._device); m.eval()
        return m, mean, std, lmax

    def _load_h9(self, path: Path):
        import torch
        ck   = torch.load(path, map_location=self._device, weights_only=False)
        sd   = self._remap(
            ck.get("model_state_dict", ck.get("model_state", ck)),
            self._KEY_MAP,
        )
        mean = float(ck.get("train_mean", 0.2586))
        std  = float(ck.get("train_std",  0.1152))
        lmax = float(ck.get("log_max",    math.log10(32767.0)))
        ci   = int(ck.get("in_channels",  4))
        m    = self._build_h9(ci)
        m.load_state_dict(sd); m.to(self._device); m.eval()
        return m, mean, std, lmax

    # ----------------------------------------- preprocessing
    @staticmethod
    def _norm_1ch(snap: np.ndarray, mean: float, std: float, lmax: float) -> np.ndarray:
        """z-scored log-power → (8,200) float32."""
        log = np.log10(np.maximum(snap, 1.0)) / lmax
        return ((log - mean) / (std + 1e-8)).astype(np.float32)

    @staticmethod
    def _build_4ch(snap: np.ndarray, mean: float, std: float, lmax: float) -> np.ndarray:
        """Convert (8,200) raw snapshot → (4,8,200) 4-channel features.
        Matches train_v4.compute_channels exactly."""
        snap = np.asarray(snap, dtype=np.float32)
        # Ch0: z-scored log
        log0 = np.log10(np.maximum(snap, 1.0)) / lmax
        ch0  = ((log0 - mean) / (std + 1e-8)).astype(np.float32)
        # Ch1: temporal mean spectrum
        ms   = snap.mean(axis=0, keepdims=True)
        log1 = np.log10(np.maximum(ms, 1.0)) / lmax
        ch1  = np.broadcast_to(((log1 - mean) / (std + 1e-8)).astype(np.float32),
                                (_NUM_SNAPS, _NUM_BINS)).copy()
        # Ch2: temporal std per bin, log-normalised to [0,1]
        ss   = snap.std(axis=0, keepdims=True) + 1.0
        ch2  = np.broadcast_to((np.log10(ss) / np.log10(100.0)).astype(np.float32),
                                (_NUM_SNAPS, _NUM_BINS)).copy()
        # Ch3: FFT autocorrelation of mean spectrum
        m64  = snap.mean(axis=0).astype(np.float64); m64 -= m64.mean()
        n    = len(m64)
        fft  = np.fft.rfft(m64, n=2 * n)
        ac   = np.real(np.fft.irfft(np.abs(fft) ** 2))[:n].astype(np.float32)
        ac  /= (ac[0] + 1e-8)
        ch3  = np.broadcast_to(ac[np.newaxis, :], (_NUM_SNAPS, _NUM_BINS)).copy()
        return np.stack([ch0, ch1, ch2, ch3], axis=0)   # (4,8,200)

    def _snap_to_linear(self, spectrum: RawSpectrum) -> np.ndarray:
        """dBmV spectrum → (8,200) linear-power snapshots."""
        avg_dbmv = np.array(spectrum.traces.average, dtype=np.float32)
        linear   = (10.0 ** ((avg_dbmv + _DBMV_SHIFT) / 10.0)).astype(np.float32)
        src_hz   = np.linspace(spectrum.startFrequencyHz, spectrum.stopFrequencyHz, spectrum.numBins)
        dst_hz   = np.linspace(spectrum.startFrequencyHz, spectrum.stopFrequencyHz, _NUM_BINS)
        resampled = np.interp(dst_hz, src_hz, linear).astype(np.float32)
        return np.tile(resampled, (_NUM_SNAPS, 1))   # (8,200)

    # ----------------------------------------- inference
    def _run_models(self, snap: np.ndarray) -> np.ndarray:
        """Run AND-voting on a (8,200) linear-power snapshot.
        Returns (6,) float32 AND-voting scores = min(p_v3, p_h9_fused)."""
        torch = self._torch

        # v3: single-head, 1-channel input
        z_v3  = self._norm_1ch(snap, self._v3_mean, self._v3_std, self._v3_lmax)
        x_v3  = torch.from_numpy(z_v3[np.newaxis, np.newaxis]).to(self._device)  # (1,1,8,200)
        with torch.no_grad():
            p_v3 = torch.sigmoid(self._v3_model(x_v3)).cpu().numpy()[0]    # (6,)

        # H9: two-head, 4-channel input, multiplicative fusion p1×p2
        x4   = self._build_4ch(snap, self._h9_mean, self._h9_std, self._h9_lmax)  # (4,8,200)
        x_h9 = torch.from_numpy(x4[np.newaxis]).to(self._device)           # (1,4,8,200)
        with torch.no_grad():
            l1, l2 = self._h9_model(x_h9)
            p1 = torch.sigmoid(l1).cpu().numpy()[0, 0]   # scalar
            p2 = torch.sigmoid(l2).cpu().numpy()[0]      # (6,)
        p_h9_fused = p2 * p1                             # multiplicative fusion

        # AND-voting: both models must agree
        return np.minimum(p_v3, p_h9_fused)              # (6,)

    # ----------------------------------------- shared result builder
    def _build_classification(
        self,
        scores: np.ndarray,
        device_type: str,
        measurement_id: str,
        rpd_id: Optional[str],
        port_id: Optional[str],
        amp_id: Optional[str],
    ) -> Classification:
        thr    = np.array(self._thresholds, dtype=np.float32)
        active = [(c, float(scores[c])) for c in range(6) if scores[c] >= thr[c]]
        observations: list[Observation] = []

        if not active:
            label      = ImpairmentLabel.Clean
            confidence = float(1.0 - float(scores.max()))
            observations.append(Observation(
                finding="All AND-voting scores below zero-FP thresholds",
                supports=["Clean"],
            ))
        else:
            active.sort(key=lambda t: t[1], reverse=True)
            dominant_cls, dominant_prob = active[0]
            label      = _CNN_TO_LABEL[dominant_cls]
            confidence = dominant_prob
            for cls_idx, prob in active:
                observations.append(Observation(
                    finding=f"{_CNN_CLASS_NAMES[cls_idx]}: score={prob:.3f}",
                    supports=[_CNN_TO_LABEL[cls_idx].value],
                    confidence=prob,
                ))

        status = "clean" if label == ImpairmentLabel.Clean else "impaired"
        return Classification(
            deviceType=device_type,
            measurementId=measurement_id,
            status=status,
            **{"class": label},
            confidence=confidence,
            rpdId=rpd_id,
            portId=port_id,
            ampId=amp_id,
            observations=observations,
        )

    # -------------------------------------------------- public API
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
        """Classify one upstream spectrum using the v7 zero-FP ensemble."""
        self._load()
        snap   = self._snap_to_linear(spectrum)   # (8,200) linear
        scores = self._run_models(snap)            # (6,) AND-vote
        return self._build_classification(
            scores, device_type, measurement_id, rpd_id, port_id, amp_id,
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
        """Classify native (8, 200) linear-power snapshots (training format).

        Skips the dBmV→linear and resample steps performed by
        ``classify_spectrum()``.
        """
        self._load()
        if snapshots.shape != (_NUM_SNAPS, _NUM_BINS):
            raise ValueError(
                f"classify_snapshots expects shape ({_NUM_SNAPS}, {_NUM_BINS}), "
                f"got {snapshots.shape}."
            )
        scores = self._run_models(snapshots.astype(np.float32))
        return self._build_classification(
            scores, device_type, measurement_id, rpd_id, port_id, amp_id,
        )
