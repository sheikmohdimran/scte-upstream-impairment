"""Upstream spectrum simulator (Step 3).

Generates realistic-ish upstream ``RawSpectrum`` payloads (5-85 MHz, 256 bins) for the
``Clean`` baseline and the primary ``CPD`` anomaly (raised noise floor + discrete
distortion products). Other labels are stubbed so the classifier interface is exercised.

The numbers come from ``config/cpd_signatures.yaml`` and are PROVISIONAL until Irene/Bhaskar
confirm the real CPD signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum import RawSpectrum, SpectrumTraces

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@dataclass
class SimConfig:
    start_hz: int
    stop_hz: int
    num_bins: int
    cpd: dict
    clean: dict
    ingress: dict
    narrowband: dict

    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> "SimConfig":
        config_dir = config_dir or _DEFAULT_CONFIG_DIR
        cap = yaml.safe_load((config_dir / "capture_defaults.yaml").read_text())["capture"]
        sig = yaml.safe_load((config_dir / "cpd_signatures.yaml").read_text())
        return cls(
            start_hz=cap["start_frequency_hz"],
            stop_hz=cap["stop_frequency_hz"],
            num_bins=cap["num_bins"],
            cpd=sig["cpd"],
            clean=sig["clean"],
            ingress=sig["ingress"],
            narrowband=sig["narrowband_interference"],
        )


class SpectrumSimulator:
    """Deterministic (seeded) synthetic upstream spectrum generator."""

    def __init__(self, config: Optional[SimConfig] = None, seed: int = 42):
        self.cfg = config or SimConfig.load()
        self.rng = np.random.RandomState(seed)

    # ---- helpers --------------------------------------------------------
    def _freqs(self) -> np.ndarray:
        span = self.cfg.stop_hz - self.cfg.start_hz
        idx = np.arange(self.cfg.num_bins)
        return self.cfg.start_hz + (idx + 0.5) * span / self.cfg.num_bins

    def _baseline(self, severity: float = 1.0) -> np.ndarray:
        floor = self.cfg.clean["noise_floor_dbmv"]
        var = self.cfg.clean["variance_db"] * severity
        return floor + self.rng.normal(0, var, self.cfg.num_bins)

    def _to_spectrum(self, avg: np.ndarray) -> RawSpectrum:
        # max/min hold bracket the average with realistic spread.
        spread = np.abs(self.rng.normal(2.0, 0.5, self.cfg.num_bins))
        max_hold = avg + spread
        min_hold = avg - np.abs(self.rng.normal(1.0, 0.3, self.cfg.num_bins))
        return RawSpectrum(
            startFrequencyHz=self.cfg.start_hz,
            stopFrequencyHz=self.cfg.stop_hz,
            numBins=self.cfg.num_bins,
            traces=SpectrumTraces(
                maxHold=[round(float(x), 3) for x in max_hold],
                minHold=[round(float(x), 3) for x in min_hold],
                average=[round(float(x), 3) for x in avg],
            ),
        )

    # ---- public API -----------------------------------------------------
    def generate(
        self,
        label: ImpairmentLabel = ImpairmentLabel.Clean,
        severity: float = 1.0,
    ) -> RawSpectrum:
        """Generate one spectrum for ``label``."""
        avg = self._baseline()
        freqs = self._freqs()

        if label == ImpairmentLabel.Clean:
            pass
        elif label == ImpairmentLabel.CPD:
            c = self.cfg.cpd
            in_band = (freqs >= c["band_start_hz"]) & (freqs <= c["band_stop_hz"])
            jitter = self.rng.normal(0, c["severity_jitter_db"])
            avg[in_band] += (c["floor_rise_db"] * severity) + jitter
            # discrete distortion-product comb
            f = c["band_start_hz"]
            while f <= c["band_stop_hz"]:
                bin_i = int((f - self.cfg.start_hz) / (self.cfg.stop_hz - self.cfg.start_hz) * self.cfg.num_bins)
                if 0 <= bin_i < self.cfg.num_bins:
                    avg[bin_i] += c["comb_peak_db"] * severity
                f += c["comb_spacing_hz"]
        elif label == ImpairmentLabel.Ingress:
            g = self.cfg.ingress
            in_band = (freqs >= g["band_start_hz"]) & (freqs <= g["band_stop_hz"])
            avg[in_band] += g["level_db"] * severity
        elif label == ImpairmentLabel.NarrowbandInterference:
            n = self.cfg.narrowband
            in_band = np.abs(freqs - n["center_hz"]) <= n["width_hz"]
            avg[in_band] += n["level_db"] * severity
        else:
            # generic raised-energy stub for the remaining labels
            avg += self.rng.uniform(3, 6) * severity

        return self._to_spectrum(avg)
