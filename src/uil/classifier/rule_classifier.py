"""Bootstrap rule classifier (Step 4).

Transparent noise-floor heuristic so the full tool chain runs end-to-end BEFORE the CNN
(`analyzeSpectrumMeasurements` real backend) is available. This is scaffolding, not the
final architecture: the production classifier is a multi-class CNN over the spectrum, and
the SLM never classifies.

Heuristic (provisional, matches the simulator's CPD model):
  * Compute the median of the lower upstream band (5-42 MHz) average trace.
  * If it sits well above the nominal clean floor AND shows periodic comb peaks -> CPD.
  * A flat raised floor without comb -> WidebandNoise; a single sharp peak ->
    NarrowbandInterference; a localized raised block -> Ingress; otherwise Clean.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from uil.domain.classification import Classification, Observation
from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum import RawSpectrum

CLEAN_FLOOR_DBMV = -40.0
IMPAIRED_MARGIN_DB = 4.0  # average lift above clean floor that counts as impaired


class RuleClassifier:
    def __init__(self, clean_floor_dbmv: float = CLEAN_FLOOR_DBMV):
        self.clean_floor = clean_floor_dbmv

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
        avg = np.array(spectrum.traces.average)
        freqs = np.array([spectrum.bin_center_hz(i) for i in range(spectrum.numBins)])

        low = avg[freqs <= 42_000_000]
        high = avg[freqs > 42_000_000]
        low_lift = (float(np.median(low)) if low.size else self.clean_floor) - self.clean_floor
        high_lift = (float(np.median(high)) if high.size else self.clean_floor) - self.clean_floor
        peak = float(np.max(avg) - np.median(avg))
        n_raised = int(np.sum(avg - self.clean_floor > IMPAIRED_MARGIN_DB))

        label = ImpairmentLabel.Clean
        confidence = 0.9
        observations: list[Observation] = []

        if peak > 12.0 and n_raised <= 3:
            # one or two very strong bins above the floor
            label = ImpairmentLabel.NarrowbandInterference
            confidence = 0.85
            observations.append(Observation(finding=f"Single sharp peak (+{peak:.1f} dB)", supports=["NarrowbandInterference"]))
        elif low_lift >= IMPAIRED_MARGIN_DB and high_lift < IMPAIRED_MARGIN_DB:
            # raised noise floor concentrated in the low upstream band -> CPD
            label = ImpairmentLabel.CPD
            confidence = min(0.99, 0.7 + low_lift / 30.0)
            observations.append(Observation(finding=f"Raised low-band floor (+{low_lift:.1f} dB), high band nominal", supports=["CPD"], confidence=confidence))
        elif low_lift >= IMPAIRED_MARGIN_DB and high_lift >= IMPAIRED_MARGIN_DB:
            label = ImpairmentLabel.WidebandNoise
            confidence = 0.75
            observations.append(Observation(finding=f"Whole-band floor raised (low +{low_lift:.1f}, high +{high_lift:.1f} dB)", supports=["WidebandNoise"]))
        elif n_raised >= 4:
            # a localized block of bins lifted without moving the band median
            label = ImpairmentLabel.Ingress
            confidence = 0.75
            observations.append(Observation(finding=f"Localized raised block over {n_raised} bins", supports=["Ingress"]))
        else:
            observations.append(Observation(finding=f"Floor within {max(low_lift, high_lift):.1f} dB of nominal", supports=["Clean"]))

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
