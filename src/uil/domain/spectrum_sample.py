"""Spectrum sample domain models for SpectrumSampleGenerator (T7/T8 tools).

Format: 8 time-snapshots × 200 frequency bins, linear power units (5-85 MHz).
This is the native training format for the v1 CNN — no conversion needed for
CnnClassifier.classify_snapshots().
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ImpairmentType(str, Enum):
    """Generator-level impairment types (snake_case, 7 values).

    Maps to ImpairmentLabel via _LABEL_TO_IMPAIRMENT_TYPE in
    spectrum_sample_generator.py. Used internally; callers of T7/T8 use
    the existing ImpairmentLabel CamelCase values.
    """

    clean              = "clean"
    cpd                = "cpd"
    ingress_narrowband = "ingress_narrowband"
    ingress_burst      = "ingress_burst"
    impulse_noise      = "impulse_noise"
    micro_reflection   = "micro_reflection"
    amplitude_tilt     = "amplitude_tilt"


class DeviceSpecification(BaseModel):
    """Internal device specification — not directly exposed to callers.

    Constructed by T7/T8 methods after prefix validation and label mapping.
    deviceType is set explicitly by the calling method (not inferred here).
    """

    deviceId:    str
    deviceType:  Literal["RPD", "AMP"]
    impairments: list[ImpairmentType] = Field(default_factory=lambda: [ImpairmentType.clean])
    severity:    Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SpectrumSample(BaseModel):
    """One spectrum sample: 8 snapshots × 200 bins, linear power units.

    Field names are camelCase to match the existing RawSpectrum convention
    so model_dump() produces JSON-ready keys without aliases.
    """

    deviceId:         str
    deviceType:       Literal["RPD", "AMP"]
    impairments:      list[str]           # ImpairmentType.value strings (snake_case)
    severity:         float
    snapshots:        list[list[float]]   # shape: 8 rows × 200 cols, linear power
    startFrequencyHz: int = 5_000_000
    stopFrequencyHz:  int = 85_000_000    # band edge; matches existing simulator convention
    numBins:          int = 200
    numSnapshots:     int = 8
    powerUnit:        Literal["linear"] = "linear"
    timestamp:        str

    @model_validator(mode="after")
    def _check_shape(self) -> "SpectrumSample":
        if len(self.snapshots) != self.numSnapshots:
            raise ValueError(
                f"snapshots has {len(self.snapshots)} rows, expected {self.numSnapshots}"
            )
        for i, row in enumerate(self.snapshots):
            if len(row) != self.numBins:
                raise ValueError(
                    f"snapshots[{i}] has {len(row)} columns, expected {self.numBins}"
                )
        return self


class GroupSpectrumResult(BaseModel):
    """Result of a batch generate_group() call."""

    runSeverity:  float
    deviceCount:  int
    samples:      list[SpectrumSample]
    generatedAt:  str
