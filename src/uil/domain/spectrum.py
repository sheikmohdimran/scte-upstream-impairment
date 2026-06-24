"""Raw upstream spectrum — mirrors `_defs.schema.json#/$defs/rawSpectrum`.

The SLM never sees these payloads; they live behind the reference-surface handles.
Each trace array has length == ``numBins``. Bin ``i`` center frequency is::

    startFrequencyHz + (i + 0.5) * (stopFrequencyHz - startFrequencyHz) / numBins
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SpectrumTraces(BaseModel):
    """Per-bin power traces. All three arrays have length == numBins."""

    maxHold: list[float]
    minHold: list[float]
    average: list[float]


class RawSpectrum(BaseModel):
    """A single upstream spectrum capture."""

    startFrequencyHz: int = Field(description="Low edge of the captured band.")
    stopFrequencyHz: int = Field(description="High edge of the captured band.")
    numBins: int = Field(ge=1)
    traces: SpectrumTraces

    @model_validator(mode="after")
    def _check_lengths(self) -> "RawSpectrum":
        for name in ("maxHold", "minHold", "average"):
            arr = getattr(self.traces, name)
            if len(arr) != self.numBins:
                raise ValueError(
                    f"traces.{name} length {len(arr)} != numBins {self.numBins}"
                )
        if self.stopFrequencyHz <= self.startFrequencyHz:
            raise ValueError("stopFrequencyHz must be > startFrequencyHz")
        return self

    def bin_center_hz(self, i: int) -> float:
        """Center frequency of bin ``i``."""
        span = self.stopFrequencyHz - self.startFrequencyHz
        return self.startFrequencyHz + (i + 0.5) * span / self.numBins
