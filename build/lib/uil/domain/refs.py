"""Device + measurement references — mirrors `_defs.schema.json`.

``DeviceRef`` is a discriminated union (RPD vs AMP). ``MeasurementRef`` is the
reference-surface handle returned by ``getRPDSpectrumMeasurements`` and carried in
classifications.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class RpdDeviceRef(BaseModel):
    deviceType: Literal["RPD"] = "RPD"
    rpdId: str
    portId: str

    model_config = {"extra": "forbid"}


class AmpDeviceRef(BaseModel):
    deviceType: Literal["AMP"] = "AMP"
    ampId: str
    portId: Optional[str] = Field(
        default=None, description="Amp output leg. Omit to reference the whole amp."
    )

    model_config = {"extra": "forbid"}


DeviceRef = Union[RpdDeviceRef, AmpDeviceRef]


class MeasurementRef(BaseModel):
    """Handle to a stored spectrum measurement (RPD port or amp leg)."""

    measurementId: str
    deviceType: Literal["RPD", "AMP"]
    measurementType: Literal["upstream_spectrum"] = "upstream_spectrum"
    timestamp: str
    rpdId: Optional[str] = None
    portId: Optional[str] = None
    ampId: Optional[str] = None
    validUntil: Optional[str] = None
