"""Localization output — mirrors the symbolic output of
`reference/localizeUpstreamSpectrumImpairmentSource.schema.json`.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from uil.domain.labels import ImpairmentLabel
from uil.domain.refs import DeviceRef

LocationType = Literal["span", "device", "branch"]


class LikelySourceLocation(BaseModel):
    locationType: LocationType
    description: str
    upstreamBoundaryDevice: Optional[DeviceRef] = None
    downstreamBoundaryDevice: Optional[DeviceRef] = None


class CandidateLocation(BaseModel):
    locationType: LocationType
    description: str
    score: float = Field(ge=0, le=1)
    reason: Optional[str] = None


class RecommendedNextAction(BaseModel):
    action: str
    reason: Optional[str] = None
    targetDevices: list[str] = Field(default_factory=list)


class LocalizationResult(BaseModel):
    """Symbolic localization result (status == 'success')."""

    status: Literal["success"] = "success"
    impairmentType: ImpairmentLabel
    localizationStatus: Literal["localized", "low_confidence"]
    confidence: float = Field(ge=0, le=1)
    candidateLocations: list[CandidateLocation]
    likelySourceLocation: Optional[LikelySourceLocation] = None
    supportingDevices: list[DeviceRef] = Field(default_factory=list)
    cleanBoundaryDevices: list[DeviceRef] = Field(default_factory=list)
    uncertainDevices: list[DeviceRef] = Field(default_factory=list)
    recommendedNextAction: Optional[RecommendedNextAction] = None
