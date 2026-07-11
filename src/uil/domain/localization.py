"""Localization output.

Two layers:

* The **internal** localizer result (:class:`LocalizationResult` and friends) is what
  :class:`~uil.localizer.graph_localizer.GraphLocalizer` produces. It keeps the impairment
  evidence inline (``supportingDevices`` etc.) and a single ``likelySourceLocation`` so the
  deterministic localizer stays easy to test and reason about.
* The **public** MCP contract (:class:`PublicLocalizationResult`) mirrors
  ``reference/localizeUpstreamSpectrumImpairmentSource.schema.json`` — the CableLabs shape.
  Each candidate carries measurable ``upstreamBoundaryDevice`` /
  ``downstreamBoundaryDevices`` refs, and the evidence device lists move **behind opaque
  handles** (``supportingDevicesRef`` / ``cleanBoundaryDevicesRef`` / ``uncertainDevicesRef``)
  so the SLM never holds the inline lists. :func:`build_public_localization` converts the
  internal result into this contract, creating the handles via an injected callback.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from uil.domain.labels import ImpairmentLabel
from uil.domain.refs import DeviceRef, RpdDeviceRef

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
    """Internal localizer result (evidence inline). NOT the public wire contract."""

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


# --- Public MCP contract (CableLabs shape) --------------------------------------------------
class BoundaryCandidate(BaseModel):
    """One ``candidateLocations[]`` item of the public localize output.

    Boundaries are measurable RPD/AMP device refs. The impairment evidence lives behind
    opaque backend handles instead of inline device arrays.
    """

    locationType: LocationType
    score: float = Field(ge=0, le=1)
    upstreamBoundaryDevice: DeviceRef
    downstreamBoundaryDevices: list[DeviceRef] = Field(min_length=1)
    supportingDevicesRef: Optional[str] = None
    cleanBoundaryDevicesRef: Optional[str] = None
    uncertainDevicesRef: Optional[str] = None
    description: Optional[str] = None
    reason: Optional[str] = None


class PublicLocalizationResult(BaseModel):
    """Public ``localizeUpstreamSpectrumImpairmentSource`` success output (status == 'success')."""

    status: Literal["success"] = "success"
    impairmentType: ImpairmentLabel
    localizationStatus: Literal["localized", "low_confidence"]
    confidence: float = Field(ge=0, le=1)
    candidateLocations: list[BoundaryCandidate] = Field(default_factory=list)


def build_public_localization(
    rpd_id: str,
    port_id: str,
    result: LocalizationResult,
    put_handle: Callable[[str, object], str],
) -> PublicLocalizationResult:
    """Convert an internal :class:`LocalizationResult` into the public MCP contract.

    ``put_handle(prefix, value) -> handle`` stores an evidence device list in the backend and
    returns the opaque handle placed in the candidate's ``*Ref`` field. When no downstream
    boundary can be established (e.g. no impaired device confirmed) an empty
    ``candidateLocations`` list is returned, which the schema permits.
    """
    supporting = list(result.supportingDevices)
    clean = list(result.cleanBoundaryDevices)
    uncertain = list(result.uncertainDevices)
    likely = result.likelySourceLocation

    downstream: list[DeviceRef] = list(supporting)
    if not downstream and likely is not None and likely.downstreamBoundaryDevice is not None:
        downstream = [likely.downstreamBoundaryDevice]

    candidates: list[BoundaryCandidate] = []
    if downstream:
        upstream: DeviceRef = (
            (likely.upstreamBoundaryDevice if likely is not None and likely.upstreamBoundaryDevice is not None else None)
            or (clean[0] if clean else None)
            or RpdDeviceRef(rpdId=rpd_id, portId=port_id)
        )
        location_type: LocationType = (
            likely.locationType if likely is not None
            else (result.candidateLocations[0].locationType if result.candidateLocations else "branch")
        )
        description = (
            likely.description if likely is not None
            else (result.candidateLocations[0].description if result.candidateLocations else None)
        )
        candidates.append(BoundaryCandidate(
            locationType=location_type,
            score=round(result.confidence, 2),
            upstreamBoundaryDevice=upstream,
            downstreamBoundaryDevices=downstream,
            supportingDevicesRef=put_handle("supportdev", [d.model_dump(exclude_none=True) for d in supporting]) if supporting else None,
            cleanBoundaryDevicesRef=put_handle("cleandev", [d.model_dump(exclude_none=True) for d in clean]) if clean else None,
            uncertainDevicesRef=put_handle("uncertaindev", [d.model_dump(exclude_none=True) for d in uncertain]) if uncertain else None,
            description=description,
            reason=result.candidateLocations[0].reason if result.candidateLocations else None,
        ))

    return PublicLocalizationResult(
        impairmentType=result.impairmentType,
        localizationStatus=result.localizationStatus,
        confidence=round(result.confidence, 2),
        candidateLocations=candidates,
    )

