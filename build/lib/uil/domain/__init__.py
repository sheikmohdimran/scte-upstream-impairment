"""Domain models — Pydantic v2 mirrors of `schemas/tools/_defs.schema.json`.

These models are the typed contract used across the simulator, classifier, localizer,
mock MCP server and the orchestrator. They are validated against the JSON schemas in
`tests/test_schema_conformance.py`.
"""

from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum import RawSpectrum, SpectrumTraces
from uil.domain.refs import DeviceRef, MeasurementRef
from uil.domain.classification import Classification, Observation
from uil.domain.localization import (
    CandidateLocation,
    LikelySourceLocation,
    LocalizationResult,
    RecommendedNextAction,
)
from uil.domain.errors import ErrorResponse

__all__ = [
    "ImpairmentLabel",
    "RawSpectrum",
    "SpectrumTraces",
    "DeviceRef",
    "MeasurementRef",
    "Classification",
    "Observation",
    "CandidateLocation",
    "LikelySourceLocation",
    "LocalizationResult",
    "RecommendedNextAction",
    "ErrorResponse",
]
