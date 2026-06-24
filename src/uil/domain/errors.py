"""Error response — mirrors `_defs.schema.json#/$defs/errorResponse`.

Per-tool error code enums are defined here so the orchestrator can branch on them.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Per-tool error code sets (from each tool's reference outputSchema).
RPD_SPECTRUM_ERRORS = {
    "MEASUREMENT_UNAVAILABLE",
    "DEVICE_UNREACHABLE",
    "INVALID_RPD_PORT",
    "STALE_DATA_ONLY",
    "PERMISSION_DENIED",
}
ANALYZE_ERRORS = {
    "CLASSIFICATION_FAILED",
    "INVALID_MEASUREMENT_REF",
    "MEASUREMENT_TOO_STALE",
    "UNSUPPORTED_MEASUREMENT_TYPE",
    "INSUFFICIENT_SIGNAL_QUALITY",
}
AMPS_IN_SEGMENT_ERRORS = {
    "TOPOLOGY_UNAVAILABLE",
    "INVALID_RPD_PORT",
    "EMPTY_SEGMENT",
    "STALE_TOPOLOGY",
    "PERMISSION_DENIED",
}
AMP_SPECTRUM_ERRORS = {
    "AMP_MEASUREMENTS_UNAVAILABLE",
    "DEVICE_UNREACHABLE",
    "INVALID_AMP_ID",
    "INVALID_AMP_LIST_REF",
    "TOO_MANY_DEVICES_REQUESTED",
    "STALE_DATA_ONLY",
    "PERMISSION_DENIED",
}
LOCALIZE_ERRORS = {
    "INSUFFICIENT_LOCALIZATION_EVIDENCE",
    "TOPOLOGY_UNAVAILABLE",
    "INVALID_CLASSIFICATION_SET_REF",
    "NO_IMPAIRMENT_CONFIRMED",
    "CONFLICTING_CLASSIFICATIONS",
    "UNSUPPORTED_IMPAIRMENT_TYPE",
}


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    errorCode: str
    message: Optional[str] = None
    # Only present for localize INVALID_CLASSIFICATION_SET_REF.
    invalidClassificationSetRefs: Optional[list[str]] = Field(default=None, min_length=1)
