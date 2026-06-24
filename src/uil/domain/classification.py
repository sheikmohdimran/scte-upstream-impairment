"""Classification result — mirrors `_defs.schema.json#/$defs/classification`.

This is the raw (inline) classification produced by the classifier behind
``analyzeSpectrumMeasurements``. On the reference surface it is hidden behind a
``classificationSetRef`` handle.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from uil.domain.labels import ImpairmentLabel


class Observation(BaseModel):
    finding: str
    supports: list[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class Classification(BaseModel):
    """A single per-measurement classification."""

    # ``class`` is a Python keyword -> expose as ``klass`` with alias "class".
    model_config = ConfigDict(populate_by_name=True)

    deviceType: Literal["RPD", "AMP"]
    measurementId: str
    status: Literal["impaired", "clean"]
    klass: ImpairmentLabel = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    rpdId: Optional[str] = None
    portId: Optional[str] = None
    ampId: Optional[str] = None
    observations: list[Observation] = Field(default_factory=list)
