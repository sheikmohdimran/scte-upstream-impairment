"""Impairment labels — mirrors `_defs.schema.json#/$defs/impairmentLabel`."""

from enum import Enum


class ImpairmentLabel(str, Enum):
    """The 9 upstream impairment classes the classifier can emit."""

    Clean = "Clean"
    CPD = "CPD"
    Ingress = "Ingress"
    ImpulseNoise = "ImpulseNoise"
    WidebandNoise = "WidebandNoise"
    NarrowbandInterference = "NarrowbandInterference"
    Ripple = "Ripple"
    Suckout = "Suckout"
    UnknownImpairment = "UnknownImpairment"
