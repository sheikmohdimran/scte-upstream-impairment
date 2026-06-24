"""Step 0 — schema conformance.

Validates that:
  1. every vendored `.schema.json` is itself a valid JSON Schema, and
  2. our Pydantic domain models produce instances that validate against the relevant
     `$defs` in `_defs.schema.json` (round-trip), and
  3. representative tool outputs from the mock server validate against the tool
     reference `outputSchema`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, RefResolver

from uil.domain.classification import Classification
from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum import RawSpectrum, SpectrumTraces

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas" / "tools"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _defs_resolver() -> RefResolver:
    defs = _load(SCHEMA_DIR / "_defs.schema.json")
    # Map both the relative ref used in tool files and the $id to the defs doc.
    store = {
        "../_defs.schema.json": defs,
        "_defs.schema.json": defs,
        defs.get("$id", "_defs.schema.json"): defs,
    }
    return RefResolver(base_uri="", referrer=defs, store=store)


@pytest.mark.parametrize("schema_path", sorted(SCHEMA_DIR.rglob("*.schema.json")))
def test_schema_is_valid_jsonschema(schema_path: Path) -> None:
    Draft202012Validator.check_schema(_load(schema_path))


def test_rawspectrum_validates_against_defs() -> None:
    defs = _load(SCHEMA_DIR / "_defs.schema.json")
    schema = {"$ref": "#/$defs/rawSpectrum", "$defs": defs["$defs"]}
    spec = RawSpectrum(
        startFrequencyHz=5_000_000, stopFrequencyHz=85_000_000, numBins=4,
        traces=SpectrumTraces(maxHold=[1, 2, 3, 4], minHold=[0, 0, 0, 0], average=[0.5, 1, 1.5, 2]),
    )
    Draft202012Validator(schema).validate(spec.model_dump())


def test_classification_validates_against_defs() -> None:
    defs = _load(SCHEMA_DIR / "_defs.schema.json")
    schema = {"$ref": "#/$defs/classification", "$defs": defs["$defs"]}
    c = Classification(
        deviceType="AMP", measurementId="meas-1", status="impaired",
        **{"class": ImpairmentLabel.CPD}, confidence=0.91, ampId="A2",
    )
    # by_alias so "class" key is emitted; exclude_none so optional ids stay absent.
    Draft202012Validator(schema).validate(c.model_dump(by_alias=True, mode="json", exclude_none=True))


def test_rpd_tool_success_output_conforms(server) -> None:
    schema = _load(SCHEMA_DIR / "reference" / "getRPDSpectrumMeasurements.schema.json")["outputSchema"]
    out = server.getRPDSpectrumMeasurements("RPD-1", "P1")
    Draft202012Validator(schema, resolver=_defs_resolver()).validate(out)


def test_analyze_tool_success_output_conforms(server) -> None:
    schema = _load(SCHEMA_DIR / "reference" / "analyzeSpectrumMeasurements.schema.json")["outputSchema"]
    rpd = server.getRPDSpectrumMeasurements("RPD-1", "P1")
    out = server.analyzeSpectrumMeasurements(measurementRefs=[rpd["measurementRef"]])
    Draft202012Validator(schema, resolver=_defs_resolver()).validate(out)


def test_amp_measurements_partial_success_contains_failed_refs(server) -> None:
    schema = _load(SCHEMA_DIR / "reference" / "getAmpSpectrumMeasurements.schema.json")["outputSchema"]
    amps = server.getAllAmpsInSegment("RPD-1", "P1")
    # Force a partial-success path by failing one known amp.
    server.scn.faults.amp_failed_ports = {"A4"}
    out = server.getAmpSpectrumMeasurements(ampListRef=amps["ampListRef"])
    Draft202012Validator(schema, resolver=_defs_resolver()).validate(out)
    assert out["status"] == "partial_success"
    assert out["failedCount"] == 1
    assert "failedDevicesRef" in out


def test_reference_outputs_never_include_raw_spectrum_arrays(server) -> None:
    out = server.getRPDSpectrumMeasurements("RPD-1", "P1")
    text = str(out)
    assert "maxHold" not in text
    assert "minHold" not in text
    assert "average" not in text
