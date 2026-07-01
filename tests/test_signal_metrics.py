"""Tests for getSignalMetrics (T8) — Phase 6 (6 tests)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, RefResolver

from uil.domain.labels import ImpairmentLabel
from uil.mcp_server.server import MockMcpServer, Scenario

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas" / "tools"


def _server(severity: float = 1.0) -> MockMcpServer:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
        severity=severity,
    )
    return MockMcpServer(scenario)


def _defs_resolver() -> RefResolver:
    defs  = json.loads((SCHEMA_DIR / "_defs.schema.json").read_text())
    store = {
        "../_defs.schema.json": defs,
        "_defs.schema.json":    defs,
        defs.get("$id", "_defs.schema.json"): defs,
    }
    return RefResolver(base_uri="", referrer=defs, store=store)


def test_deterministic() -> None:
    server = _server()
    r1 = server.getSignalMetrics(modemId="rpd-001", windowSec=60)
    # fresh server, same inputs
    r2 = _server().getSignalMetrics(modemId="rpd-001", windowSec=60)
    assert r1["upstreamTxPower"]    == r2["upstreamTxPower"]
    assert r1["downstreamSnr"]      == r2["downstreamSnr"]
    assert r1["uncorrectablesRate"] == r2["uncorrectablesRate"]
    assert r1["observationSummary"] == r2["observationSummary"]


def test_severity_changes_output() -> None:
    r_lo = _server(severity=0.1).getSignalMetrics(modemId="rpd-001")
    r_hi = _server(severity=0.9).getSignalMetrics(modemId="rpd-001")
    # Different severity → different md5 seed → different scalar values
    assert r_lo["observationSummary"] != r_hi["observationSummary"]


def test_reference_hides_snapshots() -> None:
    out = _server().getSignalMetrics(modemId="rpd-001")
    assert "snapshots" not in out
    assert "spectrum"  not in out


def test_raw_artifact_resolvable_8x200() -> None:
    server = _server()
    out    = server.getSignalMetrics(modemId="rpd-001", windowSec=60)
    assert out["status"] == "success"
    sample = server.store.get(out["rawArtifactRef"])
    assert len(sample.snapshots) == 8
    assert all(len(row) == 200 for row in sample.snapshots)


def test_field_ranges() -> None:
    out = _server().getSignalMetrics(modemId="rpd-001", windowSec=60)
    assert 35.0 <= out["upstreamTxPower"]    <= 55.0
    assert 30.0 <= out["downstreamSnr"]      <= 50.0
    assert  0.0 <= out["uncorrectablesRate"] <= 1.0


def test_schema_conformance_t8_reference() -> None:
    schema_doc = json.loads(
        (SCHEMA_DIR / "reference" / "getSignalMetrics.schema.json").read_text()
    )
    ref_schema = schema_doc["outputSchema"]
    out = _server().getSignalMetrics(modemId="rpd-001", windowSec=60)
    Draft202012Validator(ref_schema, resolver=_defs_resolver()).validate(out)


def test_invalid_device_id_returns_error() -> None:
    out = _server().getSignalMetrics(modemId="cm-001")
    assert out["status"] == "error"
    assert out["errorCode"] == "INVALID_DEVICE_ID"


def test_window_sec_variation_changes_output() -> None:
    s1 = _server().getSignalMetrics(modemId="rpd-001", windowSec=60)
    s2 = _server().getSignalMetrics(modemId="rpd-001", windowSec=3600)
    # Different windowSec → different md5 seed → different scalars
    assert s1["observationSummary"] != s2["observationSummary"]


def test_schema_conformance_t8_raw() -> None:
    """SpectrumSample serialised against raw/getSignalMetrics schema."""
    defs       = json.loads((SCHEMA_DIR / "_defs.schema.json").read_text())
    schema_doc = json.loads((SCHEMA_DIR / "raw" / "getSignalMetrics.schema.json").read_text())
    raw_schema = schema_doc["outputSchema"]
    store_map  = {
        "../_defs.schema.json": defs,
        "_defs.schema.json":    defs,
        defs.get("$id", "_defs.schema.json"): defs,
    }
    resolver = RefResolver(base_uri="", referrer=defs, store=store_map)

    server  = _server()
    ref_out = server.getSignalMetrics(modemId="rpd-001", windowSec=60)
    sample  = server.store.get(ref_out["rawArtifactRef"])
    raw_out = {
        "status":             "success",
        "modemId":            "rpd-001",
        "windowSec":          60,
        "upstreamTxPower":    ref_out["upstreamTxPower"],
        "downstreamSnr":      ref_out["downstreamSnr"],
        "uncorrectablesRate": ref_out["uncorrectablesRate"],
        "observationSummary": ref_out["observationSummary"],
        "spectrum":           sample.model_dump(),
    }
    Draft202012Validator(raw_schema, resolver=resolver).validate(raw_out)
