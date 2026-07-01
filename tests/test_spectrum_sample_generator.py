"""Tests for SpectrumSampleGenerator — Phase 6 (10 tests)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, RefResolver

from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum_sample import DeviceSpecification, ImpairmentType, SpectrumSample
from uil.mcp_server.server import MockMcpServer, Scenario
from uil.sim.spectrum_sample_generator import (
    NUM_BINS,
    T_SNAPS,
    _LABEL_TO_IMPAIRMENT_TYPE,
    SpectrumSampleGenerator,
)

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas" / "tools"


def _generator() -> SpectrumSampleGenerator:
    return SpectrumSampleGenerator()


def _spec(imp: ImpairmentType, sev: float | None = None) -> DeviceSpecification:
    return DeviceSpecification(
        deviceId="rpd-001", deviceType="RPD",
        impairments=[imp], severity=sev,
    )


@pytest.mark.parametrize("imp", list(ImpairmentType))
def test_all_7_types_produce_8x200(imp: ImpairmentType) -> None:
    sample = _generator().generate(_spec(imp), run_severity=0.5)
    assert len(sample.snapshots) == T_SNAPS
    assert all(len(row) == NUM_BINS for row in sample.snapshots)


def test_deterministic_same_inputs() -> None:
    gen = _generator()
    spec = _spec(ImpairmentType.cpd, sev=0.7)
    s1 = gen.generate(spec, run_severity=0.7)
    s2 = gen.generate(spec, run_severity=0.7)
    assert s1.snapshots == s2.snapshots


def test_different_severity_differs() -> None:
    gen = _generator()
    s_lo = gen.generate(DeviceSpecification(deviceId="rpd-001", deviceType="RPD",
                                             impairments=[ImpairmentType.cpd], severity=0.1),
                        run_severity=0.1)
    s_hi = gen.generate(DeviceSpecification(deviceId="rpd-001", deviceType="RPD",
                                             impairments=[ImpairmentType.cpd], severity=0.9),
                        run_severity=0.9)
    assert s_lo.snapshots != s_hi.snapshots


def test_per_device_severity_overrides_run() -> None:
    gen = _generator()
    # device severity=0.1 should override run_severity=1.0
    spec_override = DeviceSpecification(deviceId="rpd-002", deviceType="RPD",
                                         impairments=[ImpairmentType.cpd], severity=0.1)
    spec_run      = DeviceSpecification(deviceId="rpd-002", deviceType="RPD",
                                         impairments=[ImpairmentType.cpd], severity=None)
    s_override = gen.generate(spec_override, run_severity=1.0)
    s_run_low  = gen.generate(spec_run, run_severity=0.1)
    # Both should use effective severity=0.1 and produce the same output
    assert s_override.snapshots == s_run_low.snapshots


def test_group_preserves_order_and_count() -> None:
    gen   = _generator()
    specs = [
        DeviceSpecification(deviceId=f"rpd-{i:03d}", deviceType="RPD",
                             impairments=[ImpairmentType.clean])
        for i in range(3)
    ]
    result = gen.generate_group(specs, run_severity=1.0, max_devices=50)
    assert result.deviceCount == 3
    assert len(result.samples) == 3
    for i, s in enumerate(result.samples):
        assert s.deviceId == f"rpd-{i:03d}"


def test_group_max_devices_enforced() -> None:
    gen   = _generator()
    specs = [
        DeviceSpecification(deviceId=f"rpd-{i:03d}", deviceType="RPD",
                             impairments=[ImpairmentType.clean])
        for i in range(51)
    ]
    with pytest.raises(ValueError, match="exceeds limit"):
        gen.generate_group(specs, run_severity=1.0, max_devices=50)


def test_snapshots_in_valid_range() -> None:
    gen    = _generator()
    sample = gen.generate(_spec(ImpairmentType.cpd, sev=1.0), run_severity=1.0)
    mat    = np.array(sample.snapshots)
    assert mat.min() >= 0.0
    assert mat.max() <= 32767.0


def test_label_bridge_cpd() -> None:
    assert _LABEL_TO_IMPAIRMENT_TYPE[ImpairmentLabel.CPD] == ImpairmentType.cpd


def test_label_bridge_all_9_labels() -> None:
    for label in ImpairmentLabel:
        assert label in _LABEL_TO_IMPAIRMENT_TYPE, f"Missing bridge for {label}"


def test_amp_prefix_produces_amp_device_type() -> None:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    )
    server = MockMcpServer(scenario)
    out    = server.getDeviceSpectrumSamples(devices=[{"deviceId": "amp-042"}])
    assert out["status"] == "success"
    result = server.store.get(out["sampleSetRef"])
    assert result.samples[0].deviceType == "AMP"


def test_invalid_device_id_prefix_returns_error() -> None:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    )
    server = MockMcpServer(scenario)
    out    = server.getDeviceSpectrumSamples(devices=[{"deviceId": "cm-001"}])
    assert out["status"] == "error"
    assert out["errorCode"] == "INVALID_DEVICE_ID"


def test_too_many_devices_returns_error() -> None:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
        max_devices=3,
    )
    server  = MockMcpServer(scenario)
    devices = [{"deviceId": f"rpd-{i:03d}"} for i in range(4)]
    out     = server.getDeviceSpectrumSamples(devices=devices)
    assert out["status"] == "error"
    assert out["errorCode"] == "TOO_MANY_DEVICES"


def test_max_devices_boundary_exactly_at_limit_succeeds() -> None:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
        max_devices=5,
    )
    server  = MockMcpServer(scenario)
    devices = [{"deviceId": f"rpd-{i:03d}"} for i in range(5)]   # exactly at limit
    out     = server.getDeviceSpectrumSamples(devices=devices)
    assert out["status"] == "success"
    assert out["deviceCount"] == 5


def test_severity_zero_produces_floor_only() -> None:
    spec   = DeviceSpecification(deviceId="rpd-z", deviceType="RPD",
                                  impairments=[ImpairmentType.cpd], severity=0.0)
    sample = _generator().generate(spec, run_severity=0.0)
    mat    = np.array(sample.snapshots)
    # At sev=0, CPD adds 0 * floor_rise → result is floor only
    assert mat.mean() < 20.0      # near floor (~8), not elevated like real CPD


def test_spectrum_sample_shape_validator_rejects_wrong_rows() -> None:
    with pytest.raises(Exception, match="8"):
        SpectrumSample(
            deviceId="rpd-bad", deviceType="RPD",
            impairments=["clean"], severity=1.0,
            snapshots=[[0.0] * 200] * 5,          # 5 rows instead of 8
            startFrequencyHz=5_000_000, stopFrequencyHz=85_000_000,
            numBins=200, numSnapshots=8, powerUnit="linear",
            timestamp="2026-07-01T00:00:00+00:00",
        )


def test_spectrum_sample_shape_validator_rejects_wrong_cols() -> None:
    with pytest.raises(Exception, match="100"):
        SpectrumSample(
            deviceId="rpd-bad", deviceType="RPD",
            impairments=["clean"], severity=1.0,
            snapshots=[[0.0] * 100] * 8,          # 100 cols instead of 200
            startFrequencyHz=5_000_000, stopFrequencyHz=85_000_000,
            numBins=200, numSnapshots=8, powerUnit="linear",
            timestamp="2026-07-01T00:00:00+00:00",
        )


def test_invalid_impairment_label_returns_error() -> None:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    )
    server = MockMcpServer(scenario)
    out    = server.getDeviceSpectrumSamples(devices=[
        {"deviceId": "rpd-001", "impairments": ["NotALabel"]}
    ])
    assert out["status"] == "error"
    assert out["errorCode"] == "INVALID_IMPAIRMENT"


def test_multi_label_device_shape() -> None:
    """Two impairments on one device — combined synthesis must still return 8x200."""
    spec = DeviceSpecification(
        deviceId="rpd-multi", deviceType="RPD",
        impairments=[ImpairmentType.cpd, ImpairmentType.micro_reflection],
    )
    sample = _generator().generate(spec, run_severity=0.7)
    assert len(sample.snapshots) == T_SNAPS
    assert all(len(row) == NUM_BINS for row in sample.snapshots)
    assert set(sample.impairments) == {"cpd", "micro_reflection"}


def test_schema_conformance_t7_raw() -> None:
    """GroupSpectrumResult serialised against raw/getDeviceSpectrumSamples schema."""
    defs       = json.loads((SCHEMA_DIR / "_defs.schema.json").read_text())
    schema_doc = json.loads((SCHEMA_DIR / "raw" / "getDeviceSpectrumSamples.schema.json").read_text())
    raw_schema = schema_doc["outputSchema"]
    store = {
        "../_defs.schema.json": defs,
        "_defs.schema.json":    defs,
        defs.get("$id", "_defs.schema.json"): defs,
    }
    resolver = RefResolver(base_uri="", referrer=defs, store=store)

    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    )
    server = MockMcpServer(scenario)
    ref_out = server.getDeviceSpectrumSamples(devices=[
        {"deviceId": "rpd-001", "impairments": ["CPD"]},
    ])
    result  = server.store.get(ref_out["sampleSetRef"])
    raw_out = {
        "status":      "success",
        "runSeverity": result.runSeverity,
        "deviceCount": result.deviceCount,
        "samples":     [s.model_dump() for s in result.samples],
    }
    Draft202012Validator(raw_schema, resolver=resolver).validate(raw_out)


def test_schema_conformance_t7_reference() -> None:
    schema_path = SCHEMA_DIR / "reference" / "getDeviceSpectrumSamples.schema.json"
    defs        = json.loads((SCHEMA_DIR / "_defs.schema.json").read_text())
    schema_doc  = json.loads(schema_path.read_text())
    ref_schema  = schema_doc["outputSchema"]
    store = {
        "../_defs.schema.json": defs,
        "_defs.schema.json":    defs,
        defs.get("$id", "_defs.schema.json"): defs,
    }
    resolver = RefResolver(base_uri="", referrer=defs, store=store)

    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    )
    server = MockMcpServer(scenario)
    out    = server.getDeviceSpectrumSamples(devices=[
        {"deviceId": "rpd-001", "impairments": ["CPD"]},
        {"deviceId": "amp-001", "impairments": ["Clean"]},
    ])
    assert out["status"] == "success"
    Draft202012Validator(ref_schema, resolver=resolver).validate(out)
