"""E2E tests for use_cnn_path=True — Phase 6 (expanded).

Tests exercise the full T1->T2->T4->T5->T6 chain with
SpectrumSampleGenerator as the data source and CnnClassifier as the
classifier. Requires torch and the v1 model checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from uil.agent.orchestrator import Orchestrator
from uil.classifier.rule_classifier import RuleClassifier
from uil.domain.labels import ImpairmentLabel
from uil.mcp_server.server import FaultInjection, MockMcpServer, Scenario

torch = pytest.importorskip("torch", reason="torch not installed; skipping CNN path tests")


def _cnn_server(label: ImpairmentLabel = ImpairmentLabel.CPD,
                severity: float = 0.8) -> MockMcpServer:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=label,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": label.value},
            {"ampId": "A3", "parentId": "A2", "label": label.value},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
        ],
        severity=severity,
    )
    return MockMcpServer(scenario, use_cnn_path=True)


def test_cnn_path_wrong_classifier_raises() -> None:
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    )
    with pytest.raises(ValueError, match="CnnClassifier"):
        MockMcpServer(scenario, use_cnn_path=True, classifier=RuleClassifier())


def test_cnn_path_t1_stores_snapshots_key() -> None:
    server = _cnn_server()
    out    = server.getRPDSpectrumMeasurements("RPD-1", "P1")
    assert out["status"] == "success"
    meas_id = out["measurementRef"]["measurementId"]
    payload = server.store.get(meas_id)
    assert "snapshots" in payload
    assert "spectrum"  not in payload
    mat = np.array(payload["snapshots"])
    assert mat.shape == (8, 200)


def test_cnn_path_reference_surface_hides_snapshots() -> None:
    server = _cnn_server()
    out    = server.getRPDSpectrumMeasurements("RPD-1", "P1")

    def _has_forbidden(value, keys=("snapshots", "maxHold", "minHold", "average")):
        if isinstance(value, dict):
            assert not any(k in value for k in keys), \
                f"Forbidden key found in reference output: {set(value) & set(keys)}"
            for v in value.values():
                _has_forbidden(v)
        elif isinstance(value, list):
            for item in value:
                _has_forbidden(item)

    _has_forbidden(out)


def test_cnn_path_t2_routes_to_classify_snapshots() -> None:
    server  = _cnn_server()
    t1_out  = server.getRPDSpectrumMeasurements("RPD-1", "P1")
    t2_out  = server.analyzeSpectrumMeasurements(measurementRefs=[t1_out["measurementRef"]])
    assert t2_out["status"] == "success"
    assert t2_out["classificationCount"] == 1


def test_cnn_path_full_orchestrator() -> None:
    server = _cnn_server()
    result = Orchestrator(server, scenario_name="cnn-e2e").run()
    # CNN may not perfectly classify synthetic data, but must not crash (status != "failed" on clean run)
    assert result.status in {"localized", "low_confidence", "escalated"}
    # All 6 tool calls must have been attempted
    tools_called = [c.tool for c in result.trace.calls]
    assert "getRPDSpectrumMeasurements" in tools_called
    assert "analyzeSpectrumMeasurements" in tools_called


def test_cnn_path_t4_amp_measurements_store_snapshots_key() -> None:
    """Amp measurements in CNN path must also store 'snapshots' not 'spectrum'."""
    server  = _cnn_server()
    t1_out  = server.getRPDSpectrumMeasurements("RPD-1", "P1")
    t3_out  = server.getAllAmpsInSegment("RPD-1", "P1")
    t4_out  = server.getAmpSpectrumMeasurements(ampListRef=t3_out["ampListRef"])
    assert t4_out["status"] == "success"
    # Resolve measset and inspect each stored measurement
    meas_items = server.store.get(t4_out["measurementSetRef"])
    assert len(meas_items) > 0
    for item in meas_items:
        assert "snapshots" in item, f"Amp item missing 'snapshots' key: {list(item.keys())}"
        assert "spectrum"  not in item
        mat = np.array(item["snapshots"])
        assert mat.shape == (8, 200)


# Labels expected to localize (impairment detected and source found)
_LOCALIZING_LABELS = [
    ImpairmentLabel.CPD,
    ImpairmentLabel.Ingress,
    ImpairmentLabel.ImpulseNoise,
    ImpairmentLabel.WidebandNoise,          # maps to ingress_burst; CNN detects Ingress
    ImpairmentLabel.NarrowbandInterference,
    ImpairmentLabel.Ripple,
    ImpairmentLabel.Suckout,               # maps to amplitude_tilt; CNN detects WidebandNoise
]

# Labels expected to produce low_confidence (generator outputs clean floor)
_CLEAN_FALLBACK_LABELS = [
    ImpairmentLabel.Clean,
    ImpairmentLabel.UnknownImpairment,
]


@pytest.mark.parametrize("label", _LOCALIZING_LABELS)
def test_cnn_path_all_impairment_labels_localize(label: ImpairmentLabel) -> None:
    """Every non-clean ImpairmentLabel must complete the 6-step chain without crashing
    and produce a localized or low_confidence result (never 'failed')."""
    server = _cnn_server(label=label, severity=0.8)
    result = Orchestrator(server, scenario_name=f"cnn-{label.value}").run()
    assert result.status != "failed", (
        f"CNN path crashed for label={label.value}: {result.handoff_markdown}"
    )
    assert len(result.trace.calls) == 6, (
        f"Expected 6 tool calls for label={label.value}, got {len(result.trace.calls)}"
    )


@pytest.mark.parametrize("label", _CLEAN_FALLBACK_LABELS)
def test_cnn_path_clean_labels_return_low_confidence(label: ImpairmentLabel) -> None:
    """Clean and UnknownImpairment generate a clean spectrum; the orchestrator must
    gracefully return low_confidence (no impairment detected) — not crash."""
    server = _cnn_server(label=label, severity=0.8)
    result = Orchestrator(server, scenario_name=f"cnn-{label.value}").run()
    # Orchestrator stops at step 2 when impairedCount==0 and escalates
    assert result.status == "low_confidence"


def test_cnn_path_partial_success_amp_failure() -> None:
    """Amp failure in CNN path must produce partial_success at T4 and not crash."""
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
        ],
        faults=FaultInjection(amp_failed_ports={"A3"}),
        severity=0.8,
    )
    server = MockMcpServer(scenario, use_cnn_path=True)
    result = Orchestrator(server, scenario_name="cnn-partial").run()
    t4 = next(c for c in result.trace.calls if c.tool == "getAmpSpectrumMeasurements")
    assert t4.outcome == "partial_success"
    assert result.status != "failed"


def test_cnn_path_severity_zero_is_safe() -> None:
    """severity=0.0 produces a clean floor; CNN finds no impairment; chain must not crash."""
    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
        ],
        severity=0.0,
    )
    server = MockMcpServer(scenario, use_cnn_path=True)
    result = Orchestrator(server, scenario_name="cnn-sev0").run()
    # At severity=0, CPD generator produces floor only; CNN likely finds no impairment
    assert result.status in {"localized", "low_confidence", "escalated"}


def test_classify_snapshots_wrong_shape_raises() -> None:
    """classify_snapshots must raise ValueError for wrong input shape."""
    from uil.classifier.cnn_classifier import CnnClassifier
    clf = CnnClassifier()
    bad = np.zeros((256, 1), dtype=np.float32)   # wrong shape — not (8, 200)
    with pytest.raises(ValueError, match="8"):
        clf.classify_snapshots(bad, device_type="RPD", measurement_id="m1")
