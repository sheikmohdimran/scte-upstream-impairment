"""Step 5/7/8 — end-to-end orchestration, error recovery, traces."""

from uil.agent.orchestrator import Orchestrator
from uil.domain.labels import ImpairmentLabel
from uil.mcp_server.server import FaultInjection, MockMcpServer, Scenario


def _scenario(**faults) -> Scenario:
    return Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
        ],
        faults=FaultInjection(**faults),
    )


def test_happy_path_localizes_cpd() -> None:
    res = Orchestrator(MockMcpServer(_scenario()), "happy").run()
    assert res.status == "localized"
    assert res.localization["impairmentType"] == "CPD"
    # 6 tool calls on the happy path.
    assert len(res.trace.calls) == 6
    assert res.trace.final_status == "localized"


def test_partial_success_still_completes() -> None:
    res = Orchestrator(MockMcpServer(_scenario(amp_failed_ports={"A4"})), "partial").run()
    # A4 fails to measure but is on the clean branch; localization still proceeds.
    amp_call = next(c for c in res.trace.calls if c.tool == "getAmpSpectrumMeasurements")
    assert amp_call.outcome == "partial_success"
    assert res.status in {"localized", "low_confidence"}


def test_rpd_unreachable_escalates_with_handoff() -> None:
    res = Orchestrator(MockMcpServer(_scenario(rpd_unreachable=True)), "unreachable").run()
    assert res.status == "failed"
    assert res.handoff_markdown is not None
    assert "Diagnosis Summary" in res.handoff_markdown
    assert res.trace.label == "negative"


def test_stale_rpd_triggers_remeasure_then_recovers_or_escalates() -> None:
    # STALE_DATA_ONLY is transient -> orchestrator retries up to max_remeasure.
    server = MockMcpServer(_scenario(rpd_stale_only=True))
    res = Orchestrator(server, "stale", max_remeasure=1).run()
    rpd_calls = [c for c in res.trace.calls if c.tool == "getRPDSpectrumMeasurements"]
    assert len(rpd_calls) == 2  # original + 1 re-measure
    assert res.status == "failed"


def test_measurement_unavailable_once_retries_and_completes() -> None:
    server = MockMcpServer(_scenario(rpd_measurement_unavailable_once=True))
    res = Orchestrator(server, "measurement-unavailable", max_remeasure=1).run()
    rpd_calls = [c for c in res.trace.calls if c.tool == "getRPDSpectrumMeasurements"]
    assert len(rpd_calls) == 2
    assert res.status in {"localized", "low_confidence"}


def test_handoff_contains_required_sections() -> None:
    res = Orchestrator(MockMcpServer(_scenario(rpd_unreachable=True)), "handoff-sections").run()
    assert res.handoff_markdown is not None
    assert "## Diagnosis Summary" in res.handoff_markdown
    assert "## Evidence" in res.handoff_markdown
    assert "## Recommended Next Action" in res.handoff_markdown


def test_trace_is_serializable_and_holds_only_handles() -> None:
    res = Orchestrator(MockMcpServer(_scenario()), "trace").run()
    js = res.trace.to_jsonl()
    assert "measurementRef" in js or "classificationSetRef" in js
    # raw trace arrays must never leak into the SLM-facing trace
    assert "maxHold" not in js and "average" not in js
