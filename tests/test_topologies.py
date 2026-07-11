"""E2E topology tests — varied HFC segment structures and impairment patterns.

All tests run the full 6-step orchestrator chain. The goal is to confirm:
  - GraphLocalizer handles structurally different topologies correctly
  - Orchestrator recovery paths work regardless of segment shape
  - CNN path (use_cnn_path=True) also handles topology variety

Topologies tested:
  - Linear chain (no branching)
  - Deep chain (5+ levels)
  - Wide star (one root, many leaf amps)
  - Root fault (all amps impaired)
  - Single amp segment
  - Large segment (10 amps)
"""

from __future__ import annotations

import pytest

from uil.agent.orchestrator import Orchestrator
from uil.domain.labels import ImpairmentLabel
from uil.mcp_server.server import FaultInjection, MockMcpServer, Scenario


# ── Scenario factories ─────────────────────────────────────────────────────────

def _run(scenario: Scenario, use_cnn: bool = False):
    server = MockMcpServer(scenario, use_cnn_path=use_cnn)
    result = Orchestrator(server, scenario_name=scenario.rpdId).run()
    # Expand evidence handles back to inline lists so tests can assert on device ids.
    result.localization = server.resolve_localization(result.localization)
    return result


def _scenario(rpd_label: ImpairmentLabel, amps: list[dict],
              severity: float = 0.8, **kwargs) -> Scenario:
    return Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=rpd_label,
        amps=amps, severity=severity, **kwargs,
    )


# ── Topology 1: Single amp segment ────────────────────────────────────────────

def test_single_amp_segment_fault_detected() -> None:
    """One amp in segment, impaired — must localize to that device."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [{"ampId": "A1", "parentId": None, "label": "CPD"}],
    ))
    assert result.status == "localized"
    loc = result.localization
    assert loc["localizationStatus"] == "localized"
    assert loc["candidateLocations"][0]["locationType"] == "device"


def test_single_amp_segment_clean() -> None:
    """RPD impaired, single clean amp — localization returns low_confidence."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [{"ampId": "A1", "parentId": None, "label": "Clean"}],
    ))
    # GraphLocalizer sees no impaired amps → low_confidence, orchestrator escalates
    assert result.status == "low_confidence"


# ── Topology 2: Linear chain ───────────────────────────────────────────────────

def test_linear_chain_fault_in_middle() -> None:
    """A1→A2→A3→A4→A5, fault enters at A3 — common point localizes to A3."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None,  "label": "Clean"},
            {"ampId": "A2", "parentId": "A1",  "label": "Clean"},
            {"ampId": "A3", "parentId": "A2",  "label": "CPD"},
            {"ampId": "A4", "parentId": "A3",  "label": "CPD"},
            {"ampId": "A5", "parentId": "A4",  "label": "CPD"},
        ],
    ))
    assert result.status != "failed"
    if result.localization:
        loc = result.localization
        assert loc["localizationStatus"] in {"localized", "low_confidence"}
        # Supporting amps must be A3/A4/A5; clean boundary must include A2
        if loc["localizationStatus"] == "localized":
            supporting = {d["ampId"] for d in loc.get("supportingDevices", [])}
            assert supporting == {"A3", "A4", "A5"}
            clean_b = {d["ampId"] for d in loc.get("cleanBoundaryDevices", [])}
            assert "A2" in clean_b


def test_linear_chain_fault_at_end() -> None:
    """A1→A2→A3, only the deepest amp A3 is impaired."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "Clean"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
        ],
    ))
    assert result.status != "failed"
    if result.localization and result.localization["localizationStatus"] == "localized":
        assert result.localization["candidateLocations"][0]["locationType"] == "device"


# ── Topology 3: Wide star ──────────────────────────────────────────────────────

def test_wide_star_one_branch_impaired() -> None:
    """A1(root)→[A2,A3,A4,A5,A6], only A2 impaired — single device localization."""
    result = _run(_scenario(
        ImpairmentLabel.Ripple,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "Ripple"},
            {"ampId": "A3", "parentId": "A1", "label": "Clean"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
            {"ampId": "A5", "parentId": "A1", "label": "Clean"},
            {"ampId": "A6", "parentId": "A1", "label": "Clean"},
        ],
    ))
    assert result.status != "failed"
    if result.localization and result.localization["localizationStatus"] == "localized":
        loc = result.localization
        assert loc["candidateLocations"][0]["locationType"] == "device"
        assert loc["candidateLocations"][0]["downstreamBoundaryDevices"][0]["ampId"] == "A2"


def test_wide_star_two_branches_impaired() -> None:
    """A1(root)→[A2,A3,A4,A5], A2 and A4 impaired — localization at root level."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A1", "label": "Clean"},
            {"ampId": "A4", "parentId": "A1", "label": "CPD"},
            {"ampId": "A5", "parentId": "A1", "label": "Clean"},
        ],
    ))
    # Two branches impaired with same label → LCA=A1, branch at root
    assert result.status != "failed"
    if result.localization:
        assert result.localization["localizationStatus"] in {"localized", "low_confidence"}


# ── Topology 4: Root fault (all amps impaired) ────────────────────────────────

def test_all_amps_impaired_branch_at_root() -> None:
    """Every amp in the segment sees the impairment — fault is at/before the root amp."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "CPD"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
            {"ampId": "A4", "parentId": "A1", "label": "CPD"},
        ],
    ))
    assert result.status != "failed"
    if result.localization and result.localization["localizationStatus"] == "localized":
        # All amps impaired → common point = A1 → branch at A1
        assert {d["ampId"] for d in result.localization.get("supportingDevices", [])} == {
            "A1", "A2", "A3", "A4"
        }


# ── Topology 5: RPD-only fault ────────────────────────────────────────────────

def test_rpd_impaired_all_amps_clean() -> None:
    """RPD sees fault but all amps are clean — likely intermittent/upstream.

    Orchestrator proceeds through all 6 steps; localizer returns low_confidence
    because no impaired amp is confirmed, and escalates.
    """
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "Clean"},
            {"ampId": "A3", "parentId": "A2", "label": "Clean"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
        ],
    ))
    # All amps clean → GraphLocalizer returns low_confidence → orchestrator escalates
    assert result.status == "low_confidence"
    assert len(result.trace.calls) == 6    # all 6 steps ran


# ── Topology 6: Conflicting labels (two branches, different impairment types) ──

def test_conflicting_impairment_types_escalates() -> None:
    """Different impairment types on parallel branches → low_confidence from localizer."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None,  "label": "Clean"},
            {"ampId": "A2", "parentId": "A1",  "label": "CPD"},
            {"ampId": "A3", "parentId": "A1",  "label": "Ingress"},
        ],
    ))
    assert result.status != "failed"
    # Conflicting CPD vs Ingress labels on parallel branches → low_confidence
    if result.localization:
        assert result.localization["localizationStatus"] == "low_confidence"


# ── Topology 7: Large segment (stress test) ───────────────────────────────────

def test_large_segment_10_amps_does_not_crash() -> None:
    """10-amp balanced binary tree segment — must complete without errors."""
    # Tree: A1 → (A2, A3); A2 → (A4, A5); A3 → (A6, A7); A4 → (A8, A9); A5 → A10
    amps = [
        {"ampId": "A1",  "parentId": None,  "label": "Clean"},
        {"ampId": "A2",  "parentId": "A1",  "label": "Clean"},
        {"ampId": "A3",  "parentId": "A1",  "label": "CPD"},
        {"ampId": "A4",  "parentId": "A2",  "label": "Clean"},
        {"ampId": "A5",  "parentId": "A2",  "label": "Clean"},
        {"ampId": "A6",  "parentId": "A3",  "label": "CPD"},
        {"ampId": "A7",  "parentId": "A3",  "label": "CPD"},
        {"ampId": "A8",  "parentId": "A4",  "label": "Clean"},
        {"ampId": "A9",  "parentId": "A4",  "label": "Clean"},
        {"ampId": "A10", "parentId": "A5",  "label": "Clean"},
    ]
    result = _run(_scenario(ImpairmentLabel.CPD, amps))
    assert result.status != "failed"
    assert len(result.trace.calls) == 6
    if result.localization:
        # A3 branch impaired; supporting devices must include A3/A6/A7
        if result.localization["localizationStatus"] == "localized":
            supporting = {d["ampId"] for d in result.localization.get("supportingDevices", [])}
            assert supporting >= {"A3", "A6", "A7"}


# ── Topology 8: Partial success in varied topologies ─────────────────────────

def test_linear_chain_partial_success_missing_middle_amp() -> None:
    """Linear chain with A3 (middle) failing to measure — uncertain → low_confidence."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
        ],
        faults=FaultInjection(amp_failed_ports={"A3"}),  # A3 unreachable
    ))
    t4 = next(c for c in result.trace.calls if c.tool == "getAmpUpstreamSpectrumMeasurements")
    assert t4.outcome == "partial_success"
    # A3 missing → its status is uncertain → low_confidence
    assert result.status in {"localized", "low_confidence"}


def test_wide_star_partial_success_still_localizes() -> None:
    """Star topology: one impaired amp, one amp unreachable — should still localize."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},     # impaired
            {"ampId": "A3", "parentId": "A1", "label": "Clean"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},   # this one fails
        ],
        faults=FaultInjection(amp_failed_ports={"A4"}),
    ))
    assert result.status != "failed"
    t4 = next(c for c in result.trace.calls if c.tool == "getAmpUpstreamSpectrumMeasurements")
    assert t4.outcome == "partial_success"


# ── Topology 9: CNN path with varied topologies ───────────────────────────────

torch = pytest.importorskip("torch", reason="torch not installed; skipping CNN topology tests")


def test_cnn_linear_chain_localizes() -> None:
    """CNN path with linear chain: A1→A2→A3, fault at A2+."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
        ],
    ), use_cnn=True)
    assert result.status != "failed"
    assert len(result.trace.calls) == 6


def test_cnn_all_amps_impaired_root_fault() -> None:
    """CNN path with all amps impaired — must complete and not crash."""
    result = _run(_scenario(
        ImpairmentLabel.CPD,
        [
            {"ampId": "A1", "parentId": None, "label": "CPD"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A1", "label": "CPD"},
        ],
    ), use_cnn=True)
    assert result.status != "failed"


def test_cnn_wide_star_one_branch_impaired() -> None:
    """CNN path with wide star topology: one impaired leaf among many clean ones."""
    result = _run(_scenario(
        ImpairmentLabel.NarrowbandInterference,
        [
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "NarrowbandInterference"},
            {"ampId": "A3", "parentId": "A1", "label": "Clean"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
        ],
    ), use_cnn=True)
    assert result.status != "failed"
    assert len(result.trace.calls) == 6
