"""Orchestrator (Step 5) — the SLM's job, as a deterministic state machine for now.

Runs the fixed 6-step sequence against the mock MCP server, holding ONLY handles + counts
in state (mirroring the reference surface). It records a :class:`ToolCallTrace` (Step 8) and,
on ambiguity/unrecoverable error, emits a human-handoff summary (Step 7).

A real LangGraph + local-SLM implementation can replace `run()` while keeping the same
tool calls and recovery policy; the decision points below are exactly what the SLM will own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from uil.agent.handoff import build_handoff_summary
from uil.agent.trace import ToolCallTrace
from uil.mcp_server.server import MockMcpServer


@dataclass
class OrchestratorResult:
    status: str  # localized | low_confidence | escalated | failed
    localization: Optional[dict[str, Any]] = None
    handoff_markdown: Optional[str] = None
    trace: ToolCallTrace = field(default_factory=lambda: ToolCallTrace(scenario=""))


class Orchestrator:
    def __init__(self, server: MockMcpServer, scenario_name: str = "scenario", max_remeasure: int = 1) -> None:
        self.s = server
        self.name = scenario_name
        self.max_remeasure = max_remeasure

    def run(self) -> OrchestratorResult:
        trace = ToolCallTrace(scenario=self.name)
        rpd_id, port_id = self.s.scn.rpdId, self.s.scn.portId

        # --- Step 1: RPD spectrum (with bounded re-measure on transient errors) ---
        rpd_meas_ref = None
        for attempt in range(self.max_remeasure + 1):
            r1 = self.s.getRPDSpectrumMeasurements(rpd_id, port_id)
            outcome = "success" if r1.get("status") == "success" else "error"
            trace.add(tool="getRPDSpectrumMeasurements", arguments={"rpdId": rpd_id, "portId": port_id},
                      result=r1, outcome=outcome,
                      decision="Capture RPD upstream spectrum to confirm the impairment.")
            if outcome == "success":
                rpd_meas_ref = r1["measurementRef"]
                break
            if r1.get("errorCode") not in {"STALE_DATA_ONLY", "MEASUREMENT_UNAVAILABLE"}:
                break  # non-transient -> stop retrying
        if rpd_meas_ref is None:
            return self._escalate(trace, "RPD spectrum capture failed (non-recoverable).", "failed")

        # --- Step 2: classify RPD (kept for step 6) ---
        r2 = self.s.analyzeSpectrumMeasurements(measurementRefs=[rpd_meas_ref])
        trace.add(tool="analyzeSpectrumMeasurements", arguments={"measurementRefs": [rpd_meas_ref["measurementId"]]},
                  result=r2, outcome=r2.get("status", "error"),
                  decision="Classify the RPD capture; keep its classificationSetRef for localization.")
        if r2.get("status") != "success":
            return self._escalate(trace, "RPD classification failed.", "failed")
        rpd_class_ref = r2["classificationSetRef"]
        if r2.get("impairedCount", 0) == 0:
            return self._escalate(trace, "RPD shows clean — no impairment to localize (possibly intermittent).",
                                  "low_confidence")
        impairment_type = next((c for c in r2.get("classesPresent", []) if c != "Clean"), "UnknownImpairment")

        # --- Step 3: enumerate amps ---
        r3 = self.s.getAllAmpsInSegment(rpd_id, port_id)
        trace.add(tool="getAllAmpsInSegment", arguments={"rpdId": rpd_id, "portId": port_id},
                  result=r3, outcome=r3.get("status", "error"),
                  decision="List every amp in the leg (one-shot, not recursive).")
        if r3.get("status") != "success":
            return self._escalate(trace, "Topology unavailable; cannot localize within the segment.", "failed")
        amp_list_ref = r3["ampListRef"]

        # --- Step 4: measure all amp spectra (tolerate partial_success) ---
        r4 = self.s.getAmpSpectrumMeasurements(ampListRef=amp_list_ref)
        trace.add(tool="getAmpSpectrumMeasurements", arguments={"ampListRef": amp_list_ref},
                  result=r4, outcome=r4.get("status", "error"),
                  decision="Capture spectra at all amp legs at once via the list handle.")
        if r4.get("status") not in {"success", "partial_success"}:
            return self._escalate(trace, "Amp spectrum capture failed for the whole segment.", "failed")
        amp_set_ref = r4["measurementSetRef"]
        if r4.get("failedCount", 0):
            # proceed but note: missing legs become uncertainDevices in localization
            trace.calls[-1].decision += f" {r4['failedCount']} leg(s) failed — proceeding; they become uncertain."

        # --- Step 5: classify amps ---
        r5 = self.s.analyzeSpectrumMeasurements(measurementSetRef=amp_set_ref)
        trace.add(tool="analyzeSpectrumMeasurements", arguments={"measurementSetRef": amp_set_ref},
                  result=r5, outcome=r5.get("status", "error"),
                  decision="Classify the amp measurement set in one call.")
        if r5.get("status") != "success":
            return self._escalate(trace, "Amp classification failed.", "failed")
        amp_class_ref = r5["classificationSetRef"]

        # --- Step 6: localize ---
        r6 = self.s.localizeUpstreamSpectrumImpairmentSource(
            rpd_id, port_id, impairment_type, [rpd_class_ref, amp_class_ref])
        outcome = "success" if r6.get("status") == "success" else "error"
        trace.add(tool="localizeUpstreamSpectrumImpairmentSource",
                  arguments={"rpdId": rpd_id, "portId": port_id, "impairmentType": impairment_type,
                             "classificationSetRefs": [rpd_class_ref, amp_class_ref]},
                  result=r6, outcome=outcome,
                  decision="Pool RPD + amp classifications and run common-point localization.")
        if outcome != "success":
            return self._escalate(trace, f"Localization error: {r6.get('errorCode')}.", "failed", r6)

        loc_status = r6.get("localizationStatus")
        if loc_status == "low_confidence":
            return self._escalate(trace, "Localization is low-confidence (multi-anomaly or missing data).",
                                  "low_confidence", r6)
        trace.final_status = "localized"
        return OrchestratorResult(status="localized", localization=r6, trace=trace)

    def _escalate(self, trace: ToolCallTrace, reason: str, status: str,
                  localization: Optional[dict] = None) -> OrchestratorResult:
        trace.final_status = "escalated" if status != "failed" else "failed"
        trace.label = "negative"
        md = build_handoff_summary(self.name, trace, reason, localization)
        return OrchestratorResult(status=status, localization=localization, handoff_markdown=md, trace=trace)


def _demo() -> None:
    """Minimal end-to-end CPD demo with one injected partial_success (Step 7)."""
    from uil.domain.labels import ImpairmentLabel
    from uil.mcp_server.server import FaultInjection, Scenario

    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},      # closest to RPD
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},        # fault enters here
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},        # downstream, also sees it
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},      # other branch, clean
        ],
        faults=FaultInjection(amp_failed_ports={"A4"}),  # force a partial_success
    )
    server = MockMcpServer(scenario)
    result = Orchestrator(server, scenario_name="demo-CPD-partial").run()
    print(f"STATUS: {result.status}")
    if result.localization:
        loc = result.localization
        print(f"impairment: {loc.get('impairmentType')} | localizationStatus: {loc.get('localizationStatus')} "
              f"| confidence: {loc.get('confidence')}")
        if loc.get("likelySourceLocation"):
            print("likely source:", loc["likelySourceLocation"]["description"])
    if result.handoff_markdown:
        print("\n--- HUMAN HANDOFF ---\n" + result.handoff_markdown)
    print("\n--- TRACE (Step 8) ---")
    print(result.trace.to_jsonl())


if __name__ == "__main__":
    _demo()
