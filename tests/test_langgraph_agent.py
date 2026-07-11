"""Phase 3 tests: LangGraph agent wiring + SLM-facing toolset, no live SLM required.

The agent is driven by a small *scripted* tool-calling model so the full LangGraph loop runs
deterministically offline. A real OpenAI-compatible SLM is a drop-in for the scripted model.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from uil.agent.langgraph_agent import LangGraphAgent, McpToolset, build_tools
from uil.agent.trace import ToolCallTrace
from uil.mcp_server.server import MockMcpServer, Scenario


def test_toolset_surface_flows_handles(cpd_scenario: Scenario) -> None:
    """The SLM-facing toolset moves only string handles and captures localization."""
    server = MockMcpServer(cpd_scenario)
    trace = ToolCallTrace(scenario="t")
    ts = McpToolset(server, trace)

    r1 = json.loads(ts.get_rpd_spectrum())
    assert r1["status"] == "success" and isinstance(r1["measurementId"], str)

    r2 = json.loads(ts.analyze_spectrum(measurementIds=[r1["measurementId"]]))
    assert r2["impairedCount"] == 1 and "CPD" in r2["classesPresent"]
    rpd_set = r2["classificationSetRef"]

    r3 = json.loads(ts.get_all_amps())
    r4 = json.loads(ts.get_amp_spectra(ampListRef=r3["ampListRef"]))
    assert r4["status"] == "success"

    r5 = json.loads(ts.analyze_spectrum(measurementSetRef=r4["measurementSetRef"]))
    amp_set = r5["classificationSetRef"]

    r6 = json.loads(ts.localize("CPD", [rpd_set, amp_set]))
    assert r6["impairmentType"] == "CPD"
    assert r6["localizationStatus"] == "localized"
    assert ts.last_localization == r6
    assert len(trace.calls) == 6


def test_build_tools_exposes_five_named_tools(cpd_scenario: Scenario) -> None:
    ts = McpToolset(MockMcpServer(cpd_scenario), ToolCallTrace(scenario="t"))
    tools = build_tools(ts)
    names = {t.name for t in tools}
    assert names == {
        "getRPDSpectrumMeasurements",
        "analyzeSpectrumMeasurements",
        "getAllAmpsInSegment",
        "getAmpUpstreamSpectrumMeasurements",
        "localizeUpstreamSpectrumImpairmentSource",
    }


class _ScriptedSLM(BaseChatModel):
    """A minimal deterministic stand-in for a tool-calling SLM.

    Reimplements the 6-step happy path by reading prior ToolMessages, so the LangGraph loop is
    exercised end-to-end without a network endpoint.
    """

    @property
    def _llm_type(self) -> str:
        return "scripted-slm"

    def bind_tools(self, tools, **kwargs):  # create_react_agent calls this
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    @staticmethod
    def _next(messages):
        parsed = [(m.name, json.loads(m.content)) for m in messages if isinstance(m, ToolMessage)]

        def call(name, args):
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c-{uuid.uuid4().hex[:8]}"}])

        def last(name):
            for n, d in reversed(parsed):
                if n == name:
                    return d
            return None

        n = len(parsed)
        if n == 0:
            return call("getRPDSpectrumMeasurements", {})
        if n == 1:
            return call("analyzeSpectrumMeasurements", {"measurementIds": [last("getRPDSpectrumMeasurements")["measurementId"]]})
        if n == 2:
            return call("getAllAmpsInSegment", {})
        if n == 3:
            return call("getAmpUpstreamSpectrumMeasurements", {"ampListRef": last("getAllAmpsInSegment")["ampListRef"]})
        if n == 4:
            return call("analyzeSpectrumMeasurements", {"measurementSetRef": last("getAmpUpstreamSpectrumMeasurements")["measurementSetRef"]})
        if n == 5:
            analyses = [d for nm, d in parsed if nm == "analyzeSpectrumMeasurements"]
            imp = next((c for c in analyses[0]["classesPresent"] if c != "Clean"), "UnknownImpairment")
            return call(
                "localizeUpstreamSpectrumImpairmentSource",
                {"impairmentType": imp,
                 "classificationSetRefs": [analyses[0]["classificationSetRef"], analyses[1]["classificationSetRef"]]},
            )
        loc = last("localizeUpstreamSpectrumImpairmentSource")
        cands = loc.get("candidateLocations") or [{}]
        return AIMessage(
            content=f"Localized {loc['impairmentType']} ({loc['localizationStatus']}, conf={loc['confidence']}): "
                    f"{cands[0].get('description', cands[0].get('locationType', 'n/a'))}"
        )


def test_langgraph_agent_end_to_end_with_scripted_slm(cpd_scenario: Scenario) -> None:
    pytest.importorskip("langgraph")
    agent = LangGraphAgent(MockMcpServer(cpd_scenario), llm=_ScriptedSLM(), scenario_name="cpd-scripted")
    result = agent.run()

    assert result.trace.final_status == "localized"
    assert result.localization and result.localization["impairmentType"] == "CPD"
    assert [c.tool for c in result.trace.calls] == [
        "getRPDSpectrumMeasurements",
        "analyzeSpectrumMeasurements",
        "getAllAmpsInSegment",
        "getAmpUpstreamSpectrumMeasurements",
        "analyzeSpectrumMeasurements",
        "localizeUpstreamSpectrumImpairmentSource",
    ]
    assert "CPD" in result.final_message


# ── Live SLM integration over a REAL plant (opt-in) ──────────────────────────
def _real_plant_port3_scenario():
    """Real 16-amp scoped segment of plant-1 (port 3) with a mid-tree CPD fault."""
    from pathlib import Path

    from uil.domain.labels import ImpairmentLabel
    from uil.localizer.graph_localizer import Topology

    fix = Path(__file__).resolve().parent / "fixtures" / "network-topology-example-1.json"
    doc = json.loads(fix.read_text(encoding="utf-8"))
    root, port, fault = "0000000001", "3", "0000001418"
    topo = Topology.from_data_package(root, port, doc)
    kids: dict[str, list[str]] = {}
    for nid, n in topo.amps.items():
        if n.parentId:
            kids.setdefault(n.parentId, []).append(nid)
    impaired, stack, seen = set(), [fault], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if topo.amps[cur].deviceType == "AMP":
            impaired.add(cur)
        stack.extend(kids.get(cur, []))
    scenario = Scenario(
        rpdId=root, portId=port, rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": a, "label": "CPD" if a in impaired else "Clean"} for a in topo.amp_ids],
        topology_doc=doc,
    )
    return scenario, root, port, fault, impaired


@pytest.mark.skipif(
    not (os.getenv("SLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
    reason="live SLM endpoint not configured (set SLM_BASE_URL to run)",
)
def test_langgraph_agent_real_plant_live_slm() -> None:
    """Opt-in: drive a REAL 16-amp plant segment through a live SLM.

    Asserts the run reaches a valid terminal state without crashing and that the SLM never
    receives the amp-id list (only the count + handle). If the SLM completes localization, the
    boundary must reference the planted fault subtree; if it hallucinates ids / stalls, the
    contract's error handling must escalate gracefully (both are acceptable here — this test
    guards the plumbing, not SLM accuracy).
    """
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_openai")

    scenario, root, port, fault, impaired = _real_plant_port3_scenario()
    server = MockMcpServer(scenario)
    agent = LangGraphAgent(server, scenario_name="real-plant-live")
    alarm = {
        "alarmId": "ALM-LIVE", "severity": "major", "alarmType": "highUpstreamFecErrors",
        "direction": "upstream",
        "entity": {"type": "rpdPort", "rpdId": root, "portId": port},
    }
    result = agent.run_from_alarm(alarm)

    assert result.trace.calls, "agent made no tool calls"
    assert result.trace.final_status in {"localized", "low_confidence", "escalated"}

    # Reference-surface invariant: getAllAmpsInSegment exposes only status/handle/count.
    for c in result.trace.calls:
        if c.tool == "getAllAmpsInSegment" and c.result.get("status") == "success":
            assert set(c.result.keys()) <= {"status", "ampListRef", "ampCount"}
            assert c.result["ampCount"] == 16

    loc = server.resolve_localization(result.localization)
    if loc and loc.get("status") == "success" and loc.get("localizationStatus") == "localized":
        supporting = {d["ampId"] for d in loc.get("supportingDevices", [])}
        assert fault in supporting, "localized but boundary missed the planted fault"

