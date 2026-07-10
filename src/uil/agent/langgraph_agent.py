"""Phase 3: real SLM + LangGraph orchestrator (drop-in for the deterministic Orchestrator).

The 5 MCP reference-surface tools are exposed to a tool-calling chat model via LangChain
``StructuredTool``s, and a LangGraph ReAct agent drives them. The model only ever sees
**handles + counts** (never raw spectra) and owns the off-happy-path decisions: error
recovery, ``partial_success``, re-measure, and human escalation.

The chat model is **pluggable**: any OpenAI-compatible endpoint works (local llama.cpp /
vLLM / Ollama's OpenAI shim, or a hosted API). Point it at one via env vars
(``SLM_BASE_URL``/``OPENAI_BASE_URL``, ``SLM_MODEL``, ``OPENAI_API_KEY``) or pass your own
``BaseChatModel``. Fine-tuning is intentionally out of scope here.

The deterministic :class:`~uil.agent.orchestrator.Orchestrator` remains the oracle this agent
can be graded against.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from uil.agent.handoff import build_handoff_summary
from uil.agent.trace import ToolCallTrace
from uil.mcp_server.server import MockMcpServer

SYSTEM_PROMPT = """\
You are an autonomous diagnostic agent that localizes the physical source of an UPSTREAM
spectrum impairment in a DOCSIS cable segment (RPD port -> amplifier cascade). You drive a
fixed set of measurement/analysis tools. You only ever see opaque handles and counts, never
raw spectra.

Follow this happy-path sequence, but adapt when tools return errors or partial results:
  1. getRPDSpectrumMeasurements(rpdId, portId) -> a measurementId for the RPD.
  2. analyzeSpectrumMeasurements(measurementIds=[that id]) -> a classificationSetRef for the
     RPD plus impairedCount and classesPresent. If impairedCount == 0 the segment looks clean;
     stop and report that (the fault may be intermittent).
  3. getAllAmpsInSegment(rpdId, portId) -> an ampListRef and ampCount for the whole leg.
  4. getAmpSpectrumMeasurements(ampListRef=that ref) -> a measurementSetRef. This may come back
     as partial_success with a failedCount; that is acceptable - proceed with what was measured.
  5. analyzeSpectrumMeasurements(measurementSetRef=that ref) -> a classificationSetRef for the amps.
  6. localizeUpstreamSpectrumImpairmentSource(rpdId, portId, impairmentType,
     classificationSetRefs=[the RPD set from step 2, the amp set from step 5]) -> the result.
     Use the dominant non-Clean label from step 2's classesPresent as impairmentType.

Recovery rules:
  - On a transient error (STALE_DATA_ONLY, MEASUREMENT_UNAVAILABLE) retry the SAME call ONCE.
  - On a non-transient error (DEVICE_UNREACHABLE, TOPOLOGY_UNAVAILABLE, PERMISSION_DENIED, etc.)
    do NOT loop; stop and escalate with a short reason.
  - If localization comes back localizationStatus == "low_confidence" (multi-anomaly, conflicting
    labels, or missing measurements), do NOT keep calling tools; escalate to a human.

When you are done, reply with a final plain-text summary that states: the impairmentType, the
localizationStatus, the confidence, and the likely source description (or, if you stopped early,
why you escalated). Do not call any more tools after that.
"""


# --- SLM-facing tool argument schemas (handles only; no nested objects) ---------------------
class _RpdArgs(BaseModel):
    rpdId: str = Field(description="RPD identifier, e.g. 'RPD-1'.")
    portId: str = Field(description="Upstream port identifier, e.g. 'P1'.")


class _AnalyzeArgs(BaseModel):
    measurementIds: Optional[list[str]] = Field(
        default=None, description="List of single measurement ids (use for the RPD capture)."
    )
    measurementSetRef: Optional[str] = Field(
        default=None, description="A measurement-set handle (use for the amp captures)."
    )


class _AllAmpsArgs(BaseModel):
    rpdId: str
    portId: str


class _AmpMeasArgs(BaseModel):
    ampListRef: Optional[str] = Field(default=None, description="Amp-list handle from getAllAmpsInSegment.")
    ampIds: Optional[list[str]] = Field(default=None, description="Explicit amp ids (alternative to ampListRef).")


class _LocalizeArgs(BaseModel):
    rpdId: str
    portId: str
    impairmentType: str = Field(description="Dominant non-Clean impairment label, e.g. 'CPD'.")
    classificationSetRefs: list[str] = Field(description="[RPD classificationSetRef, amp classificationSetRef].")


class McpToolset:
    """Adapts the mock MCP server to SLM-friendly tools (string handles in, flat dicts out).

    Records every call into a :class:`ToolCallTrace` and remembers the final localization.
    """

    def __init__(self, server: MockMcpServer, trace: ToolCallTrace) -> None:
        self.s = server
        self.trace = trace
        self.last_localization: Optional[dict[str, Any]] = None

    def _record(self, tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
        self.trace.add(
            tool=tool,
            arguments=arguments,
            result=result,
            outcome=result.get("status", "error"),
            decision="(SLM-driven)",
        )
        return json.dumps(result)

    def get_rpd_spectrum(self, rpdId: str, portId: str) -> str:
        r = self.s.getRPDSpectrumMeasurements(rpdId, portId)
        if r.get("status") == "success":
            ref = r["measurementRef"]
            out = {"status": "success", "measurementId": ref["measurementId"], "deviceType": ref["deviceType"]}
        else:
            out = r
        return self._record("getRPDSpectrumMeasurements", {"rpdId": rpdId, "portId": portId}, out)

    def analyze_spectrum(
        self, measurementIds: Optional[list[str]] = None, measurementSetRef: Optional[str] = None
    ) -> str:
        if measurementIds is not None:
            r = self.s.analyzeSpectrumMeasurements(measurementRefs=[{"measurementId": m} for m in measurementIds])
        else:
            r = self.s.analyzeSpectrumMeasurements(measurementSetRef=measurementSetRef)
        args = {"measurementIds": measurementIds, "measurementSetRef": measurementSetRef}
        return self._record("analyzeSpectrumMeasurements", args, r)

    def get_all_amps(self, rpdId: str, portId: str) -> str:
        r = self.s.getAllAmpsInSegment(rpdId, portId)
        return self._record("getAllAmpsInSegment", {"rpdId": rpdId, "portId": portId}, r)

    def get_amp_spectra(self, ampListRef: Optional[str] = None, ampIds: Optional[list[str]] = None) -> str:
        r = self.s.getAmpSpectrumMeasurements(ampListRef=ampListRef, ampIds=ampIds)
        return self._record("getAmpSpectrumMeasurements", {"ampListRef": ampListRef, "ampIds": ampIds}, r)

    def localize(self, rpdId: str, portId: str, impairmentType: str, classificationSetRefs: list[str]) -> str:
        r = self.s.localizeUpstreamSpectrumImpairmentSource(rpdId, portId, impairmentType, classificationSetRefs)
        if r.get("status") == "success":
            self.last_localization = r
        args = {
            "rpdId": rpdId, "portId": portId, "impairmentType": impairmentType,
            "classificationSetRefs": classificationSetRefs,
        }
        return self._record("localizeUpstreamSpectrumImpairmentSource", args, r)


def build_tools(toolset: McpToolset) -> list:
    """Wrap the toolset methods as LangChain StructuredTools."""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            func=toolset.get_rpd_spectrum, name="getRPDSpectrumMeasurements",
            description="Capture the upstream spectrum at an RPD port. Returns a measurementId.",
            args_schema=_RpdArgs,
        ),
        StructuredTool.from_function(
            func=toolset.analyze_spectrum, name="analyzeSpectrumMeasurements",
            description=(
                "Classify one or more upstream spectrum captures. Pass measurementIds for the RPD "
                "capture, or measurementSetRef for the amp set. Returns a classificationSetRef, "
                "impairedCount and classesPresent."
            ),
            args_schema=_AnalyzeArgs,
        ),
        StructuredTool.from_function(
            func=toolset.get_all_amps, name="getAllAmpsInSegment",
            description="List every amplifier in the RPD leg (one-shot, not recursive). Returns an ampListRef.",
            args_schema=_AllAmpsArgs,
        ),
        StructuredTool.from_function(
            func=toolset.get_amp_spectra, name="getAmpSpectrumMeasurements",
            description=(
                "Capture upstream spectra at all amps via an ampListRef. May return partial_success "
                "with a failedCount. Returns a measurementSetRef."
            ),
            args_schema=_AmpMeasArgs,
        ),
        StructuredTool.from_function(
            func=toolset.localize, name="localizeUpstreamSpectrumImpairmentSource",
            description=(
                "Run common-point localization over the pooled RPD + amp classification sets. "
                "Returns the likely source location and localizationStatus."
            ),
            args_schema=_LocalizeArgs,
        ),
    ]


def build_chat_model(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
):
    """Build a pluggable OpenAI-compatible chat model.

    Reads ``SLM_MODEL``, ``SLM_BASE_URL``/``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` from the
    environment when arguments are omitted. ``api_key`` defaults to ``"EMPTY"`` so local
    endpoints that ignore auth still work.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or os.getenv("SLM_MODEL", "local-model"),
        base_url=base_url or os.getenv("SLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        temperature=temperature,
    )


@dataclass
class AgentRunResult:
    final_message: str
    localization: Optional[dict[str, Any]]
    trace: ToolCallTrace
    handoff_markdown: Optional[str] = None
    messages: list[Any] = field(default_factory=list)


class LangGraphAgent:
    """A LangGraph ReAct agent that orchestrates the 5 MCP tools with a pluggable SLM."""

    def __init__(self, server: MockMcpServer, llm: Any = None, scenario_name: str = "scenario",
                 recursion_limit: int = 30) -> None:
        try:  # LangGraph >= 1.0 moved this to langchain.agents (uses system_prompt=)
            from langchain.agents import create_agent as _create_agent
            _prompt_kw = "system_prompt"
        except ImportError:  # pragma: no cover - older langgraph (uses prompt=)
            from langgraph.prebuilt import create_react_agent as _create_agent
            _prompt_kw = "prompt"

        self.server = server
        self.scenario_name = scenario_name
        self.recursion_limit = recursion_limit
        self.trace = ToolCallTrace(scenario=scenario_name)
        self.toolset = McpToolset(server, self.trace)
        self.llm = llm if llm is not None else build_chat_model()
        self.graph = _create_agent(self.llm, build_tools(self.toolset), **{_prompt_kw: SYSTEM_PROMPT})
        from uil.agent.trace_store import TraceStore
        self.trace_store = TraceStore()
        self._trigger: Optional[dict[str, Any]] = None

    def default_goal(self) -> str:
        return (
            f"An upstream impairment alarm fired on RPD {self.server.scn.rpdId} "
            f"port {self.server.scn.portId}. Localize the physical source and report it."
        )

    def run_from_alarm(self, alarm: dict, goal: Optional[str] = None) -> AgentRunResult:
        """Gate the alarm first (deterministic oracle), then run the SLM only on a positive.

        Mirrors :meth:`Orchestrator.run_from_alarm` so the trigger decision is reproducible
        and the SLM is not invoked on obvious negatives.
        """
        from uil.agent.trigger import AlarmTrigger

        decision = AlarmTrigger().evaluate(alarm)
        self._trigger = {
            "verdict": decision.verdict, "reason": decision.reason,
            "alarmId": alarm.get("alarmId"), "alarmType": alarm.get("alarmType"),
        }
        if not decision.should_trigger:
            self.trace.final_status = "not_triggered"
            self.trace_store.record(self.trace, extra={"trigger": self._trigger})
            return AgentRunResult(
                final_message=f"No workflow triggered: {decision.reason}.",
                localization=None, trace=self.trace, messages=[],
            )
        args = decision.tool_call["arguments"]
        alarm_goal = goal or (
            f"An upstream impairment alarm ({alarm.get('alarmType')}) fired on RPD "
            f"{args['rpdId']} port {args['portId']}. Localize the physical source and report it."
        )
        return self.run(goal=alarm_goal)

    def run(self, goal: Optional[str] = None) -> AgentRunResult:
        out = self.graph.invoke(
            {"messages": [("user", goal or self.default_goal())]},
            config={"recursion_limit": self.recursion_limit},
        )
        messages = out["messages"]
        final = messages[-1].content if messages else ""
        loc = self.toolset.last_localization
        status = loc.get("localizationStatus") if loc else "escalated"
        self.trace.final_status = status or "escalated"
        handoff = None
        if status != "localized":
            self.trace.label = "negative"
            handoff = build_handoff_summary(
                self.scenario_name, self.trace,
                reason=f"Agent ended without confident localization (status={status}).",
                localization=loc,
            )
        extra = {"trigger": self._trigger} if self._trigger is not None else None
        self.trace_store.record(self.trace, extra=extra)
        return AgentRunResult(
            final_message=final if isinstance(final, str) else str(final),
            localization=loc, trace=self.trace, handoff_markdown=handoff, messages=messages,
        )

def _main() -> None:
    """Run the CPD demo scenario against a configured OpenAI-compatible SLM endpoint."""
    from uil.domain.labels import ImpairmentLabel
    from uil.mcp_server.server import FaultInjection, Scenario

    if not (os.getenv("SLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")):
        raise SystemExit(
            "Set SLM_BASE_URL (or OPENAI_BASE_URL) to an OpenAI-compatible endpoint, "
            "and SLM_MODEL to the served model name. Example:\n"
            "  export SLM_BASE_URL=http://localhost:8000/v1\n"
            "  export SLM_MODEL=Qwen2.5-7B-Instruct\n"
            "  export OPENAI_API_KEY=EMPTY"
        )

    scenario = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
        ],
        faults=FaultInjection(amp_failed_ports={"A4"}),
    )
    agent = LangGraphAgent(MockMcpServer(scenario), scenario_name="langgraph-CPD")
    result = agent.run()
    print("=== FINAL MESSAGE ===\n" + result.final_message)
    print(f"\nlocalization status: {result.trace.final_status}")
    if result.localization and result.localization.get("likelySourceLocation"):
        print("likely source:", result.localization["likelySourceLocation"]["description"])
    print(f"\n=== TOOL CALLS ({len(result.trace.calls)}) ===")
    for c in result.trace.calls:
        print(f"  {c.step}. {c.tool:42} {c.outcome}")
    if result.handoff_markdown:
        print("\n=== HUMAN HANDOFF ===\n" + result.handoff_markdown)


if __name__ == "__main__":
    _main()
