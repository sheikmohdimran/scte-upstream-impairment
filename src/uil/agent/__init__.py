"""SLM orchestration agent (Steps 5, 7, 8)."""

from uil.agent.orchestrator import Orchestrator, OrchestratorResult
from uil.agent.trace import ToolCall, ToolCallTrace
from uil.agent.handoff import build_handoff_summary
from uil.agent.trigger import AlarmTrigger, TriggerDecision, UPSTREAM_FEC_ALARM_TYPES
from uil.agent.trace_store import TraceStore

__all__ = [
    "Orchestrator",
    "OrchestratorResult",
    "ToolCall",
    "ToolCallTrace",
    "build_handoff_summary",
    "AlarmTrigger",
    "TriggerDecision",
    "UPSTREAM_FEC_ALARM_TYPES",
    "TraceStore",
]

# Phase 3 LangGraph agent is optional (needs langgraph/langchain). Import lazily so the
# package still imports without those extras installed.
try:  # pragma: no cover - exercised when extras are present
    from uil.agent.langgraph_agent import LangGraphAgent, McpToolset, build_chat_model

    __all__ += ["LangGraphAgent", "McpToolset", "build_chat_model"]
except ImportError:
    pass
