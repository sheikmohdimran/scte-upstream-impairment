"""Tool-call trace format (Step 8).

A compact, serializable record of every tool call the agent makes, suitable for:
  * audit / debugging, and
  * generating positive/negative fine-tuning examples for the SLM orchestrator.

Each step captures the tool name, the arguments the SLM chose, the (reference-surface)
result, and the agent's decision rationale. Raw spectra never appear here — only handles.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    step: int
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    decision: str = Field(description="Why the agent made this call / what it concluded.")
    outcome: str = Field(description="success | error | partial_success")


class ToolCallTrace(BaseModel):
    scenario: str
    calls: list[ToolCall] = Field(default_factory=list)
    final_status: str = ""  # localized | low_confidence | escalated | failed
    label: str = "positive"  # positive (good orchestration) | negative (recoverable error path)

    def add(self, **kwargs: Any) -> None:
        self.calls.append(ToolCall(step=len(self.calls) + 1, **kwargs))

    def to_jsonl(self) -> str:
        return self.model_dump_json()
