"""Alarm trigger gate — the workflow entry point (before Step 1).

Decides whether an incoming alarm should start the upstream-impairment localization
workflow:

* **Positive** alarm  -> ``verdict == "call"``; the workflow's first tool call is
  ``getRPDSpectrumMeasurements(rpdId, portId)`` seeded from the alarm's RPD port.
* **Negative** alarm  -> ``verdict == "noCall"``; the workflow is not started.

Rule (derived from the ``eval_CPD`` alarm set): trigger **iff** the alarm is an upstream
FEC-error alarm raised on an RPD upstream port. Both conditions are required — the hard
negatives include upstream FEC alarms on a *cable modem* (wrong entity) and non-FEC
upstream alarms on an RPD port (wrong alarm type), and neither should trigger.

This is the deterministic oracle; the SLM (LangGraph agent) must reproduce the same
decision on the same alarm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Upstream FEC-error alarm types that indicate a possible physical-layer impairment
# and therefore warrant spectrum localization.
UPSTREAM_FEC_ALARM_TYPES = frozenset(
    {
        "highUpstreamFecErrors",
        "highUpstreamCorrectables",
        "highUpstreamUncorrectables",
        "highUpstreamCorrectablesAndUncorrectables",
    }
)

_FIRST_TOOL = "getRPDSpectrumMeasurements"


@dataclass
class TriggerDecision:
    """Outcome of evaluating a single alarm against the trigger gate."""

    verdict: str  # "call" | "noCall"
    tool_call: Optional[dict]  # {"name", "arguments"} when verdict == "call", else None
    reason: str

    @property
    def should_trigger(self) -> bool:
        return self.verdict == "call"


class AlarmTrigger:
    """Deterministic gate that maps an alarm payload to a trigger decision."""

    def evaluate(self, alarm: dict) -> TriggerDecision:
        entity = alarm.get("entity") or {}
        entity_type = entity.get("type")
        alarm_type = alarm.get("alarmType")

        if entity_type != "rpdPort":
            return TriggerDecision(
                "noCall", None,
                f"entity type {entity_type!r} is not an RPD upstream port",
            )
        if alarm_type not in UPSTREAM_FEC_ALARM_TYPES:
            return TriggerDecision(
                "noCall", None,
                f"alarmType {alarm_type!r} is not an upstream FEC-error alarm",
            )

        rpd_id = entity.get("rpdId")
        port_id = entity.get("portId")
        if not rpd_id or not port_id:
            return TriggerDecision(
                "noCall", None,
                "RPD port alarm is missing rpdId/portId; cannot seed the workflow",
            )

        return TriggerDecision(
            "call",
            {"name": _FIRST_TOOL, "arguments": {"rpdId": rpd_id, "portId": port_id}},
            f"upstream FEC alarm {alarm_type!r} on RPD port {rpd_id}/{port_id}",
        )
