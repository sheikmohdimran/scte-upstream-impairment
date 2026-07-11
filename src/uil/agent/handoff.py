"""Human-handoff artifact (Step 7).

When localization is ambiguous (multi-anomaly, conflicting labels, low confidence, or
unrecoverable error), the agent escalates to a human with a readable Markdown summary of
what it measured, classified, and why it stopped.
"""

from __future__ import annotations

from typing import Any, Optional

from uil.agent.trace import ToolCallTrace


def build_handoff_summary(
    scenario: str,
    trace: ToolCallTrace,
    reason: str,
    localization: Optional[dict[str, Any]] = None,
) -> str:
    lines = [
        "## Diagnosis Summary",
        "",
        f"- scenario: {scenario}",
        f"**Escalation reason:** {reason}",
        "",
        "## Evidence",
        "### Tool-call timeline",
    ]
    for c in trace.calls:
        lines.append(f"- Step {c.step} `{c.tool}` → **{c.outcome}** — {c.decision}")

    if localization:
        lines += [
            "",
            "### Localization (partial / low-confidence)",
            f"- status: `{localization.get('localizationStatus', 'n/a')}`",
            f"- impairmentType: `{localization.get('impairmentType', 'n/a')}`",
            f"- confidence: {localization.get('confidence', 'n/a')}",
        ]
        candidates = localization.get("candidateLocations") or []
        if candidates:
            top = candidates[0]
            up = top.get("upstreamBoundaryDevice") or {}
            downs = top.get("downstreamBoundaryDevices") or []
            up_id = up.get("rpdId") or up.get("ampId")
            down_ids = [d.get("rpdId") or d.get("ampId") for d in downs]
            desc = top.get("description") or top.get("locationType")
            lines.append(f"- top candidate: {desc} (score {top.get('score', 'n/a')})")
            lines.append(f"- boundary: upstream `{up_id}` -> downstream {down_ids}")
            if top.get("uncertainDevicesRef"):
                lines.append(
                    f"- uncertain devices behind handle `{top['uncertainDevicesRef']}` "
                    "(resolve backend-side for ids)"
                )

    lines += [
        "",
        "## Recommended Next Action",
        "Review the conflicting/low-confidence evidence above and dispatch a technician or "
        "trigger a targeted re-measurement of the listed devices.",
    ]
    return "\n".join(lines)
