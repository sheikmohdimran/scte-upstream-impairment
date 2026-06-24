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
        f"# Diagnosis Summary — {scenario}",
        "",
        f"**Escalation reason:** {reason}",
        "",
        "## Tool-call timeline",
    ]
    for c in trace.calls:
        lines.append(f"- Step {c.step} `{c.tool}` → **{c.outcome}** — {c.decision}")

    if localization:
        lines += [
            "",
            "## Localization (partial / low-confidence)",
            f"- status: `{localization.get('localizationStatus', 'n/a')}`",
            f"- impairmentType: `{localization.get('impairmentType', 'n/a')}`",
            f"- confidence: {localization.get('confidence', 'n/a')}",
        ]
        likely = localization.get("likelySourceLocation")
        if likely:
            lines.append(f"- likely source: {likely.get('description')}")
        uncertain = localization.get("uncertainDevices") or []
        if uncertain:
            ids = [d.get("ampId") for d in uncertain]
            lines.append(f"- unmeasured / uncertain devices: {ids}")

    lines += [
        "",
        "## Recommended human action",
        "Review the conflicting/low-confidence evidence above and dispatch a technician or "
        "trigger a targeted re-measurement of the listed devices.",
    ]
    return "\n".join(lines)
