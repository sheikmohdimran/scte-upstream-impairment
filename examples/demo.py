"""Runnable demo: two scenarios end-to-end through the mock MCP chain.

  1. Clean topology with a CPD fault on one branch -> fully localized.
  2. Same, but with an injected amp measurement failure -> partial_success ->
     low-confidence -> human handoff (the error-recovery path the paper highlights).

Run: python examples/demo.py   (from the project root, with src on PYTHONPATH)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uil.agent.orchestrator import Orchestrator  # noqa: E402
from uil.domain.labels import ImpairmentLabel  # noqa: E402
from uil.mcp_server.server import FaultInjection, MockMcpServer, Scenario  # noqa: E402


def base_scenario(**faults) -> Scenario:
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


def show(name: str, res) -> None:
    print(f"\n========== {name} → {res.status} ==========")
    for c in res.trace.calls:
        print(f"  {c.step}. {c.tool:<38} {c.outcome}")
    if res.localization:
        loc = res.localization
        print(f"  impairment={loc['impairmentType']} status={loc['localizationStatus']} "
              f"confidence={loc['confidence']}")
        if loc.get("candidateLocations"):
            top = loc["candidateLocations"][0]
            print("  source:", top.get("description") or top.get("locationType"))
    if res.handoff_markdown:
        print("  [human handoff artifact generated]")


if __name__ == "__main__":
    show("Scenario 1: full localization", Orchestrator(MockMcpServer(base_scenario()), "full").run())
    show("Scenario 2: partial_success + escalation",
         Orchestrator(MockMcpServer(base_scenario(amp_failed_ports={"A3"})), "partial").run())
