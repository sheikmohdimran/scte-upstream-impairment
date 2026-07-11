"""Real-plant, SLM-driven end-to-end demo.

Runs the LangGraph + SLM orchestrator against a REAL CableLabs plant (scoped to one RPD port),
with a fault injected at a chosen real amplifier. Shows exactly how the code and the SLM divide
the work on a complex topology:

  * The SLM only ever sees opaque handles + counts (never the 16-amp list, never raw spectra),
    so long numeric amp ids can't be hallucinated into tool arguments.
  * The deterministic backend resolves handles, measures every amp in the scoped segment at once,
    classifies, and runs graph common-point localization over the full passive-aware topology.

Independent (non-circular) oracle: the fault is planted at amp X, so X and its amp-descendants
are impaired; the localized boundary must reference X's subtree. We check that here.

Usage:
    export SLM_BASE_URL=http://0.0.0.0:8000/v1
    export SLM_MODEL=google/gemma-4-E4B-it
    export OPENAI_API_KEY=EMPTY
    python examples/demo_real_plant_slm.py [--port 3] [--fault-amp <ampId>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from uil.domain.labels import ImpairmentLabel
from uil.localizer.graph_localizer import Topology
from uil.mcp_server.server import MockMcpServer, Scenario

_PLANT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "network-topology-example-1.json"
_ROOT_ID = "0000000001"  # RfSource "FN1"


def _amp_descendants(topo: Topology, fault: str) -> set[str]:
    """Amp ids in fault's downstream subtree (fault ∪ amp-descendants), within the scoped topo."""
    kids: dict[str, list[str]] = {}
    for nid, n in topo.amps.items():
        if n.parentId:
            kids.setdefault(n.parentId, []).append(nid)
    seen, stack, out = set(), [fault], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if topo.amps[cur].deviceType == "AMP":
            out.add(cur)
        stack.extend(kids.get(cur, []))
    return out


def build_scenario(doc: dict, port_id: str, fault_amp: str | None) -> tuple[Scenario, str, set[str]]:
    topo = Topology.from_data_package(_ROOT_ID, port_id, doc)
    amp_ids = topo.amp_ids
    # Default fault: the amp with the largest downstream amp-subtree (a clear branch fault).
    if fault_amp is None:
        fault_amp = max(amp_ids, key=lambda a: len(_amp_descendants(topo, a)))
    impaired = _amp_descendants(topo, fault_amp)
    amps = [{"ampId": a, "label": "CPD" if a in impaired else "Clean"} for a in amp_ids]
    scn = Scenario(
        rpdId=_ROOT_ID, portId=port_id, rpd_label=ImpairmentLabel.CPD,
        amps=amps, topology_doc=doc,
    )
    return scn, fault_amp, impaired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="3", help="RfPort portId under FN1 (1=56 amps, 3=16, 2=2)")
    ap.add_argument("--fault-amp", default=None, help="amp id to fault (default: largest subtree)")
    args = ap.parse_args()

    if not (os.getenv("SLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")):
        raise SystemExit("Set SLM_BASE_URL (e.g. http://0.0.0.0:8000/v1) and SLM_MODEL first.")

    doc = json.loads(_PLANT.read_text(encoding="utf-8"))
    scn, fault_amp, impaired = build_scenario(doc, args.port, args.fault_amp)
    server = MockMcpServer(scn)

    print(f"Plant: {_PLANT.name}  RPD=FN1({_ROOT_ID})  port={args.port}")
    print(f"Scoped segment: {len(scn.amps)} amps  |  injected fault at amp {fault_amp}")
    print(f"Impaired subtree ({len(impaired)} amps): {sorted(impaired)}")
    print(f"SLM: {os.getenv('SLM_MODEL')} @ {os.getenv('SLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')}")
    print("-" * 72)

    from uil.agent.langgraph_agent import LangGraphAgent

    agent = LangGraphAgent(server, scenario_name=f"real-plant-port{args.port}")
    alarm = {
        "alarmId": "ALM-REAL-1", "severity": "major", "alarmType": "highUpstreamFecErrors",
        "direction": "upstream",
        "entity": {"type": "rpdPort", "rpdId": _ROOT_ID, "portId": args.port},
    }
    result = agent.run_from_alarm(alarm)

    print("SLM-driven tool calls (SLM sees only the italicised handles/counts):")
    for c in result.trace.calls:
        r = c.result
        summary = {k: r[k] for k in ("status", "ampCount", "measurementSetRef", "ampListRef",
                                     "classificationSetRef", "impairedCount", "failedCount",
                                     "errorCode")
                   if isinstance(r, dict) and k in r}
        print(f"  {c.step}. {c.tool:38} -> {c.outcome:12} args={c.arguments} {summary}")

    loc = server.resolve_localization(result.localization)
    print("-" * 72)
    if loc and loc.get("status") == "success":
        cand = loc["candidateLocations"][0] if loc.get("candidateLocations") else {}
        supporting = {d.get("ampId") for d in loc.get("supportingDevices", [])}
        up = cand.get("upstreamBoundaryDevice", {})
        downs = [d.get("ampId") or d.get("rpdId") for d in cand.get("downstreamBoundaryDevices", [])]
        print(f"localizationStatus : {loc['localizationStatus']}  (confidence={loc['confidence']})")
        print(f"top candidate      : {cand.get('locationType')}  upstream={up.get('rpdId') or up.get('ampId')}  downstream={downs}")
        print(f"supporting devices : {sorted(supporting)}")
        ok = fault_amp in supporting or fault_amp in set(downs)
        print(f"ORACLE fault {fault_amp} referenced in localization: {'PASS' if ok else 'MISS'}")
    else:
        print(f"No localization (final_status={result.trace.final_status}).")
        if result.handoff_markdown:
            print(result.handoff_markdown)
    print("-" * 72)
    print("Final SLM message:\n" + (result.final_message or "(none)"))


if __name__ == "__main__":
    main()
