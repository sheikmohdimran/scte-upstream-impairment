"""Parse the CableLabs RF plant data-package into normalized device nodes.

The real topology (``getAllAmpsInSegment`` / Chai's topology tool) is a Frictionless
data-package: ``resources[].data`` holds a flat ``components[]`` list plus a directed
``edges[]`` list (``{"source","target"}`` id pairs). This module turns that into the
parent/child device graph the :class:`~uil.localizer.graph_localizer.GraphLocalizer`
consumes, collapsing the passive connector fabric (ports + cables) so only the RF-active
and passive *devices* remain as nodes.

--------------------------------------------------------------------------------
ASSUMPTIONS (isolated here on purpose; tracked in docs/operations/open-decisions.md).
Change these here — nothing else in the localizer should hard-code plant semantics.

    A1. Root anchor: the ``RfSource`` node (the fiber node / RPD location) is the upstream
        root. Impairments propagate upstream toward it. Confirmed by transcript
        (Randy/Irene) that a topology tool exists; the RPD<->RfSource binding is NOT yet
        confirmed by CableLabs — see CLARIFICATION C1.
    A2. Edge orientation is DOWNSTREAM (source -> target flows away from the headend toward
        subscribers). We reverse it to build upstream parent links. Verified against both
        sample files (RfSource has only outgoing edges).
    A3. Segment scope: the whole graph under the root is treated as one segment/leg. Per
        ``portId`` slicing (one RfSource may have several ports/legs) is NOT yet wired —
        see CLARIFICATION C2.
    A4. Roles by component ``type``:
          - measured devices  : RfAmp                      -> node, deviceType "AMP"
          - passive devices   : RfSplitter/RfTap/RfCoupler/RfPowerInserter -> node, kept
                                 as candidate common points
          - connectors        : RfPort/RfCable             -> collapsed (cable length is
                                 accumulated into distance)
          - leaves            : device (subscriber homes)  -> dropped (not measured)
          - root              : RfSource                   -> node, deviceType "RPD"
        Unknown/other types are treated as pass-through connectors (defensive default).
    A5. Device identity is the component ``id`` (strings like "0000000017"). ``name`` is
        carried through for human-readable handoff only. Whether the real amp identifier
        is ``id`` or ``name`` is NOT confirmed — see CLARIFICATION C3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Component-type role sets (A4). Kept as module constants so they are easy to amend.
ROOT_TYPES = {"RfSource"}
AMP_TYPES = {"RfAmp"}
PASSIVE_TYPES = {"RfSplitter", "RfTap", "RfCoupler", "RfPowerInserter"}
CONNECTOR_TYPES = {"RfPort", "RfCable"}
LEAF_TYPES = {"device"}


@dataclass
class PlantNode:
    """A normalized plant device (root/amp/passive) after connectors are collapsed."""

    id: str
    deviceType: str  # "RPD" (root), "AMP", or the passive component type (e.g. "RfSplitter")
    name: Optional[str] = None
    parentId: Optional[str] = None
    children: list[str] = field(default_factory=list)
    distanceFromRpdMeters: Optional[float] = None


def _extract_data(doc: dict) -> dict:
    """Accept either the full data-package or an already-unwrapped ``data`` dict."""
    if "components" in doc and "edges" in doc:
        return doc
    resources = doc.get("resources")
    if resources:
        data = resources[0].get("data", {})
        if "components" in data:
            return data
    raise ValueError("topology document has no components/edges")


def _cable_length(component: dict) -> float:
    try:
        return float(component.get("length", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_data_package(doc: dict) -> dict:
    """Return ``{"rootId": str | None, "nodes": {id: PlantNode}}``.

    Connectors (ports/cables) and leaf homes are collapsed away; cable ``length`` is
    summed into each device's ``distanceFromRpdMeters``.
    """
    data = _extract_data(doc)
    components = {c["id"]: c for c in data["components"] if "id" in c}
    edges = data.get("edges", [])

    # Downstream adjacency: source -> [targets] (A2).
    downstream: dict[str, list[str]] = {}
    indeg: dict[str, int] = {cid: 0 for cid in components}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in components and t in components:
            downstream.setdefault(s, []).append(t)
            indeg[t] = indeg.get(t, 0) + 1

    def _type(cid: str) -> str:
        return components.get(cid, {}).get("type", "")

    def _is_significant(cid: str) -> bool:
        t = _type(cid)
        return t in ROOT_TYPES or t in AMP_TYPES or t in PASSIVE_TYPES

    # Roots: prefer RfSource; fall back to any significant node with no incoming edge (A1).
    roots = [cid for cid in components if _type(cid) in ROOT_TYPES and indeg.get(cid, 0) == 0]
    if not roots:
        roots = [cid for cid in components if _type(cid) in ROOT_TYPES]
    if not roots:
        roots = [cid for cid in components if _is_significant(cid) and indeg.get(cid, 0) == 0]

    nodes: dict[str, PlantNode] = {}

    def _norm_type(cid: str) -> str:
        t = _type(cid)
        if t in ROOT_TYPES:
            return "RPD"
        if t in AMP_TYPES:
            return "AMP"
        return t  # passive types keep their plant type

    def _register(cid: str, parent_sig: Optional[str], accum_len: float) -> None:
        if cid in nodes:
            return  # first path wins (defensive against meshed graphs)
        comp = components[cid]
        parent_dist = nodes[parent_sig].distanceFromRpdMeters if parent_sig in nodes else 0.0
        dist = 0.0 if parent_sig is None else (parent_dist or 0.0) + accum_len
        nodes[cid] = PlantNode(
            id=cid,
            deviceType=_norm_type(cid),
            name=comp.get("name"),
            parentId=parent_sig,
            distanceFromRpdMeters=round(dist, 2),
        )

    # Iterative DFS collapsing connectors; carry (node, nearest significant ancestor,
    # accumulated cable length since that ancestor).
    visited_edges: set[tuple[str, str]] = set()
    for root in roots:
        _register(root, None, 0.0)
        stack: list[tuple[str, str, float]] = [(root, root, 0.0)]
        while stack:
            nid, parent_sig, accum = stack.pop()
            for child in downstream.get(nid, []):
                if (nid, child) in visited_edges:
                    continue
                visited_edges.add((nid, child))
                t = _type(child)
                if t in LEAF_TYPES:
                    continue  # subscriber home — drop (A4)
                if t in CONNECTOR_TYPES or not (_is_significant(child)):
                    # pass through connectors / unknown types; accumulate cable length
                    add = _cable_length(components[child]) if t == "RfCable" else 0.0
                    stack.append((child, parent_sig, accum + add))
                else:
                    _register(child, parent_sig, accum)
                    stack.append((child, child, 0.0))

    # Backfill children from parent links.
    for n in nodes.values():
        if n.parentId and n.parentId in nodes:
            nodes[n.parentId].children.append(n.id)

    return {"rootId": roots[0] if roots else None, "nodes": nodes}
