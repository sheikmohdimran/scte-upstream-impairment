"""Graph-theory common-point localizer (Step 6).

Upstream signal flows from the leaf amps toward the RPD (the tree root). An impairment
injected at a point propagates upstream, so every device on the path from the fault to the
RPD "sees" it, while devices on other branches stay clean. We therefore localize by finding
the **common point** where the impaired region meets the clean region.

Topology comes from ``getAllAmpsInSegment`` raw output (``amps[]`` with
``parentId``/``children``/``distanceFromRpdMeters``). ``parentId == None`` means the amp hangs
directly off the RPD port.

This is a deterministic prototype, NOT an LLM step. Output mirrors
``localizeUpstreamSpectrumImpairmentSource``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from uil.domain.classification import Classification
from uil.domain.labels import ImpairmentLabel
from uil.domain.localization import (
    CandidateLocation,
    LikelySourceLocation,
    LocalizationResult,
    RecommendedNextAction,
)
from uil.domain.refs import AmpDeviceRef, RpdDeviceRef
from uil.localizer.plant_topology import parse_data_package


@dataclass
class AmpNode:
    ampId: str
    parentId: Optional[str] = None
    children: list[str] = field(default_factory=list)
    distanceFromRpdMeters: Optional[float] = None
    deviceType: str = "AMP"  # "AMP" (measured), "RPD" (root), or passive plant type
    name: Optional[str] = None


@dataclass
class Topology:
    rpdId: str
    portId: str
    amps: dict[str, AmpNode]

    @classmethod
    def from_amp_list(cls, rpd_id: str, port_id: str, amps: list[dict]) -> "Topology":
        nodes = {a["ampId"]: AmpNode(**{k: a.get(k) for k in ("ampId", "parentId", "children", "distanceFromRpdMeters")}) for a in amps}
        for n in nodes.values():
            if n.children is None:
                n.children = []
        # backfill children if only parentId given
        for n in nodes.values():
            if n.parentId and n.parentId in nodes and n.ampId not in nodes[n.parentId].children:
                nodes[n.parentId].children.append(n.ampId)
        return cls(rpdId=rpd_id, portId=port_id, amps=nodes)

    @classmethod
    def from_data_package(cls, rpd_id: str, port_id: str, doc: dict) -> "Topology":
        """Build a topology from the CableLabs RF plant data-package.

        Passive devices (splitter/tap/coupler/power-inserter) are kept as nodes so the
        common point can land on them; connectors (ports/cables) and subscriber homes are
        collapsed away. See :mod:`uil.localizer.plant_topology` for the parsing assumptions.
        """
        parsed = parse_data_package(doc, rpd_id=rpd_id, port_id=port_id)
        nodes = {
            nid: AmpNode(
                ampId=nid,
                parentId=pn.parentId,
                children=list(pn.children),
                distanceFromRpdMeters=pn.distanceFromRpdMeters,
                deviceType=pn.deviceType,
                name=pn.name,
            )
            for nid, pn in parsed["nodes"].items()
        }
        return cls(rpdId=rpd_id, portId=port_id, amps=nodes)

    @property
    def amp_ids(self) -> list[str]:
        """Ids of measured (RfAmp) devices only — passives/root are excluded."""
        return sorted(nid for nid, n in self.amps.items() if n.deviceType == "AMP")

    def ref_for(self, node_id: str):
        """Build a wire-compatible ref for a node.

        Passive nodes stay in the graph for common-point analysis, but the public MCP
        contract permits only measurable RPD/AMP references. Walk upstream to the nearest
        measurable ancestor when the internal common point is passive.
        """
        node = self.amps.get(node_id)
        seen: set[str] = set()
        while node is not None and node.ampId not in seen:
            seen.add(node.ampId)
            if node.deviceType == "RPD":
                return RpdDeviceRef(rpdId=self.rpdId, portId=self.portId)
            if node.deviceType == "AMP":
                return AmpDeviceRef(ampId=node.ampId)
            node = self.amps.get(node.parentId) if node.parentId else None
        return RpdDeviceRef(rpdId=self.rpdId, portId=self.portId)

    def path_to_rpd(self, amp_id: str) -> list[str]:
        """Amp ids from ``amp_id`` up to (and excluding) the RPD root."""
        path, cur, seen = [], amp_id, set()
        while cur and cur in self.amps and cur not in seen:
            seen.add(cur)
            path.append(cur)
            cur = self.amps[cur].parentId
        return path


class GraphLocalizer:
    def localize(
        self,
        rpd_id: str,
        port_id: str,
        impairment_type: ImpairmentLabel,
        classifications: list[Classification],
        topology: Topology,
    ) -> LocalizationResult:
        amp_class = {c.ampId: c for c in classifications if c.deviceType == "AMP" and c.ampId}
        impaired = {aid for aid, c in amp_class.items() if c.status == "impaired"}
        clean = {aid for aid, c in amp_class.items() if c.status == "clean"}
        measured = set(amp_class)
        # measured-capable devices (amps) in topology that we never got a measurement for
        uncertain = [a for a in topology.amp_ids if a not in measured]

        # Conflicting / multi-anomaly: impaired devices carry more than one distinct
        # non-Clean label (across RPD + amps). Common-point analysis assumes a single
        # propagating fault, so disagreeing labels mean we cannot trust the boundary ->
        # report low_confidence and recommend human escalation.
        impaired_labels = {
            c.klass
            for c in classifications
            if c.status == "impaired" and c.klass != ImpairmentLabel.Clean
        }
        if len(impaired_labels) > 1:
            return self._conflicting(
                impairment_type, impaired_labels, amp_class, impaired, uncertain
            )

        if not impaired:
            return LocalizationResult(
                impairmentType=impairment_type,
                localizationStatus="low_confidence",
                confidence=0.2,
                candidateLocations=[],
                uncertainDevices=[AmpDeviceRef(ampId=a) for a in uncertain],
                recommendedNextAction=RecommendedNextAction(
                    action="re-measure",
                    reason="No impaired amp confirmed; impairment may be intermittent.",
                ),
            )

        # Common point: deepest amp that is an ancestor-or-self of every impaired amp.
        common = self._common_point(topology, impaired)

        supporting = [AmpDeviceRef(ampId=a) for a in sorted(impaired)]
        # Clean boundary = clean amps immediately upstream of the impaired cluster.
        clean_boundary = self._clean_boundary(topology, impaired, clean)
        if common and common in clean:
            # impairment enters below a clean device -> span between them
            likely = LikelySourceLocation(
                locationType="span",
                description=f"Span downstream of clean amp {common} feeding impaired amps {sorted(impaired)}",
                upstreamBoundaryDevice=topology.ref_for(common),
                downstreamBoundaryDevice=AmpDeviceRef(ampId=sorted(impaired)[0]),
            )
            loc_type, desc, score = "span", likely.description, 0.85
        elif len(impaired) == 1:
            only = next(iter(impaired))
            likely = LikelySourceLocation(
                locationType="device",
                description=f"Single impaired amp {only}",
                downstreamBoundaryDevice=AmpDeviceRef(ampId=only),
            )
            loc_type, desc, score = "device", likely.description, 0.9
        else:
            anchor = common or sorted(impaired)[0]
            likely = LikelySourceLocation(
                locationType="branch",
                description=f"Branch at/under {anchor} (common point of impaired amps {sorted(impaired)})",
                upstreamBoundaryDevice=topology.ref_for(anchor),
            )
            loc_type, desc, score = "branch", likely.description, 0.8

        # Lower confidence if we are missing measurements that could change the boundary.
        confidence = score - (0.15 if uncertain else 0.0)
        status = "localized" if confidence >= 0.6 and not uncertain else "low_confidence"

        candidates = [CandidateLocation(locationType=loc_type, description=desc, score=round(confidence, 2), reason="common-point analysis")]
        if uncertain:
            candidates.append(
                CandidateLocation(
                    locationType="device",
                    description=f"Unmeasured amps could shift the boundary: {sorted(uncertain)}",
                    score=0.3,
                    reason="missing measurements",
                )
            )

        next_action = None
        if uncertain:
            next_action = RecommendedNextAction(
                action="re-measure",
                reason="Measure the unmeasured amps to tighten the boundary.",
                targetDevices=sorted(uncertain),
            )

        return LocalizationResult(
            impairmentType=impairment_type,
            localizationStatus=status,
            confidence=round(confidence, 2),
            candidateLocations=candidates,
            likelySourceLocation=likely,
            supportingDevices=supporting,
            cleanBoundaryDevices=[AmpDeviceRef(ampId=a) for a in clean_boundary],
            uncertainDevices=[AmpDeviceRef(ampId=a) for a in uncertain],
            recommendedNextAction=next_action,
        )

    def _conflicting(
        self,
        impairment_type: ImpairmentLabel,
        impaired_labels: set[ImpairmentLabel],
        amp_class: dict[str, Classification],
        impaired: set[str],
        uncertain: list[str],
    ) -> LocalizationResult:
        labels = sorted(l.value for l in impaired_labels)
        return LocalizationResult(
            impairmentType=impairment_type,
            localizationStatus="low_confidence",
            confidence=0.3,
            candidateLocations=[
                CandidateLocation(
                    locationType="branch",
                    description=f"Multiple impairment types present: {labels}; common-point analysis is unreliable.",
                    score=0.3,
                    reason="conflicting classifications",
                )
            ],
            supportingDevices=[AmpDeviceRef(ampId=a) for a in sorted(impaired)],
            uncertainDevices=[AmpDeviceRef(ampId=a) for a in uncertain],
            recommendedNextAction=RecommendedNextAction(
                action="escalate",
                reason=f"Conflicting impairment labels {labels} across impaired devices; hand off to a human.",
                targetDevices=sorted(impaired),
            ),
        )

    @staticmethod
    def _common_point(topology: Topology, impaired: set[str]) -> Optional[str]:
        """LCA of the impaired amps toward the RPD root (the shared upstream point)."""
        paths = [list(reversed(topology.path_to_rpd(a))) for a in impaired]  # root->amp
        if not paths:
            return None
        common = None
        for tup in zip(*paths):
            if all(x == tup[0] for x in tup):
                common = tup[0]
            else:
                break
        return common

    @staticmethod
    def _clean_boundary(topology: Topology, impaired: set[str], clean: set[str]) -> list[str]:
        boundary = set()
        for a in impaired:
            parent = topology.amps[a].parentId if a in topology.amps else None
            if parent in clean:
                boundary.add(parent)
        return sorted(boundary)
