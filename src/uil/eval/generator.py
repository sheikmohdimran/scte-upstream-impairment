"""Seeded random scenario generator with independent ground truth.

The generator plants a fault at a *chosen* node/span in a randomly-shaped amplifier tree and
records, from first principles, what a correct agent must do:

  * the trigger decision (from how the alarm is built),
  * whether localization should even be attempted,
  * the impairment type,
  * the device the localizer should point at (the planted fault's common point), and
  * which devices should be reported uncertain (from injected measurement failures).

Ground truth is derived from the *known* topology + planted fault — NOT by running the
``GraphLocalizer`` — so grading does not test the localizer against itself.

Physics recap (see ``uil.localizer.graph_localizer``): the RPD is the tree root; amps hang
below it via ``parentId``. An impairment injected at a node propagates to that node and all of
its **descendants** (their upstream signal routes through the fault point), while other
branches stay clean. The correct "common point" is therefore the planted fault node itself
(``branch``/``device``), or — when the fault sits on the span feeding several sibling
subtrees — the clean parent above them (``span``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from uil.domain.labels import ImpairmentLabel
from uil.mcp_server.server import FaultInjection, Scenario

# CPD is the only impairment with a modelled simulator signature that round-trips through
# the RuleClassifier back to the same label. Other labels are stubbed (they classify as
# Clean or drift to a different class), so restricting to CPD keeps ground truth honest.
_IMPAIRMENTS = [ImpairmentLabel.CPD]

_UPSTREAM_FEC_ALARMS = [
    "highUpstreamFecErrors",
    "highUpstreamCorrectables",
    "highUpstreamUncorrectables",
    "highUpstreamCorrectablesAndUncorrectables",
]

# Canonical happy-path tool order (used by the grader for subsequence checks).
FULL_CHAIN = [
    "getRPDSpectrumMeasurements",
    "analyzeSpectrumMeasurements",
    "getAllAmpsInSegment",
    "getAmpUpstreamSpectrumMeasurements",
    "analyzeSpectrumMeasurements",
    "localizeUpstreamSpectrumImpairmentSource",
]

# The fault dimensions we can generate.
DIMENSIONS = [
    "single_device",
    "branch",
    "span",
    "partial_failure",
    "clean",
    "non_transient_error",
    "transient_recovery",
    "transient_fail",
    "not_triggered",
]


@dataclass
class GroundTruth:
    """The independently-derived correct outcome for one generated case."""

    case_id: str
    dimension: str
    trigger_verdict: str                    # "call" | "noCall"
    expects_localize_call: bool             # should tool 6 be reached?
    acceptable_statuses: set[str]           # normalized statuses that count as correct
    expected_impairment: Optional[str]      # dominant non-Clean label, or None
    fault_origin: Optional[str]             # planted fault node id
    expected_source_ids: list[str]          # device ids the localizer should reference
    expected_source_type: Optional[str]     # "device" | "branch" | "span" | None
    impaired_amps: list[str]                # amps that should classify impaired
    expected_uncertain: list[str]           # amps that should be reported uncertain
    expected_tool_sequence: list[str]       # deterministic reference sequence
    min_tool_calls: int
    max_tool_calls: int
    expected_rpd_calls: int = 1             # for recovery grading


@dataclass
class GeneratedCase:
    scenario: Scenario
    alarm: dict
    ground_truth: GroundTruth
    seed: int = 0
    notes: str = ""


# --------------------------------------------------------------------------------------------
# Tree helpers (operate on the ``Scenario.amps`` list-of-dicts topology).
# --------------------------------------------------------------------------------------------
def _children_map(amps: list[dict]) -> dict[str, list[str]]:
    kids: dict[str, list[str]] = {a["ampId"]: [] for a in amps}
    for a in amps:
        p = a.get("parentId")
        if p in kids:
            kids[p].append(a["ampId"])
    return kids


def _descendants(root: str, kids: dict[str, list[str]]) -> list[str]:
    """``root`` plus every node below it (pre-order)."""
    out, stack = [], [root]
    seen: set[str] = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        stack.extend(kids.get(n, []))
    return out


class ScenarioGenerator:
    """Deterministic, seedable generator of ``GeneratedCase`` triples."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    # -- topology -----------------------------------------------------------------------------
    def _random_tree(self, n_amps: int) -> list[dict]:
        """A random rooted amp tree. ``A1`` hangs off the RPD (``parentId=None``)."""
        amps = [{"ampId": "A1", "parentId": None, "label": "Clean"}]
        for i in range(2, n_amps + 1):
            parent = self.rng.choice(amps)["ampId"]
            amps.append({"ampId": f"A{i}", "parentId": parent, "label": "Clean"})
        return amps

    def _tree_with_branching(self, n_amps: int, min_root_children: int = 2) -> list[dict]:
        """Random tree guaranteed to give the root at least ``min_root_children`` subtrees.

        Needed for ``span`` cases where the clean common point must have >=2 impaired subtrees.
        """
        amps = [{"ampId": "A1", "parentId": None, "label": "Clean"}]
        # Force a few direct children of A1 first.
        forced = min(min_root_children, max(0, n_amps - 1))
        idx = 2
        for _ in range(forced):
            amps.append({"ampId": f"A{idx}", "parentId": "A1", "label": "Clean"})
            idx += 1
        while idx <= n_amps:
            parent = self.rng.choice(amps)["ampId"]
            amps.append({"ampId": f"A{idx}", "parentId": parent, "label": "Clean"})
            idx += 1
        return amps

    # -- alarm --------------------------------------------------------------------------------
    def _alarm(self, rpd_id: str, port_id: str, triggering: bool) -> dict:
        if triggering:
            return {
                "alarmId": f"ALM-{self.rng.randint(10000, 99999)}",
                "alarmType": self.rng.choice(_UPSTREAM_FEC_ALARMS),
                "direction": "upstream",
                "entity": {"type": "rpdPort", "rpdId": rpd_id, "portId": port_id},
            }
        # A hard negative: right alarm family, wrong entity — or right entity, wrong alarm.
        if self.rng.random() < 0.5:
            return {
                "alarmId": f"ALM-{self.rng.randint(10000, 99999)}",
                "alarmType": self.rng.choice(_UPSTREAM_FEC_ALARMS),
                "direction": "upstream",
                "entity": {"type": "cableModem", "macAddress": "00:11:22:33:44:55"},
            }
        return {
            "alarmId": f"ALM-{self.rng.randint(10000, 99999)}",
            "alarmType": "highDownstreamSnr",
            "direction": "downstream",
            "entity": {"type": "rpdPort", "rpdId": rpd_id, "portId": port_id},
        }

    # -- per-dimension builders ---------------------------------------------------------------
    def generate_case(self, dimension: str, index: int = 0) -> GeneratedCase:
        if dimension not in DIMENSIONS:
            raise ValueError(f"unknown dimension {dimension!r}")
        builder = getattr(self, f"_gen_{dimension}")
        rpd_id = f"RPD-{self.seed}-{index:03d}"
        port_id = "P1"
        return builder(rpd_id, port_id, index)

    def generate_suite(self, per_dimension: int = 2,
                       dimensions: Optional[list[str]] = None) -> list[GeneratedCase]:
        dims = dimensions or DIMENSIONS
        cases: list[GeneratedCase] = []
        for d in dims:
            for i in range(per_dimension):
                cases.append(self.generate_case(d, index=len(cases)))
        return cases

    # ----- localizable cases -----------------------------------------------------------------
    def _gen_single_device(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(3, 6)
        amps = self._random_tree(n)
        kids = _children_map(amps)
        leaves = [a["ampId"] for a in amps if not kids[a["ampId"]]]
        fault = self.rng.choice(leaves)
        imp = self.rng.choice(_IMPAIRMENTS)
        self._set_label(amps, fault, imp)
        scn = self._scenario(rpd_id, port_id, imp, amps)
        gt = GroundTruth(
            case_id=f"single_device-{index}", dimension="single_device",
            trigger_verdict="call", expects_localize_call=True,
            acceptable_statuses={"localized"},
            expected_impairment=imp.value, fault_origin=fault,
            expected_source_ids=[fault], expected_source_type="device",
            impaired_amps=[fault], expected_uncertain=[],
            expected_tool_sequence=list(FULL_CHAIN), min_tool_calls=6, max_tool_calls=9,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes=f"leaf fault at {fault}")

    def _gen_branch(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(4, 7)
        amps = self._random_tree(n)
        kids = _children_map(amps)
        internal = [a["ampId"] for a in amps if kids[a["ampId"]]]
        fault = self.rng.choice(internal)
        imp = self.rng.choice(_IMPAIRMENTS)
        impaired = _descendants(fault, kids)
        for aid in impaired:
            self._set_label(amps, aid, imp)
        scn = self._scenario(rpd_id, port_id, imp, amps)
        gt = GroundTruth(
            case_id=f"branch-{index}", dimension="branch",
            trigger_verdict="call", expects_localize_call=True,
            acceptable_statuses={"localized"},
            expected_impairment=imp.value, fault_origin=fault,
            expected_source_ids=[fault], expected_source_type="branch",
            impaired_amps=sorted(impaired), expected_uncertain=[],
            expected_tool_sequence=list(FULL_CHAIN), min_tool_calls=6, max_tool_calls=9,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes=f"branch fault at {fault} -> impaired {sorted(impaired)}")

    def _gen_span(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(5, 8)
        amps = self._tree_with_branching(n, min_root_children=2)
        kids = _children_map(amps)
        # Clean common point = A1; impair two of its subtrees, leave A1 clean.
        subtrees = [c for c in kids["A1"]]
        self.rng.shuffle(subtrees)
        chosen = subtrees[:2]
        imp = self.rng.choice(_IMPAIRMENTS)
        impaired: list[str] = []
        for c in chosen:
            impaired.extend(_descendants(c, kids))
        for aid in impaired:
            self._set_label(amps, aid, imp)
        # A1 stays clean -> localizer reports a span below A1.
        scn = self._scenario(rpd_id, port_id, imp, amps)
        gt = GroundTruth(
            case_id=f"span-{index}", dimension="span",
            trigger_verdict="call", expects_localize_call=True,
            acceptable_statuses={"localized"},
            expected_impairment=imp.value, fault_origin="A1",
            expected_source_ids=["A1"], expected_source_type="span",
            impaired_amps=sorted(impaired), expected_uncertain=[],
            expected_tool_sequence=list(FULL_CHAIN), min_tool_calls=6, max_tool_calls=9,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes=f"span below clean A1 feeding {sorted(impaired)}")

    def _gen_partial_failure(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        # Branch topology so at least one clean amp always exists to fail: impair ONE of A1's
        # subtrees, leaving A1 and the sibling subtree(s) clean.
        n = self.rng.randint(5, 8)
        amps = self._tree_with_branching(n, min_root_children=2)
        kids = _children_map(amps)
        subtrees = list(kids["A1"])
        self.rng.shuffle(subtrees)
        fault = subtrees[0]
        imp = self.rng.choice(_IMPAIRMENTS)
        impaired = set(_descendants(fault, kids))
        for aid in impaired:
            self._set_label(amps, aid, imp)
        # Fail measurement of a CLEAN amp on another branch -> becomes uncertain.
        clean_amps = [a["ampId"] for a in amps if a["ampId"] not in impaired]
        failed = self.rng.choice(clean_amps)
        faults = FaultInjection(amp_failed_ports={failed})
        scn = self._scenario(rpd_id, port_id, imp, amps, faults)
        gt = GroundTruth(
            case_id=f"partial_failure-{index}", dimension="partial_failure",
            trigger_verdict="call", expects_localize_call=True,
            acceptable_statuses={"low_confidence"},
            expected_impairment=imp.value, fault_origin=fault,
            expected_source_ids=[fault], expected_source_type=None,
            impaired_amps=sorted(impaired),
            expected_uncertain=[failed],
            expected_tool_sequence=list(FULL_CHAIN), min_tool_calls=6, max_tool_calls=9,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes=f"fault {fault}; unreachable {failed}")

    # ----- non-localizable / control cases ---------------------------------------------------
    def _gen_clean(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(3, 6)
        amps = self._random_tree(n)  # all Clean
        scn = self._scenario(rpd_id, port_id, ImpairmentLabel.Clean, amps)
        gt = GroundTruth(
            case_id=f"clean-{index}", dimension="clean",
            trigger_verdict="call", expects_localize_call=False,
            acceptable_statuses={"low_confidence", "unresolved"},
            expected_impairment=None, fault_origin=None,
            expected_source_ids=[], expected_source_type=None,
            impaired_amps=[], expected_uncertain=[],
            expected_tool_sequence=["getRPDSpectrumMeasurements", "analyzeSpectrumMeasurements"],
            min_tool_calls=2, max_tool_calls=4,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes="RPD + all amps clean (intermittent)")

    def _gen_non_transient_error(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(3, 6)
        amps = self._random_tree(n)
        imp = self.rng.choice(_IMPAIRMENTS)
        faults = FaultInjection(rpd_unreachable=True)
        scn = self._scenario(rpd_id, port_id, imp, amps, faults)
        gt = GroundTruth(
            case_id=f"non_transient_error-{index}", dimension="non_transient_error",
            trigger_verdict="call", expects_localize_call=False,
            acceptable_statuses={"unresolved"},
            expected_impairment=None, fault_origin=None,
            expected_source_ids=[], expected_source_type=None,
            impaired_amps=[], expected_uncertain=[],
            expected_tool_sequence=["getRPDSpectrumMeasurements"],
            min_tool_calls=1, max_tool_calls=2, expected_rpd_calls=1,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes="RPD unreachable (non-transient)")

    def _gen_transient_recovery(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(4, 7)
        amps = self._random_tree(n)
        kids = _children_map(amps)
        internal = [a["ampId"] for a in amps if kids[a["ampId"]]]
        fault = self.rng.choice(internal)
        imp = self.rng.choice(_IMPAIRMENTS)
        impaired = _descendants(fault, kids)
        for aid in impaired:
            self._set_label(amps, aid, imp)
        faults = FaultInjection(rpd_measurement_unavailable_once=True)
        scn = self._scenario(rpd_id, port_id, imp, amps, faults)
        gt = GroundTruth(
            case_id=f"transient_recovery-{index}", dimension="transient_recovery",
            trigger_verdict="call", expects_localize_call=True,
            acceptable_statuses={"localized"},
            expected_impairment=imp.value, fault_origin=fault,
            expected_source_ids=[fault], expected_source_type="branch",
            impaired_amps=sorted(impaired), expected_uncertain=[],
            expected_tool_sequence=["getRPDSpectrumMeasurements"] + list(FULL_CHAIN),
            min_tool_calls=7, max_tool_calls=10, expected_rpd_calls=2,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes=f"one transient RPD miss then recover; fault {fault}")

    def _gen_transient_fail(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(3, 6)
        amps = self._random_tree(n)
        imp = self.rng.choice(_IMPAIRMENTS)
        faults = FaultInjection(rpd_stale_only=True)
        scn = self._scenario(rpd_id, port_id, imp, amps, faults)
        gt = GroundTruth(
            case_id=f"transient_fail-{index}", dimension="transient_fail",
            trigger_verdict="call", expects_localize_call=False,
            acceptable_statuses={"unresolved"},
            expected_impairment=None, fault_origin=None,
            expected_source_ids=[], expected_source_type=None,
            impaired_amps=[], expected_uncertain=[],
            expected_tool_sequence=["getRPDSpectrumMeasurements", "getRPDSpectrumMeasurements"],
            min_tool_calls=2, max_tool_calls=3, expected_rpd_calls=2,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, True), gt, self.seed,
                             notes="RPD stale-only (retry then give up)")

    def _gen_not_triggered(self, rpd_id: str, port_id: str, index: int) -> GeneratedCase:
        n = self.rng.randint(3, 6)
        amps = self._random_tree(n)
        imp = self.rng.choice(_IMPAIRMENTS)
        scn = self._scenario(rpd_id, port_id, imp, amps)
        gt = GroundTruth(
            case_id=f"not_triggered-{index}", dimension="not_triggered",
            trigger_verdict="noCall", expects_localize_call=False,
            acceptable_statuses={"not_triggered"},
            expected_impairment=None, fault_origin=None,
            expected_source_ids=[], expected_source_type=None,
            impaired_amps=[], expected_uncertain=[],
            expected_tool_sequence=[], min_tool_calls=0, max_tool_calls=0,
            expected_rpd_calls=0,
        )
        return GeneratedCase(scn, self._alarm(rpd_id, port_id, False), gt, self.seed,
                             notes="alarm should not trigger the workflow")

    # -- shared -------------------------------------------------------------------------------
    @staticmethod
    def _set_label(amps: list[dict], amp_id: str, label: ImpairmentLabel) -> None:
        for a in amps:
            if a["ampId"] == amp_id:
                a["label"] = label.value
                return

    @staticmethod
    def _scenario(rpd_id: str, port_id: str, rpd_label: ImpairmentLabel,
                  amps: list[dict], faults: Optional[FaultInjection] = None) -> Scenario:
        return Scenario(
            rpdId=rpd_id, portId=port_id, rpd_label=rpd_label,
            amps=amps, faults=faults or FaultInjection(),
        )
