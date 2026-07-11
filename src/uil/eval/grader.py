"""Rubric-based grader: score an agent run against generated ground truth.

The agent under test is a *tool-calling orchestrator*; the classification/localization math is
deterministic. So we grade **orchestration**, not the localizer's arithmetic:

Hard criteria (all must pass for the case to PASS):
  * ``trigger_correct``      — right call/noCall decision.
  * ``localize_presence``    — called tool 6 iff localization was warranted.
  * ``no_false_localize``    — did not fabricate a confident ``localized`` when none was warranted.
  * ``source_match``         — the reported likely source references the planted fault
                               (only when a single fault location is expected).

Soft criteria (reported, not gating):
  * ``status_match``   — normalized final status is acceptable.
  * ``impairment_match`` — reported impairmentType matches the planted label.
  * ``uncertain_match`` — the low-confidence result emits an ``uncertainDevicesRef`` handle.
  * ``sequence_order`` — tool order is a valid (retry-tolerant) subsequence of the 6-step chain.
  * ``recovery``       — transient errors retried once; non-transient not looped.
  * ``efficiency``     — tool-call count within the expected band.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from uil.agent.trace import ToolCallTrace
from uil.eval.generator import FULL_CHAIN, GroundTruth

_LOCALIZE_TOOL = "localizeUpstreamSpectrumImpairmentSource"
_RPD_TOOL = "getRPDSpectrumMeasurements"


def normalize_status(status: Optional[str]) -> str:
    """Collapse the two agents' vocabularies into shared buckets.

    Orchestrator emits ``failed``; the SLM emits ``escalated`` when it stops without a
    confident localization. Both mean "no localization produced" -> ``unresolved``.
    """
    s = (status or "").lower()
    if s == "localized":
        return "localized"
    if s == "low_confidence":
        return "low_confidence"
    if s == "not_triggered":
        return "not_triggered"
    if s in {"failed", "escalated", ""}:
        return "unresolved"
    return s


@dataclass
class CriterionResult:
    name: str
    passed: bool
    hard: bool
    detail: str = ""


@dataclass
class CaseVerdict:
    case_id: str
    dimension: str
    passed: bool
    criteria: list[CriterionResult]
    tool_calls: int
    final_status: str
    notes: str = ""

    @property
    def hard_failures(self) -> list[str]:
        return [c.name for c in self.criteria if c.hard and not c.passed]

    @property
    def soft_failures(self) -> list[str]:
        return [c.name for c in self.criteria if not c.hard and not c.passed]


def _device_ids(obj: Any) -> set[str]:
    """Recursively collect ampId/rpdId values from a nested localization dict."""
    ids: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"ampId", "rpdId"} and isinstance(v, str):
                ids.add(v)
            else:
                ids |= _device_ids(v)
    elif isinstance(obj, list):
        for item in obj:
            ids |= _device_ids(item)
    return ids


def _has_key(obj: Any, key: str) -> bool:
    """Recursively test whether ``key`` appears anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key(item, key) for item in obj)
    return False


def _collapse_consecutive(seq: list[str]) -> list[str]:
    out: list[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _is_subsequence(sub: list[str], full: list[str]) -> bool:
    it = iter(full)
    return all(any(x == f for f in it) for x in sub)


def grade_case(
    gt: GroundTruth,
    trigger_verdict: str,
    trace: ToolCallTrace,
    localization: Optional[dict],
    final_status: Optional[str] = None,
) -> CaseVerdict:
    """Grade one agent run against its ground truth."""
    tool_seq = [c.tool for c in trace.calls]
    n_calls = len(tool_seq)
    status = normalize_status(final_status if final_status is not None else trace.final_status)
    localize_called = _LOCALIZE_TOOL in tool_seq
    loc_status = (localization or {}).get("localizationStatus")

    crit: list[CriterionResult] = []

    # --- HARD -------------------------------------------------------------------------------
    crit.append(CriterionResult(
        "trigger_correct", trigger_verdict == gt.trigger_verdict, True,
        f"got {trigger_verdict!r}, expected {gt.trigger_verdict!r}",
    ))

    crit.append(CriterionResult(
        "localize_presence", localize_called == gt.expects_localize_call, True,
        f"localize_called={localize_called}, expected={gt.expects_localize_call}",
    ))

    # Must not report a confident localization when none is warranted.
    false_localize = (not gt.expects_localize_call) and loc_status == "localized"
    crit.append(CriterionResult(
        "no_false_localize", not false_localize, True,
        "fabricated localized result" if false_localize else "ok",
    ))

    # Source match only when a single fault location is expected.
    if gt.expected_source_ids and gt.expects_localize_call:
        # Boundary devices live in candidateLocations[].downstreamBoundaryDevices (the
        # confirmed/supporting devices); evidence handles are opaque, so scan candidates.
        reported = _device_ids((localization or {}).get("candidateLocations"))
        hit = any(sid in reported for sid in gt.expected_source_ids)
        crit.append(CriterionResult(
            "source_match", hit, True,
            f"expected one of {gt.expected_source_ids}, reported {sorted(reported)}",
        ))

    # --- SOFT -------------------------------------------------------------------------------
    crit.append(CriterionResult(
        "status_match", status in gt.acceptable_statuses, False,
        f"got {status!r}, acceptable {sorted(gt.acceptable_statuses)}",
    ))

    if gt.expected_impairment and localization:
        got_imp = localization.get("impairmentType")
        crit.append(CriterionResult(
            "impairment_match", got_imp == gt.expected_impairment, False,
            f"got {got_imp!r}, expected {gt.expected_impairment!r}",
        ))

    if gt.expected_uncertain and localization:
        # Uncertain devices are exposed only as an opaque handle (uncertainDevicesRef); the
        # SLM-facing surface no longer inlines their ids. Assert the handle is emitted so the
        # low-confidence result acknowledges the unmeasured/failed devices.
        has_ref = _has_key(localization, "uncertainDevicesRef")
        crit.append(CriterionResult(
            "uncertain_match", has_ref, False,
            "uncertainDevicesRef handle present" if has_ref
            else "expected an uncertainDevicesRef handle, none emitted",
        ))

    collapsed = _collapse_consecutive(tool_seq)
    seq_ok = _is_subsequence(collapsed, FULL_CHAIN) if collapsed else (gt.min_tool_calls == 0)
    crit.append(CriterionResult(
        "sequence_order", seq_ok, False,
        f"sequence {collapsed} not a subsequence of the 6-step chain" if not seq_ok else "ok",
    ))

    rpd_calls = tool_seq.count(_RPD_TOOL)
    recovery_ok = rpd_calls == gt.expected_rpd_calls
    crit.append(CriterionResult(
        "recovery", recovery_ok, False,
        f"{_RPD_TOOL} called {rpd_calls}x, expected {gt.expected_rpd_calls}x",
    ))

    eff_ok = gt.min_tool_calls <= n_calls <= gt.max_tool_calls
    crit.append(CriterionResult(
        "efficiency", eff_ok, False,
        f"{n_calls} tool calls, band [{gt.min_tool_calls}, {gt.max_tool_calls}]",
    ))

    passed = all(c.passed for c in crit if c.hard)
    return CaseVerdict(
        case_id=gt.case_id, dimension=gt.dimension, passed=passed,
        criteria=crit, tool_calls=n_calls, final_status=status,
    )
