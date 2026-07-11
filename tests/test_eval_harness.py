"""Offline validation of the eval harness (no live SLM required).

Two things are proven here:
  1. The generator's *ground truth* agrees with the deterministic ``Orchestrator`` (the
     reference implementation). If the reference disagrees, either the ground truth or the
     reference is wrong — both must be fixed before trusting SLM scores.
  2. The grader's mechanics (status normalization, subsequence, id extraction) behave.

The live gemma4 grading lives in ``examples/eval_slm.py`` (needs an endpoint).
"""

from __future__ import annotations

import pytest

from uil.agent.orchestrator import Orchestrator
from uil.eval.generator import DIMENSIONS, ScenarioGenerator
from uil.eval.grader import (
    CriterionResult,
    grade_case,
    normalize_status,
)
from uil.eval.grader import _device_ids, _is_subsequence  # noqa: PLC2701 (internal helpers)
from uil.mcp_server.server import MockMcpServer


def _run_reference(case):
    """Run the deterministic Orchestrator and adapt its result for the grader."""
    server = MockMcpServer(case.scenario)
    result = Orchestrator(server, scenario_name=case.ground_truth.case_id).run_from_alarm(case.alarm)
    trigger_verdict = result.trigger.verdict if result.trigger else "noCall"
    return grade_case(
        case.ground_truth,
        trigger_verdict=trigger_verdict,
        trace=result.trace,
        localization=server.resolve_localization(result.localization),
        final_status=result.status,
    )


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_reference_orchestrator_matches_ground_truth(seed: int) -> None:
    """Every generated case must PASS when driven by the deterministic reference."""
    gen = ScenarioGenerator(seed=seed)
    cases = gen.generate_suite(per_dimension=2)
    failures = []
    for case in cases:
        verdict = _run_reference(case)
        if not verdict.passed:
            failures.append((case.ground_truth.case_id, verdict.hard_failures,
                             [c.detail for c in verdict.criteria if c.hard and not c.passed]))
    assert not failures, f"reference disagreed with ground truth: {failures}"


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_each_dimension_is_generatable_and_sound(dimension: str) -> None:
    gen = ScenarioGenerator(seed=99)
    case = gen.generate_case(dimension, index=0)
    assert case.ground_truth.dimension == dimension
    verdict = _run_reference(case)
    assert verdict.passed, (
        f"{dimension}: hard failures {verdict.hard_failures} "
        f"({[c.detail for c in verdict.criteria if c.hard and not c.passed]})"
    )


def test_reference_soft_criteria_are_clean_on_happy_path() -> None:
    """On the deterministic reference, soft criteria (sequence/recovery/efficiency/status) hold.

    Covers the localizable paths plus the control paths so ground-truth generation bugs
    (e.g. expecting low_confidence when no failure was actually injected) are caught.
    """
    gen = ScenarioGenerator(seed=5)
    for dim in ("single_device", "branch", "span", "partial_failure", "clean",
                "transient_recovery", "transient_fail", "non_transient_error", "not_triggered"):
        case = gen.generate_case(dim)
        verdict = _run_reference(case)
        assert verdict.passed
        # the reference should be tidy: no soft failures on these deterministic paths
        assert not verdict.soft_failures, f"{dim}: soft failures {verdict.soft_failures}"


# --- grader unit tests ----------------------------------------------------------------------
def test_normalize_status_buckets() -> None:
    assert normalize_status("localized") == "localized"
    assert normalize_status("low_confidence") == "low_confidence"
    assert normalize_status("escalated") == "unresolved"
    assert normalize_status("failed") == "unresolved"
    assert normalize_status(None) == "unresolved"
    assert normalize_status("not_triggered") == "not_triggered"


def test_subsequence_is_retry_tolerant() -> None:
    chain = [
        "getRPDSpectrumMeasurements", "analyzeSpectrumMeasurements",
        "getAllAmpsInSegment", "getAmpUpstreamSpectrumMeasurements",
        "analyzeSpectrumMeasurements", "localizeUpstreamSpectrumImpairmentSource",
    ]
    assert _is_subsequence(["getRPDSpectrumMeasurements", "analyzeSpectrumMeasurements"], chain)
    assert _is_subsequence(chain, chain)
    assert not _is_subsequence(["localizeUpstreamSpectrumImpairmentSource",
                                "getRPDSpectrumMeasurements"], chain)


def test_device_id_extraction_is_recursive() -> None:
    loc = {"likelySourceLocation": {"upstreamBoundaryDevice": {"ampId": "A2"},
                                    "downstreamBoundaryDevice": {"ampId": "A3"}}}
    assert _device_ids(loc["likelySourceLocation"]) == {"A2", "A3"}


def test_grader_flags_false_localize() -> None:
    """A fabricated 'localized' on a case that should not localize is a hard fail."""
    gen = ScenarioGenerator(seed=3)
    case = gen.generate_case("clean")
    # Simulate a bad agent that (wrongly) reports a confident localization.
    from uil.agent.trace import ToolCallTrace
    trace = ToolCallTrace(scenario="bad")
    verdict = grade_case(
        case.ground_truth, trigger_verdict="call", trace=trace,
        localization={"localizationStatus": "localized"}, final_status="localized",
    )
    assert not verdict.passed
    assert "no_false_localize" in verdict.hard_failures


def test_generator_does_not_import_the_localizer() -> None:
    """Ground truth must be independent of GraphLocalizer.

    Guard against future coupling: the generator must not import the localizer (or its
    Topology helpers), otherwise 'source_match' would risk checking the localizer against
    itself instead of against the independently-planted fault node.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "uil" / "eval" / "generator.py"
    tree = ast.parse(src.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    forbidden = {"uil.localizer.graph_localizer", "uil.localizer.plant_topology", "uil.localizer"}
    leaked = imported & forbidden
    assert not leaked, f"generator must not import the localizer, found: {sorted(leaked)}"
