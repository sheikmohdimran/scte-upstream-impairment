"""SLM evaluation harness: random scenario generation + ground-truth grading.

This package builds randomized upstream-impairment scenarios with a *known planted fault*
(the ground truth), then grades an agent's tool-calling behaviour against that truth.

- :mod:`uil.eval.generator` — seeded random topology + fault injection + ground truth.
- :mod:`uil.eval.grader`    — rubric-based per-case verdict (trigger / sequence / recovery /
  localization) independent of the deterministic localizer's internal math.

The classification/localization math is deterministic, so this harness tests *orchestration*:
did the agent trigger correctly, call the fixed 6-step chain in order, recover from
errors/partial results, and drive the localizer to the correct planted fault?
"""

from uil.eval.generator import GeneratedCase, GroundTruth, ScenarioGenerator
from uil.eval.grader import CaseVerdict, CriterionResult, grade_case, normalize_status

__all__ = [
    "GeneratedCase",
    "GroundTruth",
    "ScenarioGenerator",
    "CaseVerdict",
    "CriterionResult",
    "grade_case",
    "normalize_status",
]
