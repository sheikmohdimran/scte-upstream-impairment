"""Trigger-gate evaluation against the real ``eval_CPD`` alarm set.

Each case in ``tests/fixtures/alarms_eval.json`` carries an ``expectedVerdict``
(``call``/``noCall``) and an ``expectedToolCall`` (the workflow's first tool call, or
``null``). The gate must reproduce both exactly — including the hard negatives that share
fields with the positives (upstream FEC alarms on a cable modem, non-FEC upstream alarms on
an RPD port).
"""

import json
from pathlib import Path

import pytest

from uil.agent.trigger import AlarmTrigger

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "alarms_eval.json"


def _load_cases() -> list[dict]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return data["evalCases"]


CASES = _load_cases()


def test_fixture_has_positive_and_negative_cases() -> None:
    verdicts = {c["expectedVerdict"] for c in CASES}
    assert verdicts == {"call", "noCall"}
    assert len(CASES) == 26


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_trigger_matches_expected_verdict(case: dict) -> None:
    decision = AlarmTrigger().evaluate(case["alarm"])

    assert decision.verdict == case["expectedVerdict"], (
        f"{case['id']} ({case['description']}): "
        f"got {decision.verdict!r} — {decision.reason}"
    )

    expected_tool = case["expectedToolCall"]
    if expected_tool is None:
        assert decision.tool_call is None
    else:
        assert decision.tool_call == {
            "name": expected_tool["name"],
            "arguments": expected_tool["arguments"],
        }, f"{case['id']}: first tool call mismatch"


def test_gate_is_perfect_on_the_eval_set() -> None:
    trigger = AlarmTrigger()
    wrong = [
        c["id"]
        for c in CASES
        if trigger.evaluate(c["alarm"]).verdict != c["expectedVerdict"]
    ]
    assert not wrong, f"misclassified cases: {wrong}"
