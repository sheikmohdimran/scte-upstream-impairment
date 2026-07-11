"""Grade a live SLM (gemma4 via an OpenAI-compatible endpoint) on generated scenarios.

Generates randomized upstream-impairment cases with known ground truth, runs each through the
LangGraph SLM agent, and scores orchestration quality against the rubric in
``uil.eval.grader``. The deterministic ``Orchestrator`` is validated separately by
``tests/test_eval_harness.py``; this script is the *SLM under test*.

Usage (endpoint must be up):

    export SLM_BASE_URL=http://0.0.0.0:8000/v1
    export SLM_MODEL=google/gemma-4-E4B-it
    export OPENAI_API_KEY=EMPTY
    PYTHONPATH=src python3 examples/eval_slm.py --per-dimension 2 --seed 0

Options:
    --seed N              generator seed (default 0)
    --per-dimension N     cases per fault dimension (default 2)
    --dimensions a,b,c    restrict to a subset of dimensions
    --verbose             print per-criterion detail for every case
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict

# Allow running straight from the repo without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uil.agent.langgraph_agent import LangGraphAgent  # noqa: E402
from uil.eval.generator import DIMENSIONS, GeneratedCase, ScenarioGenerator  # noqa: E402
from uil.eval.grader import CaseVerdict, grade_case  # noqa: E402
from uil.mcp_server.server import MockMcpServer  # noqa: E402


def _sum_tokens(messages: list) -> tuple[int, int, int, int]:
    """Sum input/output/total token usage and count LLM turns across the run's messages.

    Relies on LangChain ``AIMessage.usage_metadata`` (populated when the OpenAI-compatible
    endpoint returns usage). Returns zeros when the provider omits usage.
    """
    inp = out = tot = turns = 0
    for m in messages:
        if m.__class__.__name__ == "AIMessage":
            turns += 1
        um = getattr(m, "usage_metadata", None)
        if um:
            inp += int(um.get("input_tokens", 0) or 0)
            out += int(um.get("output_tokens", 0) or 0)
            tot += int(um.get("total_tokens", 0) or 0)
    return inp, out, tot, turns


def _run_slm(case: GeneratedCase) -> tuple[CaseVerdict, dict]:
    server = MockMcpServer(case.scenario)
    agent = LangGraphAgent(server, scenario_name=case.ground_truth.case_id)
    t0 = time.perf_counter()
    result = agent.run_from_alarm(case.alarm)
    latency = time.perf_counter() - t0
    # The agent gates the alarm deterministically; a not_triggered run == noCall.
    trigger_verdict = "noCall" if result.trace.final_status == "not_triggered" else "call"
    verdict = grade_case(
        case.ground_truth,
        trigger_verdict=trigger_verdict,
        trace=result.trace,
        localization=result.localization,
        final_status=result.trace.final_status,
    )
    inp, out, tot, turns = _sum_tokens(result.messages)
    metrics = {
        "latency_s": latency, "input_tokens": inp, "output_tokens": out,
        "total_tokens": tot, "llm_turns": turns, "tool_calls": len(result.trace.calls),
        "triggered": trigger_verdict == "call",
    }
    return verdict, metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade a live SLM on generated scenarios.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-dimension", type=int, default=2)
    ap.add_argument("--dimensions", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not (os.getenv("SLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")):
        print("Set SLM_BASE_URL (or OPENAI_BASE_URL), SLM_MODEL and OPENAI_API_KEY first.",
              file=sys.stderr)
        return 2

    dims = [d.strip() for d in args.dimensions.split(",") if d.strip()] or list(DIMENSIONS)
    gen = ScenarioGenerator(seed=args.seed)
    cases = gen.generate_suite(per_dimension=args.per_dimension, dimensions=dims)

    model = os.getenv("SLM_MODEL", "local-model")
    print(f"Grading SLM={model!r} on {len(cases)} cases "
          f"(seed={args.seed}, {args.per_dimension}/dimension)\n")

    by_dim_pass: dict[str, int] = defaultdict(int)
    by_dim_total: dict[str, int] = defaultdict(int)
    soft_counts: dict[str, int] = defaultdict(int)
    verdicts: list[CaseVerdict] = []
    metrics_all: list[dict] = []

    for case in cases:
        gt = case.ground_truth
        by_dim_total[gt.dimension] += 1
        try:
            verdict, metrics = _run_slm(case)
        except Exception as exc:  # noqa: BLE001 - report, keep going
            print(f"  [ERROR] {gt.case_id}: {type(exc).__name__}: {exc}")
            verdicts.append(CaseVerdict(gt.case_id, gt.dimension, False, [], 0, "error"))
            continue
        verdicts.append(verdict)
        metrics_all.append(metrics)
        if verdict.passed:
            by_dim_pass[gt.dimension] += 1
        for name in verdict.soft_failures:
            soft_counts[name] += 1

        mark = "PASS" if verdict.passed else "FAIL"
        line = (f"  [{mark}] {gt.case_id:<26} status={verdict.final_status:<14} "
                f"calls={verdict.tool_calls} lat={metrics['latency_s']:.2f}s "
                f"tok={metrics['total_tokens']}")
        if not verdict.passed:
            line += f"  hard_fail={verdict.hard_failures}"
        print(line)
        if args.verbose or not verdict.passed:
            for c in verdict.criteria:
                if not c.passed:
                    tier = "HARD" if c.hard else "soft"
                    print(f"          - {tier} {c.name}: {c.detail}")

    # ---- summary ---------------------------------------------------------------------------
    total_pass = sum(v.passed for v in verdicts)
    print("\n=== SUMMARY BY DIMENSION ===")
    for d in dims:
        t = by_dim_total[d]
        if not t:
            continue
        p = by_dim_pass[d]
        print(f"  {d:<22} {p}/{t}")
    print(f"\nOVERALL: {total_pass}/{len(verdicts)} cases passed "
          f"({100.0 * total_pass / max(1, len(verdicts)):.1f}%)")
    if soft_counts:
        print("\nSoft-criterion misses (non-gating):")
        for name, n in sorted(soft_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {n}")

    # ---- efficiency (latency + tokens) -----------------------------------------------------
    if metrics_all:
        triggered = [m for m in metrics_all if m["triggered"]]
        lat = [m["latency_s"] for m in metrics_all]
        lat_trig = [m["latency_s"] for m in triggered] or lat
        tok_trig = [m["total_tokens"] for m in triggered]
        out_trig = [m["output_tokens"] for m in triggered]
        turns_trig = [m["llm_turns"] for m in triggered]
        have_tokens = any(tok_trig)
        print("\n=== EFFICIENCY ===")
        print(f"  cases: {len(metrics_all)} total, {len(triggered)} triggered (ran the SLM)")
        print(f"  latency/resolution  median={statistics.median(lat_trig):.2f}s "
              f"mean={statistics.fmean(lat_trig):.2f}s max={max(lat_trig):.2f}s")
        if turns_trig:
            print(f"  llm turns/resolution  mean={statistics.fmean(turns_trig):.1f}")
        if have_tokens:
            print(f"  tokens/resolution   total mean={statistics.fmean(tok_trig):.0f} "
                  f"(output mean={statistics.fmean(out_trig):.0f})")
            print(f"  tokens total (triggered): {sum(tok_trig)}")
        else:
            print("  tokens/resolution   [endpoint returned no usage_metadata]")

    return 0 if total_pass == len(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
