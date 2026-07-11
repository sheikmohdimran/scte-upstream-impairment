# SLM Evaluation Harness

Randomized, ground-truth-graded evaluation of the tool-calling agent (the SLM) against the
fixed upstream-impairment localization workflow.

- Code: `src/uil/eval/generator.py`, `src/uil/eval/grader.py`
- Offline self-check (no LLM): `tests/test_eval_harness.py`
- Live SLM run: `examples/eval_slm.py` (`make eval-slm`)

## 1. Purpose

The spectrum generator / CNN classifier is still open, so the classifier's *accuracy* cannot
yet be tested on realistic spectra. This harness deliberately targets a **different axis**:
the **orchestration** the SLM owns. It answers:

> Given a plant with a *known planted fault*, does the agent trigger correctly, drive the
> fixed 6-step MCP chain in order, recover from errors / partial results, and steer the
> deterministic localizer to the correct fault location?

Because the classification/localization math is deterministic, we can plant a fault, let the
existing simulator + rule classifier round-trip it, and grade the agent purely on how well it
orchestrates the tools.

## 2. Roles (what is graded vs what is ground truth)

| Component | Role |
|---|---|
| **Generator ground truth** | The correct answer. Derived from the *planted* fault + tree, independently of the localizer. |
| **Deterministic `Orchestrator`** | Reference implementation of the flow. Must always match ground truth (the offline self-check). |
| **`LangGraphAgent` (gemma4)** | System under test. Graded against ground truth. |

The SLM is **never** the oracle. It is the candidate being scored. The localization math it
drives (`GraphLocalizer` inside tool 6) is deterministic and separately unit-tested
(`tests/test_localizer.py`).

## 3. What the SLM is scored on

The SLM is a *tool-calling orchestrator*; it does not compute the fault location itself. It can
only be wrong by mis-orchestrating. The rubric (`grade_case`):

**Hard criteria (all must pass for a case to PASS):**
- `trigger_correct` — right call / noCall decision at the alarm gate.
- `localize_presence` — called tool 6 **iff** localization was warranted.
- `no_false_localize` — did not fabricate a confident `localized` when none was warranted.
- `source_match` — the reported likely source references the planted fault node (only when a
  single fault location is expected).

**Soft criteria (reported, non-gating):**
- `status_match` — normalized final status is acceptable.
- `impairment_match` — reported `impairmentType` equals the planted label (CPD).
- `uncertain_match` — reported uncertain devices include the injected measurement failures.
- `sequence_order` — tool order is a retry-tolerant subsequence of the 6-step chain.
- `recovery` — transient errors retried once; non-transient not looped.
- `efficiency` — tool-call count within the expected band.

## 4. Scenario generation

`ScenarioGenerator(seed)` builds `(alarm, Scenario, GroundTruth)` triples. The topology is a
randomly-shaped amplifier tree emitted as the legacy `Scenario.amps` list
(`{ampId, parentId, label}`); `A1` hangs off the RPD port.

**Fault physics** (mirrors `uil.localizer.graph_localizer`): the RPD is the tree root, amps
hang below via `parentId`. An impairment injected at a node propagates to that node and all of
its **descendants** (their upstream signal routes through the fault point); other branches
stay clean. Therefore the correct common point is the planted node itself (`branch`/`device`),
or the clean parent above several impaired sibling subtrees (`span`).

### Fault dimensions

| Dimension | What it injects | Expected outcome |
|---|---|---|
| `single_device` | fault at one leaf amp | `localized`, source = that leaf |
| `branch` | fault at an internal amp (+ descendants) | `localized`, source = that amp |
| `span` | two of A1's subtrees impaired, A1 clean | `localized`, span below A1 |
| `partial_failure` | one subtree impaired + a clean sibling amp fails to measure | `low_confidence`, failed amp uncertain |
| `clean` | RPD + all amps clean | no localization (intermittent) |
| `non_transient_error` | RPD unreachable | unresolved after 1 call |
| `transient_recovery` | RPD unavailable once, then recovers | `localized` after 1 retry |
| `transient_fail` | RPD stale-only (never recovers) | unresolved after retry |
| `not_triggered` | alarm on wrong entity / non-FEC type | `not_triggered`, 0 tool calls |

## 5. Assumptions and constraints

1. **CPD-only impairments.** CPD is the only label the `SpectrumSimulator` models *and* the
   `RuleClassifier` round-trips back to the same label. Ingress / Wideband / Narrowband are
   detectable but their simulator paths are stubs; ImpulseNoise / Ripple / Suckout classify as
   Clean. Restricting to CPD keeps ground truth honest. When the CNN + full spectrum generator
   land, `_IMPAIRMENTS` can be widened and the multi-label conflict dimension re-added.
2. **Non-CNN path.** The harness uses `MockMcpServer(..., use_cnn_path=False)`, i.e. the
   deterministic simulator + `RuleClassifier`, so the planted label round-trips exactly.
3. **Ground truth is independent of the localizer.** `generator.py` never imports
   `GraphLocalizer` / `Topology`; the expected source is the node the generator planted. An
   import-guard test (`test_generator_does_not_import_the_localizer`) enforces this.
4. **Legacy `amps[]` topology.** The first version uses the parent/child amp tree, not the
   CableLabs `components[]/edges[]` data-package. A data-package generator backend is a natural
   follow-up (exercising `plant_topology.py` + passive common points).
5. **Cross-agent status vocabulary.** The `Orchestrator` emits `failed`; the SLM emits
   `escalated` when it stops without a confident localization. The grader normalizes both to
   `unresolved` (see `normalize_status`).
6. **Determinism.** Both the generator (`random.Random(seed)`) and the mock server
   (`np.random.RandomState(seed)`) are seeded, so a seed reproduces the exact suite.
7. **Trigger gate is deterministic in both agents.** The SLM is not invoked on obvious
   negatives; a `not_triggered` run is treated as a `noCall` verdict for grading.

## 6. Running

Offline self-check (validates ground truth against the deterministic reference; no endpoint):

```bash
make test                      # includes tests/test_eval_harness.py
python3 -m pytest tests/test_eval_harness.py -q
```

Live SLM grading (requires an OpenAI-compatible endpoint):

```bash
export SLM_BASE_URL=http://0.0.0.0:8000/v1
export SLM_MODEL=google/gemma-4-E4B-it
export OPENAI_API_KEY=EMPTY
make eval-slm PER=3 SEED=0
# or: PYTHONPATH=src python3 examples/eval_slm.py --per-dimension 3 --seed 0 --verbose
```

Test to run the offline suite, then the 56‑amp segment through the live SLM to confirm binding fixes the id‑hallucination.

```
python -m pytest -q 2>&1 | tail -4 && echo "=== 56-amp segment via live SLM (bound ids) ===" && SLM_BASE_URL=http://0.0.0.0:8000/v1 SLM_MODEL=google/gemma-4-E4B-it OPENAI_API_KEY=EMPTY UIL_TRACE_DISABLE=1 python examples/demo_real_plant_slm.py --port 1 --fault-amp 0000000214 2>&1 | grep -E "Scoped|tool calls|[0-9]\. get|[0-9]\. analyze|[0-9]\. localize|localizationStatus|top candidate|ORACLE"
```



## 7. Output

Per-case lines plus a per-dimension summary and an overall pass rate; failing cases print the
offending hard/soft criteria. Example (gemma4 `google/gemma-4-E4B-it`, seed 0, 3/dimension):

```
=== SUMMARY BY DIMENSION ===
  single_device          3/3
  branch                 3/3
  span                   3/3
  partial_failure        3/3
  clean                  3/3
  non_transient_error    3/3
  transient_recovery     3/3
  transient_fail         3/3
  not_triggered          3/3

OVERALL: 27/27 cases passed (100.0%)

Soft-criterion misses (non-gating):
  recovery: 3     # gemma4 retries the stale-data error twice instead of once
```

The `recovery` soft miss is a genuine behavioral observation, not a failure: gemma4 retries a
transient stale-data error more than the reference's single retry. Because it still terminates
correctly (`unresolved`), the hard criteria pass.

## 8. Interpreting results

- **Hard-criteria pass rate** is the headline "did the model drive the workflow correctly."
- **Soft-criteria misses** are behavioral signals (over-retrying, extra calls, status wording)
  worth tracking but not gating.
- A `source_match` failure with an empty reported source usually means the agent never reached
  a successful tool 6 (skipped a step, passed the wrong handle, or gave up early) — or, in the
  deterministic reference, that the planted label did not classify as impaired (the signal that
  drove the CPD-only decision).
