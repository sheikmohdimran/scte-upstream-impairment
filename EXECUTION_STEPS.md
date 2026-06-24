# Execution Steps — Upstream Impairment Localization (Intel POC)

Progress tracker for the 8 build steps. Update the **Status** and **Notes** columns as work
proceeds. Source of truth for the architecture is `../SCTE-Plan-Corrected.md`; the tool contract
is vendored under `schemas/tools/`.

Status legend: `TODO` · `IN PROGRESS` · `DONE` · `BLOCKED`

---

## Step tracker

| # | Step | Phase | Depends on CableLabs? | Status | Notes |
|---|------|-------|-----------------------|--------|-------|
| 0 | **Schemas as code**: vendor `schemas/tools/**`, Pydantic v2 models that round-trip the schemas, CI schema-conformance test | 0 | No | DONE | Models in `src/uil/domain/`; conformance test in `tests/test_schema_conformance.py` |
| 2 | **Mock MCP server + handle store**: 5 tools on the reference surface (handles + counts), backend handle→raw resolution | 1 | No | DONE | `src/uil/mcp_server/` (in-process server + handle store) |
| 3 | **CPD synthetic generator**: upstream `rawSpectrum` (256 bins, 5–85 MHz); Clean vs CPD (raised noise floor) | 1 | Partial (signature spec from Irene) | DONE | `src/uil/sim/spectrum_simulator.py`; provisional CPD params in `config/cpd_signatures.yaml` |
| 4 | **Bootstrap rule classifier**: noise-floor threshold so chain runs E2E before CNN | 1 | No | DONE | `src/uil/classifier/rule_classifier.py` (transparent scaffolding, not final) |
| 6 | **Graph-theory localizer prototype**: common-point analysis over topology → span/device/branch | 2 | Partial (topology availability) | DONE | `src/uil/localizer/graph_localizer.py`; now also flags conflicting/multi-anomaly labels → escalate |
| 5 | **Orchestrator skeleton**: 6-step sequence + conditional error / partial_success / re-measure / escalate; holds only handles | 3 | No | DONE | `src/uil/agent/orchestrator.py` (deterministic oracle) **+ real LangGraph agent** `src/uil/agent/langgraph_agent.py` (pluggable OpenAI-compatible SLM; fine-tuning skipped) |
| 7 | **Error-injection harness + human-handoff artifact**: force DEVICE_UNREACHABLE / partial_success / conflicting labels; SLM "Diagnosis Summary" markdown | 3 | No | DONE | Fault injection in mock server; `src/uil/agent/handoff.py` |
| 8 | **Tool-call trace format**: fine-tuning trace schema; auto-generate positive/negative traces from sim | 3 | No | DONE | `src/uil/agent/trace.py`; emitted by orchestrator |

> Note: the first build pass (this commit) lands a runnable skeleton of **all 8 steps** end-to-end
> on synthetic data. Subsequent passes deepen each (real CNN, LangGraph, richer fault models).

---

## First demo milestone
End-to-end **CPD localization on synthetic data** through the mock MCP chain, including one injected
`partial_success`. Run: `make demo` (deterministic orchestrator) or `make agent-demo` (real
LangGraph + SLM; needs `SLM_BASE_URL`/`SLM_MODEL`).

## Phase 3 — real SLM + LangGraph (fine-tuning skipped)
- `src/uil/agent/langgraph_agent.py`: LangGraph ReAct agent over the 5 MCP tools, exposed to a
  **pluggable** OpenAI-compatible chat model. The SLM only sees handles + counts and owns the
  off-happy-path decisions (retry / partial_success / escalate).
- Point it at any endpoint: `export SLM_BASE_URL=http://localhost:8000/v1 SLM_MODEL=<name> OPENAI_API_KEY=EMPTY`,
  then `make agent-demo`. The deterministic `Orchestrator` stays as the grading oracle.
- Offline test uses a scripted tool-calling model so the LangGraph loop is covered without a network
  endpoint (`tests/test_langgraph_agent.py`).

## Definition of done per step
- **0**: `pytest tests/test_schema_conformance.py` green — every tool success/error example validates
  against its `.schema.json`, and Pydantic models round-trip.
- **2**: all 5 tools callable; handles resolve; raw stays out of the reference outputs.
- **3**: Clean stays within variance; CPD raises the floor by the configured margin; deterministic seed.
- **4**: classifier separates Clean vs CPD on synthetic spectra; emits `class` + `confidence`.
- **6**: single CPD on a known span → correct `likelySourceLocation` + boundary device sets.
- **5**: full 6-step sequence runs autonomously and returns a localization or an escalation.
- **7**: injected errors trigger recovery / escalation; handoff produces a readable summary.
- **8**: traces emitted in the agreed schema; positive + negative examples generated from sim.

## Open dependencies on CableLabs (track against asks)
- CPD spectral signature spec (Irene) → tightens Step 3.
- `getAllAmpsInSegment` topology (`amps[]` parent/children/distance) (Randy) → confirms Step 6 inputs.
- Trigger contract (SNMP trap / Kafka) (Randy) → Phase 4.
- `impairmentType` selection rule for localize step 6 (Randy/Irene).
- CNN reuse details / sample captures (Irene + Bhaskar) → replaces Step 4.

## Repo publication and verification status
- Git worktree branch for implementation: `impl-plan-2026-06-24`.
- GitHub org creation attempted for `Intel-Sandbox/scte-upstream-impairment` and blocked by org IP allow-list (HTTP 403).
- Fallback publication target used: `sheikmohdimran/scte-upstream-impairment` (private) to keep execution unblocked.
- Consolidated repository documentation completed in `README.md` plus:
  - `docs/architecture/contract-and-data-surfaces.md`
  - `docs/operations/open-decisions.md`
- Validation evidence on this branch:
  - `/home/sdp/vllm-env/bin/python -m pytest tests/test_readme_consolidation.py tests/test_schema_conformance.py tests/test_localizer.py tests/test_agent_e2e.py -q`
  - Result: 29 passed.
